"""Matryoshka Medusa head with BitNet-style TERNARY weight quantization.

Per Gemini's suggestion: we store heads as F16 (16 bpw). BitNet base is I2_S
(~2 bpw). Training heads natively as ternary + matryoshka gives us:
  - 8× memory reduction vs F16 (direct CPU bandwidth win)
  - Dynamic rank via matryoshka
  - Combined: ~65× bandwidth reduction at rank 256 vs full F16 head

Uses straight-through estimator for gradients: forward sees quantized weights,
backward passes through as if unquantized.
"""
import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train import CachedHiddenDataset, collate_cached
from torch.utils.data import DataLoader

SEQ_LEN = 2048; HIDDEN = 2560; VOCAB = 128256
RANKS = [64, 256, 1024, 2560]
RANK_WEIGHTS = {64: 3.0, 256: 2.0, 1024: 1.0, 2560: 1.0}
RANKS_PER_STEP = 2


def weight_quant_ste(w):
    """BitNet-style ternary quantization with straight-through estimator.

    Quantize to {-s, 0, +s} where s is chosen by abs-mean scaling.
    Forward uses quantized; backward passes through as if identity.
    """
    # Per-matrix scale (absmean scaling, matches BitNet convention)
    s = 1.0 / w.abs().mean().clamp_min(1e-5)
    w_q = (w * s).round().clamp(-1, 1) / s
    # Straight-through: gradient flows through as if w_q == w
    return w + (w_q - w).detach()


class TernaryMatryoshkaHead(nn.Module):
    def __init__(self, hidden_size, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_in = nn.Parameter(torch.empty(hidden_size, hidden_size, dtype=dtype))
        self.W_out = nn.Parameter(torch.zeros(hidden_size, hidden_size, dtype=dtype))
        nn.init.kaiming_uniform_(self.W_in, a=5 ** 0.5)

    def forward_at_rank(self, hidden, rank, lm_head_weight, pos_indices=None,
                        quantize=True):
        x = hidden
        W_in_r = self.W_in[:, :rank]
        W_out_r = self.W_out[:rank, :]
        if quantize:
            W_in_r = weight_quant_ste(W_in_r)
            W_out_r = weight_quant_ste(W_out_r)
        hidden_proj = F.silu(x @ W_in_r)
        residual = hidden_proj @ W_out_r
        x_out = x + residual

        if pos_indices is not None:
            B, T, H = x_out.shape
            idx = pos_indices.view(B, -1, 1).expand(-1, -1, H)
            x_out = torch.gather(x_out, 1, idx)

        if lm_head_weight.shape[0] > lm_head_weight.shape[1]:
            w = lm_head_weight.t().contiguous()
        else:
            w = lm_head_weight
        return x_out @ w


def ternary_matryoshka_loss(head, hidden, targets, lm_head_weight, shift=2,
                             loss_positions=256):
    B, T, H = hidden.shape
    max_valid = T - shift - 3
    if loss_positions > 0 and loss_positions < max_valid:
        perm = torch.randperm(max_valid, device=hidden.device)[:loss_positions]
        pos_indices = perm.sort().values.unsqueeze(0).expand(B, -1).contiguous()
    else:
        pos_indices = None

    sampled = [max(RANKS)]
    others = [r for r in RANKS if r != max(RANKS)]
    np.random.shuffle(others)
    sampled += others[:RANKS_PER_STEP - 1]

    total = torch.zeros((), dtype=hidden.dtype, device=hidden.device)
    accs = {}
    for R in sampled:
        logits = head.forward_at_rank(hidden, R, lm_head_weight, pos_indices,
                                       quantize=True)
        if pos_indices is not None:
            target_sel = torch.gather(targets, 1, pos_indices + shift)
        else:
            target_sel = targets[:, shift:shift + logits.shape[1]]
            logits = logits[:, :target_sel.shape[1]]
        loss_R = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), target_sel.reshape(-1))
        total = total + RANK_WEIGHTS[R] * loss_R
        with torch.no_grad():
            accs[R] = (logits.argmax(-1) == target_sel).float().mean().item()
    return total / sum(RANK_WEIGHTS[r] for r in sampled), accs


def main():
    torch.manual_seed(0)
    device = "cpu"; dtype = torch.bfloat16

    lm_head = torch.load("data/lm_head.pt", map_location=device,
                         weights_only=True).to(dtype).t().contiguous()

    # Full-scale cache
    ds = CachedHiddenDataset(
        "data/hidden_gguf_v2.bin", "data/tokens.bin", SEQ_LEN, HIDDEN)
    loader = DataLoader(ds, batch_size=1, shuffle=True,
                        collate_fn=collate_cached, num_workers=0, drop_last=True)
    print(f"[ternary] dataset: {len(ds)} seqs (pilot)")

    head = TernaryMatryoshkaHead(HIDDEN, dtype=dtype).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=0)

    max_steps = 800; grad_accum = 4; log_every = 25
    step, accum, loss_accum = 0, 0, 0.0
    t0 = time.time()
    di = iter(loader)
    print(f"[ternary] ranks {RANKS} | {max_steps} steps, grad_accum={grad_accum}")

    while step < max_steps:
        try:
            batch = next(di)
        except StopIteration:
            di = iter(loader); batch = next(di)
        hidden, targets = batch
        hidden = hidden.to(device); targets = targets.to(device)
        with torch.cpu.amp.autocast(dtype=torch.bfloat16):
            loss, accs = ternary_matryoshka_loss(head, hidden, targets, lm_head)
            (loss / grad_accum).backward()
        loss_accum += loss.item(); accum += 1
        if accum < grad_accum: continue
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        step += 1; accum = 0
        if step % log_every == 0:
            dt = time.time() - t0
            avg = loss_accum / (log_every * grad_accum); loss_accum = 0; t0 = time.time()
            acc_str = "  ".join(f"R{r}:{accs[r]:.2f}" if r in accs else f"R{r}:--"
                                for r in RANKS)
            print(f"step {step:4d} | loss {avg:.3f} | {dt/log_every:.2f}s | {acc_str}",
                  flush=True)

    os.makedirs("checkpoints/ternary_full", exist_ok=True)
    torch.save({"W_in": head.W_in.detach(), "W_out": head.W_out.detach(),
                "ranks": RANKS, "ternary": True},
               "checkpoints/ternary_full/head_step800.pt")
    print("[ternary] saved full")


if __name__ == "__main__":
    main()
