"""Matryoshka Medusa head pilot.

Train a single Medusa head with nested-rank matryoshka loss. At each step,
compute the loss at multiple rank operating points and sum them. This forces
the first R components to form a valid predictor for each R, so truncating
at any rank yields a working head.

If successful: pilot head at rank 32 should beat SVD-truncated rank 32 by
a large margin. That's the breakthrough that enables dynamic-rank routing.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import PackedTokenDataset
from train import CachedHiddenDataset, collate_cached
from torch.utils.data import DataLoader

SEQ_LEN = 2048
HIDDEN = 2560
VOCAB = 128256

# Nested ranks: train to work at each of these
RANKS = [64, 256, 1024, 2560]
# Loss weight per rank (give lower ranks more weight to prioritize compression)
RANK_WEIGHTS = {64: 3.0, 256: 2.0, 1024: 1.0, 2560: 1.0}
# Stochastic: each step, sample K ranks from the set (cheaper than all-ranks)
RANKS_PER_STEP = 2


class MatryoshkaMedusaHead(nn.Module):
    """Single matryoshka head with nested-rank forward.

    W_in: [H, H] — first R columns used at rank R
    W_out: [H, H] — first R rows used at rank R
    """
    def __init__(self, hidden_size, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_in = nn.Parameter(torch.empty(hidden_size, hidden_size, dtype=dtype))
        self.W_out = nn.Parameter(torch.zeros(hidden_size, hidden_size, dtype=dtype))
        nn.init.kaiming_uniform_(self.W_in, a=5 ** 0.5)
        # W_out zero-init: head starts as identity passthrough

    def forward_at_rank(self, hidden, rank, lm_head_weight, pos_indices=None):
        """hidden: [B, T, H].  Returns logits: [B, T_or_P, V]"""
        x = hidden
        W_in_r = self.W_in[:, :rank]    # [H, R]
        W_out_r = self.W_out[:rank, :]  # [R, H]
        hidden_proj = F.silu(x @ W_in_r)    # [B, T, R]
        residual = hidden_proj @ W_out_r     # [B, T, H]
        x_out = x + residual

        if pos_indices is not None:
            B, T, H = x_out.shape
            idx = pos_indices.view(B, -1, 1).expand(-1, -1, H)
            x_out = torch.gather(x_out, 1, idx)

        # Project to vocab via shared lm_head
        if lm_head_weight.shape[0] > lm_head_weight.shape[1]:
            w = lm_head_weight.t().contiguous()
        else:
            w = lm_head_weight
        logits = x_out @ w  # [B, T_or_P, V]
        return logits


def matryoshka_loss(head, hidden, targets, lm_head_weight, num_heads=1,
                    shift=2, ranks=RANKS, weights=RANK_WEIGHTS,
                    loss_positions=256, ranks_per_step=RANKS_PER_STEP):
    """Stochastic matryoshka loss: sample ranks_per_step ranks per call.

    Over many steps, every rank gets trained without the K× slowdown of
    computing loss at every rank every step.
    """
    B, T, H = hidden.shape
    max_valid = T - shift - num_heads + 1
    if loss_positions > 0 and loss_positions < max_valid:
        perm = torch.randperm(max_valid, device=hidden.device)[:loss_positions]
        pos_indices = perm.sort().values.unsqueeze(0).expand(B, -1).contiguous()
        P = loss_positions
    else:
        pos_indices = None
        P = T

    # Sample ranks_per_step random ranks from the set (always include the max rank)
    all_ranks_list = list(ranks)
    sampled_ranks = [max(all_ranks_list)]
    others = [r for r in all_ranks_list if r != max(all_ranks_list)]
    if ranks_per_step - 1 > 0:
        np.random.shuffle(others)
        sampled_ranks += others[:ranks_per_step - 1]

    total_loss = torch.zeros((), dtype=hidden.dtype, device=hidden.device)
    per_rank_accs = {}
    for R in sampled_ranks:
        logits = head.forward_at_rank(hidden, R, lm_head_weight, pos_indices)
        if pos_indices is not None:
            target_idx = pos_indices + shift
            target_sel = torch.gather(targets, 1, target_idx)
        else:
            target_sel = targets[:, shift:shift + P]
            logits = logits[:, :target_sel.shape[1], :]
        loss_R = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_sel.reshape(-1),
        )
        total_loss = total_loss + weights[R] * loss_R
        with torch.no_grad():
            acc_R = (logits.argmax(dim=-1) == target_sel).float().mean().item()
            per_rank_accs[R] = acc_R
    total_loss = total_loss / sum(weights[r] for r in sampled_ranks)
    return total_loss, per_rank_accs


def main():
    torch.manual_seed(0)
    device = "cpu"
    dtype = torch.bfloat16

    lm_head = torch.load("data/lm_head.pt", map_location=device,
                         weights_only=True).to(dtype)
    # Pre-transpose for contiguous matmul
    lm_head_t = lm_head.t().contiguous()

    # Load cached pilot dataset (first 5 sequences)
    ds = CachedHiddenDataset(
        "data/hidden_gguf_v2_pilot.bin", "data/tokens.bin", SEQ_LEN, HIDDEN,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=True,
                        collate_fn=collate_cached, num_workers=0, drop_last=True)
    print(f"[matryoshka] dataset size: {len(ds)}")

    head = MatryoshkaMedusaHead(HIDDEN, dtype=dtype).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=0)

    max_steps = 300
    grad_accum = 4
    log_every = 20
    loss_accum = 0.0
    step = 0; accum = 0
    t0 = time.time()
    data_iter = iter(loader)

    print(f"[matryoshka] training ranks: {RANKS}")
    print(f"[matryoshka] weights: {[f'{RANK_WEIGHTS[r]:.2f}' for r in RANKS]}")
    print(f"[matryoshka] starting training: {max_steps} steps, grad_accum={grad_accum}")

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        hidden, targets = batch
        hidden = hidden.to(device); targets = targets.to(device)

        with torch.cpu.amp.autocast(dtype=torch.bfloat16):
            loss, per_rank_acc = matryoshka_loss(head, hidden, targets, lm_head_t)
            loss_for_backward = loss / grad_accum

        loss_for_backward.backward()
        loss_accum += loss.item()
        accum += 1

        if accum < grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        step += 1; accum = 0

        if step % log_every == 0:
            dt = time.time() - t0
            avg_loss = loss_accum / (log_every * grad_accum)
            loss_accum = 0.0; t0 = time.time()
            # Accuracy only available for ranks sampled this step
            acc_str = "  ".join(
                [f"R={r}:{per_rank_acc[r]:.2f}" if r in per_rank_acc else f"R={r}:--"
                 for r in RANKS]
            )
            print(f"step {step:4d} | loss {avg_loss:.4f} | {dt/log_every:.2f}s/step | {acc_str}", flush=True)

    os.makedirs("checkpoints/matryoshka_pilot", exist_ok=True)
    torch.save({"W_in": head.W_in.detach(), "W_out": head.W_out.detach()},
               "checkpoints/matryoshka_pilot/head_step500.pt")
    print(f"[matryoshka] saved checkpoint")


if __name__ == "__main__":
    main()
