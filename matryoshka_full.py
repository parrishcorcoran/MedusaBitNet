"""Full-scale matryoshka training on 1000-seq cache.

Stochastic matryoshka with wider rank range. 1000 steps target.
"""
import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train import CachedHiddenDataset, collate_cached
from torch.utils.data import DataLoader
from matryoshka_pilot import MatryoshkaMedusaHead

SEQ_LEN = 2048; HIDDEN = 2560; VOCAB = 128256
RANKS = [32, 128, 512, 2560]            # 4 ranks spanning the range
RANK_WEIGHTS = {32: 4.0, 128: 2.0, 512: 1.0, 2560: 1.0}
RANKS_PER_STEP = 2  # always include max rank + 1 random other


def matryoshka_loss(head, hidden, targets, lm_head_weight, shift=2,
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
        logits = head.forward_at_rank(hidden, R, lm_head_weight, pos_indices)
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

    ds = CachedHiddenDataset(
        "data/hidden_gguf_v2.bin", "data/tokens.bin", SEQ_LEN, HIDDEN)
    loader = DataLoader(ds, batch_size=1, shuffle=True,
                        collate_fn=collate_cached, num_workers=0, drop_last=True)
    print(f"[mat-full] dataset: {len(ds)} seqs")

    head = MatryoshkaMedusaHead(HIDDEN, dtype=dtype).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=0)

    max_steps = 800; grad_accum = 4; log_every = 25
    step, accum, loss_accum = 0, 0, 0.0
    t0 = time.time()
    di = iter(loader)

    print(f"[mat-full] ranks {RANKS} weights {RANK_WEIGHTS}")
    print(f"[mat-full] training {max_steps} steps, grad_accum={grad_accum}")

    while step < max_steps:
        try:
            batch = next(di)
        except StopIteration:
            di = iter(loader); batch = next(di)
        hidden, targets = batch
        hidden = hidden.to(device); targets = targets.to(device)

        with torch.cpu.amp.autocast(dtype=torch.bfloat16):
            loss, accs = matryoshka_loss(head, hidden, targets, lm_head)
            (loss / grad_accum).backward()
        loss_accum += loss.item(); accum += 1
        if accum < grad_accum: continue
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        step += 1; accum = 0
        if step % log_every == 0:
            dt = time.time() - t0
            avg = loss_accum / (log_every * grad_accum); loss_accum = 0; t0 = time.time()
            acc_str = "  ".join(f"R{r}:{accs.get(r, None):.2f}"
                                if r in accs else f"R{r}:--"
                                for r in RANKS)
            print(f"step {step:4d} | loss {avg:.3f} | {dt/log_every:.2f}s | {acc_str}",
                  flush=True)

    os.makedirs("checkpoints/matryoshka_full", exist_ok=True)
    torch.save({"W_in": head.W_in.detach(), "W_out": head.W_out.detach(),
                "ranks": RANKS, "weights": RANK_WEIGHTS},
               "checkpoints/matryoshka_full/head_step800.pt")
    print("[mat-full] saved")


if __name__ == "__main__":
    main()
