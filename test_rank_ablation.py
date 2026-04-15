"""Rank ablation on trained Medusa heads.

For each head, SVD w_in and w_out, truncate to rank R, measure accuracy.
Answers: what's the minimum rank that preserves head quality?
"""
import numpy as np
import torch
import torch.nn.functional as F
from model import MedusaHeads

SEQ_LEN = 2048
HIDDEN  = 2560
VOCAB   = 128256


def truncate_svd(W, rank):
    """Return rank-R SVD approximation of W. W: [H_in, H_out]."""
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    return (U_r * S_r.unsqueeze(0)) @ Vh_r


def apply_head_with_truncated_weights(hidden, w_in_r, w_out_r):
    """Single Medusa head forward with custom (rank-reduced) weights.

    hidden: [T, H] bf16
    w_in_r, w_out_r: [H, H] float32 (rank-reduced reconstructions)
    """
    x = hidden.float()
    hidden_proj = F.silu(x @ w_in_r)
    x_new = x + hidden_proj @ w_out_r
    return x_new  # [T, H]


def main():
    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    w_in_all = ckpt["heads"]["w_in"]   # [1, 4, 2560, 2560]
    w_out_all = ckpt["heads"]["w_out"] # [1, 4, 2560, 2560]
    num_layers, num_heads, H, _ = w_in_all.shape
    print(f"[rank] loaded heads: layers={num_layers} heads={num_heads} H={H}")

    # Look at singular value spectrum first
    print("\n=== Singular value spectrum (top-k / sum) per head ===")
    print(f"{'head':>4} {'layer':>5}   ", end="")
    ks = [16, 32, 64, 128, 256, 512, 1024]
    for k in ks:
        print(f"top{k:<5}", end=" ")
    print()
    for k in range(num_heads):
        for l in range(num_layers):
            Win = w_in_all[l, k].float()
            U, S, Vh = torch.linalg.svd(Win, full_matrices=False)
            total = S.sum().item()
            cum = torch.cumsum(S, dim=0) / total
            print(f"{k:>4} {l:>5} W_in ", end="")
            for kk in ks:
                print(f"{cum[kk-1].item():>7.3f}", end=" ")
            print()

    # Load lm_head and cached hidden states for accuracy eval
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)
    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    n_test_seqs = 12  # held-out sequences (last of our 48)

    # We'll evaluate head-0 specifically at various ranks.
    ranks = [8, 16, 32, 64, 128, 256, 512, 1024, 2560]
    results = {}  # rank -> accuracy

    print("\n=== Head-0 accuracy at various ranks ===")
    print(f"{'rank':>6} {'params':>10} {'compress':>10} {'acc@1':>10}")
    print("-" * 45)

    # Baseline: full rank
    w_in_full = w_in_all[0, 0].float()
    w_out_full = w_out_all[0, 0].float()

    # Gather data once
    all_hiddens = []
    all_vpred = []
    for si in range(36, 36 + n_test_seqs):
        off = si * per_seq
        chunk = hidden_mm[off:off + per_seq]
        h = (torch.from_numpy(chunk.copy()).view(torch.bfloat16)
                   .view(SEQ_LEN, HIDDEN))
        vlogits = h[1:].to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
        vpred = vlogits.float().argmax(dim=-1)
        all_hiddens.append(h)
        all_vpred.append(vpred)

    for R in ranks:
        if R >= H:
            w_in_r = w_in_full
            w_out_r = w_out_full
            params = 2 * H * H
        else:
            w_in_r = truncate_svd(w_in_full, R)
            w_out_r = truncate_svd(w_out_full, R)
            # Parameter count of rank-R factorization: 2 * (H*R + R*H) = 4HR
            params = 4 * H * R

        compress_ratio = (2 * H * H) / params if params > 0 else 1.0

        total_correct = 0
        total_count = 0
        with torch.no_grad():
            for h, vpred in zip(all_hiddens, all_vpred):
                x_new = apply_head_with_truncated_weights(h, w_in_r, w_out_r)
                logits_h0 = x_new.to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
                pred_h0 = logits_h0.float().argmax(dim=-1)  # [T]
                valid = SEQ_LEN - 2
                # head-0 at anchor t predicts token at t+2. vpred[t] is greedy at t+2.
                correct = (pred_h0[:valid] == vpred[:valid]).float().sum().item()
                total_correct += correct
                total_count += valid

        acc = total_correct / total_count
        results[R] = acc
        print(f"{R:>6} {params:>10,d} {compress_ratio:>10.2f}x {acc:>10.4f}")

    # Identify knee: largest R such that acc drops by no more than 1% from full
    full_acc = results[2560]
    print(f"\nFull-rank accuracy: {full_acc:.4f}")
    print(f"\nRank required to preserve accuracy within:")
    for drop_pct in [0.1, 0.5, 1.0, 2.0, 5.0]:
        threshold = full_acc * (1 - drop_pct / 100)
        candidates = [R for R, a in results.items() if a >= threshold]
        if candidates:
            min_R = min(candidates)
            compress = (2 * H * H) / (4 * H * min_R) if min_R < H else 1.0
            print(f"  ≤{drop_pct:>4.1f}% drop -> rank {min_R:>4}  ({compress:>.1f}x compression)")


if __name__ == "__main__":
    main()
