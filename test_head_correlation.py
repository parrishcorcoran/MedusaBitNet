"""Measure error correlation between the 4 trained Medusa heads.

If error patterns are highly correlated, joint training is making heads
redundant, and a multi-seed or multi-input training approach would widen
the ensemble gate frontier. If error patterns are uncorrelated, we're
already near the ensemble ceiling and need more diverse inputs (multi-layer)
to make progress.

Each head k at anchor t-k predicts position t+2. For each position, we
have 4 binary correctness scores (one per head). We compute:
  - pairwise error correlation
  - distribution of "how many of 4 correct"
  - joint accuracy P(all 4 correct)
  - vs independence baseline: prod(P(each correct))
"""
import numpy as np
import torch
import torch.nn.functional as F
from model import MedusaHeads

SEQ_LEN = 2048
HIDDEN  = 2560
VOCAB   = 128256


def main():
    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    heads = MedusaHeads(HIDDEN, VOCAB, 4, 1, dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])

    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    n_test_seqs = 16

    correct_per_head = [[], [], [], []]  # head k -> list of bool per position

    with torch.no_grad():
        for si in range(n_test_seqs):
            off = si * per_seq
            chunk = hidden_mm[off:off + per_seq]
            h = (torch.from_numpy(chunk.copy()).view(torch.bfloat16)
                       .view(SEQ_LEN, HIDDEN))

            logits_all = heads(h.unsqueeze(0), lm_head)  # [1, T, K, V]
            preds_all = logits_all.argmax(dim=-1)[0]     # [T, K]

            # vanilla_all[j] = greedy prediction for position j+2
            vanilla_logits = h[1:].to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
            vanilla_all = vanilla_logits.float().argmax(dim=-1)  # [T-1], idx j -> pos j+2

            t_start, t_end = 3, SEQ_LEN - 2
            ts = torch.arange(t_start, t_end)
            vanilla = vanilla_all[ts]  # target at pos t+2

            # head k at anchor t-k predicts pos t+2
            for k in range(4):
                pred_k = preds_all[ts - k, k]
                correct_k = (pred_k == vanilla).numpy()
                correct_per_head[k].append(correct_k)

    # Stack across sequences
    correct_per_head = [np.concatenate(c) for c in correct_per_head]
    N = len(correct_per_head[0])
    correct_matrix = np.stack(correct_per_head, axis=0).astype(np.float32)  # [4, N]

    marginal = correct_matrix.mean(axis=1)
    print(f"Per-head accuracy:  {[f'{m:.3f}' for m in marginal]}")

    # Pairwise correlation (of the BINARY correctness variables)
    print(f"\nPairwise Pearson correlation (correctness variables):")
    print("     H0    H1    H2    H3")
    for i in range(4):
        row = [f"{np.corrcoef(correct_matrix[i], correct_matrix[j])[0,1]:.3f}"
               for j in range(4)]
        print(f"H{i}  " + "  ".join(row))

    # Joint distribution: how often are k of 4 correct?
    counts = correct_matrix.sum(axis=0)  # [N], values 0..4
    print(f"\nDistribution of 'how many of 4 heads correct':")
    for k in range(5):
        n = (counts == k).sum()
        print(f"  {k} correct: {n:>6} ({100*n/N:5.2f}%)")

    # All correct joint probability vs independence
    p_all_observed = (counts == 4).mean()
    p_all_independent = np.prod(marginal)
    print(f"\nP(all 4 correct):")
    print(f"  observed:    {p_all_observed:.4f}")
    print(f"  independence baseline: {p_all_independent:.4f}")
    print(f"  ratio observed/independence: {p_all_observed / max(p_all_independent, 1e-9):.2f}x")
    print(f"  (ratio >> 1 means errors are correlated — heads fail together)")

    # None correct: how often is EVERY head wrong?
    p_none_observed = (counts == 0).mean()
    p_none_independent = np.prod(1 - marginal)
    print(f"\nP(all 4 wrong):")
    print(f"  observed:    {p_none_observed:.4f}")
    print(f"  independence baseline: {p_none_independent:.4f}")
    print(f"  ratio observed/independence: {p_none_observed / max(p_none_independent, 1e-9):.2f}x")


if __name__ == "__main__":
    main()
