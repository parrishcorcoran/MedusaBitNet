"""Router-based dynamic rank simulation on existing heads.

For each token:
  - Ask at each rank R: does head-0-truncated-to-R match vanilla greedy?
  - Find the smallest R that matches (label: "minimum sufficient rank")
  - Train a router on cheap pre-backbone features to predict that label
  - Evaluate: expected compute per token with routed ranks vs fixed full-rank

This is a PROXY for matryoshka+router — limited by SVD truncation's quality
drop at low ranks, but tests whether router-based dynamic compute works.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import MedusaHeads

import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_cheap_aperture import (
    get_boundary_token_ids, dist_to_last, rolling_mean, lagged_diff,
)

SEQ_LEN = 2048
HIDDEN  = 2560
VOCAB   = 128256
TOKENIZER_DIR = "/home/cpinchington/MedusaBitNet/models/bitnet-b1.58-2B-4T"
RANKS = [256, 512, 1024, 2560]  # the usable range for SVD truncation


def truncate_svd(W, rank):
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    return (U[:, :rank] * S[:rank].unsqueeze(0)) @ Vh[:rank, :]


def head_forward(hidden, w_in, w_out):
    x = hidden.float()
    x_new = x + F.silu(x @ w_in) @ w_out
    return x_new


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    enders, newlines = get_boundary_token_ids(tok)

    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    w_in_full = ckpt["heads"]["w_in"][0, 0].float()
    w_out_full = ckpt["heads"]["w_out"][0, 0].float()

    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    tokens_mm = np.memmap("data/tokens.bin", dtype=np.uint32, mode="r")
    n_seqs = 48; seq_split = 36

    # Precompute truncated weights
    truncated_wins = {R: (truncate_svd(w_in_full, R) if R < HIDDEN else w_in_full)
                      for R in RANKS}
    truncated_wouts = {R: (truncate_svd(w_out_full, R) if R < HIDDEN else w_out_full)
                       for R in RANKS}

    all_correctness = {R: [] for R in RANKS}  # for each rank, did it match vanilla?
    all_cheap_feat = []
    split_flags = []

    with torch.no_grad():
        for si in range(n_seqs):
            off = si * per_seq
            chunk = hidden_mm[off:off + per_seq]
            h = (torch.from_numpy(chunk.copy()).view(torch.bfloat16)
                       .view(SEQ_LEN, HIDDEN))
            token_ids = tokens_mm[si * SEQ_LEN : (si + 1) * SEQ_LEN]

            # For each rank, compute head-0 prediction at anchor t for target t+2
            rank_preds = {}
            for R in RANKS:
                x_new = head_forward(h, truncated_wins[R], truncated_wouts[R])
                logits = x_new.to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
                rank_preds[R] = logits.float().argmax(dim=-1).numpy()  # [T]

            # Vanilla greedy (label source)
            vlogits = h[1:].to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
            vpred = vlogits.float().argmax(dim=-1).numpy()  # [T-1] -> j maps to pos j+2

            # Need cheap features (pre-backbone, using PAST confidences from full-rank head)
            logits_full = (head_forward(h, w_in_full, w_out_full)
                           .to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16))
            probs_full = F.softmax(logits_full.float(), dim=-1)
            confs_full = probs_full.max(dim=-1).values.numpy()  # for past_conf features

            valid = SEQ_LEN - 2
            t_start, t_end = 6, valid
            ts = np.arange(t_start, t_end, dtype=np.int64)

            # Correctness per rank: rank_preds[R][t] == vpred[t] (target t+2)
            for R in RANKS:
                correct = (rank_preds[R][ts] == vpred[ts]).astype(np.int8)
                all_correctness[R].append(correct)

            # Cheap features (zero-backbone-at-t, uses PAST confidences)
            past_confs = np.concatenate([[confs_full[0]], confs_full])[:-1]
            rc10 = rolling_mean(past_confs, 10)[ts]
            rc50 = rolling_mean(past_confs, 50)[ts]
            lag5 = lagged_diff(past_confs, 5)[ts]

            de = dist_to_last(token_ids.tolist(), enders).astype(np.float32)
            dn = dist_to_last(token_ids.tolist(), newlines).astype(np.float32)
            de_at = np.log1p(np.minimum(de[ts + 1], 200.0))
            dn_at = np.log1p(np.minimum(dn[ts + 1], 200.0))
            rel_pos = (ts.astype(np.float32) / valid)

            feat = np.stack([rc10, rc50, lag5, de_at, dn_at, rel_pos], axis=1).astype(np.float32)
            all_cheap_feat.append(feat)
            split_flags.append(0 if si < seq_split else 1)

    X = np.concatenate(all_cheap_feat, axis=0)
    correctness = {R: np.concatenate(all_correctness[R]) for R in RANKS}
    is_test = np.concatenate([
        np.full(len(all_cheap_feat[i]), split_flags[i], dtype=np.int8)
        for i in range(n_seqs)
    ])
    train_mask = is_test == 0; test_mask = is_test == 1
    N = len(X)
    print(f"features: {X.shape}  train={train_mask.sum()}  test={test_mask.sum()}")

    # Per-rank marginal accuracy
    print("\n=== Per-rank marginal accuracy on TEST ===")
    for R in RANKS:
        print(f"  rank {R:>5}: acc = {correctness[R][test_mask].mean():.4f}")

    # Oracle: what's the expected "minimum rank needed" if we could choose perfectly?
    # For each position, find the smallest R where correctness[R]=1
    min_rank_needed = np.full(N, 99999, dtype=np.int32)
    for R in sorted(RANKS):
        first_match = (min_rank_needed == 99999) & (correctness[R] == 1)
        min_rank_needed[first_match] = R
    # "none work" positions: we'd have to run full rank anyway (or skip)
    none_work = (min_rank_needed == 99999)
    print(f"\nOracle stats on TEST:")
    test_mrn = min_rank_needed[test_mask]
    for R in RANKS:
        frac = (test_mrn == R).mean()
        print(f"  min rank {R:>5}: {frac:.4f} fraction")
    frac_none = (test_mrn == 99999).mean()
    print(f"  none of tested ranks works: {frac_none:.4f}")

    # Expected compute with ORACLE routing (lower bound on what router can achieve)
    rank_cost = {R: (4 * HIDDEN * R) if R < HIDDEN else (2 * HIDDEN * HIDDEN) for R in RANKS}
    # When no rank works, fall back to full (or higher) — approximate with full cost
    baseline_cost = rank_cost[2560]  # always full-rank
    oracle_cost_per_pos = np.array([
        rank_cost[R] if R != 99999 else baseline_cost
        for R in test_mrn
    ])
    expected_cost_oracle = oracle_cost_per_pos.mean()
    print(f"\nCompute per token (4*H*R params transferred):")
    print(f"  Always full-rank:       {baseline_cost:,}")
    print(f"  Oracle routing:         {expected_cost_oracle:,.0f}  ({baseline_cost/expected_cost_oracle:.2f}x savings)")

    # Now: train a router on cheap features to predict min_rank_needed
    # Frame as classification over {256, 512, 1024, 2560} (drop "none" -> lump with 2560)
    y_router = np.zeros(N, dtype=np.int64)
    rank_to_class = {R: i for i, R in enumerate(RANKS)}  # 256->0, 512->1, 1024->2, 2560->3
    for i in range(N):
        R = min_rank_needed[i]
        if R == 99999:
            y_router[i] = rank_to_class[2560]
        else:
            y_router[i] = rank_to_class[R]

    mu = X[train_mask].mean(axis=0); sd = X[train_mask].std(axis=0) + 1e-6
    Xn = (X - mu) / sd

    router = nn.Sequential(
        nn.Linear(X.shape[1], 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, len(RANKS)),
    )
    opt = torch.optim.Adam(router.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.from_numpy(Xn[train_mask])
    yt = torch.from_numpy(y_router[train_mask])
    Xe = torch.from_numpy(Xn[test_mask])
    for epoch in range(40):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 4096):
            idx = perm[i:i+4096]
            logits = router(Xt[idx])
            loss = F.cross_entropy(logits, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    router.eval()
    with torch.no_grad():
        pred_class = router(Xe).argmax(dim=-1).numpy()
    pred_rank = np.array([RANKS[c] for c in pred_class])

    # Evaluate: apply predicted rank, check if that rank was correct at that position
    actual_correct = np.zeros(test_mask.sum(), dtype=np.int8)
    test_idx = np.where(test_mask)[0]
    for i, orig_idx in enumerate(test_idx):
        R = pred_rank[i]
        actual_correct[i] = correctness[R][orig_idx]

    fidelity = actual_correct.mean()
    router_cost_per_pos = np.array([rank_cost[R] for R in pred_rank])
    expected_cost_router = router_cost_per_pos.mean()

    print(f"\nTrained router on TEST:")
    print(f"  Fidelity (router's rank matches vanilla): {fidelity:.4f}")
    print(f"  Expected compute: {expected_cost_router:,.0f}  ({baseline_cost/expected_cost_router:.2f}x savings vs full)")

    # Router rank distribution
    print(f"\n  Router rank distribution on test:")
    for R in RANKS:
        print(f"    rank {R:>5}: {(pred_rank == R).mean():.4f}")


if __name__ == "__main__":
    main()
