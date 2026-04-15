"""Token-reuse apertures — H2O at the lexical level.

Hypothesis (per user): current token is likely to recur. Features:
  1. Token frequency in recent window (how common IS current token?)
  2. Is current token a "heavy hitter" in recent context?
  3. Cumulative count of this token in the full sequence so far
  4. Distribution of distances-to-next-occurrence (via retroactive labels)
  5. Zipfian rank of token in recent window

Orthogonal to hidden-state neighborhood because it operates on token IDs,
not on state vectors.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features
from test_unified_estimator_v2 import UnifiedMLP, frontier, LAMBDA_TARGETS

SEQ_LEN = 2048


def build_token_reuse_features(n_seqs=48, window=100):
    tokens_mm = np.memmap("data/tokens.bin", dtype=np.uint32, mode="r")
    feats = []
    for si in range(n_seqs):
        toks = tokens_mm[si * SEQ_LEN : (si + 1) * SEQ_LEN]

        # Feature 1: count of current token in last 'window' tokens
        # Feature 2: rank (inverse of frequency rank) of current token in window
        # Feature 3: cumulative count so far
        # Feature 4: is top-10% frequent in recent window (heavy hitter)

        freq_in_window = np.zeros(SEQ_LEN, dtype=np.float32)
        rank_in_window = np.zeros(SEQ_LEN, dtype=np.float32)
        cum_count = np.zeros(SEQ_LEN, dtype=np.float32)
        is_heavy_hitter = np.zeros(SEQ_LEN, dtype=np.float32)
        distinct_in_window = np.zeros(SEQ_LEN, dtype=np.float32)

        # Running count using dict
        from collections import Counter
        total_counter = Counter()
        for i in range(SEQ_LEN):
            lo = max(0, i - window + 1)
            window_toks = toks[lo:i+1]
            window_counter = Counter(window_toks.tolist())
            tid = int(toks[i])
            freq_in_window[i] = window_counter[tid]
            distinct_in_window[i] = len(window_counter)

            # Rank: how many unique tokens have count >= current token's count
            # Lower rank = more frequent
            sorted_counts = sorted(window_counter.values(), reverse=True)
            try:
                rank = sorted_counts.index(window_counter[tid]) + 1
            except ValueError:
                rank = 1
            rank_in_window[i] = rank

            # Heavy hitter: current token is in top-10% of unique tokens by frequency
            # For window of 100 with maybe 70 unique, top-10% = top 7 tokens
            n_unique = len(sorted_counts)
            threshold_rank = max(1, n_unique // 10)
            is_heavy_hitter[i] = 1.0 if rank <= threshold_rank else 0.0

            total_counter[tid] += 1
            cum_count[i] = total_counter[tid]

        valid = SEQ_LEN - 2
        ts = np.arange(6, valid, dtype=np.int64)
        feat = np.stack([
            np.log1p(freq_in_window[ts + 1]),
            np.log1p(rank_in_window[ts + 1]),
            np.log1p(cum_count[ts + 1]),
            is_heavy_hitter[ts + 1],
            np.log1p(distinct_in_window[ts + 1]),
        ], axis=1).astype(np.float32)
        feats.append(feat)
    return np.concatenate(feats, axis=0)


def main():
    X, y, train_mask, test_mask = build_features()
    print(f"existing features: {X.shape}")

    print("\nBuilding token-reuse features...")
    import time; t0 = time.time()
    tr = build_token_reuse_features()
    print(f"  {tr.shape} in {time.time()-t0:.1f}s")

    def train_eval_(X_all, feat_idx, label):
        mu = X_all[train_mask][:, feat_idx].mean(axis=0)
        sd = X_all[train_mask][:, feat_idx].std(axis=0) + 1e-6
        Xn = (X_all[:, feat_idx] - mu) / sd
        torch.manual_seed(0)
        net = UnifiedMLP(n_feat=len(feat_idx), hidden=64)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        Xt = torch.from_numpy(Xn[train_mask]); yt = torch.from_numpy(y[train_mask])
        Xe = torch.from_numpy(Xn[test_mask])
        for _ in range(40):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 4096):
                bi = perm[i:i+4096]
                loss = F.binary_cross_entropy_with_logits(net(Xt[bi]), yt[bi])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(Xe)).numpy()
        print(f"\n=== {label} ===")
        for λ, skip, fid in frontier(p, y[test_mask], LAMBDA_TARGETS):
            print(f"  λ={λ:.2f}  skip={skip:.4f}  fidelity={fid:.4f}")
        return net, Xn

    train_eval_(X, list(range(17)), "Baseline 17 features")

    X_plus = np.concatenate([X, tr], axis=1)
    train_eval_(X_plus, list(range(17, 22)), "Token-reuse ONLY (5 features)")
    net, Xn = train_eval_(X_plus, list(range(22)),
                           "Baseline + Token-reuse (22 features)")

    # Feature importance
    Xe_r = torch.from_numpy(Xn[test_mask]).clone().requires_grad_(True)
    torch.sigmoid(net(Xe_r)).sum().backward()
    grads = Xe_r.grad.abs().mean(dim=0).numpy()
    names = [
        "content_conf", "content_entropy", "logit_gap", "purity", "top3_cov",
        "top10_cov", "rc10", "rc50", "conf_deriv", "conf_lag1", "conf_lag5",
        "dist_period_log", "dist_newline_log", "rel_pos",
        "agreement_count", "conf_var", "conf_min",
        "token_freq_window", "token_rank_window", "token_cumcount",
        "is_heavy_hitter", "distinct_in_window",
    ]
    print("\n=== Feature importance (22 features) ===")
    order = np.argsort(-grads)
    for idx in order:
        marker = " ★" if idx >= 17 else ""
        print(f"  {names[idx]:>22}  |grad|={grads[idx]:.4f}{marker}")


if __name__ == "__main__":
    main()
