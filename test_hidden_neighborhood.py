"""Hidden state neighborhood distance — a geometric aperture on the token state.

For each position, maintain a FIFO of recent hidden states (say last 50).
Compute:
  - min distance to any recent hidden state
  - mean distance to recent hidden states
  - distance to k-th nearest

Intuition (electron cloud analog): if current hidden state is NEAR recent
states, we're in a familiar region of state space — model is cycling, token
likely predictable. If FAR from recent, we're in novel territory — token
likely unpredictable.

This is a GEOMETRIC aperture, orthogonal to any softmax-derived signal.
Add to unified MLP and see if frontier pushes past the current ceiling.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features
from test_unified_estimator_v2 import UnifiedMLP, frontier, LAMBDA_TARGETS

SEQ_LEN = 2048; HIDDEN = 2560


def build_neighborhood_features(fifo_size=50, n_seqs=48):
    """For each position, compute distances to recent hidden states in FIFO."""
    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    feats = []
    for si in range(n_seqs):
        off = si * per_seq
        chunk = hidden_mm[off:off + per_seq]
        # Convert bf16 (uint16) to float32, in chunks to save memory
        raw32 = chunk.astype(np.uint32) << 16
        h = raw32.view(np.float32).reshape(SEQ_LEN, HIDDEN)
        h_t = torch.from_numpy(h)

        # Use a sliding window: for position i, compute distances to positions
        # i-fifo_size..i-1. Cheap: torch cdist on smaller chunks.
        feat_seq = np.zeros((SEQ_LEN, 3), dtype=np.float32)
        for i in range(1, SEQ_LEN):
            lo = max(0, i - fifo_size)
            fifo = h_t[lo:i]  # [K, H]
            cur = h_t[i:i+1]  # [1, H]
            dists = torch.cdist(cur, fifo).squeeze(0)  # [K]
            if len(dists) == 0:
                feat_seq[i] = [0, 0, 0]
            else:
                feat_seq[i, 0] = float(dists.min())       # nearest
                feat_seq[i, 1] = float(dists.mean())       # mean
                feat_seq[i, 2] = float(torch.median(dists))  # median

        valid = SEQ_LEN - 2
        t_start, t_end = 6, valid
        ts = np.arange(t_start, t_end, dtype=np.int64)
        # Use feature at position t+1 (context up through t+1 for prediction at t+2)
        feats.append(feat_seq[ts + 1])

    return np.concatenate(feats, axis=0)


def main():
    X, y, train_mask, test_mask = build_features()
    print(f"existing features: {X.shape}")

    print("Building hidden-state neighborhood features (this takes a moment)...")
    import time; t0 = time.time()
    nbr = build_neighborhood_features()
    print(f"neighborhood features: {nbr.shape} in {time.time()-t0:.1f}s")

    # Combine
    X_plus = np.concatenate([X, nbr], axis=1)
    all_names = [
        "content_conf", "content_entropy", "logit_gap", "purity", "top3_cov",
        "top10_cov", "rc10", "rc50", "conf_deriv", "conf_lag1", "conf_lag5",
        "dist_period_log", "dist_newline_log", "rel_pos",
        "agreement_count", "conf_var", "conf_min",
        "nbr_min_dist", "nbr_mean_dist", "nbr_median_dist",
    ]

    def train_eval(idx, label):
        mu = X_plus[train_mask][:, idx].mean(axis=0)
        sd = X_plus[train_mask][:, idx].std(axis=0) + 1e-6
        Xn = (X_plus[:, idx] - mu) / sd
        net = UnifiedMLP(n_feat=len(idx), hidden=64)
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

    train_eval(list(range(17)), "Baseline 17 features")
    train_eval(list(range(17, 20)), "Neighborhood ONLY (3 features)")
    net, Xn = train_eval(list(range(20)), "All 20 (17 + 3 neighborhood)")

    # Feature importance
    Xe_r = torch.from_numpy(Xn[test_mask]).clone().requires_grad_(True)
    torch.sigmoid(net(Xe_r)).sum().backward()
    grads = Xe_r.grad.abs().mean(dim=0).numpy()
    print("\n=== Feature importance (20-feature model) ===")
    order = np.argsort(-grads)
    for idx in order:
        marker = "★ NEW" if idx >= 17 else ""
        print(f"  {all_names[idx]:>20}  |grad|={grads[idx]:.4f}  {marker}")


if __name__ == "__main__":
    main()
