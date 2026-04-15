"""Clean 8-feature MLP (one representative per independent dimension).

If the frontier stays ~same as the 17-feature MLP, the 4-dimensions-not-17
claim is empirically supported.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features
from test_unified_estimator_v2 import UnifiedMLP, frontier, LAMBDA_TARGETS


ALL_NAMES = [
    "content_conf", "content_entropy",
    "logit_gap", "purity", "top3_cov", "top10_cov",
    "rc10", "rc50", "conf_deriv",
    "conf_lag1", "conf_lag5",
    "dist_period_log", "dist_newline_log", "rel_pos",
    "agreement_count", "conf_var", "conf_min",
]
MINIMAL_NAMES = [
    "logit_gap",          # Dim 1 (sharpness, highest-importance member)
    "content_entropy",    # Dim 1 (different aperture within cluster, provides smoothing)
    "agreement_count",    # Dim 2 (cross-aperture, orthogonal)
    "conf_min",           # Dim 2 (cross-aperture, partial independence)
    "dist_period_log",    # Dim 3 (structural)
    "rel_pos",            # Dim 3 (structural)
    "rc50",               # Dim 4 (trajectory, long window)
    "conf_lag5",          # Dim 4 (trajectory, derivative)
]
MINIMAL_IDX = [ALL_NAMES.index(n) for n in MINIMAL_NAMES]


def train_frontier(X, y, train_mask, test_mask, feat_idx, epochs=40, seed=0):
    torch.manual_seed(seed)
    mu = X[train_mask][:, feat_idx].mean(axis=0)
    sd = X[train_mask][:, feat_idx].std(axis=0) + 1e-6
    Xn = (X[:, feat_idx] - mu) / sd

    net = UnifiedMLP(n_feat=len(feat_idx), hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.from_numpy(Xn[train_mask]); yt = torch.from_numpy(y[train_mask])
    Xe = torch.from_numpy(Xn[test_mask])
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 4096):
            idx = perm[i:i+4096]
            logit = net(Xt[idx])
            loss = F.binary_cross_entropy_with_logits(logit, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(Xe)).numpy()
    return frontier(p, y[test_mask], LAMBDA_TARGETS)


def main():
    X, y, train_mask, test_mask = build_features()
    print(f"features built: {X.shape}, train={train_mask.sum()}, test={test_mask.sum()}\n")

    all_idx = list(range(17))

    # (C_v2) All 17 features
    print("=== 17 features (the kitchen sink) ===")
    for λ, skip, fid in train_frontier(X, y, train_mask, test_mask, all_idx):
        print(f"  λ={λ:.2f}  skip={skip:.4f}  fidelity={fid:.4f}")

    # (C_minimal) 8 features, one rep per independent dimension
    print(f"\n=== 8 minimal features {MINIMAL_NAMES} ===")
    for λ, skip, fid in train_frontier(X, y, train_mask, test_mask, MINIMAL_IDX):
        print(f"  λ={λ:.2f}  skip={skip:.4f}  fidelity={fid:.4f}")

    # Drop logit_gap (use content_entropy as sole sharpness rep)
    idx_5 = [ALL_NAMES.index(n) for n in
             ["content_entropy", "agreement_count", "dist_period_log", "rel_pos", "rc50"]]
    print(f"\n=== 5 ultra-minimal features (one per dimension) ===")
    for λ, skip, fid in train_frontier(X, y, train_mask, test_mask, idx_5):
        print(f"  λ={λ:.2f}  skip={skip:.4f}  fidelity={fid:.4f}")


if __name__ == "__main__":
    main()
