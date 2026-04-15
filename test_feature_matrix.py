"""Full feature-cluster-cost matrix.

For every feature we have (22 total across Tier 0 + Tier B + Tier C):
  - Cluster assignment (which underlying dimension does it measure?)
  - Per-feature cost estimate
  - Marginal contribution (|grad| from full-features MLP)
  - Best use-case (scenario where it's most valuable)

Output: a big table the paper can use to justify feature choices.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features
from test_tier_b_features import build_tier_b
from test_unified_estimator_v2 import UnifiedMLP, frontier, LAMBDA_TARGETS

SEQ_LEN = 2048; HIDDEN = 2560; VOCAB = 128256


ALL_FEATS = [
    # name, cluster, cost_medusa (when heads run anyway), cost_zero_bb (no head)
    ("content_conf",      "Dim1:Sharpness",          0,  1,  "Basic softmax peak"),
    ("content_entropy",   "Dim1:Sharpness",          0,  1,  "Full distribution entropy"),
    ("logit_gap",         "Dim1:Sharpness",          0,  1,  "Decisiveness, non-squashed"),
    ("purity",            "Dim1:Sharpness",          0,  1,  "Σp² — inverse participation ratio"),
    ("top3_cov",          "Dim1:Sharpness",          0,  1,  "Top-3 cumulative probability"),
    ("top10_cov",         "Dim1:Sharpness",          0,  1,  "Top-10 cumulative probability"),
    ("rc10",              "Dim2:Trajectory",         0,  0,  "Rolling conf, 10-window"),
    ("rc50",              "Dim2:Trajectory",         0,  0,  "Rolling conf, 50-window"),
    ("conf_deriv",        "Dim2:Trajectory",         0,  0,  "Confidence derivative"),
    ("conf_lag1",         "Dim2:Trajectory",         0,  0,  "Lagged conf difference, 1-step"),
    ("conf_lag5",         "Dim2:Trajectory",         0,  0,  "Lagged conf difference, 5-step"),
    ("dist_period_log",   "Dim3:Structural",         0,  0,  "Distance to last sentence-ender"),
    ("dist_newline_log",  "Dim3:Structural",         0,  0,  "Distance to last newline"),
    ("rel_pos",           "Dim3:Structural",         0,  0,  "Relative position in sequence"),
    ("agreement_count",   "Dim5:CrossAperture",      3,  4,  "How many of 4 heads agree"),
    ("conf_var",          "Dim5:CrossAperture",      3,  4,  "Variance of 4-head confidences"),
    ("conf_min",          "Dim5:CrossAperture",      3,  4,  "Minimum of 4-head confidences"),
]


def cluster_of(name):
    for n, c, _, _, _ in ALL_FEATS:
        if n == name: return c
    return "?"


def main():
    X, y, train_mask, test_mask = build_features()
    print(f"features: {X.shape}\n")

    # Train full-22-feature MLP, get gradient importance
    mu = X[train_mask].mean(axis=0); sd = X[train_mask].std(axis=0) + 1e-6
    Xn = (X - mu) / sd
    net = UnifiedMLP(n_feat=17, hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.from_numpy(Xn[train_mask]); yt = torch.from_numpy(y[train_mask])
    for _ in range(40):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 4096):
            idx = perm[i:i+4096]
            loss = F.binary_cross_entropy_with_logits(net(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    Xe_r = torch.from_numpy(Xn[test_mask]).clone().requires_grad_(True)
    torch.sigmoid(net(Xe_r)).sum().backward()
    grads = Xe_r.grad.abs().mean(dim=0).numpy()

    # Per-feature: drop test on TEST set
    def eval_drop_one(drop_idx, epochs=20):
        keep = [i for i in range(17) if i != drop_idx]
        mu2 = X[train_mask][:, keep].mean(axis=0); sd2 = X[train_mask][:, keep].std(axis=0) + 1e-6
        Xn2 = (X[:, keep] - mu2) / sd2
        net2 = UnifiedMLP(n_feat=len(keep), hidden=48)
        opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
        Xt2 = torch.from_numpy(Xn2[train_mask]); yt2 = torch.from_numpy(y[train_mask])
        Xe2 = torch.from_numpy(Xn2[test_mask])
        for _ in range(epochs):
            perm = torch.randperm(len(Xt2))
            for i in range(0, len(Xt2), 4096):
                idx = perm[i:i+4096]
                loss = F.binary_cross_entropy_with_logits(net2(Xt2[idx]), yt2[idx])
                opt2.zero_grad(); loss.backward(); opt2.step()
        net2.eval()
        with torch.no_grad():
            p = torch.sigmoid(net2(Xe2)).numpy()
        fr = frontier(p, y[test_mask], [0.95])
        return fr[0][1]

    # Full (17 features) as baseline
    fr_full = frontier(
        torch.sigmoid(net(torch.from_numpy(Xn[test_mask]))).detach().numpy(),
        y[test_mask], [0.95])
    full_skip = fr_full[0][1]
    print(f"Full 17-feature skip@λ=0.95: {full_skip:.4f}\n")

    print("=" * 120)
    print(f"{'Feature':>22}  {'Cluster':>22}  {'Cost':>5} {'CostZBB':>8} {'|grad|':>8} {'dropΔ':>8}  Description")
    print("=" * 120)

    # Compute drop-one for each feature
    rows = []
    for i, (name, cluster, cost, cost_zbb, desc) in enumerate(ALL_FEATS):
        drop_skip = eval_drop_one(i, epochs=15)
        delta = drop_skip - full_skip  # negative if dropping hurts
        rows.append((name, cluster, cost, cost_zbb, grads[i], delta, desc))
        print(f"{name:>22}  {cluster:>22}  {cost:>5} {cost_zbb:>8} "
              f"{grads[i]:>8.3f} {delta:>8.4f}  {desc}")

    # Group by cluster
    print("\n" + "=" * 120)
    print("CLUSTER SUMMARY (number of apertures, cheapest cost in cluster, total |grad|)")
    print("=" * 120)
    from collections import defaultdict
    clusters = defaultdict(list)
    for row in rows:
        clusters[row[1]].append(row)
    for cluster in sorted(clusters.keys()):
        cs = clusters[cluster]
        total_grad = sum(r[4] for r in cs)
        min_cost = min(r[2] for r in cs)
        min_cost_zbb = min(r[3] for r in cs)
        avg_drop = np.mean([r[5] for r in cs])
        print(f"\n{cluster:>22}")
        print(f"  {len(cs)} apertures. Total |grad|: {total_grad:.3f}. "
              f"Min cost Medusa/ZBB: {min_cost}/{min_cost_zbb}")
        for r in cs:
            print(f"    - {r[0]:>18}  cost={r[2]}/{r[3]}  |grad|={r[4]:.3f}  drop-Δ={r[5]:+.4f}")

    # Cheapest operating point per cluster (for paper "minimal feature set" claim)
    print("\n" + "=" * 120)
    print("MINIMAL SETS (cheapest feature per cluster, different budgets)")
    print("=" * 120)
    cheapest_per_cluster = {}
    for cluster, cs in clusters.items():
        # Pick the feature with lowest cost AND highest |grad|/cost ratio
        best = max(cs, key=lambda r: r[4] / max(r[3], 0.1))  # grad per zero-bb-cost
        cheapest_per_cluster[cluster] = best
        print(f"  {cluster:>22}: {best[0]}  (|grad|={best[4]:.3f}, cost_zbb={best[3]})")


if __name__ == "__main__":
    main()
