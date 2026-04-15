"""Greedy feature selection + cost analysis.

Two scenarios:
  A) 'Free' gate (heads aren't run if gate skips) — only structural/trajectory
     features available. Cheapest gate possible.
  B) 'Medusa gate' — heads run anyway, so all features are free marginal cost.

For each scenario, greedy forward-select features one at a time by which
gives the biggest gain in skip rate at λ=0.95 on held-out. Produces Pareto
curves of (features used, skip achieved).

This answers: 'did we pick the right 5 features, or were cheaper/better
ones available?'
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features
from test_unified_estimator_v2 import UnifiedMLP, frontier, LAMBDA_TARGETS


ALL_NAMES = [
    "content_conf", "content_entropy",         # 0, 1 — need head forward
    "logit_gap", "purity", "top3_cov", "top10_cov",  # 2-5 — need head forward
    "rc10", "rc50", "conf_deriv",              # 6-8 — free (past confs)
    "conf_lag1", "conf_lag5",                  # 9-10 — free
    "dist_period_log", "dist_newline_log",     # 11-12 — free
    "rel_pos",                                 # 13 — free
    "agreement_count", "conf_var", "conf_min", # 14-16 — need ALL 4 heads
]

# Cost model (in abstract head-forward units):
# - Free (just past context / token stream): cost = 0
# - Needs head-0 forward (we run head-0 anyway for emission): cost = 0 (marginal)
# - Needs additional heads 1-3: cost = 3 (three extra head forwards)
COST = {
    "content_conf": 0, "content_entropy": 0,
    "logit_gap": 0, "purity": 0, "top3_cov": 0, "top10_cov": 0,
    "rc10": 0, "rc50": 0, "conf_deriv": 0,
    "conf_lag1": 0, "conf_lag5": 0,
    "dist_period_log": 0, "dist_newline_log": 0, "rel_pos": 0,
    "agreement_count": 3, "conf_var": 3, "conf_min": 3,
}
# For scenario A (zero-backbone gate), head forward itself costs:
COST_ZERO_BB = {
    "content_conf": 1, "content_entropy": 1,
    "logit_gap": 1, "purity": 1, "top3_cov": 1, "top10_cov": 1,
    "rc10": 0, "rc50": 0, "conf_deriv": 0,
    "conf_lag1": 0, "conf_lag5": 0,
    "dist_period_log": 0, "dist_newline_log": 0, "rel_pos": 0,
    "agreement_count": 4, "conf_var": 4, "conf_min": 4,
}


def train_frontier(X, y, train_mask, test_mask, feat_idx, epochs=30, seed=0):
    torch.manual_seed(seed)
    if len(feat_idx) == 0:
        # Null model: just use base rate
        return [(λ, 0.0, 0.0) for λ in LAMBDA_TARGETS]
    mu = X[train_mask][:, feat_idx].mean(axis=0)
    sd = X[train_mask][:, feat_idx].std(axis=0) + 1e-6
    Xn = (X[:, feat_idx] - mu) / sd

    net = UnifiedMLP(n_feat=len(feat_idx), hidden=48)
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


def greedy_select(X, y, train_mask, test_mask, available_idx, target_metric_idx=2):
    """Greedy forward selection by skip rate at target fidelity (0.95 default).
    Returns list of (step, feat_idx, skip_rate)."""
    selected = []
    order = []
    for step in range(len(available_idx)):
        best = (-1, -1)
        for idx in available_idx:
            if idx in selected: continue
            trial = selected + [idx]
            fr = train_frontier(X, y, train_mask, test_mask, trial, epochs=20)
            skip = fr[target_metric_idx][1]
            if skip > best[0]:
                best = (skip, idx)
        if best[1] < 0: break
        selected.append(best[1])
        order.append((step + 1, best[1], best[0]))
        print(f"  step {step+1}: add {ALL_NAMES[best[1]]:>20}  skip@λ=0.95 = {best[0]:.4f}")
    return order


def main():
    X, y, train_mask, test_mask = build_features()
    print(f"features: {X.shape}")

    # Scenario B: everything available, greedy forward selection
    print("\n=== Greedy forward selection, all 17 features available ===")
    all_idx = list(range(17))
    order = greedy_select(X, y, train_mask, test_mask, all_idx, target_metric_idx=2)

    # Show plateau point
    prev_skip = 0.0
    plateau_at = None
    for i, (_, _, skip) in enumerate(order):
        if skip - prev_skip < 0.001 and i >= 3:
            plateau_at = i
            break
        prev_skip = skip
    if plateau_at:
        print(f"\n  Plateau: {plateau_at} features give most of the signal")

    # Scenario A: only FREE features (no head forward at this position)
    print("\n=== Free-only features (zero-head-forward gate) ===")
    free_idx = [i for i, name in enumerate(ALL_NAMES) if COST_ZERO_BB[name] == 0]
    print(f"  {len(free_idx)} free features available: {[ALL_NAMES[i] for i in free_idx]}")
    order_free = greedy_select(X, y, train_mask, test_mask, free_idx, target_metric_idx=2)

    # Scenario C: head-0 features free (single head forward), no multi-head
    print("\n=== Head-0 + free features (single head forward, no multi-head agreement) ===")
    head0_or_free = [i for i, name in enumerate(ALL_NAMES)
                     if COST_ZERO_BB[name] <= 1]
    print(f"  {len(head0_or_free)} features available: {[ALL_NAMES[i] for i in head0_or_free]}")
    order_head0 = greedy_select(X, y, train_mask, test_mask, head0_or_free,
                                  target_metric_idx=2)

    # Summary
    print(f"\n=== Summary: skip rate at λ=0.95 by feature budget ===")
    print(f"{'feats':>6} {'all':>12} {'head0+free':>12} {'free_only':>12}")
    for i in range(1, max(len(order), len(order_head0), len(order_free)) + 1):
        all_skip = order[i-1][2] if i <= len(order) else order[-1][2]
        h0_skip = order_head0[i-1][2] if i <= len(order_head0) else order_head0[-1][2] if order_head0 else 0
        fr_skip = order_free[i-1][2] if i <= len(order_free) else order_free[-1][2] if order_free else 0
        print(f"{i:>6} {all_skip:>12.4f} {h0_skip:>12.4f} {fr_skip:>12.4f}")

    # Also check: what's our current 5-feature "one-per-cluster" set's skip?
    my_5 = [ALL_NAMES.index(n) for n in
            ["content_entropy", "agreement_count", "dist_period_log", "rel_pos", "rc50"]]
    fr_my = train_frontier(X, y, train_mask, test_mask, my_5)
    # And: greedy-5 (top 5 by forward selection)
    greedy_5 = [o[1] for o in order[:5]]
    fr_greedy = train_frontier(X, y, train_mask, test_mask, greedy_5)
    # And: cheapest 5
    # Pick 5 features with lowest cost that greedily improve skip
    fr_cheap = train_frontier(X, y, train_mask, test_mask, [o[1] for o in order_free[:5]])

    print(f"\n=== 5-feature comparison (skip@λ=0.95) ===")
    print(f"  My one-per-cluster pick:     skip = {fr_my[2][1]:.4f}")
    print(f"    features: {[ALL_NAMES[i] for i in my_5]}")
    print(f"  Greedy-best 5:               skip = {fr_greedy[2][1]:.4f}")
    print(f"    features: {[ALL_NAMES[i] for i in greedy_5]}")
    print(f"  Best free-only 5:            skip = {fr_cheap[2][1]:.4f}  (no head forwards at t)")
    print(f"    features: {[ALL_NAMES[o[1]] for o in order_free[:5]]}")


if __name__ == "__main__":
    main()
