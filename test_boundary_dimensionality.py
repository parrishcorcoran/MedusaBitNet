"""How many independent dimensions does our boundary-layer measurement space have?

If we PCA the full feature matrix (all the signals we've measured), how many
principal components carry meaningful variance? That number is our empirical
estimate of the effective dimensionality of the boundary layer measurement.

Prediction from thesis: ~4-5 independent dimensions based on cluster analysis.
If PCA confirms ≤5 meaningful dimensions, the boundary layer IS low-dimensional
in a measurable sense, and we should stop looking for more dimensions and
focus on sharper apertures on the existing ones.
"""
import numpy as np
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features


def main():
    X, y, train_mask, test_mask = build_features()
    print(f"feature matrix: {X.shape}")

    # Normalize each feature (otherwise PCA is dominated by scale)
    mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-6
    Xn = (X - mu) / sd

    # PCA via SVD of centered matrix
    U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
    # Eigenvalues of cov = (S^2) / (N-1)
    eigs = S ** 2 / (len(Xn) - 1)
    total_var = eigs.sum()
    cum_var = np.cumsum(eigs) / total_var

    print(f"\n=== Principal components ordered by variance captured ===")
    print(f"{'PC':>4} {'eigenvalue':>12} {'fraction':>10} {'cumulative':>12}")
    print("-" * 45)
    for i, (lam, cum) in enumerate(zip(eigs, cum_var)):
        print(f"{i+1:>4} {lam:>12.4f} {eigs[i]/total_var:>10.4f} {cum:>12.4f}")
        if cum > 0.99: break

    print(f"\nCount of PCs needed to capture:")
    for target in [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]:
        n = int(np.searchsorted(cum_var, target)) + 1
        print(f"  {int(target*100)}% variance: {n} components")

    # Participation ratio (effective dimensionality)
    p = eigs / total_var
    pr = 1.0 / (p ** 2).sum()
    print(f"\nParticipation ratio (effective dimensionality): {pr:.2f}")
    print("  (A measure of how many components carry comparable variance)")

    # Look at top 5 PCs — which ORIGINAL features do they mix?
    feat_names = [
        "content_conf", "content_entropy",
        "logit_gap", "purity", "top3_cov", "top10_cov",
        "rc10", "rc50", "conf_deriv",
        "conf_lag1", "conf_lag5",
        "dist_period_log", "dist_newline_log", "rel_pos",
        "agreement_count", "conf_var", "conf_min",
    ]
    print(f"\n=== Top 5 principal components: which features dominate each? ===")
    for pc_idx in range(min(5, Vt.shape[0])):
        loadings = Vt[pc_idx]
        order = np.argsort(-np.abs(loadings))
        print(f"\nPC{pc_idx+1} (fraction {eigs[pc_idx]/total_var:.3f}):")
        for j in order[:5]:
            print(f"  {feat_names[j]:>20}  loading={loadings[j]:+.3f}")

    # Answer the headline question
    k_effective = int(np.searchsorted(cum_var, 0.90)) + 1
    print(f"\n*** BOUNDARY LAYER DIMENSIONALITY (90% variance): {k_effective} dimensions ***")


if __name__ == "__main__":
    main()
