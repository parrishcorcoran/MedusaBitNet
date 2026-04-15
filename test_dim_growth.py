"""How has effective dimensionality grown as we added physics apertures?

Computes participation ratio (effective dim) for:
  - Original 17 features
  - + Tier B
  - + Tier C
  - + All round-2 physics (cluster, velocity, moments, etc.)
  - + Round-7 holographic (event_horizon, fisher_proxy, uncert, etc.)
  - + Token-reuse (lexical H2O)

Shows the dimensionality trajectory as we add genuinely-orthogonal apertures.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features
from test_physics_apertures import (
    aperture_velocity_and_accel, aperture_cluster,
)
from test_physics_more import (
    softmax_higher_moments, hidden_state_norm_features, free_energy_analog,
)
from test_hidden_neighborhood import build_neighborhood_features
from test_holographic import build_holographic_features
from test_tier_b_features import build_tier_b
from test_token_reuse import build_token_reuse_features


def pr(X):
    """Participation ratio."""
    mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-6
    Xn = (X - mu) / sd
    U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
    eigs = S ** 2 / (len(Xn) - 1)
    p = eigs / eigs.sum()
    return 1.0 / (p ** 2).sum(), eigs / eigs.sum()


def main():
    print("Building all feature sets...")
    X_base, y, train_mask, test_mask = build_features()

    # Baseline 17
    pr17, ev = pr(X_base)
    print(f"\n17 baseline features:")
    print(f"  participation ratio = {pr17:.2f}")

    # + Tier B (5 features)
    import numpy as np, torch
    from transformers import AutoTokenizer
    # Use an existing tier B build function — need to import tokens / confs
    tokens_mm = np.memmap("data/tokens.bin", dtype=np.uint32, mode="r")

    print("\nGathering head-0 confidences per seq (needed for Tier B)...")
    from model import MedusaHeads
    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    heads = MedusaHeads(2560, 128256, 4, 1, dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    SEQ_LEN = 2048; HIDDEN = 2560
    per_seq = SEQ_LEN * HIDDEN
    confs_full = []
    with torch.no_grad():
        for si in range(48):
            off = si * per_seq
            chunk = hidden_mm[off:off + per_seq]
            h_raw = chunk.astype(np.uint32) << 16
            h = h_raw.view(np.float32).reshape(SEQ_LEN, HIDDEN)
            h_t = torch.from_numpy(h).to(torch.bfloat16)
            logits = heads(h_t.unsqueeze(0), lm_head)[0, :, 0, :].float()
            cf = F.softmax(logits, dim=-1).max(-1).values.numpy()
            confs_full.append(cf)

    tier_b = build_tier_b(tokens_mm, confs_full)
    X_bB = np.concatenate([X_base, tier_b], axis=1)
    pr_B, _ = pr(X_bB)
    print(f"\n+ Tier B (5 features, n=22):")
    print(f"  participation ratio = {pr_B:.2f}")

    # + Round-2 physics (velocity, cluster, moments, norm, FE)
    print("\nBuilding round-2 physics apertures...")
    va = aperture_velocity_and_accel()
    cl = aperture_cluster()
    mo = softmax_higher_moments()
    hn = hidden_state_norm_features()
    fe = free_energy_analog(cl)
    round2 = np.concatenate([va, cl, mo, hn, fe], axis=1)
    X_bR2 = np.concatenate([X_base, round2], axis=1)
    pr_R2, _ = pr(X_bR2)
    print(f"\n+ Round-2 physics ({round2.shape[1]} features, n={X_bR2.shape[1]}):")
    print(f"  participation ratio = {pr_R2:.2f}")

    # + Neighborhood
    print("\nBuilding neighborhood...")
    nbr = build_neighborhood_features()
    X_bN = np.concatenate([X_bR2, nbr], axis=1)
    pr_N, _ = pr(X_bN)
    print(f"\n+ Neighborhood ({nbr.shape[1]} features, n={X_bN.shape[1]}):")
    print(f"  participation ratio = {pr_N:.2f}")

    # + Holographic
    print("\nBuilding holographic...")
    hol = build_holographic_features()
    X_bH = np.concatenate([X_bN, hol], axis=1)
    pr_H, _ = pr(X_bH)
    print(f"\n+ Holographic ({hol.shape[1]} features, n={X_bH.shape[1]}):")
    print(f"  participation ratio = {pr_H:.2f}")

    # + Token reuse
    print("\nBuilding token-reuse...")
    tr = build_token_reuse_features()
    X_bT = np.concatenate([X_bH, tr], axis=1)
    pr_T, eigs_T = pr(X_bT)
    print(f"\n+ Token reuse ({tr.shape[1]} features, n={X_bT.shape[1]}):")
    print(f"  participation ratio = {pr_T:.2f}")

    # Summary
    print(f"\n=== Dimensionality growth ===")
    print(f"  17 baseline:                     PR = {pr17:.2f}")
    print(f"  + Tier B:                         PR = {pr_B:.2f}")
    print(f"  + Round-2 physics:                PR = {pr_R2:.2f}")
    print(f"  + Neighborhood:                   PR = {pr_N:.2f}")
    print(f"  + Holographic:                    PR = {pr_H:.2f}")
    print(f"  + Token-reuse (final):            PR = {pr_T:.2f}")
    print(f"\n  Growth: {pr17:.2f} -> {pr_T:.2f}  = +{(pr_T/pr17 - 1)*100:.0f}%")
    print(f"\n  Intrinsic manifold dimension:  14.4 (measured by TwoNN on raw hidden states)")
    print(f"  Current gap: {14.4 - pr_T:.2f} dimensions still unapertured")


if __name__ == "__main__":
    main()
