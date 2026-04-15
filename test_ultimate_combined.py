"""MEGA combined test: every productive aperture from the overnight session.

Feature groups:
  - Baseline 17 (original)
  - Tier B (trigram, bigram, vocab, dist_same, conf_var20)
  - Neighborhood (3)
  - Round-2 physics: velocity, cluster, moments, hidden_norm, FE
  - Token reuse (5, H2O lexical)
  - Round-4: KL trajectory, self-attn, entropy flux
  - Round-7 holographic: event_horizon, fisher_proxy, uncert_product, path_integral, surface/bulk
  - Round-8: phase_residual, rg_div, superposition
  - Layer-wise: norm_29, cos_15_29, etc

Goal: measure the actual frontier when ALL apertures are combined.
Also measure participation ratio — should grow past 11.14 if layer-wise etc. add new dimensions.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features
from test_physics_apertures import (
    aperture_velocity_and_accel, aperture_cluster
)
from test_physics_more import (
    softmax_higher_moments, hidden_state_norm_features, free_energy_analog,
)
from test_hidden_neighborhood import build_neighborhood_features
from test_holographic import build_holographic_features
from test_tier_b_features import build_tier_b
from test_token_reuse import build_token_reuse_features
from test_physics_round8 import (
    aperture_phase_svd, aperture_superposition_structure, aperture_rg_multiscale,
)
from test_layer_wise import build_layer_features
from test_unified_estimator_v2 import UnifiedMLP, frontier, LAMBDA_TARGETS
from model import MedusaHeads


SEQ_LEN = 2048; HIDDEN = 2560; VOCAB = 128256


def pr(X):
    mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-6
    Xn = (X - mu) / sd
    U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
    eigs = S ** 2 / (len(Xn) - 1)
    p = eigs / eigs.sum()
    return 1.0 / (p ** 2).sum()


def main():
    import time
    X_base, y, train_mask, test_mask = build_features()

    # Collect all feature arrays
    print("Collecting feature arrays...")
    tokens_mm = np.memmap("data/tokens.bin", dtype=np.uint32, mode="r")
    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    heads = MedusaHeads(HIDDEN, VOCAB, 4, 1, dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)
    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
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

    t0 = time.time()
    tier_b = build_tier_b(tokens_mm, confs_full); print(f"  tier_b {tier_b.shape} in {time.time()-t0:.1f}s")
    t0 = time.time()
    va = aperture_velocity_and_accel(); print(f"  velocity {va.shape} in {time.time()-t0:.1f}s")
    t0 = time.time()
    cl = aperture_cluster(); print(f"  cluster {cl.shape} in {time.time()-t0:.1f}s")
    t0 = time.time()
    mo = softmax_higher_moments(); print(f"  moments {mo.shape} in {time.time()-t0:.1f}s")
    t0 = time.time()
    hn = hidden_state_norm_features(); print(f"  hnorm {hn.shape} in {time.time()-t0:.1f}s")
    fe = free_energy_analog(cl); print(f"  FE {fe.shape}")
    t0 = time.time()
    nbr = build_neighborhood_features(); print(f"  neighborhood {nbr.shape} in {time.time()-t0:.1f}s")
    t0 = time.time()
    hol = build_holographic_features(); print(f"  holographic {hol.shape} in {time.time()-t0:.1f}s")
    t0 = time.time()
    tr = build_token_reuse_features(); print(f"  token reuse {tr.shape} in {time.time()-t0:.1f}s")
    t0 = time.time()
    phase = aperture_phase_svd(); print(f"  phase {phase.shape} in {time.time()-t0:.1f}s")
    t0 = time.time()
    rg = aperture_rg_multiscale(); print(f"  RG {rg.shape} in {time.time()-t0:.1f}s")
    t0 = time.time()
    sup = aperture_superposition_structure(); print(f"  superposition {sup.shape} in {time.time()-t0:.1f}s")
    t0 = time.time()
    layer = build_layer_features(); print(f"  layer-wise {layer.shape} in {time.time()-t0:.1f}s")

    X_all = np.concatenate([
        X_base, tier_b, va, cl, mo, hn, fe, nbr, hol, tr, phase, rg, sup, layer,
    ], axis=1)
    n = X_all.shape[1]
    print(f"\n=== ULTIMATE COMBINED: {n} features ===")

    mu = X_all[train_mask].mean(axis=0); sd = X_all[train_mask].std(axis=0) + 1e-6
    Xn = (X_all - mu) / sd
    torch.manual_seed(0)
    net = UnifiedMLP(n_feat=n, hidden=128)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.from_numpy(Xn[train_mask]); yt = torch.from_numpy(y[train_mask])
    Xe = torch.from_numpy(Xn[test_mask])
    for ep in range(60):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 4096):
            bi = perm[i:i+4096]
            loss = F.binary_cross_entropy_with_logits(net(Xt[bi]), yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(Xe)).numpy()
    print(f"\nFrontier on held-out ({n} features):")
    for λ, skip, fid in frontier(p, y[test_mask], LAMBDA_TARGETS):
        print(f"  λ={λ:.2f}  skip={skip:.4f}  fidelity={fid:.4f}")

    # Participation ratio
    pr_all = pr(X_all)
    print(f"\nParticipation ratio of {n} features: {pr_all:.2f}")
    print(f"Intrinsic manifold dim: 14.4")
    print(f"Coverage: {pr_all/14.4*100:.1f}% of intrinsic dimensionality")


if __name__ == "__main__":
    main()
