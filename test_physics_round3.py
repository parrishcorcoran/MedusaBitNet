"""Third batch of physics-derived apertures.

  A. Local intrinsic dimension - neighborhood tangent-space rank
  B. Offset similarity spectrum - cosine(h_t, h_{t-k}) for multiple k
  C. Prediction sensitivity - argmax change under small perturbation
  D. Directional persistence - velocity direction consistency over last K
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features
from test_physics_apertures import load_hiddens_f32, train_eval
from test_unified_estimator_v2 import UnifiedMLP, frontier, LAMBDA_TARGETS
from model import MedusaHeads

SEQ_LEN = 2048; HIDDEN = 2560; VOCAB = 128256


def aperture_offset_similarity(n_seqs=48):
    """Cosine similarity between h_t and h_{t-k} for k in {1, 5, 10, 50}."""
    feats = []
    for si in range(n_seqs):
        h = load_hiddens_f32(si)
        h_t = torch.from_numpy(h)
        h_norm = h_t / (h_t.norm(dim=-1, keepdim=True) + 1e-9)
        # Precompute shifted dot products
        cos_k = {}
        for k in [1, 5, 10, 50]:
            shifted = torch.zeros_like(h_norm)
            shifted[k:] = h_norm[:-k]
            cos_k[k] = (h_norm * shifted).sum(dim=-1).numpy()
            cos_k[k][:k] = 0  # no data for early positions
        valid = SEQ_LEN - 2
        ts = np.arange(6, valid, dtype=np.int64)
        feat = np.stack([cos_k[1][ts + 1], cos_k[5][ts + 1],
                         cos_k[10][ts + 1], cos_k[50][ts + 1]], axis=1).astype(np.float32)
        feats.append(feat)
    return np.concatenate(feats, axis=0)


def aperture_local_intrinsic_dim(n_seqs=48, K=30):
    """Local intrinsic dim at each position via TwoNN-like measure on
    the K-nearest recent hidden states (FIFO of 50)."""
    feats = []
    for si in range(n_seqs):
        h = load_hiddens_f32(si)
        h_t = torch.from_numpy(h)
        local_dim = np.zeros(SEQ_LEN, dtype=np.float32)
        local_spread = np.zeros(SEQ_LEN, dtype=np.float32)
        for i in range(K + 1, SEQ_LEN):
            lo = max(0, i - 50)
            window = h_t[lo:i]  # recent states
            cur = h_t[i:i+1]
            dists = torch.cdist(cur, window).squeeze(0)
            if len(dists) < 3:
                continue
            sorted_d, _ = torch.sort(dists)
            r1 = sorted_d[0]; r2 = sorted_d[1]
            if r1 > 1e-6 and r2 > r1:
                mu = float(r2 / r1)
                if mu > 1 + 1e-6:
                    local_dim[i] = float(1 / (np.log(mu) + 1e-9))
            # Also: spread (std of neighbor distances)
            local_spread[i] = float(sorted_d[:10].std())
        valid = SEQ_LEN - 2
        ts = np.arange(6, valid, dtype=np.int64)
        feat = np.stack([
            np.clip(local_dim[ts + 1], 0, 100),
            local_spread[ts + 1],
        ], axis=1).astype(np.float32)
        feats.append(feat)
    return np.concatenate(feats, axis=0)


def aperture_prediction_sensitivity(n_seqs=48, noise_scale=0.05):
    """How much does head-0's argmax change if we add small Gaussian noise
    to the hidden state? Measures susceptibility / response function."""
    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    heads = MedusaHeads(HIDDEN, VOCAB, 4, 1, dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    feats = []
    torch.manual_seed(0)
    with torch.no_grad():
        for si in range(n_seqs):
            h = load_hiddens_f32(si)
            h_t = torch.from_numpy(h).to(torch.bfloat16)
            # Clean logits and prediction
            logits_clean = heads(h_t.unsqueeze(0), lm_head)[0, :, 0, :].float()
            pred_clean = logits_clean.argmax(dim=-1)
            conf_clean = F.softmax(logits_clean, dim=-1).max(-1).values
            # Perturbed hidden state
            noise = torch.randn_like(h_t) * noise_scale * h_t.abs().mean()
            h_noisy = h_t + noise.to(h_t.dtype)
            logits_noisy = heads(h_noisy.unsqueeze(0), lm_head)[0, :, 0, :].float()
            pred_noisy = logits_noisy.argmax(dim=-1)
            conf_noisy = F.softmax(logits_noisy, dim=-1).max(-1).values
            # Sensitivity features
            pred_unchanged = (pred_clean == pred_noisy).float().numpy()
            conf_change = (conf_clean - conf_noisy).abs().numpy()
            # Logit distance between clean and noisy
            logit_l2 = (logits_clean - logits_noisy).norm(dim=-1).numpy()

            valid = SEQ_LEN - 2
            ts = np.arange(6, valid, dtype=np.int64)
            feat = np.stack([
                pred_unchanged[ts],
                conf_change[ts],
                np.log1p(logit_l2[ts]),
            ], axis=1).astype(np.float32)
            feats.append(feat)
    return np.concatenate(feats, axis=0)


def aperture_directional_persistence(n_seqs=48, K=5):
    """Is velocity vector consistent over last K steps?

    For each step, velocity = h_t - h_{t-1}. Cosine similarity between
    current velocity and avg of last K velocities = persistence.
    """
    feats = []
    for si in range(n_seqs):
        h = load_hiddens_f32(si)
        h_t = torch.from_numpy(h)
        vel = torch.zeros_like(h_t)
        vel[1:] = h_t[1:] - h_t[:-1]
        vel_norm = vel / (vel.norm(dim=-1, keepdim=True) + 1e-9)
        persistence = np.zeros(SEQ_LEN, dtype=np.float32)
        for i in range(K + 1, SEQ_LEN):
            recent_vel_avg = vel_norm[i-K:i].mean(dim=0)
            recent_vel_avg = recent_vel_avg / (recent_vel_avg.norm() + 1e-9)
            persistence[i] = float((vel_norm[i] * recent_vel_avg).sum())
        valid = SEQ_LEN - 2
        ts = np.arange(6, valid, dtype=np.int64)
        feat = persistence[ts + 1].reshape(-1, 1).astype(np.float32)
        feats.append(feat)
    return np.concatenate(feats, axis=0)


def main():
    X, y, train_mask, test_mask = build_features()
    print(f"existing features: {X.shape}")

    print("\n[A] Offset similarity spectrum...")
    t0 = time.time(); off = aperture_offset_similarity(); print(f"    {off.shape} in {time.time()-t0:.1f}s")
    print("\n[B] Local intrinsic dimension...")
    t0 = time.time(); lid = aperture_local_intrinsic_dim(); print(f"    {lid.shape} in {time.time()-t0:.1f}s")
    print("\n[C] Prediction sensitivity...")
    t0 = time.time(); sens = aperture_prediction_sensitivity(); print(f"    {sens.shape} in {time.time()-t0:.1f}s")
    print("\n[D] Directional persistence...")
    t0 = time.time(); persist = aperture_directional_persistence(); print(f"    {persist.shape} in {time.time()-t0:.1f}s")

    # Baseline
    train_eval(X, y, train_mask, test_mask, list(range(17)),
               "Baseline 17 features")

    # Each ALONE
    for label, feats in [("Offset-similarity ONLY", off),
                          ("Local intrinsic dim ONLY", lid),
                          ("Prediction sensitivity ONLY", sens),
                          ("Directional persistence ONLY", persist)]:
        Xaug = np.concatenate([X, feats], axis=1)
        train_eval(Xaug, y, train_mask, test_mask,
                   list(range(17, 17 + feats.shape[1])), label)

    # Each ADDED
    for label, feats in [("Baseline + Offset-similarity", off),
                          ("Baseline + Local intrinsic dim", lid),
                          ("Baseline + Prediction sensitivity", sens),
                          ("Baseline + Directional persistence", persist)]:
        Xaug = np.concatenate([X, feats], axis=1)
        train_eval(Xaug, y, train_mask, test_mask,
                   list(range(17 + feats.shape[1])), label)

    # All round-3 combined
    X_all3 = np.concatenate([X, off, lid, sens, persist], axis=1)
    n = X_all3.shape[1]
    train_eval(X_all3, y, train_mask, test_mask, list(range(n)),
               f"Baseline + ALL round-3 apertures ({n} features)")

    # Feature importance
    mu = X_all3[train_mask].mean(axis=0); sd = X_all3[train_mask].std(axis=0) + 1e-6
    Xn = (X_all3 - mu) / sd
    torch.manual_seed(0)
    net = UnifiedMLP(n_feat=n, hidden=96)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.from_numpy(Xn[train_mask]); yt = torch.from_numpy(y[train_mask])
    for _ in range(40):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 4096):
            bi = perm[i:i+4096]
            loss = F.binary_cross_entropy_with_logits(net(Xt[bi]), yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    Xe_r = torch.from_numpy(Xn[test_mask]).clone().requires_grad_(True)
    torch.sigmoid(net(Xe_r)).sum().backward()
    grads = Xe_r.grad.abs().mean(dim=0).numpy()

    names = [
        "content_conf", "content_entropy", "logit_gap", "purity", "top3_cov",
        "top10_cov", "rc10", "rc50", "conf_deriv", "conf_lag1", "conf_lag5",
        "dist_period_log", "dist_newline_log", "rel_pos",
        "agreement_count", "conf_var", "conf_min",
        "cos_lag1", "cos_lag5", "cos_lag10", "cos_lag50",
        "local_id", "local_spread",
        "pred_unchanged", "conf_change", "logit_l2",
        "dir_persistence",
    ]
    print("\n=== Feature importance with 17 + round-3 ===")
    order = np.argsort(-grads)
    for idx in order[:20]:
        marker = " ★" if idx >= 17 else ""
        print(f"  {names[idx]:>22}  |grad|={grads[idx]:.4f}{marker}")


if __name__ == "__main__":
    main()
