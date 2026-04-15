"""Continue the physics aperture sweep — test remaining candidates.

Apertures tested:
  A. Free-energy analog:   -log p(h) estimated from cluster density
  B. Softmax skewness:     3rd moment of softmax distribution
  C. Softmax kurtosis:     4th moment (tail weight of distribution)
  D. Spectral FFT:         low/mid/high frequency power of conf trajectory
  E. Hidden state norm:    magnitude of current hidden state (is it in a 'large' region?)
  F. Norm drift:           ||h|| - ||h_{t-1}|| (size change between consecutive states)

Proxies for infrastructure-dependent ones:
  G. SVD-scale at multiple ranks (approximation of layer-wise holographic signal)
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features
from test_physics_apertures import (
    load_hiddens_f32, aperture_velocity_and_accel, aperture_return_time,
    aperture_cluster, train_eval,
)
from test_unified_estimator_v2 import UnifiedMLP, frontier, LAMBDA_TARGETS
from model import MedusaHeads

SEQ_LEN = 2048; HIDDEN = 2560; VOCAB = 128256


def softmax_higher_moments(n_seqs=48):
    """3rd and 4th central moments of head-0 softmax distribution."""
    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    heads = MedusaHeads(HIDDEN, VOCAB, 4, 1, dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    feats = []
    with torch.no_grad():
        for si in range(n_seqs):
            h = load_hiddens_f32(si)
            h_t = torch.from_numpy(h).to(torch.bfloat16)
            logits = heads(h_t.unsqueeze(0), lm_head)[0, :, 0, :].float()
            probs = F.softmax(logits, dim=-1)  # [T, V]
            # Treat probs as a distribution over tokens; compute moments around mean
            token_idx = torch.arange(VOCAB).float()
            mean_tok = (probs * token_idx).sum(-1)  # [T]
            # Central moments
            dev = token_idx.unsqueeze(0) - mean_tok.unsqueeze(1)  # [T, V]
            var = (probs * dev ** 2).sum(-1)
            m3 = (probs * dev ** 3).sum(-1)
            m4 = (probs * dev ** 4).sum(-1)
            std = var.sqrt().clamp_min(1e-6)
            skew = m3 / std ** 3
            kurt = m4 / std ** 4

            valid = SEQ_LEN - 2
            ts = np.arange(6, valid, dtype=np.int64)
            feat = np.stack([skew[ts].numpy(), kurt[ts].numpy()], axis=1).astype(np.float32)
            feats.append(feat)
    return np.concatenate(feats, axis=0)


def spectral_fft_conf(n_seqs=48):
    """FFT of rolling head-0 confidence over last 32 positions. Low/mid/high
    frequency power as 3 features."""
    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    heads = MedusaHeads(HIDDEN, VOCAB, 4, 1, dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    feats = []
    with torch.no_grad():
        for si in range(n_seqs):
            h = load_hiddens_f32(si)
            h_t = torch.from_numpy(h).to(torch.bfloat16)
            logits = heads(h_t.unsqueeze(0), lm_head)[0, :, 0, :].float()
            probs = F.softmax(logits, dim=-1)
            conf = probs.max(dim=-1).values.numpy()  # [T]

            # For each position, FFT of last 32-window
            W = 32
            fft_low = np.zeros(SEQ_LEN, dtype=np.float32)
            fft_mid = np.zeros(SEQ_LEN, dtype=np.float32)
            fft_high = np.zeros(SEQ_LEN, dtype=np.float32)
            for i in range(W, SEQ_LEN):
                window = conf[i-W:i]
                window = window - window.mean()  # detrend
                spec = np.abs(np.fft.rfft(window))  # [W/2+1] = 17
                fft_low[i] = spec[1:5].sum()    # low freqs
                fft_mid[i] = spec[5:12].sum()   # mid freqs
                fft_high[i] = spec[12:].sum()   # high freqs

            valid = SEQ_LEN - 2
            ts = np.arange(6, valid, dtype=np.int64)
            feat = np.stack([
                np.log1p(fft_low[ts + 1]),
                np.log1p(fft_mid[ts + 1]),
                np.log1p(fft_high[ts + 1]),
            ], axis=1).astype(np.float32)
            feats.append(feat)
    return np.concatenate(feats, axis=0)


def hidden_state_norm_features(n_seqs=48):
    """Hidden state L2 norm and drift (change from previous state's norm)."""
    feats = []
    for si in range(n_seqs):
        h = load_hiddens_f32(si)
        h_t = torch.from_numpy(h)
        norms = h_t.norm(dim=-1).numpy()
        # Drift
        drift = np.zeros(SEQ_LEN, dtype=np.float32)
        drift[1:] = np.abs(norms[1:] - norms[:-1])

        valid = SEQ_LEN - 2
        ts = np.arange(6, valid, dtype=np.int64)
        feat = np.stack([
            np.log1p(norms[ts + 1]),
            drift[ts + 1],
        ], axis=1).astype(np.float32)
        feats.append(feat)
    return np.concatenate(feats, axis=0)


def free_energy_analog(cluster_feats):
    """Free-energy-like signal: -log( density(h) ) — use cluster_mindist as proxy.

    cluster_feats has [mindist, cluster_entropy]. Free energy ~ log(mindist) —
    already proxied. But we can compute log(1 + min_dist * 0.01) as a bounded
    free-energy-like score.
    """
    mindist = cluster_feats[:, 0]
    entropy = cluster_feats[:, 1]
    # Free-energy-like: -log density. Higher mindist = lower density = higher FE
    free_energy = np.log1p(mindist * 0.01).astype(np.float32)
    # Also: entropy-adjusted free-energy
    fe_adjusted = (free_energy - entropy * 0.1).astype(np.float32)
    return np.stack([free_energy, fe_adjusted], axis=1)


def main():
    X, y, train_mask, test_mask = build_features()
    print(f"existing features: {X.shape}")

    # Compute previous physics apertures (we have infrastructure)
    print("\n[1] Velocity+Accel...")
    vel_accel = aperture_velocity_and_accel()
    print("[2] Return-time...")
    ret = aperture_return_time()
    print("[3] Cluster membership...")
    clust = aperture_cluster()
    print("[4] Softmax higher moments...")
    t0 = time.time(); moments = softmax_higher_moments(); print(f"    in {time.time()-t0:.1f}s")
    print("[5] Spectral FFT of confidence trajectory...")
    t0 = time.time(); spec = spectral_fft_conf(); print(f"    in {time.time()-t0:.1f}s")
    print("[6] Hidden-state norm + drift...")
    t0 = time.time(); hnorm = hidden_state_norm_features(); print(f"    in {time.time()-t0:.1f}s")
    print("[7] Free-energy analog (proxy from cluster distance)...")
    fe = free_energy_analog(clust)

    train_eval(X, y, train_mask, test_mask, list(range(17)),
               "Baseline 17 features")

    # Test each new aperture ALONE
    for label, feats in [
        ("Softmax skewness+kurtosis ONLY", moments),
        ("Spectral FFT of conf ONLY", spec),
        ("Hidden norm + drift ONLY", hnorm),
        ("Free-energy proxy ONLY", fe),
    ]:
        Xaug = np.concatenate([X, feats], axis=1)
        train_eval(Xaug, y, train_mask, test_mask,
                   list(range(17, 17 + feats.shape[1])), label)

    # Each ADDED to baseline
    for label, feats in [
        ("Baseline + Skew/Kurt", moments),
        ("Baseline + FFT spec", spec),
        ("Baseline + Norm/Drift", hnorm),
        ("Baseline + FE proxy", fe),
    ]:
        Xaug = np.concatenate([X, feats], axis=1)
        train_eval(Xaug, y, train_mask, test_mask,
                   list(range(17 + feats.shape[1])), label)

    # The mega model: ALL physics apertures tested so far
    X_mega = np.concatenate([X, vel_accel, ret, clust, moments, spec, hnorm, fe], axis=1)
    n_total = X_mega.shape[1]
    train_eval(X_mega, y, train_mask, test_mask, list(range(n_total)),
               f"MEGA: 17 baseline + 7 physics aperture sets ({n_total} features)")

    # Feature importance of mega
    mu = X_mega[train_mask].mean(axis=0); sd = X_mega[train_mask].std(axis=0) + 1e-6
    Xn = (X_mega - mu) / sd
    torch.manual_seed(0)
    net = UnifiedMLP(n_feat=n_total, hidden=96)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.from_numpy(Xn[train_mask]); yt = torch.from_numpy(y[train_mask])
    for _ in range(50):
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
        # Baseline 17
        "content_conf", "content_entropy", "logit_gap", "purity", "top3_cov",
        "top10_cov", "rc10", "rc50", "conf_deriv", "conf_lag1", "conf_lag5",
        "dist_period_log", "dist_newline_log", "rel_pos",
        "agreement_count", "conf_var", "conf_min",
        # Velocity+Accel
        "state_velocity", "state_accel",
        # Return
        "return_time_log", "similar_count_log",
        # Cluster
        "cluster_mindist", "cluster_entropy",
        # Moments
        "softmax_skew", "softmax_kurt",
        # Spectral
        "fft_low", "fft_mid", "fft_high",
        # Norm
        "hidden_norm", "norm_drift",
        # Free energy
        "free_energy", "fe_adjusted",
    ]
    print(f"\n=== Full feature importance ({n_total} features) ===")
    order = np.argsort(-grads)
    for idx in order[:20]:
        marker = " ★" if idx >= 17 else ""
        print(f"  {names[idx]:>22}  |grad|={grads[idx]:.4f}{marker}")


if __name__ == "__main__":
    main()
