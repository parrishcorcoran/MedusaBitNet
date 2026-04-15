"""Systematic physics-driven aperture sweep.

Tests 4 geometric apertures on the hidden-state manifold:

  1. State velocity:       ||h_t - h_{t-1}||             (momentum analog)
  2. State acceleration:   ||h_t - 2·h_{t-1} + h_{t-2}||  (curvature of trajectory)
  3. Return time:          steps since nearest-neighbor in recent history
  4. Cluster membership:   distance to nearest cluster center (orbital assignment)

For each, add to the 17-feature baseline and measure frontier improvement.
If multiple apertures each deliver independent gains, the physics framework
is systematically producing real signals — not one-off luck.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features
from test_unified_estimator_v2 import UnifiedMLP, frontier, LAMBDA_TARGETS

SEQ_LEN = 2048; HIDDEN = 2560


def load_hiddens_f32(seq_idx):
    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    chunk = hidden_mm[seq_idx * per_seq : (seq_idx + 1) * per_seq]
    raw32 = chunk.astype(np.uint32) << 16
    return raw32.view(np.float32).reshape(SEQ_LEN, HIDDEN)


def aperture_velocity_and_accel(n_seqs=48):
    """Compute ||h_t - h_{t-1}|| and ||h_t - 2h_{t-1} + h_{t-2}|| at each position."""
    feats = []
    for si in range(n_seqs):
        h = load_hiddens_f32(si)
        h_t = torch.from_numpy(h)
        # Velocity: shift by 1
        vel = torch.zeros(SEQ_LEN)
        vel[1:] = (h_t[1:] - h_t[:-1]).norm(dim=-1)
        # Acceleration: second difference
        accel = torch.zeros(SEQ_LEN)
        accel[2:] = (h_t[2:] - 2 * h_t[1:-1] + h_t[:-2]).norm(dim=-1)
        valid = SEQ_LEN - 2
        t_start, t_end = 6, valid
        ts = np.arange(t_start, t_end, dtype=np.int64)
        # Use features at position t+1 (context end)
        feat = np.stack([vel[ts + 1].numpy(), accel[ts + 1].numpy()], axis=1).astype(np.float32)
        feats.append(feat)
    return np.concatenate(feats, axis=0)


def aperture_return_time(n_seqs=48, tau=0.5):
    """For each position, compute how many steps since a similar state was seen.

    'Similar' = cosine similarity > threshold (tau). Returns log of return time
    and the count of similar states in last 100 positions.
    """
    feats = []
    for si in range(n_seqs):
        h = load_hiddens_f32(si)
        h_t = torch.from_numpy(h)
        # Normalize each state for cosine similarity
        h_norm = h_t / (h_t.norm(dim=-1, keepdim=True) + 1e-9)

        ret_time = np.full(SEQ_LEN, 200, dtype=np.float32)
        similar_count = np.zeros(SEQ_LEN, dtype=np.float32)

        for i in range(1, SEQ_LEN):
            lo = max(0, i - 100)
            cossim = (h_norm[lo:i] * h_norm[i:i+1]).sum(dim=-1).numpy()
            # Return time: max index where cossim > tau (most recent similar state)
            matches = np.where(cossim > tau)[0]
            if len(matches) > 0:
                most_recent = matches[-1] + lo
                ret_time[i] = min(200, i - most_recent)
                similar_count[i] = len(matches)

        valid = SEQ_LEN - 2
        t_start, t_end = 6, valid
        ts = np.arange(t_start, t_end, dtype=np.int64)
        feat = np.stack([
            np.log1p(ret_time[ts + 1]),
            np.log1p(similar_count[ts + 1]),
        ], axis=1).astype(np.float32)
        feats.append(feat)
    return np.concatenate(feats, axis=0)


def aperture_cluster(n_seqs=48, K=32, train_seqs=20):
    """K-means on hidden states from training seqs, then distance to nearest center."""
    # Gather training points
    train_points = []
    for si in range(train_seqs):
        h = load_hiddens_f32(si)
        train_points.append(h)
    train_arr = np.concatenate(train_points, axis=0).astype(np.float32)
    # Subsample for K-means speed
    rng = np.random.default_rng(0)
    sub = rng.choice(len(train_arr), size=min(20000, len(train_arr)), replace=False)
    kmeans_data = torch.from_numpy(train_arr[sub])

    # Simple K-means with torch
    # Initialize centers by choosing K random points
    idx0 = rng.choice(len(kmeans_data), size=K, replace=False)
    centers = kmeans_data[idx0].clone()
    for it in range(20):
        dists = torch.cdist(kmeans_data, centers)  # [N, K]
        assigns = dists.argmin(dim=-1)
        for k in range(K):
            mask = assigns == k
            if mask.sum() > 0:
                centers[k] = kmeans_data[mask].mean(dim=0)

    # Now compute distance to nearest center for each position in each seq
    feats = []
    for si in range(n_seqs):
        h = load_hiddens_f32(si)
        h_t = torch.from_numpy(h)
        dists = torch.cdist(h_t, centers)  # [T, K]
        min_d, _ = dists.min(dim=-1)       # [T]
        # Also: entropy of soft cluster assignment (soft = softmax(-dist))
        soft = F.softmax(-dists * 0.01, dim=-1)
        cluster_entropy = -(soft * torch.log(soft.clamp_min(1e-12))).sum(-1)

        valid = SEQ_LEN - 2
        t_start, t_end = 6, valid
        ts = np.arange(t_start, t_end, dtype=np.int64)
        feat = np.stack([
            min_d[ts + 1].numpy(),
            cluster_entropy[ts + 1].numpy(),
        ], axis=1).astype(np.float32)
        feats.append(feat)
    return np.concatenate(feats, axis=0)


def train_eval(X, y, train_mask, test_mask, feat_idx, label, epochs=30):
    if len(feat_idx) == 0:
        return
    mu = X[train_mask][:, feat_idx].mean(axis=0)
    sd = X[train_mask][:, feat_idx].std(axis=0) + 1e-6
    Xn = (X[:, feat_idx] - mu) / sd
    torch.manual_seed(0)
    net = UnifiedMLP(n_feat=len(feat_idx), hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.from_numpy(Xn[train_mask]); yt = torch.from_numpy(y[train_mask])
    Xe = torch.from_numpy(Xn[test_mask])
    for _ in range(epochs):
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


def main():
    X, y, train_mask, test_mask = build_features()
    print(f"existing features: {X.shape}")

    print("\n[1] Computing velocity+accel features...")
    t0 = time.time()
    vel_accel = aperture_velocity_and_accel()
    print(f"    shape {vel_accel.shape} in {time.time()-t0:.1f}s")

    print("\n[2] Computing return-time features...")
    t0 = time.time()
    ret = aperture_return_time()
    print(f"    shape {ret.shape} in {time.time()-t0:.1f}s")

    print("\n[3] Computing cluster-membership features (K-means K=32)...")
    t0 = time.time()
    clust = aperture_cluster()
    print(f"    shape {clust.shape} in {time.time()-t0:.1f}s")

    # Baselines
    train_eval(X, y, train_mask, test_mask, list(range(17)),
               "Baseline: 17 existing features")

    # Each aperture ALONE (to measure standalone signal)
    for label, feats in [("Velocity+Accel ONLY", vel_accel),
                          ("Return-time ONLY", ret),
                          ("Cluster-membership ONLY", clust)]:
        Xaug = np.concatenate([X, feats], axis=1)
        train_eval(Xaug, y, train_mask, test_mask,
                   list(range(17, 17 + feats.shape[1])), label)

    # Each aperture ADDED to baseline
    for label, feats in [("Baseline + Velocity+Accel", vel_accel),
                          ("Baseline + Return-time", ret),
                          ("Baseline + Cluster-membership", clust)]:
        Xaug = np.concatenate([X, feats], axis=1)
        train_eval(Xaug, y, train_mask, test_mask,
                   list(range(17 + feats.shape[1])), label)

    # All of them together
    X_all = np.concatenate([X, vel_accel, ret, clust], axis=1)
    n_total = 17 + vel_accel.shape[1] + ret.shape[1] + clust.shape[1]
    train_eval(X_all, y, train_mask, test_mask, list(range(n_total)),
               f"Baseline + ALL physics apertures ({n_total} features)")

    # Feature importance with all combined
    mu = X_all[train_mask].mean(axis=0); sd = X_all[train_mask].std(axis=0) + 1e-6
    Xn = (X_all - mu) / sd
    torch.manual_seed(0)
    net = UnifiedMLP(n_feat=n_total, hidden=64)
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

    names_all = [
        # Existing 17
        "content_conf", "content_entropy", "logit_gap", "purity", "top3_cov",
        "top10_cov", "rc10", "rc50", "conf_deriv", "conf_lag1", "conf_lag5",
        "dist_period_log", "dist_newline_log", "rel_pos",
        "agreement_count", "conf_var", "conf_min",
        # Velocity+Accel (2)
        "state_velocity", "state_accel",
        # Return-time (2)
        "return_time_log", "similar_count_log",
        # Cluster (2)
        "cluster_mindist", "cluster_entropy",
    ]
    print("\n=== Full feature importance (23 features) ===")
    order = np.argsort(-grads)
    for idx in order:
        marker = " ★ PHYSICS" if idx >= 17 else ""
        print(f"  {names_all[idx]:>22}  |grad|={grads[idx]:.4f}{marker}")


if __name__ == "__main__":
    main()
