"""Round 8: 5 more physics apertures targeting untested dimensions.

  A. Phase-angle / SVD projection: project current h onto top singular vectors of
     recent window — "phase coordinates" orthogonal to magnitude
  B. Topological H0 (connected components count at threshold) of recent window
  C. Field correlation (two-point correlation function with mean-subtract)
  D. RG multi-scale: coarse-graining at scales {1, 3, 9} — fixed-point detection
  E. Quantum superposition structure: effective rank of top-K token indices
"""
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


def aperture_phase_svd(n_seqs=48, window=20, n_components=3):
    """SVD of recent window, project current h onto top components."""
    feats = []
    for si in range(n_seqs):
        h = load_hiddens_f32(si)  # [T, H] float32
        phase_coords = np.zeros((SEQ_LEN, n_components), dtype=np.float32)
        residual = np.zeros(SEQ_LEN, dtype=np.float32)
        for i in range(window, SEQ_LEN):
            win = h[i-window:i]  # [window, H]
            # Center
            win_c = win - win.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(win_c, full_matrices=False)
            # Project current h onto top-K components
            h_cur = h[i] - win.mean(axis=0)
            coords = h_cur @ Vt[:n_components].T  # [n_components]
            phase_coords[i] = coords
            # Residual: how much of h_cur is NOT in the top-K directions
            reconstruct = coords @ Vt[:n_components]
            residual[i] = np.linalg.norm(h_cur - reconstruct)

        valid = SEQ_LEN - 2
        ts = np.arange(6, valid, dtype=np.int64)
        feat = np.concatenate([
            phase_coords[ts + 1],
            residual[ts + 1].reshape(-1, 1),
        ], axis=1).astype(np.float32)
        feats.append(feat)
    return np.concatenate(feats, axis=0)


def aperture_topology_h0(n_seqs=48, window=30, thresh_scale=1.0):
    """Number of connected components (H0 Betti) of recent window at distance
    threshold = thresh_scale * median_dist."""
    feats = []
    for si in range(n_seqs):
        h = load_hiddens_f32(si)
        h_t = torch.from_numpy(h)
        num_components = np.zeros(SEQ_LEN, dtype=np.float32)
        giant_frac = np.zeros(SEQ_LEN, dtype=np.float32)
        for i in range(window, SEQ_LEN):
            win = h_t[i-window:i]
            d = torch.cdist(win, win).numpy()
            np.fill_diagonal(d, np.inf)
            med = float(np.median(d[d < 1e9]))
            if med < 1e-6: continue
            thresh = thresh_scale * med
            # Union-find over pairs within threshold
            parent = list(range(window))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb: parent[ra] = rb
            for a in range(window):
                for b in range(a+1, window):
                    if d[a, b] < thresh:
                        union(a, b)
            roots = set(find(x) for x in range(window))
            num_components[i] = len(roots)
            # Giant component fraction
            from collections import Counter
            sizes = Counter(find(x) for x in range(window))
            giant_frac[i] = max(sizes.values()) / window

        valid = SEQ_LEN - 2
        ts = np.arange(6, valid, dtype=np.int64)
        feat = np.stack([num_components[ts + 1], giant_frac[ts + 1]], axis=1).astype(np.float32)
        feats.append(feat)
    return np.concatenate(feats, axis=0)


def aperture_field_correlation(n_seqs=48):
    """Two-point correlation function of hidden state norms at various lags.
    C(k) = <h_t · h_{t-k}> - <h_t><h_{t-k}> at lag k, normalized."""
    feats = []
    for si in range(n_seqs):
        h = load_hiddens_f32(si)
        h_t = torch.from_numpy(h)
        # Use norms as the scalar field
        norms = h_t.norm(dim=-1).numpy()
        corr_k = {k: np.zeros(SEQ_LEN, dtype=np.float32) for k in [1, 3, 10]}
        for i in range(50, SEQ_LEN):
            window_norms = norms[max(0, i-50):i]
            mean_n = window_norms.mean()
            var_n = window_norms.var() + 1e-9
            for k in [1, 3, 10]:
                if i >= k:
                    # C(k) = E[(n_t - mean)(n_{t-k} - mean)] / var
                    pairs = [(norms[j] - mean_n) * (norms[j-k] - mean_n)
                             for j in range(max(k, i-30), i)]
                    if pairs:
                        corr_k[k][i] = np.mean(pairs) / var_n

        valid = SEQ_LEN - 2
        ts = np.arange(6, valid, dtype=np.int64)
        feat = np.stack([corr_k[1][ts + 1], corr_k[3][ts + 1],
                         corr_k[10][ts + 1]], axis=1).astype(np.float32)
        feats.append(feat)
    return np.concatenate(feats, axis=0)


def aperture_rg_multiscale(n_seqs=48):
    """Coarse-grain hidden states at scales 1, 3, 9. Measure fixed-point
    behavior: do coarse-grained states match fine-grained?"""
    feats = []
    for si in range(n_seqs):
        h = load_hiddens_f32(si)
        # For scale s: average hidden states over last s positions
        # Then compare h[i] to the scale-s coarse-graining
        div_1 = np.zeros(SEQ_LEN, dtype=np.float32)  # distance at scale 1
        div_3 = np.zeros(SEQ_LEN, dtype=np.float32)
        div_9 = np.zeros(SEQ_LEN, dtype=np.float32)
        for i in range(10, SEQ_LEN):
            h_cur = h[i]
            div_1[i] = np.linalg.norm(h_cur - h[i-1])
            div_3[i] = np.linalg.norm(h_cur - h[max(0, i-3):i].mean(axis=0))
            div_9[i] = np.linalg.norm(h_cur - h[max(0, i-9):i].mean(axis=0))

        valid = SEQ_LEN - 2
        ts = np.arange(6, valid, dtype=np.int64)
        feat = np.stack([div_1[ts + 1], div_3[ts + 1], div_9[ts + 1]],
                        axis=1).astype(np.float32)
        feats.append(feat)
    return np.concatenate(feats, axis=0)


def aperture_superposition_structure(n_seqs=48, K=32):
    """Top-K token spread in embedding space.

    For each position, get top-K tokens from head-0 softmax. Look at the
    spread of their token_embd vectors in hidden space. Tight spread =
    semantically coherent superposition. Wide spread = ambiguous.
    """
    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    heads = MedusaHeads(HIDDEN, VOCAB, 4, 1, dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)
    # lm_head is [V, H] — each row is the token's embedding vector

    feats = []
    with torch.no_grad():
        for si in range(n_seqs):
            h = load_hiddens_f32(si)
            h_t = torch.from_numpy(h).to(torch.bfloat16)
            logits = heads(h_t.unsqueeze(0), lm_head)[0, :, 0, :].float()
            probs = F.softmax(logits, dim=-1)

            top_k_idx = torch.topk(probs, K, dim=-1).indices.numpy()  # [T, K]
            top_k_probs = torch.topk(probs, K, dim=-1).values.numpy()

            tok_spread = np.zeros(SEQ_LEN, dtype=np.float32)
            eff_rank = np.zeros(SEQ_LEN, dtype=np.float32)

            for i in range(SEQ_LEN):
                idx = top_k_idx[i]
                tok_embs = lm_head[idx].float().numpy()  # [K, H]
                # Spread: std of embeddings
                centroid = tok_embs.mean(axis=0)
                spread = np.linalg.norm(tok_embs - centroid, axis=-1).mean()
                tok_spread[i] = spread
                # Effective rank = exp(entropy of top-K probs)
                p = top_k_probs[i] / top_k_probs[i].sum()
                h_ent = -np.sum(p * np.log(p + 1e-12))
                eff_rank[i] = np.exp(h_ent)

            valid = SEQ_LEN - 2
            ts = np.arange(6, valid, dtype=np.int64)
            feat = np.stack([tok_spread[ts], eff_rank[ts]], axis=1).astype(np.float32)
            feats.append(feat)
    return np.concatenate(feats, axis=0)


def main():
    X, y, train_mask, test_mask = build_features()
    print(f"existing features: {X.shape}")

    import time
    print("\n[A] Phase-angle SVD projection...")
    t0 = time.time(); phase = aperture_phase_svd(); print(f"  {phase.shape} in {time.time()-t0:.1f}s")
    print("\n[B] Topological H0...")
    t0 = time.time(); topo = aperture_topology_h0(); print(f"  {topo.shape} in {time.time()-t0:.1f}s")
    print("\n[C] Field correlation...")
    t0 = time.time(); field = aperture_field_correlation(); print(f"  {field.shape} in {time.time()-t0:.1f}s")
    print("\n[D] RG multi-scale...")
    t0 = time.time(); rg = aperture_rg_multiscale(); print(f"  {rg.shape} in {time.time()-t0:.1f}s")
    print("\n[E] Superposition structure...")
    t0 = time.time(); sup = aperture_superposition_structure(); print(f"  {sup.shape} in {time.time()-t0:.1f}s")

    train_eval(X, y, train_mask, test_mask, list(range(17)),
               "Baseline 17 features")

    for label, feats in [("Phase-SVD ONLY", phase),
                          ("Topological ONLY", topo),
                          ("Field correlation ONLY", field),
                          ("RG multiscale ONLY", rg),
                          ("Superposition ONLY", sup)]:
        Xaug = np.concatenate([X, feats], axis=1)
        train_eval(Xaug, y, train_mask, test_mask,
                   list(range(17, 17 + feats.shape[1])), label)

    # All combined
    X_all = np.concatenate([X, phase, topo, field, rg, sup], axis=1)
    n = X_all.shape[1]
    train_eval(X_all, y, train_mask, test_mask, list(range(n)),
               f"Baseline + round-8 ({n} features)")

    # Feature importance
    mu = X_all[train_mask].mean(axis=0); sd = X_all[train_mask].std(axis=0) + 1e-6
    Xn = (X_all - mu) / sd
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
        "phase_c1", "phase_c2", "phase_c3", "phase_residual",
        "topo_components", "topo_giant",
        "field_corr_1", "field_corr_3", "field_corr_10",
        "rg_div1", "rg_div3", "rg_div9",
        "sup_spread", "sup_eff_rank",
    ]
    print(f"\n=== Feature importance (round-8 total {n} features) ===")
    order = np.argsort(-grads)
    for idx in order[:20]:
        marker = " ★" if idx >= 17 else ""
        print(f"  {names[idx]:>22}  |grad|={grads[idx]:.4f}{marker}")


if __name__ == "__main__":
    main()
