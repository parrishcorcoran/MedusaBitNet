"""Round 7: Holographic / black-hole / entanglement / information geometry apertures.

  A. Holographic surface/bulk: entropy ratio of last-K tokens vs next K-back
  B. Entanglement approximation: mutual info between early and late hidden states
     (via correlation of hidden-state PCA projections)
  C. Fisher information score: sensitivity of softmax under hidden-state perturbation
     (simplified: gradient-like proxy via top-k spread)
  D. Uncertainty product: conf × conf_variance
  E. Path integral proxy: cumulative log-prob of recent trajectory (action)
  F. Event horizon distance: distance from current state to "last confident state"
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features
from test_physics_apertures import load_hiddens_f32
from test_unified_estimator_v2 import UnifiedMLP, frontier, LAMBDA_TARGETS
from model import MedusaHeads

SEQ_LEN = 2048; HIDDEN = 2560; VOCAB = 128256


def build_holographic_features(n_seqs=48, surface_k=10, bulk_k=50):
    """Features based on holographic, entanglement, and uncertainty framings."""
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
            entropy = -(probs * probs.log().clamp_min(-20)).sum(-1).numpy()
            conf = probs.max(-1).values.numpy()
            log_prob_of_argmax = probs.max(-1).values.log().clamp_min(-20).numpy()

            # A. Holographic surface/bulk entropy ratio
            # entropy of last surface_k vs last bulk_k
            surface_ent = np.zeros(SEQ_LEN, dtype=np.float32)
            bulk_ent = np.zeros(SEQ_LEN, dtype=np.float32)
            sb_ratio = np.zeros(SEQ_LEN, dtype=np.float32)
            for i in range(bulk_k, SEQ_LEN):
                surf = entropy[i - surface_k:i]
                bulk = entropy[i - bulk_k:i - surface_k]
                surface_ent[i] = surf.mean()
                bulk_ent[i] = bulk.mean() if len(bulk) > 0 else 0
                sb_ratio[i] = surface_ent[i] / (bulk_ent[i] + 1e-6)

            # B. Entanglement approximation: correlation of early vs late hidden states
            # Project hidden states onto their own first PCA axis for tractability
            h_np = h  # [T, H], already float32 from load_hiddens_f32
            correlation_len = np.zeros(SEQ_LEN, dtype=np.float32)
            for i in range(bulk_k, SEQ_LEN):
                early = h_np[i-bulk_k:i-bulk_k//2]  # first half of bulk
                late = h_np[i-bulk_k//2:i]           # last half
                # Correlation of norms
                en = np.linalg.norm(early, axis=-1)
                ln_ = np.linalg.norm(late, axis=-1)
                if len(en) > 1 and len(ln_) > 1:
                    minlen = min(len(en), len(ln_))
                    c = np.corrcoef(en[:minlen], ln_[:minlen])[0, 1]
                    correlation_len[i] = c if np.isfinite(c) else 0

            # C. Fisher information-like score: spread of top-5 logits
            # High spread = sharp distinction; low spread = uncertain
            top5_logits = torch.topk(logits, 5, dim=-1).values  # [T, 5]
            fisher_proxy = top5_logits.std(dim=-1).numpy()  # [T]

            # D. Uncertainty product: conf × var(conf in last-10)
            conf_var_local = np.zeros(SEQ_LEN, dtype=np.float32)
            for i in range(10, SEQ_LEN):
                conf_var_local[i] = conf[i-10:i].std()
            uncert_product = conf * conf_var_local

            # E. Path integral proxy: cumulative log-prob of recent trajectory
            # Sum of log(max prob) for last N positions
            path_integral_ = np.zeros(SEQ_LEN, dtype=np.float32)
            for i in range(20, SEQ_LEN):
                path_integral_[i] = log_prob_of_argmax[i-20:i].sum()

            # F. Event horizon: distance from current h to h at last "confident" position
            # Most recent position where conf > 0.8
            ev_horiz = np.full(SEQ_LEN, 200.0, dtype=np.float32)
            last_confident = -1
            for i in range(SEQ_LEN):
                if last_confident >= 0:
                    d = np.linalg.norm(h_np[i] - h_np[last_confident])
                    ev_horiz[i] = min(200, d)
                if conf[i] > 0.8:
                    last_confident = i

            valid = SEQ_LEN - 2
            ts = np.arange(6, valid, dtype=np.int64)
            feat = np.stack([
                surface_ent[ts + 1],
                bulk_ent[ts + 1],
                sb_ratio[ts + 1],
                correlation_len[ts + 1],
                fisher_proxy[ts + 1],
                uncert_product[ts + 1],
                path_integral_[ts + 1],
                np.log1p(ev_horiz[ts + 1]),
            ], axis=1).astype(np.float32)
            feats.append(feat)
    return np.concatenate(feats, axis=0)


def main():
    X, y, train_mask, test_mask = build_features()
    print(f"existing features: {X.shape}")

    print("\nBuilding holographic / entanglement / info-geometry features...")
    import time; t0 = time.time()
    h_feat = build_holographic_features()
    print(f"  {h_feat.shape} in {time.time()-t0:.1f}s")

    def train_eval_(X_all, feat_idx, label):
        mu = X_all[train_mask][:, feat_idx].mean(axis=0)
        sd = X_all[train_mask][:, feat_idx].std(axis=0) + 1e-6
        Xn = (X_all[:, feat_idx] - mu) / sd
        torch.manual_seed(0)
        net = UnifiedMLP(n_feat=len(feat_idx), hidden=64)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        Xt = torch.from_numpy(Xn[train_mask]); yt = torch.from_numpy(y[train_mask])
        Xe = torch.from_numpy(Xn[test_mask])
        for _ in range(40):
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
        return net, Xn

    train_eval_(X, list(range(17)), "Baseline 17 features")

    X_plus = np.concatenate([X, h_feat], axis=1)
    train_eval_(X_plus, list(range(17, 17 + h_feat.shape[1])), "Holographic ONLY (8)")
    net, Xn = train_eval_(X_plus, list(range(17 + h_feat.shape[1])),
                           "Baseline + holographic (25 features)")

    Xe_r = torch.from_numpy(Xn[test_mask]).clone().requires_grad_(True)
    torch.sigmoid(net(Xe_r)).sum().backward()
    grads = Xe_r.grad.abs().mean(dim=0).numpy()
    names = [
        "content_conf", "content_entropy", "logit_gap", "purity", "top3_cov",
        "top10_cov", "rc10", "rc50", "conf_deriv", "conf_lag1", "conf_lag5",
        "dist_period_log", "dist_newline_log", "rel_pos",
        "agreement_count", "conf_var", "conf_min",
        "surface_ent", "bulk_ent", "sb_ratio", "corr_len", "fisher_proxy",
        "uncert_product", "path_integral", "event_horizon",
    ]
    print("\n=== Feature importance ===")
    order = np.argsort(-grads)
    for idx in order:
        marker = " ★" if idx >= 17 else ""
        print(f"  {names[idx]:>22}  |grad|={grads[idx]:.4f}{marker}")


if __name__ == "__main__":
    main()
