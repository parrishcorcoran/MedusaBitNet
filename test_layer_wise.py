"""Round 9: Layer-wise apertures (Ryu-Takayanagi depth projection).

Uses cached hidden states at layers 5, 15, and 29 (result_norm). Measures:

  A. Layer velocities: ||h_15 - h_5||, ||h_29 - h_15||
  B. Trajectory angle: cos(dir_5→15, dir_15→29) — does state continue or turn?
  C. Layer cosine similarities: cos(h_5, h_15), cos(h_15, h_29), cos(h_5, h_29)
  D. Norm evolution: ||h_l|| at each depth and their ratios
  E. Layer-wise prediction convergence: does argmax from h_5 match h_29?

Per Gemini: if hidden state "shoots straight" across layers -> laminar/predictable.
If it "wanders" (high angle between layer velocities) -> boundary/hard.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features
from test_unified_estimator_v2 import UnifiedMLP, frontier, LAMBDA_TARGETS
from model import MedusaHeads

SEQ_LEN = 2048; HIDDEN = 2560; VOCAB = 128256


def load_hidden(path, si):
    mm = np.memmap(path, dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    chunk = mm[si * per_seq : (si + 1) * per_seq]
    raw32 = chunk.astype(np.uint32) << 16
    return raw32.view(np.float32).copy().reshape(SEQ_LEN, HIDDEN)


def build_layer_features(n_seqs=48):
    """Build features using layers 5, 15, 29 (result_norm)."""
    # load lm_head for early-prediction feature
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    feats = []
    for si in range(n_seqs):
        h5 = load_hidden("data/hidden_gguf_layer5.bin", si)
        h15 = load_hidden("data/hidden_gguf_layer15.bin", si)
        h29 = load_hidden("data/hidden_gguf_v2.bin", si)  # result_norm

        # A. Layer velocities (distances between layers)
        vel_5_15 = np.linalg.norm(h15 - h5, axis=-1)  # [T]
        vel_15_29 = np.linalg.norm(h29 - h15, axis=-1)
        vel_5_29 = np.linalg.norm(h29 - h5, axis=-1)

        # B. Trajectory angle — does state continue in same direction layer-over-layer?
        dir_5_15 = h15 - h5
        dir_15_29 = h29 - h15
        # Normalize each row
        d1_n = dir_5_15 / (np.linalg.norm(dir_5_15, axis=-1, keepdims=True) + 1e-9)
        d2_n = dir_15_29 / (np.linalg.norm(dir_15_29, axis=-1, keepdims=True) + 1e-9)
        layer_angle = (d1_n * d2_n).sum(axis=-1)  # cosine similarity in [-1, 1]

        # C. Cosine similarity between layers
        def cos_sim(a, b):
            an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
            bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
            return (an * bn).sum(axis=-1)
        cos_5_15 = cos_sim(h5, h15)
        cos_15_29 = cos_sim(h15, h29)
        cos_5_29 = cos_sim(h5, h29)

        # D. Norm evolution
        n5 = np.linalg.norm(h5, axis=-1)
        n15 = np.linalg.norm(h15, axis=-1)
        n29 = np.linalg.norm(h29, axis=-1)
        norm_ratio_5_29 = n5 / (n29 + 1e-6)
        norm_ratio_15_29 = n15 / (n29 + 1e-6)

        # E. Early prediction convergence
        # Project h5 and h29 through lm_head, see if argmax matches
        with torch.no_grad():
            logits_5 = torch.from_numpy(h5).to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
            logits_29 = torch.from_numpy(h29).to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
            argmax_5 = logits_5.float().argmax(dim=-1).numpy()
            argmax_29 = logits_29.float().argmax(dim=-1).numpy()
            early_agrees = (argmax_5 == argmax_29).astype(np.float32)

        valid = SEQ_LEN - 2
        ts = np.arange(6, valid, dtype=np.int64)
        feat = np.stack([
            np.log1p(vel_5_15[ts + 1]),
            np.log1p(vel_15_29[ts + 1]),
            np.log1p(vel_5_29[ts + 1]),
            layer_angle[ts + 1],
            cos_5_15[ts + 1],
            cos_15_29[ts + 1],
            cos_5_29[ts + 1],
            np.log1p(n5[ts + 1]),
            np.log1p(n15[ts + 1]),
            np.log1p(n29[ts + 1]),
            norm_ratio_5_29[ts + 1],
            norm_ratio_15_29[ts + 1],
            early_agrees[ts + 1],
        ], axis=1).astype(np.float32)
        feats.append(feat)

    return np.concatenate(feats, axis=0)


def main():
    X, y, train_mask, test_mask = build_features()
    print(f"existing features: {X.shape}")

    print("\nBuilding layer-wise features (depth 5, 15, 29)...")
    import time; t0 = time.time()
    layer_feat = build_layer_features(n_seqs=48)
    print(f"  {layer_feat.shape} in {time.time()-t0:.1f}s")

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

    X_plus = np.concatenate([X, layer_feat], axis=1)
    train_eval_(X_plus, list(range(17, 17 + layer_feat.shape[1])),
                "Layer-wise ONLY (13 features)")
    net, Xn = train_eval_(X_plus, list(range(17 + layer_feat.shape[1])),
                           f"Baseline + layer-wise ({17 + layer_feat.shape[1]} features)")

    Xe_r = torch.from_numpy(Xn[test_mask]).clone().requires_grad_(True)
    torch.sigmoid(net(Xe_r)).sum().backward()
    grads = Xe_r.grad.abs().mean(dim=0).numpy()
    names = [
        "content_conf", "content_entropy", "logit_gap", "purity", "top3_cov",
        "top10_cov", "rc10", "rc50", "conf_deriv", "conf_lag1", "conf_lag5",
        "dist_period_log", "dist_newline_log", "rel_pos",
        "agreement_count", "conf_var", "conf_min",
        "vel_5_15", "vel_15_29", "vel_5_29", "layer_angle",
        "cos_5_15", "cos_15_29", "cos_5_29",
        "norm_5", "norm_15", "norm_29",
        "norm_ratio_5_29", "norm_ratio_15_29",
        "early_agrees",
    ]
    print("\n=== Feature importance ===")
    order = np.argsort(-grads)
    for idx in order[:20]:
        marker = " ★" if idx >= 17 else ""
        print(f"  {names[idx]:>22}  |grad|={grads[idx]:.4f}{marker}")


if __name__ == "__main__":
    main()
