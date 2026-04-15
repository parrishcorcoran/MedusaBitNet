"""Check how redundant our 17 features are. Correlations + ablation.

If two features are highly correlated (|ρ| > 0.9), they're measuring
the same dimension through different apertures — no new information.
We want orthogonal features, not redundant ones.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import MedusaHeads

# Re-import helpers from v2 script
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_unified_estimator_v2 import (
    get_boundary_token_ids, dist_to_last, rolling_mean, lagged_diff,
    UnifiedMLP, frontier, SEQ_LEN, HIDDEN, VOCAB, TOKENIZER_DIR, LAMBDA_TARGETS,
)


def build_features():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    enders, newlines = get_boundary_token_ids(tok)

    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    heads = MedusaHeads(HIDDEN, VOCAB, 4, 1, dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    tokens_mm = np.memmap("data/tokens.bin", dtype=np.uint32, mode="r")
    n_seqs = 48; seq_split = 36

    features_per_seq = []; labels_per_seq = []; split_flags = []

    with torch.no_grad():
        for si in range(n_seqs):
            off = si * per_seq
            chunk = hidden_mm[off:off + per_seq]
            h = (torch.from_numpy(chunk.copy()).view(torch.bfloat16)
                       .view(SEQ_LEN, HIDDEN))
            token_ids = tokens_mm[si * SEQ_LEN : (si + 1) * SEQ_LEN]

            logits = heads(h.unsqueeze(0), lm_head)
            logits0 = logits[0, :, 0, :].float()
            probs0 = F.softmax(logits0, dim=-1)
            confs0 = probs0.max(dim=-1).values.numpy()
            top2 = torch.topk(logits0, 2, dim=-1).values
            logit_gap = (top2[:, 0] - top2[:, 1]).numpy()
            purity = (probs0 ** 2).sum(dim=-1).numpy()
            top3_cov = torch.topk(probs0, 3, dim=-1).values.sum(dim=-1).numpy()
            top10_cov = torch.topk(probs0, 10, dim=-1).values.sum(dim=-1).numpy()
            h0_entropy = -(probs0 * torch.log(probs0.clamp_min(1e-12))).sum(-1).numpy()

            probs_all = F.softmax(logits[0].float(), dim=-1)
            confs_all = probs_all.max(dim=-1).values.numpy()
            preds_all = probs_all.argmax(dim=-1).numpy()

            vlogits = h[1:].to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
            vpred = vlogits.float().argmax(dim=-1).numpy()

            valid = SEQ_LEN - 2
            t_start, t_end = 6, valid
            ts = np.arange(t_start, t_end, dtype=np.int64)

            content_conf = confs0[ts]
            content_entropy = h0_entropy[ts]
            rc10 = rolling_mean(confs0, 10)[ts]
            rc50 = rolling_mean(confs0, 50)[ts]
            conf_deriv = rc10 - rc50
            conf_lag1 = lagged_diff(confs0, 1)[ts]
            conf_lag5 = lagged_diff(confs0, 5)[ts]

            de = dist_to_last(token_ids.tolist(), enders).astype(np.float32)
            dn = dist_to_last(token_ids.tolist(), newlines).astype(np.float32)
            de_at = np.log1p(np.minimum(de[ts + 1], 200.0))
            dn_at = np.log1p(np.minimum(dn[ts + 1], 200.0))
            rel_pos = (ts.astype(np.float32) / valid)

            h0p = preds_all[:, 0][ts]
            h1p = preds_all[:, 1][ts - 1]
            h2p = preds_all[:, 2][ts - 2]
            h3p = preds_all[:, 3][ts - 3]
            agreement = ((h1p == h0p) + (h2p == h0p) + (h3p == h0p)).astype(np.float32)
            c0 = confs_all[:, 0][ts]
            c1 = confs_all[:, 1][ts - 1]
            c2 = confs_all[:, 2][ts - 2]
            c3 = confs_all[:, 3][ts - 3]
            conf_stack = np.stack([c0, c1, c2, c3], axis=1)
            conf_var = conf_stack.var(axis=1)
            conf_min = conf_stack.min(axis=1)

            feat = np.stack([
                content_conf, content_entropy,
                logit_gap[ts], purity[ts], top3_cov[ts], top10_cov[ts],
                rc10, rc50, conf_deriv,
                conf_lag1, conf_lag5,
                de_at, dn_at, rel_pos,
                agreement, conf_var, conf_min,
            ], axis=1).astype(np.float32)

            label = (h0p == vpred[ts]).astype(np.float32)

            features_per_seq.append(feat)
            labels_per_seq.append(label)
            split_flags.append(0 if si < seq_split else 1)

    X = np.concatenate(features_per_seq, axis=0)
    y = np.concatenate(labels_per_seq, axis=0)
    is_test = np.concatenate([
        np.full(len(features_per_seq[i]), split_flags[i], dtype=np.int8)
        for i in range(n_seqs)
    ])
    return X, y, (is_test == 0), (is_test == 1)


def main():
    feat_names = [
        "content_conf", "content_entropy",
        "logit_gap", "purity", "top3_cov", "top10_cov",
        "rc10", "rc50", "conf_deriv",
        "conf_lag1", "conf_lag5",
        "dist_period_log", "dist_newline_log", "rel_pos",
        "agreement_count", "conf_var", "conf_min",
    ]

    X, y, train_mask, test_mask = build_features()
    print(f"features: {X.shape}\n")

    # Correlation matrix (pearson) on train
    Xt = X[train_mask]
    C = np.corrcoef(Xt.T)
    # Identify highly-correlated pairs
    print("=== Strongly correlated feature pairs (|ρ| > 0.90) ===")
    for i in range(len(feat_names)):
        for j in range(i+1, len(feat_names)):
            if abs(C[i, j]) > 0.90:
                print(f"  ρ={C[i,j]:+.3f}   {feat_names[i]:>20} ↔ {feat_names[j]:<20}")

    print("\n=== Moderately correlated pairs (0.70 < |ρ| < 0.90) ===")
    for i in range(len(feat_names)):
        for j in range(i+1, len(feat_names)):
            if 0.70 < abs(C[i, j]) <= 0.90:
                print(f"  ρ={C[i,j]:+.3f}   {feat_names[i]:>20} ↔ {feat_names[j]:<20}")

    # Ablation: drop each feature, re-train, measure skip at λ=0.95
    mu = Xt.mean(axis=0); sd = Xt.std(axis=0) + 1e-6
    Xn = (X - mu) / sd

    def train_and_score(feat_idx, epochs=30):
        net = UnifiedMLP(n_feat=len(feat_idx), hidden=64)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        Xtt = torch.from_numpy(Xn[train_mask][:, feat_idx])
        ytt = torch.from_numpy(y[train_mask])
        Xet = torch.from_numpy(Xn[test_mask][:, feat_idx])
        for _ in range(epochs):
            perm = torch.randperm(len(Xtt))
            for i in range(0, len(Xtt), 4096):
                idx = perm[i:i+4096]
                logit = net(Xtt[idx])
                loss = F.binary_cross_entropy_with_logits(logit, ytt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(Xet)).numpy()
        fr = frontier(p, y[test_mask], [0.95])[0]
        return fr[1]  # skip rate

    all_idx = list(range(len(feat_names)))
    print("\n=== Leave-one-out ablation, skip rate @ λ=0.95 on TEST ===")
    full_skip = train_and_score(all_idx)
    print(f"  full 17 features:           skip = {full_skip:.4f}")
    print(f"  drop each feature:")
    for k in range(len(feat_names)):
        minus_k = [i for i in all_idx if i != k]
        s = train_and_score(minus_k)
        delta = s - full_skip
        marker = "  ↓↓ HURT" if delta < -0.002 else ("  ~= noop" if abs(delta) < 0.002 else "  ↑ HELP")
        print(f"    -{feat_names[k]:>20}  skip = {s:.4f}  Δ={delta:+.4f}{marker}")


if __name__ == "__main__":
    main()
