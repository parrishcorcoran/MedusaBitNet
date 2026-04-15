"""Unified multi-dimensional entropy estimator v2 — with Tier A physics features.

Adds distribution-shape observables that physicists would think of as
different moments / observables of the same probabilistic state:

  Dim 1a (content sharpness, higher moments):
    - logit_gap       = top1_logit - top2_logit  (decisiveness)
    - purity          = sum(p^2)                  (inverse participation ratio)
    - top3_coverage   = sum of top-3 probabilities
    - top10_coverage  = sum of top-10 probabilities

  Dim 2+  (trajectory dynamics, higher-order):
    - conf_diff_lag1  = conf[t] - conf[t-1]
    - conf_diff_lag5  = conf[t] - conf[t-5]

Plus all prior features (content_conf, content_entropy, rc10, rc50, conf_deriv,
dist_period_log, dist_newline_log, rel_pos, agreement_count, conf_var,
conf_min). Total: 17 features now.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import MedusaHeads

SEQ_LEN = 2048
HIDDEN  = 2560
VOCAB   = 128256
TOKENIZER_DIR = "/home/cpinchington/MedusaBitNet/models/bitnet-b1.58-2B-4T"
LAMBDA_TARGETS = [0.85, 0.90, 0.95, 0.99]


def get_boundary_token_ids(tok):
    enders, newlines = set(), set()
    for tid in range(VOCAB):
        s = tok.decode([tid])
        if '\n' in s: newlines.add(tid)
        stripped = s.strip()
        if stripped and stripped[-1] in '.!?': enders.add(tid)
    return enders, newlines


def dist_to_last(token_ids, match_set):
    n = len(token_ids); out = np.full(n, 9999, dtype=np.int32); last = -10000
    for i in range(n):
        if last >= 0: out[i] = i - last
        if token_ids[i] in match_set: last = i
    return out


def rolling_mean(arr, window):
    out = np.zeros_like(arr)
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out[i] = (cs[i + 1] - cs[lo]) / (i + 1 - lo)
    return out


def lagged_diff(arr, lag):
    out = np.zeros_like(arr)
    out[lag:] = arr[lag:] - arr[:-lag]
    return out


class UnifiedMLP(nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def frontier(P_score, correct, targets):
    order = np.argsort(-P_score)
    sorted_c = correct[order]
    cum = np.cumsum(sorted_c)
    counts = np.arange(1, len(sorted_c) + 1)
    fid = cum / counts
    out = []
    for λ in targets:
        ok = fid >= λ
        if not ok.any():
            out.append((λ, 0.0, 0.0)); continue
        largest = np.where(ok)[0][-1]
        skip = (largest + 1) / len(sorted_c)
        out.append((λ, skip, fid[largest]))
    return out


def main():
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

            logits = heads(h.unsqueeze(0), lm_head)     # [1, T, K, V]
            logits0 = logits[0, :, 0, :].float()        # [T, V] head-0 logits
            probs0 = F.softmax(logits0, dim=-1)          # [T, V]

            # Head-0 distribution observables (Tier A)
            confs0 = probs0.max(dim=-1).values.numpy()   # [T] peak
            # logit gap: sort logits descending, take top[0]-top[1]
            top2_logits = torch.topk(logits0, 2, dim=-1).values  # [T, 2]
            logit_gap = (top2_logits[:, 0] - top2_logits[:, 1]).numpy()  # [T]
            purity = (probs0 ** 2).sum(dim=-1).numpy()    # [T]
            top3 = torch.topk(probs0, 3, dim=-1).values
            top10 = torch.topk(probs0, 10, dim=-1).values
            top3_cov = top3.sum(dim=-1).numpy()           # [T]
            top10_cov = top10.sum(dim=-1).numpy()         # [T]
            h0_entropy = -(probs0 * torch.log(probs0.clamp_min(1e-12))).sum(-1).numpy()  # [T]

            # All-head preds and confs (for Dim 5)
            probs_all = F.softmax(logits[0].float(), dim=-1)
            confs_all = probs_all.max(dim=-1).values.numpy()  # [T, K]
            preds_all = probs_all.argmax(dim=-1).numpy()      # [T, K]

            # Vanilla greedy next-token
            vlogits = h[1:].to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
            vpred = vlogits.float().argmax(dim=-1).numpy()  # [T-1]

            valid = SEQ_LEN - 2
            t_start, t_end = 6, valid  # need lag5 history, so t >= 5
            ts = np.arange(t_start, t_end, dtype=np.int64)

            # Dim 1
            content_conf = confs0[ts]
            content_entropy = h0_entropy[ts]
            # Dim 1a (new)
            feat_logit_gap = logit_gap[ts]
            feat_purity = purity[ts]
            feat_top3 = top3_cov[ts]
            feat_top10 = top10_cov[ts]

            # Dim 2
            rc10 = rolling_mean(confs0, 10)[ts]
            rc50 = rolling_mean(confs0, 50)[ts]
            conf_deriv = rc10 - rc50
            # Dim 2a (new)
            conf_lag1 = lagged_diff(confs0, 1)[ts]
            conf_lag5 = lagged_diff(confs0, 5)[ts]

            # Dim 3
            de = dist_to_last(token_ids.tolist(), enders).astype(np.float32)
            dn = dist_to_last(token_ids.tolist(), newlines).astype(np.float32)
            de_at = np.log1p(np.minimum(de[ts + 1], 200.0))
            dn_at = np.log1p(np.minimum(dn[ts + 1], 200.0))
            rel_pos = (ts.astype(np.float32) / valid)

            # Dim 5
            h0p = preds_all[:, 0][ts]
            h1p = preds_all[:, 1][ts - 1]
            h2p = preds_all[:, 2][ts - 2]
            h3p = preds_all[:, 3][ts - 3]
            agreement = ((h1p == h0p).astype(np.float32)
                         + (h2p == h0p).astype(np.float32)
                         + (h3p == h0p).astype(np.float32))
            c0 = confs_all[:, 0][ts]
            c1 = confs_all[:, 1][ts - 1]
            c2 = confs_all[:, 2][ts - 2]
            c3 = confs_all[:, 3][ts - 3]
            conf_stack = np.stack([c0, c1, c2, c3], axis=1)
            conf_var = conf_stack.var(axis=1)
            conf_min = conf_stack.min(axis=1)

            feat = np.stack([
                # Dim 1
                content_conf, content_entropy,
                # Dim 1a (TIER A)
                feat_logit_gap, feat_purity, feat_top3, feat_top10,
                # Dim 2
                rc10, rc50, conf_deriv,
                # Dim 2a (TIER A)
                conf_lag1, conf_lag5,
                # Dim 3
                de_at, dn_at, rel_pos,
                # Dim 5
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
    train_mask = is_test == 0; test_mask = is_test == 1
    print(f"[unified v2] features: {X.shape}  train={train_mask.sum()}  test={test_mask.sum()}")

    feat_names = [
        "content_conf", "content_entropy",
        "logit_gap", "purity", "top3_cov", "top10_cov",
        "rc10", "rc50", "conf_deriv",
        "conf_lag1", "conf_lag5",
        "dist_period_log", "dist_newline_log", "rel_pos",
        "agreement_count", "conf_var", "conf_min",
    ]

    mu = X[train_mask].mean(axis=0)
    sd = X[train_mask].std(axis=0) + 1e-6
    Xn = (X - mu) / sd

    # (A) Content-only
    print("\n=== (A) Content-only (raw head-0 softmax peak), TEST ===")
    for λ, skip, fid in frontier(X[:, 0][test_mask], y[test_mask], LAMBDA_TARGETS):
        print(f"  λ={λ:.2f}  skip={skip:.4f}  fidelity={fid:.4f}")

    # (C_v1 equivalent) 11 features (no Tier A)
    print("\n=== (C_v1) Unified MLP, ORIGINAL 11 features (no Tier A), TEST ===")
    v1_idx = [0, 1, 6, 7, 8, 11, 12, 13, 14, 15, 16]  # the 11 features from v1
    net_v1 = UnifiedMLP(n_feat=len(v1_idx), hidden=64)
    opt = torch.optim.Adam(net_v1.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.from_numpy(Xn[train_mask][:, v1_idx])
    yt = torch.from_numpy(y[train_mask])
    Xe = torch.from_numpy(Xn[test_mask][:, v1_idx])
    for epoch in range(40):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 4096):
            idx = perm[i:i+4096]
            logit = net_v1(Xt[idx])
            loss = F.binary_cross_entropy_with_logits(logit, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net_v1.eval()
    with torch.no_grad():
        p_v1 = torch.sigmoid(net_v1(Xe)).numpy()
    for λ, skip, fid in frontier(p_v1, y[test_mask], LAMBDA_TARGETS):
        print(f"  λ={λ:.2f}  skip={skip:.4f}  fidelity={fid:.4f}")

    # (C_v2) All 17 features with Tier A
    print("\n=== (C_v2) Unified MLP, ALL 17 features (incl Tier A), TEST ===")
    net = UnifiedMLP(n_feat=Xn.shape[1], hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.from_numpy(Xn[train_mask])
    yt = torch.from_numpy(y[train_mask])
    Xe = torch.from_numpy(Xn[test_mask])
    for epoch in range(40):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 4096):
            idx = perm[i:i+4096]
            logit = net(Xt[idx])
            loss = F.binary_cross_entropy_with_logits(logit, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p_v2 = torch.sigmoid(net(Xe)).numpy()
    for λ, skip, fid in frontier(p_v2, y[test_mask], LAMBDA_TARGETS):
        print(f"  λ={λ:.2f}  skip={skip:.4f}  fidelity={fid:.4f}")

    # Feature importance
    Xe_r = torch.from_numpy(Xn[test_mask]).clone().requires_grad_(True)
    p = torch.sigmoid(net(Xe_r))
    p.sum().backward()
    grad_abs = Xe_r.grad.abs().mean(dim=0).numpy()
    print("\nFeature importance (mean |grad|):")
    order = np.argsort(-grad_abs)
    for idx in order:
        marker = "★ NEW" if idx in [2, 3, 4, 5, 9, 10] else ""
        print(f"  {feat_names[idx]:>20}  |grad|={grad_abs[idx]:.4f}  {marker}")


if __name__ == "__main__":
    main()
