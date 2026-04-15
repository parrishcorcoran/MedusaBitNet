"""Unified multi-dimensional entropy estimator (the thesis experiment).

Fuses four orthogonal dimensions of the predictive cloud into one scalar
via a tiny MLP, trained by self-distillation (backbone's observed greedy
token as label):

  Dim 1 (content sharpness):    head-0 softmax peak, head-0 entropy
  Dim 2 (trajectory dynamics):  rolling_conf_10, rolling_conf_50, derivative
  Dim 3 (structural prior):     dist_period, dist_newline, rel_pos
  Dim 5 (cross-aperture agreement): head agreement count, head conf variance

Compares the fidelity/skip frontier of:
  (A) content-only threshold (Dim 1 alone)
  (B) joint bucket of rolling_conf x dist_period (partial Dim 2 x Dim 3)
  (C) unified MLP on all four dimensions

If (C) Pareto-dominates (A) and (B), the multi-dimensional claim is
empirically supported on held-out data.
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
    """For each target fidelity, find largest skip rate achievable."""
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
    print("[unified] loading tokenizer and boundary sets...")
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

    n_seqs = 48  # more data for MLP training
    seq_split = 36  # first 36 train, last 12 test

    features_per_seq = []
    labels_per_seq = []
    split_flags = []  # 0 = train, 1 = test

    with torch.no_grad():
        for si in range(n_seqs):
            off = si * per_seq
            chunk = hidden_mm[off:off + per_seq]
            h = (torch.from_numpy(chunk.copy()).view(torch.bfloat16)
                       .view(SEQ_LEN, HIDDEN))
            token_ids = tokens_mm[si * SEQ_LEN : (si + 1) * SEQ_LEN]

            logits = heads(h.unsqueeze(0), lm_head)  # [1, T, K, V]
            probs = F.softmax(logits[0].float(), dim=-1)  # [T, K, V]
            confs = probs.max(dim=-1).values  # [T, K]
            preds = probs.argmax(dim=-1)      # [T, K]
            # head-0 entropy
            h0_probs = probs[:, 0, :]
            h0_entropy = -(h0_probs * torch.log(h0_probs.clamp_min(1e-12))).sum(-1)  # [T]

            vlogits = h[1:].to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
            vpred = vlogits.float().argmax(dim=-1).numpy()  # [T-1]

            valid = SEQ_LEN - 2
            t_start = 3  # need head-3's anchor (t-3) >= 0
            t_end = valid
            ts = np.arange(t_start, t_end, dtype=np.int64)
            N = len(ts)

            # === Dim 1: content sharpness ===
            content_conf = confs[:, 0].numpy()[ts]            # [N]
            content_entropy = h0_entropy.numpy()[ts]          # [N]

            # === Dim 2: trajectory dynamics ===
            full_conf = confs[:, 0].numpy()
            rc10 = rolling_mean(full_conf, 10)[ts]
            rc50 = rolling_mean(full_conf, 50)[ts]
            conf_deriv = (rc10 - rc50)

            # === Dim 3: structural prior ===
            de = dist_to_last(token_ids.tolist(), enders).astype(np.float32)
            dn = dist_to_last(token_ids.tolist(), newlines).astype(np.float32)
            # Use context at t+1 (what's been emitted up to target-1)
            de_at = np.log1p(np.minimum(de[ts + 1], 200.0))
            dn_at = np.log1p(np.minimum(dn[ts + 1], 200.0))
            rel_pos = (ts.astype(np.float32) / valid)

            # === Dim 5: cross-aperture agreement ===
            head0_pred = preds[:, 0].numpy()
            head1_pred = preds[:, 1].numpy()
            head2_pred = preds[:, 2].numpy()
            head3_pred = preds[:, 3].numpy()
            h0 = head0_pred[ts]
            h1 = head1_pred[ts - 1]
            h2 = head2_pred[ts - 2]
            h3 = head3_pred[ts - 3]
            agreement_count = ((h1 == h0).astype(np.float32)
                               + (h2 == h0).astype(np.float32)
                               + (h3 == h0).astype(np.float32))  # 0..3
            c0 = confs[:, 0].numpy()[ts]
            c1 = confs[:, 1].numpy()[ts - 1]
            c2 = confs[:, 2].numpy()[ts - 2]
            c3 = confs[:, 3].numpy()[ts - 3]
            conf_stack = np.stack([c0, c1, c2, c3], axis=1)  # [N, 4]
            conf_var = conf_stack.var(axis=1)
            conf_min = conf_stack.min(axis=1)

            feat = np.stack([
                # Dim 1
                content_conf, content_entropy,
                # Dim 2
                rc10, rc50, conf_deriv,
                # Dim 3
                de_at, dn_at, rel_pos,
                # Dim 5
                agreement_count, conf_var, conf_min,
            ], axis=1).astype(np.float32)

            # Label: does head-0 predict vanilla greedy for target t+2?
            label = (h0 == vpred[ts]).astype(np.float32)

            features_per_seq.append(feat)
            labels_per_seq.append(label)
            split_flags.append(0 if si < seq_split else 1)

    X = np.concatenate(features_per_seq, axis=0)
    y = np.concatenate(labels_per_seq, axis=0)
    is_test = np.concatenate([
        np.full(len(features_per_seq[i]), split_flags[i], dtype=np.int8)
        for i in range(n_seqs)
    ])
    train_mask = is_test == 0
    test_mask = is_test == 1
    print(f"[unified] features shape: {X.shape}  train={train_mask.sum()}  test={test_mask.sum()}")
    feat_names = ["content_conf", "content_entropy",
                  "rc10", "rc50", "conf_deriv",
                  "dist_period_log", "dist_newline_log", "rel_pos",
                  "agreement_count", "conf_var", "conf_min"]

    # Normalize features by TRAIN statistics
    mu = X[train_mask].mean(axis=0)
    sd = X[train_mask].std(axis=0) + 1e-6
    Xn = (X - mu) / sd

    # --- Baseline A: content-only threshold on test ---
    content = X[:, 0]  # raw content_conf (not normalized)
    print("\n=== (A) Content-only threshold (raw head-0 softmax peak) on TEST ===")
    for λ, skip, fid in frontier(content[test_mask], y[test_mask], LAMBDA_TARGETS):
        print(f"  λ={λ:.2f}  skip={skip:.4f}  fidelity={fid:.4f}")

    # --- Baseline B: joint bucket τ from train, applied to test ---
    # Compute bucket means on train, use them as scores on test
    print("\n=== (B) Joint (rc10, dist_period) bucket mean accuracy as score, on TEST ===")
    rc = X[:, 2]; de_log = X[:, 5]
    # define buckets on train
    rc_edges = [-0.01, 0.3, 0.5, 0.7, 1.01]
    de_edges = [-0.01, np.log1p(2), np.log1p(15), 999]
    # Compute per-bucket accuracy on train; use as "score" for gate.
    score_B = np.zeros(len(X), dtype=np.float32)
    for i in range(len(rc_edges) - 1):
        for j in range(len(de_edges) - 1):
            b = ((rc >= rc_edges[i]) & (rc < rc_edges[i+1])
                 & (de_log >= de_edges[j]) & (de_log < de_edges[j+1]))
            train_in_b = b & train_mask
            if train_in_b.sum() > 50:
                bucket_acc = y[train_in_b].mean()
            else:
                bucket_acc = 0.0
            score_B[b] = bucket_acc
    for λ, skip, fid in frontier(score_B[test_mask], y[test_mask], LAMBDA_TARGETS):
        print(f"  λ={λ:.2f}  skip={skip:.4f}  fidelity={fid:.4f}")

    # --- (C) Unified MLP ---
    print("\n=== (C) Unified MLP (all 4 dimensions) ===")
    device = "cpu"
    net = UnifiedMLP(n_feat=Xn.shape[1], hidden=64).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

    Xt = torch.from_numpy(Xn[train_mask]).to(device)
    yt = torch.from_numpy(y[train_mask]).to(device)
    Xe = torch.from_numpy(Xn[test_mask]).to(device)
    ye = torch.from_numpy(y[test_mask]).to(device)

    batch_size = 4096
    n_train = len(Xt)
    n_epochs = 40
    for epoch in range(n_epochs):
        perm = torch.randperm(n_train)
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            xb = Xt[idx]; yb = yt[idx]
            logit = net(xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(idx)
        if epoch % 10 == 9 or epoch == 0:
            with torch.no_grad():
                p_test = torch.sigmoid(net(Xe)).numpy()
            print(f"  epoch {epoch+1:3d}  train_loss={total_loss/n_train:.4f}  "
                  f"test_auc~{abs(0.5 - p_test.mean()) + 0.5:.3f}")

    net.eval()
    with torch.no_grad():
        p_test = torch.sigmoid(net(Xe)).numpy()
    print("\nMLP frontier on TEST:")
    for λ, skip, fid in frontier(p_test, y[test_mask], LAMBDA_TARGETS):
        print(f"  λ={λ:.2f}  skip={skip:.4f}  fidelity={fid:.4f}")

    # Feature importance via gradient
    Xe_tensor = torch.from_numpy(Xn[test_mask]).clone().requires_grad_(True)
    p = torch.sigmoid(net(Xe_tensor))
    p.sum().backward()
    grad_abs = Xe_tensor.grad.abs().mean(dim=0).numpy()
    print("\nFeature importance (mean |grad| of P_correct w.r.t. normalized feature):")
    order = np.argsort(-grad_abs)
    for idx in order:
        print(f"  {feat_names[idx]:>20}  |grad|={grad_abs[idx]:.4f}")


if __name__ == "__main__":
    main()
