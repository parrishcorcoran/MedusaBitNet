"""Can we predict Dim 1 (head-0 confidence) and Dim 2 (agreement) from
ZERO-BACKBONE features alone? If yes, we have a pre-backbone gate and can
skip backbone entirely for predictably-easy tokens.

Features available WITHOUT running backbone at position t:
  - Past confidences (from prior backbone forwards, already cached)
  - Rolling entropy windows
  - Structural position signals
  - Recent token IDs (for ngram features)
  - Past logit gaps

Regression targets (what we'd normally need to run backbone + heads to get):
  - logit_gap at t  (a strong Dim 1 aperture)
  - agreement_count at t  (Dim 2)
  - correctness of head-0 (actually vs vanilla at t+2)

Metrics:
  - R² on continuous targets
  - AUC on binary correctness
  - Skip-rate frontier if we use the predicted score as gate

If R²(logit_gap) > ~0.5 and AUC(correctness) > 0.7, there's strong signal
in cheap features alone. That validates the pre-backbone gate idea.
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


class TinyNet(nn.Module):
    def __init__(self, n_feat, n_out=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )
    def forward(self, x):
        y = self.net(x)
        return y.squeeze(-1) if y.shape[-1] == 1 else y


def train_regressor(X_tr, y_tr, X_te, y_te, epochs=50, binary=False):
    net = TinyNet(X_tr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.from_numpy(X_tr); yt = torch.from_numpy(y_tr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 4096):
            idx = perm[i:i+4096]
            pred = net(Xt[idx])
            if binary:
                loss = F.binary_cross_entropy_with_logits(pred, yt[idx])
            else:
                loss = F.mse_loss(pred, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p_te = net(torch.from_numpy(X_te)).numpy()
    if binary:
        p_te = 1.0 / (1.0 + np.exp(-p_te))
    return p_te


def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / max(ss_tot, 1e-9)


def auc_score(y_true_bin, y_score):
    # Approx AUC via sorted ranks
    order = np.argsort(y_score)
    y_sorted = y_true_bin[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    ranks_of_pos = np.where(y_sorted == 1)[0] + 1
    auc = (ranks_of_pos.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


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

    all_cheap = []       # [N, F_cheap]
    all_target_gap = []  # [N] logit_gap at t
    all_target_agr = []  # [N] agreement_count at t
    all_target_cor = []  # [N] head-0 correct at t vs vanilla t+2
    split_flags = []

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
            confs0 = probs0.max(dim=-1).values.numpy()   # [T]
            top2 = torch.topk(logits0, 2, dim=-1).values
            logit_gap_t = (top2[:, 0] - top2[:, 1]).numpy()  # [T]

            probs_all = F.softmax(logits[0].float(), dim=-1)
            confs_all = probs_all.max(dim=-1).values.numpy()  # [T, K]
            preds_all = probs_all.argmax(dim=-1).numpy()      # [T, K]

            vlogits = h[1:].to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
            vpred = vlogits.float().argmax(dim=-1).numpy()

            valid = SEQ_LEN - 2
            t_start, t_end = 6, valid
            ts = np.arange(t_start, t_end, dtype=np.int64)

            # === TARGETS at anchor t (what we want to predict without running head/backbone at t) ===
            target_gap = logit_gap_t[ts]
            # agreement_count at anchor t
            h0p = preds_all[:, 0][ts]
            h1p = preds_all[:, 1][ts - 1]
            h2p = preds_all[:, 2][ts - 2]
            h3p = preds_all[:, 3][ts - 3]
            target_agr = ((h1p == h0p) + (h2p == h0p) + (h3p == h0p)).astype(np.float32)
            # correctness
            target_cor = (h0p == vpred[ts]).astype(np.float32)

            # === CHEAP FEATURES at anchor t (zero backbone-at-t) ===
            # All features use only info from BEFORE position t (past confs, past logit gaps, structural).
            # We use conf_at[t-1] and earlier — NOT confs0[t].

            past_confs = np.concatenate([[confs0[0]], confs0])[:-1]  # shift by 1; past_confs[t] = confs0[t-1]
            past_gaps = np.concatenate([[logit_gap_t[0]], logit_gap_t])[:-1]

            # All features:
            rc5_past = rolling_mean(past_confs, 5)[ts]
            rc10_past = rolling_mean(past_confs, 10)[ts]
            rc50_past = rolling_mean(past_confs, 50)[ts]
            conf_deriv_past = rc5_past - rc50_past
            lag1_past = lagged_diff(past_confs, 1)[ts]
            lag5_past = lagged_diff(past_confs, 5)[ts]

            rg5_past = rolling_mean(past_gaps, 5)[ts]
            rg10_past = rolling_mean(past_gaps, 10)[ts]
            rg50_past = rolling_mean(past_gaps, 50)[ts]

            de = dist_to_last(token_ids.tolist(), enders).astype(np.float32)
            dn = dist_to_last(token_ids.tolist(), newlines).astype(np.float32)
            de_at = np.log1p(np.minimum(de[ts + 1], 200.0))
            dn_at = np.log1p(np.minimum(dn[ts + 1], 200.0))
            rel_pos = (ts.astype(np.float32) / valid)

            # past-token identity features: is last token punctuation / whitespace / common?
            last_tok = token_ids[ts + 1]  # token emitted at t+1 (the current context end)
            is_space_after = np.isin(last_tok, list(newlines)).astype(np.float32)
            is_ender = np.isin(last_tok, list(enders)).astype(np.float32)

            # past agreement rate (how often did heads agree in recent window)
            all_heads_preds = preds_all
            past_agr = np.zeros(SEQ_LEN, dtype=np.float32)
            for i in range(3, SEQ_LEN):
                a = ((all_heads_preds[i-1, 1] == all_heads_preds[i, 0]).astype(float)
                     + (all_heads_preds[i-2, 2] == all_heads_preds[i, 0]).astype(float)
                     + (all_heads_preds[i-3, 3] == all_heads_preds[i, 0]).astype(float))
                past_agr[i] = a / 3.0
            # Compute rolling mean of past_agr for stability
            past_agr_rolled = rolling_mean(past_agr, 10)[ts]

            feat = np.stack([
                rc5_past, rc10_past, rc50_past, conf_deriv_past,
                lag1_past, lag5_past,
                rg5_past, rg10_past, rg50_past,
                de_at, dn_at, rel_pos,
                is_space_after, is_ender,
                past_agr_rolled,
            ], axis=1).astype(np.float32)

            all_cheap.append(feat)
            all_target_gap.append(target_gap)
            all_target_agr.append(target_agr)
            all_target_cor.append(target_cor)
            split_flags.append(0 if si < seq_split else 1)

    X = np.concatenate(all_cheap, axis=0)
    y_gap = np.concatenate(all_target_gap, axis=0)
    y_agr = np.concatenate(all_target_agr, axis=0)
    y_cor = np.concatenate(all_target_cor, axis=0)
    is_test = np.concatenate([
        np.full(len(all_cheap[i]), split_flags[i], dtype=np.int8)
        for i in range(n_seqs)
    ])
    train_mask = is_test == 0; test_mask = is_test == 1
    print(f"features: {X.shape}  train={train_mask.sum()}  test={test_mask.sum()}")
    print(f"[cheap] {X.shape[1]} features, all zero-backbone-at-position-t")

    mu = X[train_mask].mean(axis=0)
    sd = X[train_mask].std(axis=0) + 1e-6
    Xn = (X - mu) / sd

    # Normalize continuous targets
    gm = y_gap[train_mask].mean(); gs = y_gap[train_mask].std() + 1e-6
    am = y_agr[train_mask].mean(); as_ = y_agr[train_mask].std() + 1e-6

    print("\n=== Predicting logit_gap (Dim 1, raw scale) from cheap features ===")
    pred_gap = train_regressor(Xn[train_mask], (y_gap[train_mask] - gm) / gs,
                                Xn[test_mask], (y_gap[test_mask] - gm) / gs,
                                epochs=40)
    pred_gap_real = pred_gap * gs + gm
    r2 = r2_score(y_gap[test_mask], pred_gap_real)
    print(f"  R² on test: {r2:.4f}")
    print(f"  true mean={y_gap[test_mask].mean():.3f} pred mean={pred_gap_real.mean():.3f}")

    print("\n=== Predicting agreement_count (Dim 2, 0-3 scale) ===")
    pred_agr = train_regressor(Xn[train_mask], (y_agr[train_mask] - am) / as_,
                                Xn[test_mask], (y_agr[test_mask] - am) / as_,
                                epochs=40)
    pred_agr_real = pred_agr * as_ + am
    r2 = r2_score(y_agr[test_mask], pred_agr_real)
    print(f"  R² on test: {r2:.4f}")

    print("\n=== Predicting correctness (head-0 vs vanilla, binary) — the ULTIMATE gate ===")
    pred_cor = train_regressor(Xn[train_mask], y_cor[train_mask],
                                Xn[test_mask], y_cor[test_mask],
                                epochs=40, binary=True)
    auc = auc_score(y_cor[test_mask], pred_cor)
    print(f"  AUC on test: {auc:.4f}")

    # Frontier with cheap-only gate
    from test_unified_estimator_v2 import frontier, LAMBDA_TARGETS
    print("\n=== Cheap-only gate frontier (predict correctness from pre-backbone features) ===")
    for λ, skip, fid in frontier(pred_cor, y_cor[test_mask], LAMBDA_TARGETS):
        print(f"  λ={λ:.2f}  skip={skip:.4f}  fidelity={fid:.4f}")

    print("\nINTERPRETATION:")
    print("  If R²(logit_gap) > 0.4 → cheap aperture exists for Dim 1")
    print("  If R²(agreement)  > 0.3 → cheap aperture for Dim 2")
    print("  If AUC(correct)   > 0.7 → pre-backbone gate is viable at moderate fidelity")
    print("  If cheap-gate skip at λ=0.95 > 1% → zero-backbone path is empirically supported")


if __name__ == "__main__":
    main()
