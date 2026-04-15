"""Tier B cheap features — new signal dimensions the user asked for.

Goal: find signals IN GENERATED TOKENS beyond the current 4 dimensions. Try:

  1. N-gram repetition — does last 3-gram appear in recent 100 tokens?
  2. N-gram cache probability — bigram/trigram table frequency of last tokens
  3. Character-class features — ending char (letter, digit, punct, space)
  4. Token-class repetition — same token-type twice in last N?
  5. Local vocab diversity — unique-tokens / slots in last 50
  6. Distance to last identical token — repetition period

Each feature is free given the token stream. Test which add signal beyond
our existing 11-feature set.
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features

SEQ_LEN = 2048; HIDDEN = 2560; VOCAB = 128256


def build_tier_b(tokens_mm, confs_full, n_seqs=48):
    """Construct Tier B features on top of the existing feature set.

    Returns [N, F_new] array of novel features aligned with build_features() positions.
    """
    feats = []
    for si in range(n_seqs):
        toks = tokens_mm[si * SEQ_LEN : (si + 1) * SEQ_LEN]
        T = len(toks)
        # 1. N-gram repetition (last 3-gram seen in previous 100 tokens)
        trigram_rep = np.zeros(T, dtype=np.float32)
        for i in range(3, T):
            tri = (int(toks[i-3]), int(toks[i-2]), int(toks[i-1]))
            window_start = max(0, i - 100)
            seen = False
            for j in range(window_start, i - 2):
                if (int(toks[j]), int(toks[j+1]), int(toks[j+2])) == tri:
                    seen = True; break
            trigram_rep[i] = 1.0 if seen else 0.0

        # 2. Bigram frequency in window
        bigram_freq = np.zeros(T, dtype=np.float32)
        for i in range(2, T):
            bi = (int(toks[i-2]), int(toks[i-1]))
            window_start = max(0, i - 100)
            count = 0
            for j in range(window_start, i - 1):
                if (int(toks[j]), int(toks[j+1])) == bi: count += 1
            bigram_freq[i] = count / max(1, i - window_start)

        # 5. Local vocab diversity (unique/total in last 50)
        vocab_div = np.zeros(T, dtype=np.float32)
        for i in range(T):
            lo = max(0, i - 50)
            window = toks[lo:i+1]
            if len(window) > 0:
                uniq = len(set(window.tolist()))
                vocab_div[i] = uniq / len(window)

        # 6. Distance to last identical token (capped log)
        last_seen = {}
        dist_same = np.full(T, 200.0, dtype=np.float32)
        for i in range(T):
            tid = int(toks[i])
            if tid in last_seen:
                dist_same[i] = min(200, i - last_seen[tid])
            last_seen[tid] = i
        dist_same_log = np.log1p(dist_same)

        # 7. Rolling conf variance over 20-window (conf_full is per-token head-0 conf)
        if confs_full is not None and si < len(confs_full):
            cf = confs_full[si]
            conf_var20 = np.zeros(T, dtype=np.float32)
            for i in range(T):
                lo = max(0, i - 20)
                conf_var20[i] = float(np.var(cf[lo:i+1])) if i > lo else 0.0
        else:
            conf_var20 = np.zeros(T, dtype=np.float32)

        # Align with build_features() output: uses ts = [6, valid)
        valid = SEQ_LEN - 2
        t_start, t_end = 6, valid
        ts = np.arange(t_start, t_end, dtype=np.int64)
        # Tier B features are evaluated at context-end position t+1
        feat = np.stack([
            trigram_rep[ts + 1],
            bigram_freq[ts + 1],
            vocab_div[ts + 1],
            dist_same_log[ts + 1],
            conf_var20[ts + 1] if confs_full is not None else np.zeros(len(ts), dtype=np.float32),
        ], axis=1).astype(np.float32)
        feats.append(feat)

    return np.concatenate(feats, axis=0)


def main():
    # Get existing features + the full-head-0 confidences per sequence for conf_var20
    tokens_mm = np.memmap("data/tokens.bin", dtype=np.uint32, mode="r")

    # Get full-rank head-0 confs per sequence (needed for conf_var20 feature)
    print("[tier-b] gathering head-0 confidences...")
    from model import MedusaHeads
    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    heads = MedusaHeads(HIDDEN, VOCAB, 4, 1, dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    confs_full = []
    with torch.no_grad():
        for si in range(48):
            off = si * per_seq
            chunk = hidden_mm[off:off + per_seq]
            h = (torch.from_numpy(chunk.copy()).view(torch.bfloat16)
                       .view(SEQ_LEN, HIDDEN))
            logits = heads(h.unsqueeze(0), lm_head)
            probs = F.softmax(logits[0, :, 0, :].float(), dim=-1)
            cf = probs.max(dim=-1).values.numpy()
            confs_full.append(cf)

    print("[tier-b] building Tier B features...")
    tier_b = build_tier_b(tokens_mm, confs_full)
    print(f"[tier-b] shape: {tier_b.shape}")

    # Get standard features + labels
    print("[tier-b] getting existing features...")
    X, y, train_mask, test_mask = build_features()
    print(f"[tier-b] X: {X.shape} tier_b: {tier_b.shape}")

    # Concatenate
    X_plus = np.concatenate([X, tier_b], axis=1)

    # Train MLP with and without Tier B
    from test_unified_estimator_v2 import UnifiedMLP, frontier, LAMBDA_TARGETS

    def train_eval(Xfull, feat_idx, label):
        mu = Xfull[train_mask][:, feat_idx].mean(axis=0)
        sd = Xfull[train_mask][:, feat_idx].std(axis=0) + 1e-6
        Xn = (Xfull[:, feat_idx] - mu) / sd
        net = UnifiedMLP(n_feat=len(feat_idx), hidden=64)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        Xt = torch.from_numpy(Xn[train_mask]); yt = torch.from_numpy(y[train_mask])
        Xe = torch.from_numpy(Xn[test_mask])
        for _ in range(40):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 4096):
                idx = perm[i:i+4096]
                logit = net(Xt[idx])
                loss = F.binary_cross_entropy_with_logits(logit, yt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(Xe)).numpy()
        print(f"\n=== {label} ===")
        for λ, skip, fid in frontier(p, y[test_mask], LAMBDA_TARGETS):
            print(f"  λ={λ:.2f}  skip={skip:.4f}  fidelity={fid:.4f}")

    train_eval(X_plus, list(range(17)), "Baseline: 17 existing features")
    train_eval(X_plus, list(range(17, 22)), "Tier B ONLY (5 new features)")
    train_eval(X_plus, list(range(22)), "Combined: 17 existing + 5 Tier B")

    # Feature importance of Tier B in combined model
    mu = X_plus[train_mask].mean(axis=0); sd = X_plus[train_mask].std(axis=0) + 1e-6
    Xn = (X_plus - mu) / sd
    net = UnifiedMLP(n_feat=22, hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.from_numpy(Xn[train_mask]); yt = torch.from_numpy(y[train_mask])
    for _ in range(40):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 4096):
            idx = perm[i:i+4096]
            loss = F.binary_cross_entropy_with_logits(net(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    Xe_r = torch.from_numpy(Xn[test_mask]).clone().requires_grad_(True)
    torch.sigmoid(net(Xe_r)).sum().backward()
    grads = Xe_r.grad.abs().mean(dim=0).numpy()

    tier_b_names = ["trigram_rep", "bigram_freq", "vocab_div", "dist_same_log", "conf_var20"]
    feat_names_all = [
        "content_conf", "content_entropy", "logit_gap", "purity", "top3_cov", "top10_cov",
        "rc10", "rc50", "conf_deriv", "conf_lag1", "conf_lag5",
        "dist_period_log", "dist_newline_log", "rel_pos",
        "agreement_count", "conf_var", "conf_min",
    ] + tier_b_names

    print("\n=== Feature importance (combined 22-feature model) ===")
    order = np.argsort(-grads)
    for idx in order:
        marker = "★ NEW" if idx >= 17 else ""
        print(f"  {feat_names_all[idx]:>20}  |grad|={grads[idx]:.4f}  {marker}")


if __name__ == "__main__":
    main()
