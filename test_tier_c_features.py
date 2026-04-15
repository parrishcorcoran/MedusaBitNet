"""Tier C: more aggressive 'tokens already produced' and sentence-length signals.

Specifically exploring:

  - Sentence length so far (distance from sentence-start)
  - Expected-sentence-remaining (average sentence len in recent context - current len)
  - Open brackets parity (balanced-or-not state)
  - Discourse markers (specific tokens like "therefore", "finally", "thus")
  - Repetition structure (longest-common-substring with recent context)
  - Cumulative conf trajectory (are we in a "settled" region - mean of last-N confs is rising?)
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features

SEQ_LEN = 2048; HIDDEN = 2560; VOCAB = 128256
TOKENIZER_DIR = "/home/cpinchington/MedusaBitNet/models/bitnet-b1.58-2B-4T"


def get_special_sets(tok):
    enders, newlines, opens, closes = set(), set(), set(), set()
    discourse = set()
    discourse_patterns = [' Therefore', ' However', ' Finally', ' Thus',
                          ' In conclusion', ' Moreover', ' Furthermore']
    for tid in range(VOCAB):
        s = tok.decode([tid])
        if '\n' in s: newlines.add(tid)
        stripped = s.strip()
        if stripped and stripped[-1] in '.!?': enders.add(tid)
        if stripped and stripped[0] in '({[<"\'': opens.add(tid)
        if stripped and stripped[-1] in ')}]>"\'': closes.add(tid)
        for pat in discourse_patterns:
            if s.startswith(pat) or s == pat.strip():
                discourse.add(tid); break
    return enders, newlines, opens, closes, discourse


def dist_to_last(tokens, match_set):
    n = len(tokens); out = np.full(n, 9999, dtype=np.int32); last = -10000
    for i in range(n):
        if last >= 0: out[i] = i - last
        if tokens[i] in match_set: last = i
    return out


def sentence_length_so_far(tokens, enders):
    """For each position, how many tokens since last sentence-ender."""
    n = len(tokens); out = np.zeros(n, dtype=np.float32)
    last_end = -1
    for i in range(n):
        if tokens[i] in enders:
            out[i] = i - last_end
            last_end = i
        else:
            out[i] = i - last_end
    return out


def expected_sentence_remaining(tokens, enders, window=200):
    """Estimate: avg sentence length in last 'window' tokens, minus length so far."""
    n = len(tokens)
    # First, compute lengths of completed sentences in a sliding window
    sentence_ends = [i for i in range(n) if tokens[i] in enders]
    length_so_far = sentence_length_so_far(tokens, enders)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        # Find sentence ends in last `window` tokens before i
        recent_ends = [e for e in sentence_ends if e < i and e > i - window]
        if len(recent_ends) < 2:
            avg_len = 30.0  # default
        else:
            gaps = [recent_ends[j] - recent_ends[j-1] for j in range(1, len(recent_ends))]
            avg_len = float(np.mean(gaps))
        out[i] = max(0.0, avg_len - length_so_far[i])
    return out


def brackets_parity(tokens, opens, closes):
    """Running parity count of open vs close brackets (positive = unclosed)."""
    n = len(tokens); parity = np.zeros(n, dtype=np.float32)
    running = 0
    for i in range(n):
        if tokens[i] in opens: running += 1
        if tokens[i] in closes: running -= 1
        parity[i] = running
    return parity


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    enders, newlines, opens, closes, discourse = get_special_sets(tok)
    print(f"sets: enders={len(enders)} newlines={len(newlines)} "
          f"opens={len(opens)} closes={len(closes)} discourse={len(discourse)}")

    # Re-use the existing feature matrix + labels
    X, y, train_mask, test_mask = build_features()
    print(f"existing features: {X.shape}")

    # Build new Tier C features aligned with X's positions (ts = arange(3, SEQ_LEN-2))
    tokens_mm = np.memmap("data/tokens.bin", dtype=np.uint32, mode="r")
    n_seqs = 48

    new_features_per_seq = []
    for si in range(n_seqs):
        toks = tokens_mm[si * SEQ_LEN : (si + 1) * SEQ_LEN].tolist()
        T = SEQ_LEN
        sent_len = sentence_length_so_far(toks, enders)
        exp_rem = expected_sentence_remaining(toks, enders)
        parity = brackets_parity(toks, opens, closes)
        dist_disc = dist_to_last(toks, discourse).astype(np.float32)

        # Match the existing feature alignment (ts = 6..valid)
        valid = SEQ_LEN - 2
        t_start, t_end = 6, valid
        ts = np.arange(t_start, t_end, dtype=np.int64)

        feat = np.stack([
            np.log1p(np.minimum(sent_len[ts + 1], 200.0)),
            np.log1p(np.maximum(exp_rem[ts + 1], 0.0)),
            parity[ts + 1],
            np.log1p(np.minimum(dist_disc[ts + 1], 200.0)),
            # A few interaction signals: sentence_len × dist_to_period
            np.log1p(sent_len[ts + 1]) * np.log1p(200 - np.minimum(sent_len[ts + 1], 200)),
        ], axis=1).astype(np.float32)
        new_features_per_seq.append(feat)

    tier_c = np.concatenate(new_features_per_seq, axis=0)
    tier_c_names = ["sent_len_log", "exp_rem_log", "bracket_parity",
                    "dist_discourse_log", "sent_pos_interaction"]
    print(f"tier-c features: {tier_c.shape}")

    # Combine with existing
    X_plus = np.concatenate([X, tier_c], axis=1)
    feat_names_all = [
        "content_conf", "content_entropy", "logit_gap", "purity", "top3_cov",
        "top10_cov", "rc10", "rc50", "conf_deriv", "conf_lag1", "conf_lag5",
        "dist_period_log", "dist_newline_log", "rel_pos",
        "agreement_count", "conf_var", "conf_min",
    ] + tier_c_names

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
        return net, Xn

    train_eval(X_plus, list(range(17)), "Existing 17 features (baseline)")
    train_eval(X_plus, list(range(17, 22)), "Tier C ONLY")
    net, Xn = train_eval(X_plus, list(range(22)), "All 22 (17 + Tier C)")

    # Feature importance with all 22
    Xe_r = torch.from_numpy(Xn[test_mask]).clone().requires_grad_(True)
    torch.sigmoid(net(Xe_r)).sum().backward()
    grads = Xe_r.grad.abs().mean(dim=0).numpy()
    print("\n=== Feature importance with all 22 features ===")
    order = np.argsort(-grads)
    for idx in order:
        marker = "★ TIER C" if idx >= 17 else ""
        print(f"  {feat_names_all[idx]:>25}  |grad|={grads[idx]:.4f}  {marker}")


if __name__ == "__main__":
    main()
