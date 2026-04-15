"""Round 4: information-theoretic + self-consistency apertures.

  A. KL trajectory: KL(p_t || p_{t-1}) per step - distribution reshape rate
  B. Self-attention retrieval: entropy of cosine-similarity over past states
  C. Prediction self-consistency: did head-0 predict the current token k steps ago?
  D. Entropy flux: 1st and 2nd derivatives of softmax entropy
"""
import time
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


def aperture_kl_trajectory(n_seqs=48):
    """KL divergence between consecutive softmax distributions."""
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
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            # KL(P_t || P_{t-1}) = sum p_t * (log p_t - log p_{t-1})
            kl_fwd = torch.zeros(SEQ_LEN)
            # Use log-softmax for stability
            # KL(p_t || p_{t-1}) approx = sum_v p_t[v] * (log_p_t[v] - log_p_{t-1}[v])
            # Computing this is heavy if V=128k. Let's only compute KL over top-100 tokens
            # combined (union of top-100 of each).
            for t in range(1, SEQ_LEN):
                # Get top-100 from EACH, compute KL approximately on union
                top_t = torch.topk(probs[t], 100).indices
                top_prev = torch.topk(probs[t-1], 100).indices
                union = torch.unique(torch.cat([top_t, top_prev]))
                p_t = probs[t, union]
                p_prev = probs[t-1, union]
                p_t_n = p_t / p_t.sum().clamp_min(1e-9)
                p_prev_n = p_prev / p_prev.sum().clamp_min(1e-9)
                kl_fwd[t] = (p_t_n * (p_t_n.log() - p_prev_n.log().clamp_min(-20))).sum()

            valid = SEQ_LEN - 2
            ts = np.arange(6, valid, dtype=np.int64)
            feat = kl_fwd[ts + 1].numpy().reshape(-1, 1).astype(np.float32)
            feats.append(feat)
    return np.concatenate(feats, axis=0)


def aperture_self_attention(n_seqs=48, window=50):
    """For current position, cosine similarity to past window states -> distribution.
    Entropy and peak of this retrieval distribution are features."""
    feats = []
    for si in range(n_seqs):
        h = load_hiddens_f32(si)
        h_t = torch.from_numpy(h)
        h_norm = h_t / (h_t.norm(dim=-1, keepdim=True) + 1e-9)
        retrieval_entropy = np.zeros(SEQ_LEN, dtype=np.float32)
        retrieval_peak = np.zeros(SEQ_LEN, dtype=np.float32)
        retrieval_top3 = np.zeros(SEQ_LEN, dtype=np.float32)
        for i in range(window + 1, SEQ_LEN):
            window_norm = h_norm[i-window:i]
            cossim = (window_norm * h_norm[i:i+1]).sum(dim=-1)  # [window]
            # softmax over cosines (temperature=5 to sharpen)
            probs = F.softmax(cossim * 5, dim=-1)
            retrieval_entropy[i] = -(probs * probs.log().clamp_min(-20)).sum().item()
            retrieval_peak[i] = probs.max().item()
            top3 = torch.topk(probs, 3).values.sum()
            retrieval_top3[i] = top3.item()

        valid = SEQ_LEN - 2
        ts = np.arange(6, valid, dtype=np.int64)
        feat = np.stack([retrieval_entropy[ts + 1], retrieval_peak[ts + 1],
                         retrieval_top3[ts + 1]], axis=1).astype(np.float32)
        feats.append(feat)
    return np.concatenate(feats, axis=0)


def aperture_prediction_consistency(n_seqs=48):
    """Did head-0 k steps ago predict the current token?

    For each target position p, check: what did head-0 at anchor p-2 predict?
    (already the target for our normal head training). But also:
    what did head-1 at p-3 predict? head-2 at p-4? head-3 at p-5?
    Agreement across these tells us how "self-consistent" the trajectory is.
    """
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
            logits = heads(h_t.unsqueeze(0), lm_head)[0].float()  # [T, K, V]
            preds = logits.argmax(dim=-1).numpy()  # [T, K]

            valid = SEQ_LEN - 2
            t_start, t_end = 6, valid
            ts = np.arange(t_start, t_end, dtype=np.int64)
            # For each anchor t, the heads predicted positions:
            #   head-0 at t predicts t+2
            #   head-1 at t predicts t+3
            #   head-2 at t predicts t+4
            #   head-3 at t predicts t+5
            # Prediction self-consistency: for a target position p, multiple anchors
            # predict it. How many AGREE?
            # Here, for position ts (where head-0 predicts t+2), also check what
            # head-1 at t-1, head-2 at t-2, head-3 at t-3 predicted for the SAME target.

            # We already compute agreement_count in our standard features. Here,
            # we add: prediction cycle consistency — does what we PREDICTED a few
            # steps ago still hold as prediction for current position now?
            # cycle: at anchor (t-k-2), head-k predicted the content of t. Compare
            # to what head-0 at anchor t predicts for t+2 (i.e., the current frame).
            # Hmm this is the same as agreement.

            # Better: "does the predicted-head-0 for current position match what
            # was actually emitted at current position?" = head-0 accuracy at past
            # anchor. We'll compute rolling head-0 accuracy over last 20 positions.
            # High rolling accuracy = trajectory is "on track" = predictable.

            # For each position, look at head-0's prediction at anchor (t-2) and see
            # if it matched emitted token at t. Keep rolling accuracy.
            rolling_acc = np.zeros(SEQ_LEN, dtype=np.float32)
            for i in range(3, SEQ_LEN - 2):
                # head-0 at anchor i-2 predicted position i. What was emitted at i?
                # We use token_ids[i] but... we don't have emissions here, just the
                # training token stream. Use tokens_mm directly.
                pass  # skip this — too complex

            # Simpler: variance of head predictions across multi-step ago
            # At position t, collect preds[t-2, 0], preds[t-3, 1], preds[t-4, 2], preds[t-5, 3]
            # All predict token at position t. How many unique values?
            unique_count = np.zeros(SEQ_LEN, dtype=np.float32)
            for i in range(6, SEQ_LEN - 2):
                p0 = preds[i-2, 0]; p1 = preds[i-3, 1]
                p2 = preds[i-4, 2]; p3 = preds[i-5, 3]
                unique = len(set([int(p0), int(p1), int(p2), int(p3)]))
                unique_count[i] = unique

            feat = unique_count[ts + 1].reshape(-1, 1).astype(np.float32)
            feats.append(feat)
    return np.concatenate(feats, axis=0)


def aperture_entropy_flux(n_seqs=48):
    """1st and 2nd derivatives of head-0 softmax entropy."""
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

            d_entropy = np.zeros(SEQ_LEN, dtype=np.float32)
            d_entropy[1:] = entropy[1:] - entropy[:-1]
            d2_entropy = np.zeros(SEQ_LEN, dtype=np.float32)
            d2_entropy[2:] = d_entropy[2:] - d_entropy[1:-1]

            valid = SEQ_LEN - 2
            ts = np.arange(6, valid, dtype=np.int64)
            feat = np.stack([d_entropy[ts + 1], d2_entropy[ts + 1]], axis=1).astype(np.float32)
            feats.append(feat)
    return np.concatenate(feats, axis=0)


def main():
    X, y, train_mask, test_mask = build_features()
    print(f"existing features: {X.shape}")

    print("\n[A] KL trajectory (top-100 approx)...")
    t0 = time.time(); kl = aperture_kl_trajectory(); print(f"    {kl.shape} in {time.time()-t0:.1f}s")
    print("\n[B] Self-attention retrieval...")
    t0 = time.time(); att = aperture_self_attention(); print(f"    {att.shape} in {time.time()-t0:.1f}s")
    print("\n[C] Prediction self-consistency...")
    t0 = time.time(); pc = aperture_prediction_consistency(); print(f"    {pc.shape} in {time.time()-t0:.1f}s")
    print("\n[D] Entropy flux (d/dt, d²/dt²)...")
    t0 = time.time(); ef = aperture_entropy_flux(); print(f"    {ef.shape} in {time.time()-t0:.1f}s")

    train_eval(X, y, train_mask, test_mask, list(range(17)),
               "Baseline 17 features")

    for label, feats in [("KL trajectory ONLY", kl),
                          ("Self-attn retrieval ONLY", att),
                          ("Pred self-consistency ONLY", pc),
                          ("Entropy flux ONLY", ef)]:
        Xaug = np.concatenate([X, feats], axis=1)
        train_eval(Xaug, y, train_mask, test_mask,
                   list(range(17, 17 + feats.shape[1])), label)

    for label, feats in [("Baseline + KL", kl),
                          ("Baseline + Self-attn", att),
                          ("Baseline + Pred-cons", pc),
                          ("Baseline + Ent-flux", ef)]:
        Xaug = np.concatenate([X, feats], axis=1)
        train_eval(Xaug, y, train_mask, test_mask,
                   list(range(17 + feats.shape[1])), label)

    # All 4
    X_all4 = np.concatenate([X, kl, att, pc, ef], axis=1)
    n = X_all4.shape[1]
    train_eval(X_all4, y, train_mask, test_mask, list(range(n)),
               f"Baseline + ALL round-4 ({n} features)")

    # Feature importance
    mu = X_all4[train_mask].mean(axis=0); sd = X_all4[train_mask].std(axis=0) + 1e-6
    Xn = (X_all4 - mu) / sd
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
        "kl_traj",
        "retrieval_entropy", "retrieval_peak", "retrieval_top3",
        "pred_unique_count",
        "d_entropy", "d2_entropy",
    ]
    print(f"\n=== Feature importance ({n} features) ===")
    order = np.argsort(-grads)
    for idx in order[:15]:
        marker = " ★" if idx >= 17 else ""
        print(f"  {names[idx]:>22}  |grad|={grads[idx]:.4f}{marker}")


if __name__ == "__main__":
    main()
