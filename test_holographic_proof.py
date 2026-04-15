"""Gemini's proposal: empirically prove holographic redundancy.

Claim: as boundary-feature dimensionality (our cheap-aperture PCA dim) grows,
the mutual information between the boundary features and the deep-layer bulk
representation should asymptote to the SELF-ENTROPY of the final layer.

If the network is holographic, deep-bulk is redundant given boundary features.

Method:
  For a growing feature set (content-only → +trajectory → +structural → +geometric → all),
    compute proxy-MI with head-0 argmax and final layer softmax entropy.
  Show that:
    H(head0_output) is bounded (self-entropy)
    I(features, head0_output) grows with feature count
    At asymptote, I ≈ H → holographic redundancy confirmed.

Proxy for MI: prediction accuracy of head-0's argmax from features, compared
to the entropy of head-0 itself. If we can predict head-0's top-1 from cheap
features with accuracy close to 1 - H(distribution)/log(V), the cheap features
capture almost all the information.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features
from test_physics_apertures import load_hiddens_f32
from test_unified_estimator_v2 import UnifiedMLP, frontier, LAMBDA_TARGETS
from test_holographic import build_holographic_features
from model import MedusaHeads

SEQ_LEN = 2048; HIDDEN = 2560; VOCAB = 128256


def get_deep_layer_info(n_seqs=48):
    """Gather: head-0 argmax (target), top-10 distribution, self-entropy, max prob."""
    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    heads = MedusaHeads(HIDDEN, VOCAB, 4, 1, dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    all_argmax = []; all_entropy = []; all_max_prob = []
    with torch.no_grad():
        for si in range(n_seqs):
            h = load_hiddens_f32(si)
            h_t = torch.from_numpy(h).to(torch.bfloat16)
            logits = heads(h_t.unsqueeze(0), lm_head)[0, :, 0, :].float()
            probs = F.softmax(logits, dim=-1)
            ent = -(probs * probs.log().clamp_min(-20)).sum(-1)
            argmax = probs.argmax(-1)
            max_p = probs.max(-1).values

            valid = SEQ_LEN - 2
            ts = np.arange(6, valid, dtype=np.int64)
            all_argmax.append(argmax[ts].numpy())
            all_entropy.append(ent[ts].numpy())
            all_max_prob.append(max_p[ts].numpy())
    return np.concatenate(all_argmax), np.concatenate(all_entropy), np.concatenate(all_max_prob)


def train_predict_argmax(X, train_mask, test_mask, y_argmax, n_output_classes=1000, feat_idx=None):
    """Train MLP to predict head-0 argmax bucket from features.

    We can't easily predict into 128K vocab, so we bucket into cluster IDs by hashing.
    Return test accuracy.
    """
    if feat_idx is None: feat_idx = list(range(X.shape[1]))
    mu = X[train_mask][:, feat_idx].mean(axis=0)
    sd = X[train_mask][:, feat_idx].std(axis=0) + 1e-6
    Xn = (X[:, feat_idx] - mu) / sd

    # Bucket argmax into n_output_classes by modulo
    y = (y_argmax % n_output_classes).astype(np.int64)
    y_train = torch.from_numpy(y[train_mask])
    y_test = torch.from_numpy(y[test_mask])

    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(len(feat_idx), 128), torch.nn.ReLU(),
        torch.nn.Linear(128, n_output_classes),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.from_numpy(Xn[train_mask]); Xe = torch.from_numpy(Xn[test_mask])
    for _ in range(30):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 4096):
            bi = perm[i:i+4096]
            loss = F.cross_entropy(net(Xt[bi]), y_train[bi])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        preds = net(Xe).argmax(-1)
        acc = (preds == y_test).float().mean().item()
    return acc


def main():
    print("Gathering baseline features + holographic features...")
    X, y, train_mask, test_mask = build_features()
    hol = build_holographic_features()
    X_all = np.concatenate([X, hol], axis=1)
    print(f"  shape: {X_all.shape}")

    print("Gathering deep-layer (head-0) info...")
    argmax, entropy, max_prob = get_deep_layer_info()
    avg_entropy_bits = entropy.mean() / np.log(2)
    avg_max_p = max_prob.mean()
    print(f"  head-0 avg entropy: {avg_entropy_bits:.3f} bits  avg max prob: {avg_max_p:.3f}")

    # The self-information of the bulk's decision = log2(1 / top_prob)
    # At top prob = 0.37, bulk self-info = 1.43 bits
    # As MI with features -> self-info, features are fully predictive

    # Growing feature sets: measure predictive accuracy as proxy for MI
    # We'll predict argmax modulo 1000 (1000 classes, random choice baseline 0.1%)
    # Feature set size sweep: 1, 5, 10, 15, 17, 20, 25 features (by importance order)
    feat_order = [0, 1, 2, 3, 14, 11, 13, 6, 8, 7, 12, 10, 9, 4, 5, 15, 16,
                  17, 24, 21, 22, 18, 23, 19, 20]  # prioritize high-importance
    print(f"\n=== Holographic MI proof: info captured by growing feature set ===")
    print(f"{'n_feat':>8} {'accuracy':>12} {'MI_proxy_bits':>15}")
    print(f"{'baseline':>8} {'0.001':>12} {'0.0':>15}  (random over 1000 classes)")

    # Compute entropy of head-0 argmax modulo 1000 (approximates self-information)
    argmax_mod = argmax % 1000
    counts = np.bincount(argmax_mod, minlength=1000) / len(argmax_mod)
    self_entropy_mod = -np.sum(counts[counts > 0] * np.log2(counts[counts > 0]))
    print(f"{'ceiling':>8} {'N/A':>12} {self_entropy_mod:>15.3f}  (self-entropy of argmax %1000)")

    results = []
    for k in [1, 3, 5, 8, 12, 17, 20, 25]:
        feat_idx = feat_order[:k]
        acc = train_predict_argmax(X_all, train_mask, test_mask, argmax, feat_idx=feat_idx)
        # MI proxy: log2(1/ baseline) - log2(error rate)
        # Or: H(Y) - H(Y|X) ≈ H(Y) - (-log2(acc)) if acc is dominant
        mi_proxy_bits = self_entropy_mod + np.log2(max(acc, 1e-6))
        mi_proxy_bits = max(0, mi_proxy_bits)
        print(f"{k:>8} {acc:>12.4f} {mi_proxy_bits:>15.3f}")
        results.append((k, acc, mi_proxy_bits))

    # Does MI approach self-entropy as features grow?
    print(f"\n*** INTERPRETATION ***")
    print(f"  If MI(features; head-0 argmax) → H(head-0 argmax), network is holographic.")
    last_mi = results[-1][2]
    frac = last_mi / self_entropy_mod if self_entropy_mod > 0 else 0
    print(f"  At {results[-1][0]} features: MI = {last_mi:.3f} bits, ceiling = {self_entropy_mod:.3f} bits")
    print(f"  Fraction captured: {frac:.1%}")
    if frac > 0.8:
        print(f"  ✓ STRONG evidence of holographic redundancy (>80% captured)")
    elif frac > 0.5:
        print(f"  ~ Moderate evidence ({int(frac*100)}% captured)")
    else:
        print(f"  Features currently capture {int(frac*100)}% of bulk information")
        print(f"  Need more apertures (or richer ones) to hit holographic asymptote")


if __name__ == "__main__":
    main()
