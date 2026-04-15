"""Round 5: token-stream complexity and recurrence — more physics on previously
generated tokens.

  A. Lempel-Ziv compressibility: compression ratio of recent token window
  B. Recurrence plot statistics: determinism + laminarity of recent token trajectory
  C. Periodicity: autocorrelation-peak of recent token bigram stream
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features
from test_unified_estimator_v2 import UnifiedMLP, frontier, LAMBDA_TARGETS

SEQ_LEN = 2048


def lz_complexity(seq):
    """Return Lempel-Ziv complexity (number of distinct dictionary additions)."""
    n = len(seq); i = 0; dict_set = set(); c = 0
    while i < n:
        for j in range(i + 1, n + 1):
            sub = tuple(seq[i:j])
            if sub not in dict_set:
                dict_set.add(sub)
                c += 1
                i = j
                break
        else:
            c += 1; break
    return c


def build_token_dynamics(n_seqs=48, window=50):
    tokens_mm = np.memmap("data/tokens.bin", dtype=np.uint32, mode="r")
    feats = []
    for si in range(n_seqs):
        toks = tokens_mm[si * SEQ_LEN : (si + 1) * SEQ_LEN]

        lz_ratio = np.zeros(SEQ_LEN, dtype=np.float32)
        determinism = np.zeros(SEQ_LEN, dtype=np.float32)
        laminarity = np.zeros(SEQ_LEN, dtype=np.float32)
        periodicity_peak = np.zeros(SEQ_LEN, dtype=np.float32)
        periodicity_period = np.zeros(SEQ_LEN, dtype=np.float32)

        for i in range(window, SEQ_LEN):
            win = toks[i-window:i]
            # A. LZ complexity ratio
            c = lz_complexity(list(win))
            lz_ratio[i] = c / window  # lower = more compressible = more predictable

            # B. Recurrence plot: R[j,k] = 1 if win[j] == win[k]
            # Determinism: fraction of recurrent points on diagonal lines (length ≥ 2)
            # Laminarity: fraction on vertical/horizontal lines
            # For speed, compute on token-ID identity (exact match)
            R = (win[:, None] == win[None, :])
            np.fill_diagonal(R, False)  # exclude self-match
            # Determinism: fraction of 1s on off-diagonals of length ≥ 2
            det_count = 0; lam_count = 0; total = R.sum()
            if total > 0:
                for shift in range(1, min(window // 2, 15)):
                    diag = np.diag(R, k=shift)
                    for j in range(len(diag) - 1):
                        if diag[j] and diag[j+1]:
                            det_count += 2
                # Laminarity: consecutive 1s in columns
                for col in range(window):
                    column = R[:, col]
                    prev = False
                    for val in column:
                        if val and prev:
                            lam_count += 2
                        prev = val
                determinism[i] = det_count / total
                laminarity[i] = lam_count / total

            # C. Periodicity via autocorrelation of 1-lag bigram indicator
            # For each lag k, count of positions j where win[j] == win[j-k]
            auto = np.zeros(window // 2)
            for k in range(1, window // 2):
                auto[k] = np.mean(win[k:] == win[:-k])
            # Peak ignoring k=0
            if len(auto) > 1:
                peak_k = np.argmax(auto[1:]) + 1
                periodicity_period[i] = peak_k
                periodicity_peak[i] = auto[peak_k]

        valid = SEQ_LEN - 2
        ts = np.arange(6, valid, dtype=np.int64)
        feat = np.stack([
            lz_ratio[ts + 1],
            determinism[ts + 1],
            laminarity[ts + 1],
            periodicity_peak[ts + 1],
            np.log1p(periodicity_period[ts + 1]),
        ], axis=1).astype(np.float32)
        feats.append(feat)
    return np.concatenate(feats, axis=0)


def main():
    X, y, train_mask, test_mask = build_features()
    print(f"existing features: {X.shape}")

    print("\nBuilding token dynamics features (LZ + recurrence + periodicity)...")
    import time; t0 = time.time()
    td = build_token_dynamics()
    print(f"  {td.shape} in {time.time()-t0:.1f}s")

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

    X_plus = np.concatenate([X, td], axis=1)
    train_eval_(X_plus, list(range(17, 22)), "Token dynamics ONLY (5)")
    net, Xn = train_eval_(X_plus, list(range(22)), "Baseline + token dynamics")

    Xe_r = torch.from_numpy(Xn[test_mask]).clone().requires_grad_(True)
    torch.sigmoid(net(Xe_r)).sum().backward()
    grads = Xe_r.grad.abs().mean(dim=0).numpy()
    names = [
        "content_conf", "content_entropy", "logit_gap", "purity", "top3_cov",
        "top10_cov", "rc10", "rc50", "conf_deriv", "conf_lag1", "conf_lag5",
        "dist_period_log", "dist_newline_log", "rel_pos",
        "agreement_count", "conf_var", "conf_min",
        "lz_ratio", "determinism", "laminarity", "periodicity_peak", "period_log",
    ]
    print("\n=== Feature importance (22 features) ===")
    order = np.argsort(-grads)
    for idx in order:
        marker = " ★" if idx >= 17 else ""
        print(f"  {names[idx]:>22}  |grad|={grads[idx]:.4f}{marker}")


if __name__ == "__main__":
    main()
