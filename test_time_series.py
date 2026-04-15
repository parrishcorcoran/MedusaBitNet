"""Round 6: classical time-series complexity measures from nonlinear dynamics.

  A. Approximate entropy (ApEn): pattern-match regularity
  B. Sample entropy (SampEn): refined ApEn, more robust
  C. Hurst exponent: long-range dependence / memory of trajectory
  D. Multiscale entropy: entropy at scales 1, 2, 4 (coarse-graining)

All computed on the rolling confidence trajectory.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/cpinchington/MedusaBitNet")
from test_feature_redundancy import build_features
from test_physics_apertures import load_hiddens_f32
from test_unified_estimator_v2 import UnifiedMLP, frontier, LAMBDA_TARGETS
from model import MedusaHeads

SEQ_LEN = 2048; HIDDEN = 2560; VOCAB = 128256


def approximate_entropy(U, m=2, r=0.15):
    """ApEn estimator. U: 1D numpy array, m: pattern length, r: tolerance."""
    N = len(U)
    if N < m + 1: return 0.0
    r = r * np.std(U)

    def phi(m):
        patterns = np.array([U[i:i+m] for i in range(N - m + 1)])
        # For each pattern, count how many are within r
        C = 0.0
        for i in range(len(patterns)):
            matches = np.all(np.abs(patterns - patterns[i]) <= r, axis=1).sum()
            C += np.log(matches / len(patterns))
        return C / (N - m + 1)

    return phi(m) - phi(m + 1)


def hurst_exponent(x, max_lag=20):
    """Rescaled range estimator of Hurst exponent H. H=0.5 is random walk,
    H>0.5 is persistent, H<0.5 is anti-persistent."""
    x = np.asarray(x)
    N = len(x)
    if N < max_lag + 1: return 0.5
    lags = range(2, max_lag)
    rs = []
    for lag in lags:
        chunks = [x[i:i+lag] for i in range(0, N - lag, lag)]
        if not chunks: continue
        rs_chunk = []
        for c in chunks:
            mean = c.mean()
            z = np.cumsum(c - mean)
            r = z.max() - z.min()
            s = c.std() + 1e-9
            rs_chunk.append(r / s)
        if rs_chunk: rs.append(np.mean(rs_chunk))
    if len(rs) < 3: return 0.5
    lags_arr = np.log(list(lags)[:len(rs)])
    rs_arr = np.log(np.array(rs) + 1e-9)
    H = np.polyfit(lags_arr, rs_arr, 1)[0]
    return float(H)


def build_time_series_features(n_seqs=48, window=64):
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
            conf = F.softmax(logits, dim=-1).max(-1).values.numpy()

            apen = np.zeros(SEQ_LEN, dtype=np.float32)
            hurst = np.zeros(SEQ_LEN, dtype=np.float32)
            # Multiscale: entropy of conf trajectory, conf[::2] trajectory, conf[::4] trajectory
            mse_1 = np.zeros(SEQ_LEN, dtype=np.float32)
            mse_2 = np.zeros(SEQ_LEN, dtype=np.float32)
            mse_4 = np.zeros(SEQ_LEN, dtype=np.float32)

            for i in range(window, SEQ_LEN):
                win = conf[i-window:i]
                apen[i] = approximate_entropy(win, m=2, r=0.15)
                hurst[i] = hurst_exponent(win, max_lag=min(20, window // 3))
                # Multiscale entropy: std across different time scales
                mse_1[i] = np.std(win)
                if len(win) >= 2:
                    coarse2 = win[::2]
                    mse_2[i] = np.std(coarse2)
                if len(win) >= 4:
                    coarse4 = win[::4]
                    mse_4[i] = np.std(coarse4)

            valid = SEQ_LEN - 2
            ts = np.arange(6, valid, dtype=np.int64)
            feat = np.stack([
                apen[ts + 1],
                hurst[ts + 1],
                mse_1[ts + 1],
                mse_2[ts + 1],
                mse_4[ts + 1],
            ], axis=1).astype(np.float32)
            feats.append(feat)
    return np.concatenate(feats, axis=0)


def main():
    X, y, train_mask, test_mask = build_features()
    print(f"existing features: {X.shape}")

    print("\nBuilding time-series dynamics features (ApEn, Hurst, multiscale)...")
    import time; t0 = time.time()
    ts_feat = build_time_series_features()
    print(f"  {ts_feat.shape} in {time.time()-t0:.1f}s")

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

    X_plus = np.concatenate([X, ts_feat], axis=1)
    train_eval_(X_plus, list(range(17, 22)), "Time-series ONLY (5)")
    net, Xn = train_eval_(X_plus, list(range(22)), "Baseline + time-series")

    Xe_r = torch.from_numpy(Xn[test_mask]).clone().requires_grad_(True)
    torch.sigmoid(net(Xe_r)).sum().backward()
    grads = Xe_r.grad.abs().mean(dim=0).numpy()
    names = [
        "content_conf", "content_entropy", "logit_gap", "purity", "top3_cov",
        "top10_cov", "rc10", "rc50", "conf_deriv", "conf_lag1", "conf_lag5",
        "dist_period_log", "dist_newline_log", "rel_pos",
        "agreement_count", "conf_var", "conf_min",
        "apen", "hurst", "mse_1", "mse_2", "mse_4",
    ]
    print("\n=== Feature importance (22 features) ===")
    order = np.argsort(-grads)
    for idx in order:
        marker = " ★" if idx >= 17 else ""
        print(f"  {names[idx]:>22}  |grad|={grads[idx]:.4f}{marker}")


if __name__ == "__main__":
    main()
