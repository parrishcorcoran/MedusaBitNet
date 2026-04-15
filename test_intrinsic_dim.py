"""Intrinsic dimension of BitNet hidden states via TwoNN (Facco et al. 2017).

For each sampled hidden state point, compute distances to its 1st and 2nd
nearest neighbors, r1 and r2. The ratio μ = r2/r1 follows Pareto(d) where d
is the intrinsic dimension. MLE estimator: d = N / Σ log(μ_i).

This gives us a feature-free estimate of the manifold dimensionality of
BitNet's 2560-dim hidden states at layer 29.
"""
import numpy as np
import torch
import time

SEQ_LEN = 2048
HIDDEN = 2560

# Bootstrap the estimate by subsampling and repeating
N_SAMPLES = 5000   # per bootstrap
N_BOOTSTRAPS = 5


def twonn_mle(points):
    """TwoNN MLE intrinsic dim estimator.

    points: [N, D] tensor of float32 coordinates
    Returns: estimated intrinsic dimension (scalar)
    """
    N, D = points.shape
    # Full pairwise distance matrix
    dists = torch.cdist(points, points)  # [N, N]
    # Set diagonal to inf so self-distance doesn't win
    dists.fill_diagonal_(float('inf'))
    # For each point, take smallest 2 distances
    vals, _ = torch.topk(dists, k=2, dim=1, largest=False)  # [N, 2]
    r1 = vals[:, 0]
    r2 = vals[:, 1]
    # Drop cases where r1 == 0 (duplicate points) or r1 == r2 (degenerate)
    valid = (r1 > 1e-6) & (r2 > r1)
    r1 = r1[valid]; r2 = r2[valid]
    mu = r2 / r1
    # MLE
    log_mu = torch.log(mu)
    d_hat = len(mu) / log_mu.sum().item()
    return d_hat, len(mu)


def main():
    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    n_seqs = hidden_mm.size // per_seq
    total_points = n_seqs * SEQ_LEN
    print(f"available points: {n_seqs} seqs × {SEQ_LEN} tokens = {total_points:,}")

    # For fair sample, use positions from across sequences
    # Convert memmap to float32 on the fly (bf16 stored as uint16)
    rng = np.random.default_rng(42)
    estimates = []
    for boot in range(N_BOOTSTRAPS):
        print(f"\n=== Bootstrap {boot+1}/{N_BOOTSTRAPS} ===")
        t0 = time.time()
        # Sample N_SAMPLES random (seq, position) pairs
        seq_idxs = rng.integers(0, n_seqs, N_SAMPLES)
        pos_idxs = rng.integers(0, SEQ_LEN, N_SAMPLES)
        flat_idxs = seq_idxs * SEQ_LEN + pos_idxs

        # Gather those hidden states
        points_u16 = np.zeros((N_SAMPLES, HIDDEN), dtype=np.uint16)
        for i, fi in enumerate(flat_idxs):
            start = fi * HIDDEN
            points_u16[i] = hidden_mm[start:start + HIDDEN]
        # Convert bf16 (uint16) to float32
        raw32 = points_u16.astype(np.uint32) << 16
        points_f32 = raw32.view(np.float32).copy()
        points_t = torch.from_numpy(points_f32)
        t1 = time.time(); print(f"  sampled {N_SAMPLES} points in {t1-t0:.1f}s")

        d_hat, n_valid = twonn_mle(points_t)
        t2 = time.time()
        print(f"  TwoNN MLE intrinsic dim = {d_hat:.2f}  (valid: {n_valid}/{N_SAMPLES})")
        print(f"  (took {t2-t1:.1f}s)")
        estimates.append(d_hat)

    estimates = np.array(estimates)
    print(f"\n*** Intrinsic dimension across {N_BOOTSTRAPS} bootstraps ***")
    print(f"  mean:   {estimates.mean():.2f}")
    print(f"  median: {np.median(estimates):.2f}")
    print(f"  std:    {estimates.std():.2f}")
    print(f"  range:  [{estimates.min():.2f}, {estimates.max():.2f}]")

    # Also try a position-stratified sample: only positions > 50 (well into sequences)
    print(f"\n=== Extra: only positions ≥ 50 (later-in-sequence only) ===")
    seq_idxs = rng.integers(0, n_seqs, N_SAMPLES)
    pos_idxs = rng.integers(50, SEQ_LEN, N_SAMPLES)
    flat_idxs = seq_idxs * SEQ_LEN + pos_idxs
    points_u16 = np.zeros((N_SAMPLES, HIDDEN), dtype=np.uint16)
    for i, fi in enumerate(flat_idxs):
        start = fi * HIDDEN
        points_u16[i] = hidden_mm[start:start + HIDDEN]
    raw32 = points_u16.astype(np.uint32) << 16
    points_f32 = raw32.view(np.float32).copy()
    d_hat_late, _ = twonn_mle(torch.from_numpy(points_f32))
    print(f"  intrinsic dim (late positions only): {d_hat_late:.2f}")

    # Early positions (0-50)
    print(f"\n=== Extra: only positions < 50 (early-in-sequence only) ===")
    seq_idxs = rng.integers(0, n_seqs, N_SAMPLES)
    pos_idxs = rng.integers(0, 50, N_SAMPLES)
    flat_idxs = seq_idxs * SEQ_LEN + pos_idxs
    points_u16 = np.zeros((N_SAMPLES, HIDDEN), dtype=np.uint16)
    for i, fi in enumerate(flat_idxs):
        start = fi * HIDDEN
        points_u16[i] = hidden_mm[start:start + HIDDEN]
    raw32 = points_u16.astype(np.uint32) << 16
    points_f32 = raw32.view(np.float32).copy()
    d_hat_early, _ = twonn_mle(torch.from_numpy(points_f32))
    print(f"  intrinsic dim (early positions only): {d_hat_early:.2f}")


if __name__ == "__main__":
    main()
