"""Position-conditional λ-controlled gate (CALM-style calibration on position signals).

For each position feature bucket, find the local confidence threshold τ_bucket
that achieves a target fidelity λ (e.g., 95% match with vanilla greedy).
At inference, use the bucket's local τ.

If the position-conditional τ gives higher skip rate than a global τ at the
same target fidelity, position signals are useful *as calibration modifiers*
even though our earlier test showed they're weak as primary predictors.

This IS the CALM technique applied to position: adaptive-threshold-per-context
with fidelity-target-driven tuning.
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import MedusaHeads

SEQ_LEN = 2048
HIDDEN  = 2560
VOCAB   = 128256
TOKENIZER_DIR = "/home/cpinchington/MedusaBitNet/models/bitnet-b1.58-2B-4T"
LAMBDA_TARGET = 0.95  # target fidelity per bucket


def get_boundary_token_ids(tok):
    enders, newlines = set(), set()
    for tid in range(VOCAB):
        s = tok.decode([tid])
        if '\n' in s: newlines.add(tid)
        stripped = s.strip()
        if stripped and stripped[-1] in '.!?': enders.add(tid)
    return enders, newlines


def distance_to_last_match(token_ids, match_set):
    n = len(token_ids); out = np.full(n, 9999, dtype=np.int32); last = -10000
    for i in range(n):
        if last >= 0: out[i] = i - last
        if token_ids[i] in match_set: last = i
    return out


def rolling_mean_conf(conf_arr, window=10):
    """Rolling mean of confidence over the last N positions (causal)."""
    out = np.zeros_like(conf_arr)
    for i in range(len(conf_arr)):
        lo = max(0, i - window)
        out[i] = conf_arr[lo:i].mean() if i > lo else conf_arr[i]
    return out


def find_local_tau(confs, correct, target_fidelity):
    """Find smallest τ such that (correct[confs > τ]).mean() >= target_fidelity.

    Returns (tau, skip_rate, achieved_fidelity). If no τ achieves the target,
    returns (inf, 0, 0).
    """
    if len(confs) == 0:
        return float('inf'), 0.0, 0.0
    order = np.argsort(-confs)  # descending
    sorted_correct = correct[order]
    sorted_conf = confs[order]
    # Candidate τ's = unique sorted confidences (and 0 as a lower bound)
    # For efficiency, try a grid of percentile-based τ's.
    cum_correct = np.cumsum(sorted_correct)
    counts = np.arange(1, len(sorted_correct) + 1)
    fidelities = cum_correct / counts
    # Largest prefix where fidelity >= target_fidelity
    ok = fidelities >= target_fidelity
    if not ok.any():
        return float('inf'), 0.0, 0.0
    # Find the LARGEST prefix that still meets target
    largest = np.where(ok)[0][-1]
    tau = sorted_conf[largest] if largest < len(sorted_conf) else 0.0
    skip_rate = (largest + 1) / len(confs)
    fidelity = fidelities[largest]
    return tau, skip_rate, fidelity


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    enders, newlines = get_boundary_token_ids(tok)
    print(f"[gate-pos] sentence-enders={len(enders)} newlines={len(newlines)}")

    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    heads = MedusaHeads(HIDDEN, VOCAB, 4, 1, dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])
    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    tokens_mm = np.memmap("data/tokens.bin", dtype=np.uint32, mode="r")

    n_seqs = 32  # double the previous test to have more per-bucket data

    all_conf = []; all_correct = []; all_dist_e = []; all_dist_n = []
    all_rel_pos = []; all_rolling = []

    with torch.no_grad():
        for si in range(n_seqs):
            off = si * per_seq
            chunk = hidden_mm[off:off + per_seq]
            h = (torch.from_numpy(chunk.copy()).view(torch.bfloat16)
                       .view(SEQ_LEN, HIDDEN))
            token_ids = tokens_mm[si * SEQ_LEN : (si + 1) * SEQ_LEN]
            logits = heads(h.unsqueeze(0), lm_head)
            probs = F.softmax(logits[0, :, 0, :].float(), dim=-1)
            conf = probs.max(dim=-1).values.numpy()
            pred = probs.argmax(dim=-1).numpy()

            vlogits = h[1:].to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
            vpred = vlogits.float().argmax(dim=-1).numpy()

            valid = SEQ_LEN - 2
            conf_v = conf[:valid]; pred_v = pred[:valid]; vpred_v = vpred[:valid]
            correct = (pred_v == vpred_v).astype(np.float32)

            dist_e = distance_to_last_match(token_ids.tolist(), enders)[1:1+valid].astype(np.float32)
            dist_n = distance_to_last_match(token_ids.tolist(), newlines)[1:1+valid].astype(np.float32)
            rel_pos = np.arange(valid, dtype=np.float32) / valid
            rolling = rolling_mean_conf(conf_v, window=10)

            all_conf.append(conf_v); all_correct.append(correct)
            all_dist_e.append(dist_e); all_dist_n.append(dist_n)
            all_rel_pos.append(rel_pos); all_rolling.append(rolling)

    conf = np.concatenate(all_conf); correct = np.concatenate(all_correct)
    dist_e = np.concatenate(all_dist_e); dist_n = np.concatenate(all_dist_n)
    rel_pos = np.concatenate(all_rel_pos); rolling = np.concatenate(all_rolling)
    N = len(conf)
    print(f"[gate-pos] total positions: {N:,}")
    print(f"[gate-pos] overall acc: {correct.mean():.4f}\n")

    # === Global τ baseline ===
    print(f"=== Baseline: GLOBAL τ (no position conditioning), target fidelity={LAMBDA_TARGET} ===")
    tau_g, skip_g, fid_g = find_local_tau(conf, correct, LAMBDA_TARGET)
    print(f"  τ={tau_g:.4f}  skip_rate={skip_g:.4f}  fidelity={fid_g:.4f}")

    # === Bucket: distance to last sentence-ender ===
    print(f"\n=== Position-conditional τ by dist-to-period ===")
    print(f"{'bucket':>20} {'n':>8} {'τ*':>8} {'skip':>8} {'fid':>8}")
    de_buckets = [(0, 2), (2, 5), (5, 15), (15, 10000)]
    total_skipped = 0
    for lo, hi in de_buckets:
        m = (dist_e >= lo) & (dist_e < hi)
        n = int(m.sum())
        if n < 200: continue
        tau_b, skip_b, fid_b = find_local_tau(conf[m], correct[m], LAMBDA_TARGET)
        skipped = int(skip_b * n)
        total_skipped += skipped
        print(f"[{lo:3d}, {hi:5d})     {n:>8} {tau_b:>8.4f} {skip_b:>8.4f} {fid_b:>8.4f}")
    print(f"  total skip rate via dist_e buckets: {total_skipped/N:.4f}")

    # === Bucket: rolling confidence (the PHYSICAL signal the user pointed at) ===
    print(f"\n=== Position-conditional τ by rolling mean confidence (10-token window) ===")
    print(f"{'bucket':>20} {'n':>8} {'τ*':>8} {'skip':>8} {'fid':>8}")
    rc_buckets = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    total_skipped = 0
    for lo, hi in rc_buckets:
        m = (rolling >= lo) & (rolling < hi)
        n = int(m.sum())
        if n < 200: continue
        tau_b, skip_b, fid_b = find_local_tau(conf[m], correct[m], LAMBDA_TARGET)
        skipped = int(skip_b * n)
        total_skipped += skipped
        print(f"[{lo:.2f}, {hi:.2f})      {n:>8} {tau_b:>8.4f} {skip_b:>8.4f} {fid_b:>8.4f}")
    print(f"  total skip rate via rolling_conf buckets: {total_skipped/N:.4f}")

    # === Joint bucket: (rolling_conf, dist_e) — strong conditioning ===
    print(f"\n=== Joint: (rolling_conf x dist_e) position-conditional τ ===")
    print(f"{'rc':>10} {'de':>10} {'n':>8} {'τ*':>8} {'skip':>8} {'fid':>8}")
    total_skipped = 0
    for rc_lo, rc_hi in rc_buckets:
        for de_lo, de_hi in de_buckets:
            m = ((rolling >= rc_lo) & (rolling < rc_hi)
                 & (dist_e >= de_lo) & (dist_e < de_hi))
            n = int(m.sum())
            if n < 150: continue
            tau_b, skip_b, fid_b = find_local_tau(conf[m], correct[m], LAMBDA_TARGET)
            skipped = int(skip_b * n)
            total_skipped += skipped
            print(f"[{rc_lo:.1f},{rc_hi:.1f})  [{de_lo:3d},{de_hi:5d})  {n:>8} {tau_b:>8.4f} {skip_b:>8.4f} {fid_b:>8.4f}")
    print(f"  total skip rate via joint buckets: {total_skipped/N:.4f}")

    print(f"\n=== Summary at λ={LAMBDA_TARGET} ===")
    print(f"  global τ alone:              skip={skip_g:.4f}")
    print(f"  position-conditional τ:      see joint-bucket total above")
    print(f"  improvement ratio:           joint / global")


if __name__ == "__main__":
    main()
