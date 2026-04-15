"""Test the position-aware sensor hypothesis.

Bucket positions by two cheap position-based signals:
  - distance since last sentence-ender ('.', '!', '?')
  - distance since last newline

For each bucket, measure:
  - head-0 accuracy (vs vanilla greedy)
  - head-0 mean confidence
  - fraction of data landing in that bucket

Prediction (from boundary-layer thesis):
  tokens far from any sentence-ender, late in sequence, should be very
  predictable (high accuracy + high confidence). tokens right after a
  sentence-ender are starting a new sentence with wide distribution,
  should be hard. This gives a cheap Tier-0 position signal with no
  learned parameters.
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


def get_boundary_token_ids(tok):
    """Return sets of token ids for sentence-enders and newlines."""
    enders = set()
    newlines = set()
    for tid in range(VOCAB):
        s = tok.decode([tid])
        if '\n' in s:
            newlines.add(tid)
        # Any token ending in . ! or ? (just the punctuation)
        # We consider only tokens that include a sentence-ending punctuation.
        stripped = s.strip()
        if stripped and stripped[-1] in '.!?':
            enders.add(tid)
    return enders, newlines


def distance_to_last_match(token_ids, match_set):
    """For each position i, distance to the most recent j < i with
    token_ids[j] in match_set. Returns 9999 if no prior match."""
    n = len(token_ids)
    out = np.full(n, 9999, dtype=np.int32)
    last = -10000
    for i in range(n):
        if last >= 0:
            out[i] = i - last
        if token_ids[i] in match_set:
            last = i
    return out


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    print("[pos-sensor] computing boundary token sets...")
    enders, newlines = get_boundary_token_ids(tok)
    print(f"[pos-sensor] sentence-enders: {len(enders)} tokens  newlines: {len(newlines)} tokens")

    ckpt = torch.load("checkpoints/full_gguf_shift/medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    heads = MedusaHeads(HIDDEN, VOCAB, 4, 1, dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])

    lm_head = torch.load("data/lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)

    hidden_mm = np.memmap("data/hidden_gguf_v2.bin", dtype=np.uint16, mode="r")
    per_seq = SEQ_LEN * HIDDEN
    tokens_mm = np.memmap("data/tokens.bin", dtype=np.uint32, mode="r")

    n_test_seqs = 32

    all_rows = []  # (rel_pos, dist_ender, dist_newline, head_conf, correct)

    with torch.no_grad():
        for si in range(n_test_seqs):
            off = si * per_seq
            chunk = hidden_mm[off:off + per_seq]
            h = (torch.from_numpy(chunk.copy()).view(torch.bfloat16)
                       .view(SEQ_LEN, HIDDEN))
            token_ids = tokens_mm[si * SEQ_LEN : (si + 1) * SEQ_LEN]

            logits = heads(h.unsqueeze(0), lm_head)  # [1, T, K, V]
            head0_probs = F.softmax(logits[0, :, 0, :].float(), dim=-1)
            head0_conf = head0_probs.max(dim=-1).values  # [T]
            head0_pred = head0_probs.argmax(dim=-1)      # [T]

            vanilla_logits = h[1:].to(torch.bfloat16) @ lm_head.T.to(torch.bfloat16)
            vanilla_pred_all = vanilla_logits.float().argmax(dim=-1)  # [T-1], idx j -> pos j+2

            # head-0 at anchor t predicts pos t+2 (shift=2)
            # vanilla at pos p = vanilla_pred_all[p-2]. For target p=t+2, vanilla_pred_all[t].
            valid = SEQ_LEN - 2
            head0_pred_v = head0_pred[:valid]
            head0_conf_v = head0_conf[:valid]
            vanilla_v = vanilla_pred_all[:valid]

            correct = (head0_pred_v == vanilla_v).numpy().astype(np.int8)
            conf = head0_conf_v.numpy()

            # Distance signals based on TARGET token positions (t+2),
            # computed on the actual token stream up to position t+1
            # (we condition the cheap sensor on what's been emitted so far).
            # For position t (anchor), we evaluate the context tokens up to t+1
            # (inclusive). So distance_to_last_match uses token_ids[:t+2].
            dist_ender = distance_to_last_match(token_ids.tolist(), enders)  # [T]
            dist_newline = distance_to_last_match(token_ids.tolist(), newlines)

            # For anchor t (evaluating prediction of token at t+2), the relevant
            # context ended at t+1. So cheap signals should use dist_ender[t+1].
            # Ts = 0..valid-1 → use dist_ender[1..valid]
            dist_ender_at = dist_ender[1:1 + valid]
            dist_newline_at = dist_newline[1:1 + valid]
            rel_pos = np.arange(valid, dtype=np.float32) / valid  # 0..1

            all_rows.append(np.stack([
                rel_pos,
                dist_ender_at.astype(np.float32),
                dist_newline_at.astype(np.float32),
                conf.astype(np.float32),
                correct.astype(np.float32),
            ], axis=1))

    data = np.concatenate(all_rows, axis=0)
    N = len(data)
    print(f"\n[pos-sensor] scored positions: {N:,}")

    rel_pos = data[:, 0]
    dist_ender = data[:, 1]
    dist_newline = data[:, 2]
    conf = data[:, 3]
    correct = data[:, 4]

    print(f"[pos-sensor] overall head-0 acc@1: {correct.mean():.4f}")
    print(f"[pos-sensor] overall mean confidence: {conf.mean():.4f}")

    # === Bucket by distance to last sentence-ender ===
    print(f"\n=== Accuracy by distance to last sentence-ender ===")
    print(f"{'dist':>10} {'count':>8} {'fraction':>10} {'acc@1':>10} {'mean_conf':>10}")
    print("-" * 55)
    buckets = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 10000)]
    for lo, hi in buckets:
        m = (dist_ender >= lo) & (dist_ender < hi)
        n = int(m.sum())
        if n == 0: continue
        print(f"[{lo:3d}, {hi:4d}) {n:>8} {n/N:>10.3f} {correct[m].mean():>10.4f} {conf[m].mean():>10.4f}")

    # === Bucket by distance to last newline ===
    print(f"\n=== Accuracy by distance to last newline ===")
    print(f"{'dist':>10} {'count':>8} {'fraction':>10} {'acc@1':>10} {'mean_conf':>10}")
    print("-" * 55)
    for lo, hi in buckets:
        m = (dist_newline >= lo) & (dist_newline < hi)
        n = int(m.sum())
        if n == 0: continue
        print(f"[{lo:3d}, {hi:4d}) {n:>8} {n/N:>10.3f} {correct[m].mean():>10.4f} {conf[m].mean():>10.4f}")

    # === Bucket by relative position in sequence ===
    print(f"\n=== Accuracy by relative position in sequence (proxy for 'late in gen') ===")
    print(f"{'rel_pos':>15} {'count':>8} {'fraction':>10} {'acc@1':>10} {'mean_conf':>10}")
    print("-" * 60)
    rp_buckets = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    for lo, hi in rp_buckets:
        m = (rel_pos >= lo) & (rel_pos < hi)
        n = int(m.sum())
        if n == 0: continue
        print(f"[{lo:.2f}, {hi:.2f})    {n:>8} {n/N:>10.3f} {correct[m].mean():>10.4f} {conf[m].mean():>10.4f}")

    # === Joint bucket: sentence-ender distance x relative position ===
    print(f"\n=== Joint: (rel_pos, dist_ender) acc@1 (fraction) ===")
    de_edges = [0, 2, 10, 10000]
    rp_edges = [0.0, 0.3, 0.7, 1.01]
    print(f"{'':>12}", end="")
    for de_lo, de_hi in zip(de_edges[:-1], de_edges[1:]):
        print(f"{f'de[{de_lo},{de_hi})':>16}", end="")
    print()
    for rp_lo, rp_hi in zip(rp_edges[:-1], rp_edges[1:]):
        print(f"rp[{rp_lo:.2f},{rp_hi:.2f})", end="")
        for de_lo, de_hi in zip(de_edges[:-1], de_edges[1:]):
            m = ((rel_pos >= rp_lo) & (rel_pos < rp_hi)
                 & (dist_ender >= de_lo) & (dist_ender < de_hi))
            if m.sum() == 0:
                print(f"{'--':>16}", end="")
            else:
                a = correct[m].mean()
                f = m.sum() / N
                print(f"{f'{a:.3f} ({f:.3f})':>16}", end="")
        print()


if __name__ == "__main__":
    main()
