"""N-gram cache predictor at various N — measures each level of linguistic
hierarchy's predictability.

  N=1 bigram:   just the previous token → hints at morphology, inflection
  N=2 trigram:  last 2 tokens → captures phrase patterns (det+adj→noun)
  N=3 4-gram:   last 3 tokens → captures clause patterns (short SVO)
  N=4 5-gram:   last 4 tokens → captures sub-sentence structure
  N=5 6-gram:   last 5 tokens → should match almost deterministic cases

For each N, measure top-1 prediction accuracy on held-out data.
If N=4-5 gives 60%+, there's a lot of cheap signal we can harvest from
context alone without any backbone forward.
"""
import numpy as np
from collections import defaultdict

SEQ_LEN = 2048
n_seqs = 48; seq_split = 36


def main():
    tokens_mm = np.memmap("data/tokens.bin", dtype=np.uint32, mode="r")
    print("Building n-gram tables on training seqs...")

    # For each context length N, build table: N-gram → {next_tok: count}
    tables = {N: defaultdict(lambda: defaultdict(int)) for N in [1, 2, 3, 4, 5]}

    for si in range(0, seq_split):
        toks = tokens_mm[si * SEQ_LEN : (si + 1) * SEQ_LEN].astype(np.int64).tolist()
        for i in range(5, len(toks) - 1):
            nxt = toks[i + 1]
            for N in [1, 2, 3, 4, 5]:
                key = tuple(toks[i - N + 1 : i + 1])
                tables[N][key][nxt] += 1

    # For each N, top-1 predict from table on test seqs
    print(f"\n{'N':>3} {'ctx_len':>8} {'total_tests':>12} {'matched':>10} {'top1_acc':>10} {'coverage':>10}")
    print("-" * 60)

    for N in [1, 2, 3, 4, 5]:
        total = 0; correct = 0; coverage = 0
        for si in range(seq_split, n_seqs):
            toks = tokens_mm[si * SEQ_LEN : (si + 1) * SEQ_LEN].astype(np.int64).tolist()
            for i in range(5, len(toks) - 1):
                key = tuple(toks[i - N + 1 : i + 1])
                nxt = toks[i + 1]
                total += 1
                if key in tables[N]:
                    coverage += 1
                    top1 = max(tables[N][key].items(), key=lambda x: x[1])[0]
                    if top1 == nxt: correct += 1
        cov_rate = coverage / total
        acc = correct / total
        cond_acc = correct / coverage if coverage > 0 else 0
        print(f"{N:>3} {N:>8} {total:>12,} {correct:>10,} {acc:>10.4f} {cov_rate:>10.4f}")
        print(f"    conditional accuracy (only when N-gram seen): {cond_acc:.4f}")

    # Also: concentration of next-token distribution at each N
    print(f"\nAverage entropy of next-token distribution given N-gram context:")
    for N in [1, 2, 3, 4, 5]:
        entropies = []
        for key, counts in tables[N].items():
            total_c = sum(counts.values())
            if total_c < 3: continue  # ignore tiny buckets
            probs = np.array(list(counts.values())) / total_c
            h = -np.sum(probs * np.log2(probs))
            entropies.append(h)
        if entropies:
            print(f"  N={N}: mean H = {np.mean(entropies):.3f} bits, median = {np.median(entropies):.3f}")

    # Key question: at N-gram lookup HIGH CONFIDENCE (count > threshold + top-1 dominance),
    # what fraction of tokens can be skipped with 95% fidelity?
    print(f"\n=== Aggressive: skip backbone when N-gram context is highly predictive ===")
    print(f"(only trust N-gram lookup if top-1 dominance > 0.8)")
    for N in [2, 3, 4, 5]:
        total = 0; skipped = 0; correct_skips = 0
        for si in range(seq_split, n_seqs):
            toks = tokens_mm[si * SEQ_LEN : (si + 1) * SEQ_LEN].astype(np.int64).tolist()
            for i in range(5, len(toks) - 1):
                key = tuple(toks[i - N + 1 : i + 1])
                nxt = toks[i + 1]
                total += 1
                if key in tables[N]:
                    counts = tables[N][key]
                    total_c = sum(counts.values())
                    if total_c >= 3:
                        top1_t, top1_c = max(counts.items(), key=lambda x: x[1])
                        if top1_c / total_c > 0.8:
                            skipped += 1
                            if top1_t == nxt: correct_skips += 1
        skip_rate = skipped / total
        fid = correct_skips / skipped if skipped > 0 else 0
        print(f"  N={N}: skip_rate={skip_rate:.4f}, fidelity={fid:.4f}")


if __name__ == "__main__":
    main()
