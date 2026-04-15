"""Quick analysis: how often are tokens sub-word continuations?

In BPE tokenizers (GPT2, Llama3-style), tokens starting with space ' ' are
word-starts. Tokens without leading space are continuations of the previous
word.

Within-word continuations should be trivially predictable from the partial
word — 'extra' → 'ordinary' should be near-certain. If we can just TABLE
these without any model forward, that's free speedup.

Measures:
  - Fraction of tokens that are continuations
  - Accuracy of trivial cache lookup (bigram) at predicting continuations
  - Acceptance rate of head-0 specifically at word-continuation positions
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import MedusaHeads

SEQ_LEN = 2048; HIDDEN = 2560; VOCAB = 128256
TOKENIZER_DIR = "/home/cpinchington/MedusaBitNet/models/bitnet-b1.58-2B-4T"


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    # Classify every token in vocab: is it a word-start (starts with space or special) or continuation?
    print("Classifying vocab...")
    is_word_start = np.zeros(VOCAB, dtype=bool)
    is_all_letters = np.zeros(VOCAB, dtype=bool)
    for tid in range(VOCAB):
        s = tok.decode([tid])
        if not s: continue
        if s[0] in ' \n\t':
            is_word_start[tid] = True
        elif not s[0].isalpha():  # punct, digit, etc -- treat as word-start
            is_word_start[tid] = True
        # else: continuation (starts with letter, no leading space)
        if s.strip().isalpha():
            is_all_letters[tid] = True

    n_continuations = VOCAB - is_word_start.sum()
    print(f"vocab: {VOCAB}, word-starts: {is_word_start.sum()}, continuations: {n_continuations}")

    # Now measure on our data: how often do we emit a continuation?
    tokens_mm = np.memmap("data/tokens.bin", dtype=np.uint32, mode="r")
    n_seqs = 48
    total_tokens = 0
    total_continuations = 0
    total_letter_continuations = 0

    # For each position, is token[t+1] a continuation of token[t]?
    for si in range(n_seqs):
        toks = tokens_mm[si * SEQ_LEN : (si + 1) * SEQ_LEN]
        for t in toks:
            total_tokens += 1
            if not is_word_start[t]:
                total_continuations += 1
            if is_all_letters[t] and not is_word_start[t]:
                total_letter_continuations += 1

    print(f"\nOn {n_seqs} test seqs:")
    print(f"  total tokens:              {total_tokens:,}")
    print(f"  word-start tokens:         {total_tokens - total_continuations:,} ({(total_tokens - total_continuations)/total_tokens:.2%})")
    print(f"  continuation tokens:       {total_continuations:,} ({total_continuations/total_tokens:.2%})")
    print(f"  letter-only continuations: {total_letter_continuations:,} ({total_letter_continuations/total_tokens:.2%})")
    print(f"  (letter continuations = mid-word subtokens: the most predictable)")

    # Bigram cache: for each token, how often is the NEXT token a continuation
    # AND how often does a simple bigram table (build from first half of data)
    # predict it correctly?
    train_seqs = list(range(0, 32))
    test_seqs = list(range(32, 48))

    print("\nBuilding bigram table from training seqs...")
    bigram = {}  # (prev_tok, continuation?) → {next_tok: count}
    for si in train_seqs:
        toks = tokens_mm[si * SEQ_LEN : (si + 1) * SEQ_LEN]
        for i in range(len(toks) - 1):
            prev = int(toks[i])
            nxt = int(toks[i + 1])
            is_cont = not is_word_start[nxt]
            key = (prev, is_cont)
            bigram.setdefault(key, {})
            bigram[key][nxt] = bigram[key].get(nxt, 0) + 1

    # Test: for each (prev, is_cont_next) in test data, can bigram top-1 predict correctly?
    test_correct_cont = 0
    test_total_cont = 0
    test_correct_start = 0
    test_total_start = 0
    for si in test_seqs:
        toks = tokens_mm[si * SEQ_LEN : (si + 1) * SEQ_LEN]
        for i in range(len(toks) - 1):
            prev = int(toks[i])
            nxt = int(toks[i + 1])
            is_cont = not is_word_start[nxt]
            key = (prev, is_cont)
            if key in bigram:
                top1 = max(bigram[key].items(), key=lambda x: x[1])[0]
                if is_cont:
                    test_total_cont += 1
                    if top1 == nxt: test_correct_cont += 1
                else:
                    test_total_start += 1
                    if top1 == nxt: test_correct_start += 1

    print(f"\nBigram-table predictor accuracy on test seqs:")
    if test_total_cont > 0:
        print(f"  predicting CONTINUATION: {test_correct_cont}/{test_total_cont} = {test_correct_cont/test_total_cont:.2%}")
    if test_total_start > 0:
        print(f"  predicting WORD-START:  {test_correct_start}/{test_total_start} = {test_correct_start/test_total_start:.2%}")

    # Practical significance: if bigram table gets say 90% on continuations
    # and continuations are 40% of tokens, we can skip backbone on 40% * 90% = 36% of tokens
    if test_total_cont > 0 and test_total_start > 0:
        skip_rate = (total_continuations / total_tokens) * (test_correct_cont / test_total_cont)
        print(f"\n*** Potential backbone-skip rate (bigram table on continuations only): {skip_rate:.2%}")
        print(f"    (= continuation fraction × bigram accuracy) ")


if __name__ == "__main__":
    main()
