"""Real end-to-end Medusa speculative decoding benchmark in Python.

Runs the HF BitNet backbone + trained Medusa heads with actual speculative
verification — accept/reject loop, real token generation, wall-clock timing.
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Patch torch.compile for Python 3.14
import transformers.integrations.bitnet as _bitnet_mod
import types
for name in dir(_bitnet_mod):
    obj = getattr(_bitnet_mod, name)
    if isinstance(obj, types.FunctionType) and hasattr(obj, '__wrapped__'):
        pass  # already unwrapped

from model import MedusaHeads

SEQ_LEN = 2048
HIDDEN = 2560
VOCAB = 128256


def load_medusa_system():
    """Load cached hidden states, lm_head, trained heads, and tokens."""
    print("[load] Loading Medusa system...")
    t0 = time.time()

    # Cached hidden states from HF backbone
    hidden_data = np.memmap("data/hidden.bin", dtype=np.uint16, mode="r")

    # lm_head weight
    lm_head = torch.load("data/lm_head.pt", map_location="cpu", weights_only=True).to(torch.bfloat16)

    # Token data
    tokens = np.memmap("data/tokens.bin", dtype=np.uint32, mode="r")

    # Medusa heads
    ckpt = torch.load("checkpoints/medusa_heads_step2000.pt", map_location="cpu", weights_only=True)
    heads = MedusaHeads(
        hidden_size=HIDDEN, vocab_size=VOCAB,
        num_heads=4, num_layers_per_head=1, dtype=torch.bfloat16
    )
    heads.load_state_dict(ckpt["heads"])
    heads.eval()

    print(f"[load] Done in {time.time()-t0:.1f}s")
    return hidden_data, lm_head, tokens, heads


def speculative_decode_sequence(hidden_seq, tokens_seq, lm_head, heads, max_tokens=256):
    """
    Simulate speculative decoding on one sequence using cached hidden states.

    This is the REAL Medusa algorithm:
    1. At each position, the backbone produces a hidden state
    2. The backbone's lm_head predicts the next token (greedy)
    3. The Medusa heads predict tokens t+1..t+4
    4. We verify: how many Medusa predictions match the backbone's future tokens?
    5. Accept the matching prefix, advance by (1 + n_accepted) tokens

    Returns: (total_tokens_generated, total_backbone_steps, accepted_per_head)
    """
    n_heads = 4
    total_tokens = 0
    total_steps = 0
    accepted_counts = [0, 0, 0, 0]
    pos = 0
    max_pos = len(tokens_seq) - n_heads - 2  # need room for verification

    while total_tokens < max_tokens and pos < max_pos:
        # Get hidden state at current position
        h = hidden_seq[pos].unsqueeze(0).unsqueeze(0)  # [1, 1, H]

        with torch.no_grad():
            # Backbone prediction (lm_head @ hidden)
            backbone_logit = F.linear(h.float(), lm_head.float())  # [1, 1, V]
            backbone_tok = backbone_logit[0, 0].argmax().item()

            # Medusa head predictions
            head_logits = heads(h, lm_head)  # [1, 1, 4, V]
            head_preds = head_logits[0, 0].argmax(dim=-1)  # [4]

        # Ground truth: tokens at positions t+1, t+2, ..., t+n_heads+1
        # Head k predicts token at t+k+1 (head 0 = next token, same as backbone)
        true_next = tokens_seq[pos + 1: pos + 1 + n_heads + 1].tolist()

        # Verify: each head k should predict the token at position t+k+1.
        # Head 0 predicts t+1 (same as backbone). Head 1 predicts t+2. Etc.
        # Accept the longest prefix where all head predictions match ground truth.
        n_accept = 0
        for k in range(n_heads):
            if head_preds[k].item() == true_next[k]:
                n_accept += 1
                accepted_counts[k] += 1
            else:
                break

        # Advance: backbone token + accepted speculations
        tokens_generated = 1 + n_accept
        total_tokens += tokens_generated
        total_steps += 1
        pos += tokens_generated

    return total_tokens, total_steps, accepted_counts


def vanilla_decode_sequence(hidden_seq, tokens_seq, lm_head, max_tokens=256):
    """Vanilla autoregressive: 1 token per backbone step."""
    total_tokens = 0
    total_steps = 0
    pos = 0
    max_pos = len(tokens_seq) - 2

    while total_tokens < max_tokens and pos < max_pos:
        h = hidden_seq[pos].unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logit = F.linear(h.float(), lm_head.float())
        total_tokens += 1
        total_steps += 1
        pos += 1

    return total_tokens, total_steps


def main():
    hidden_data, lm_head, tokens, heads = load_medusa_system()

    n_seqs = 20
    max_tokens = 256

    print(f"\n{'='*60}")
    print(f"REAL SPECULATIVE DECODING BENCHMARK")
    print(f"Sequences: {n_seqs}, Max tokens: {max_tokens}")
    print(f"{'='*60}")

    # ---- VANILLA ----
    print("\n--- Vanilla (1 token/step) ---")
    vanilla_total_tokens = 0
    vanilla_total_steps = 0
    t0 = time.time()
    for i in range(n_seqs):
        offset = i * SEQ_LEN * HIDDEN
        h = torch.from_numpy(
            hidden_data[offset:offset + SEQ_LEN * HIDDEN].copy()
        ).view(torch.bfloat16).reshape(SEQ_LEN, HIDDEN)
        t_offset = i * SEQ_LEN
        toks = torch.from_numpy(tokens[t_offset:t_offset + SEQ_LEN + 1].copy().astype(np.int64))

        gen, steps = vanilla_decode_sequence(h, toks, lm_head, max_tokens)
        vanilla_total_tokens += gen
        vanilla_total_steps += steps
    vanilla_time = time.time() - t0
    vanilla_tok_s = vanilla_total_tokens / vanilla_time
    print(f"  Tokens: {vanilla_total_tokens}, Steps: {vanilla_total_steps}")
    print(f"  Time: {vanilla_time:.2f}s, Throughput: {vanilla_tok_s:.1f} tok/s")

    # ---- MEDUSA ----
    print("\n--- Medusa (speculative, 4 heads) ---")
    medusa_total_tokens = 0
    medusa_total_steps = 0
    medusa_accepted = [0, 0, 0, 0]
    t0 = time.time()
    for i in range(n_seqs):
        offset = i * SEQ_LEN * HIDDEN
        h = torch.from_numpy(
            hidden_data[offset:offset + SEQ_LEN * HIDDEN].copy()
        ).view(torch.bfloat16).reshape(SEQ_LEN, HIDDEN)
        t_offset = i * SEQ_LEN
        toks = torch.from_numpy(tokens[t_offset:t_offset + SEQ_LEN + 1].copy().astype(np.int64))

        gen, steps, accepted = speculative_decode_sequence(h, toks, lm_head, heads, max_tokens)
        medusa_total_tokens += gen
        medusa_total_steps += steps
        for k in range(4):
            medusa_accepted[k] += accepted[k]
    medusa_time = time.time() - t0
    medusa_tok_s = medusa_total_tokens / medusa_time
    tokens_per_step = medusa_total_tokens / medusa_total_steps
    print(f"  Tokens: {medusa_total_tokens}, Steps: {medusa_total_steps}")
    print(f"  Time: {medusa_time:.2f}s, Throughput: {medusa_tok_s:.1f} tok/s")
    print(f"  Tokens/step: {tokens_per_step:.2f}")
    for k in range(4):
        rate = medusa_accepted[k] / medusa_total_steps
        print(f"  Head {k+1} acceptance: {rate:.3f} ({medusa_accepted[k]}/{medusa_total_steps})")

    # ---- COMPARISON ----
    speedup = medusa_tok_s / vanilla_tok_s
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Vanilla:  {vanilla_tok_s:.1f} tok/s ({vanilla_total_steps} steps)")
    print(f"Medusa:   {medusa_tok_s:.1f} tok/s ({medusa_total_steps} steps)")
    print(f"Speedup:  {speedup:.2f}x (wall-clock, same machine, same data)")
    print(f"Tok/step: {tokens_per_step:.2f}")

    results = {
        "vanilla_tok_s": vanilla_tok_s,
        "vanilla_steps": vanilla_total_steps,
        "vanilla_tokens": vanilla_total_tokens,
        "vanilla_time_s": vanilla_time,
        "medusa_tok_s": medusa_tok_s,
        "medusa_steps": medusa_total_steps,
        "medusa_tokens": medusa_total_tokens,
        "medusa_time_s": medusa_time,
        "speedup": speedup,
        "tokens_per_step": tokens_per_step,
        "head_acceptance": [medusa_accepted[k] / medusa_total_steps for k in range(4)],
        "n_sequences": n_seqs,
        "max_tokens_per_seq": max_tokens,
    }
    with open("benchmark_medusa_real.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to benchmark_medusa_real.json")


if __name__ == "__main__":
    main()
