"""Cache hidden states from the GGUF backbone via llama-embedding.

Produces the same binary format as cache_hidden.py but using the GGUF
inference path instead of HuggingFace, so hidden states match what
llama-medusa sees at runtime.
"""
import os
import json
import subprocess
import struct
import time
import numpy as np
import torch
from dataset import PackedTokenDataset

EMBEDDING_BIN = "/home/cpinchington/bitnet.cpp/build/bin/llama-embedding"
MODEL_GGUF = "/home/cpinchington/MedusaBitNet/models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf"


def get_hidden_states(token_ids: list[int], tokenizer_path: str) -> np.ndarray:
    """Run llama-embedding on a token sequence and return hidden states."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    text = tok.decode(token_ids, skip_special_tokens=False)

    result = subprocess.run(
        [EMBEDDING_BIN, "-m", MODEL_GGUF, "-p", text,
         "--embd-normalize", "-1", "--embd-output-format", "json",
         "-t", "16"],
        capture_output=True, text=True, timeout=120
    )

    # Parse JSON output (skip stderr)
    stdout = result.stdout
    json_start = stdout.find('{')
    if json_start < 0:
        raise RuntimeError(f"No JSON in embedding output: {stdout[:200]}")

    data = json.loads(stdout[json_start:])
    embeddings = data["data"]

    # Each entry has an embedding of dim hidden_size
    hidden = np.array([e["embedding"] for e in embeddings], dtype=np.float32)
    return hidden


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bin_path", default="data/tokens.bin")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--n_seqs", type=int, default=50, help="number of sequences to cache")
    p.add_argument("--out", default="data/hidden_gguf.bin")
    p.add_argument("--tokenizer", default="models/bitnet-b1.58-2B-4T")
    args = p.parse_args()

    dataset = PackedTokenDataset(args.bin_path, args.seq_len)
    n = min(args.n_seqs, len(dataset))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    t0 = time.time()
    with open(args.out, "wb") as f:
        for i in range(n):
            tokens = dataset[i][:args.seq_len].tolist()  # int64 list
            try:
                hidden = get_hidden_states(tokens, args.tokenizer)
                # hidden shape: [n_tokens, hidden_size]
                # We need exactly seq_len positions
                if hidden.shape[0] < args.seq_len:
                    # Pad with zeros if tokenization produced fewer tokens
                    pad = np.zeros((args.seq_len - hidden.shape[0], hidden.shape[1]), dtype=np.float32)
                    hidden = np.concatenate([hidden, pad], axis=0)
                hidden = hidden[:args.seq_len]

                # Convert to bf16 and write as uint16 (same format as cache_hidden.py)
                h_torch = torch.from_numpy(hidden).to(torch.bfloat16)
                raw = h_torch.view(torch.int16).numpy().view(np.uint16)
                raw.tofile(f)

                dt = time.time() - t0
                rate = (i + 1) / max(dt, 1e-6)
                eta = (n - i - 1) / max(rate, 1e-6)
                print(f"[cache_gguf] {i+1}/{n} seq  {rate:.3f} seq/s  elapsed={dt:.0f}s  eta={eta:.0f}s",
                      flush=True)
            except Exception as e:
                print(f"[cache_gguf] ERROR seq {i}: {e}", flush=True)
                # Write zeros for failed sequences
                zeros = np.zeros(args.seq_len * 2560, dtype=np.uint16)
                zeros.tofile(f)

    dt = time.time() - t0
    print(f"[cache_gguf] done: {n} seqs in {dt:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
