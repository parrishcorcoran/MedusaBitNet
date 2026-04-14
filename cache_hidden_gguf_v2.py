"""Cache GGUF hidden states for Medusa head training.

Supersedes the broken cache_hidden_gguf.py (which called llama-embedding and
got pooled sentence embeddings, not per-token residual stream).

Uses llama-hidden-dump to capture the 'norm' tensor (post-output-RMSNorm,
the same signal that feeds the lm_head at inference time) per token per
sequence, then converts to bf16 and writes in the same layout as hidden.bin
(the existing HF cache) so dataset.py/train.py work unchanged.

Layout of output:
    [n_seqs, SEQ_LEN, HIDDEN] bf16 row-major  (matches data/hidden.bin)
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

HIDDEN_DUMP = "/home/cpinchington/bitnet.cpp/build/bin/llama-hidden-dump"
MODEL_GGUF  = "/home/cpinchington/models/official-bitnet/ggml-model-i2_s.gguf"
TOKENS_BIN  = "/home/cpinchington/MedusaBitNet/data/tokens.bin"

SEQ_LEN = 2048
HIDDEN  = 2560


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end",   type=int, default=5, help="exclusive")
    p.add_argument("--out",   required=True, help="output bin path")
    p.add_argument("--tensor", default="result_norm",
                   help="tensor to capture. Default 'result_norm' matches the "
                        "tap point llama-medusa's C++ code uses for h_final.")
    p.add_argument("--threads", type=int, default=16)
    args = p.parse_args()

    tok_mem = np.memmap(TOKENS_BIN, dtype=np.uint32, mode="r")
    total_seqs = len(tok_mem) // SEQ_LEN
    start = max(0, args.start)
    end   = min(args.end, total_seqs)
    n = end - start
    print(f"[cache_gguf_v2] caching {n} sequences [{start}, {end}) -> {args.out}")
    print(f"[cache_gguf_v2] tensor={args.tensor} threads={args.threads}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    tokens_tmp = "/tmp/cache_gguf_tokens.bin"
    dump_tmp   = "/tmp/cache_gguf_dump.bin"

    with open(args.out, "wb") as fout:
        for i in range(start, end):
            ids = tok_mem[i * SEQ_LEN : (i + 1) * SEQ_LEN].astype(np.uint32)
            ids.tofile(tokens_tmp)

            r = subprocess.run(
                [HIDDEN_DUMP, "-m", MODEL_GGUF, "-p", "x",
                 "--tokens-file", tokens_tmp, "--tensor", args.tensor,
                 "--dump-out", dump_tmp, "-t", str(args.threads),
                 "-c", "4096", "-b", "2048", "-ub", "2048"],
                capture_output=True, timeout=600,
            )
            if r.returncode != 0:
                print(f"[cache_gguf_v2] ERROR seq {i}: {r.stderr.decode()[-300:]}")
                # Write zeros to preserve layout
                np.zeros(SEQ_LEN * HIDDEN, dtype=np.uint16).tofile(fout)
                continue

            # Dump is [HIDDEN, N_TOKENS] f32. Reshape to [N_TOKENS, HIDDEN].
            # (ggml tensor shape ne[0]=HIDDEN, ne[1]=N; binary is row-major over
            # ne[0] first, then ne[1] — so byte layout is tok0[h0..H-1], tok1[...], ...)
            raw = np.fromfile(dump_tmp, dtype=np.float32)
            if raw.size != SEQ_LEN * HIDDEN:
                print(f"[cache_gguf_v2] WARN seq {i}: got {raw.size} floats, expected {SEQ_LEN*HIDDEN}")
                # Truncate or pad
                if raw.size < SEQ_LEN * HIDDEN:
                    pad = np.zeros(SEQ_LEN * HIDDEN - raw.size, dtype=np.float32)
                    raw = np.concatenate([raw, pad])
                raw = raw[:SEQ_LEN * HIDDEN]
            arr = raw.reshape(SEQ_LEN, HIDDEN)

            # Convert f32 -> bf16 (stored as uint16) to match hidden.bin layout
            t = torch.from_numpy(arr).to(torch.bfloat16)
            bf = t.view(torch.int16).numpy().view(np.uint16)
            bf.tofile(fout)

            elapsed = time.time() - t0
            done = i - start + 1
            rate = done / max(elapsed, 1e-6)
            eta = (n - done) / max(rate, 1e-6)
            print(f"[cache_gguf_v2] seq {i:5d} done  ({done}/{n})  "
                  f"{rate:.3f} seq/s  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                  flush=True)

    elapsed = time.time() - t0
    print(f"[cache_gguf_v2] done: {n} seqs in {elapsed:.0f}s -> {args.out}")
    print(f"[cache_gguf_v2] size: {os.path.getsize(args.out)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
