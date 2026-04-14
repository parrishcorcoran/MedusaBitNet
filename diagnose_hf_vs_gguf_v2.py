"""Diagnostic: compare cached HF hidden states to GGUF l_out-29 on identical tokens.

Uses the already-computed HF cache in data/hidden.bin rather than reloading the
HF model (which hits a transformers BitNet packing bug).
"""
import numpy as np
import subprocess
import struct
from pathlib import Path
from transformers import AutoTokenizer

HF_HIDDEN  = "/home/cpinchington/MedusaBitNet/data/hidden.bin"
TOKENS_BIN = "/home/cpinchington/MedusaBitNet/data/tokens.bin"
GGUF       = "/home/cpinchington/models/official-bitnet/ggml-model-i2_s.gguf"
HIDDEN_DUMP = "/home/cpinchington/bitnet.cpp/build/bin/llama-hidden-dump"
TOKENIZER  = "/home/cpinchington/MedusaBitNet/models/bitnet-b1.58-2B-4T"

SEQ_LEN = 2048
HIDDEN = 2560
N_PROBE_SEQS = 2
N_PROBE_TOKENS = 64  # per seq — enough for a statistical signal, cheap


def load_hf_seq(seq_idx: int, n_tokens: int) -> np.ndarray:
    # hidden.bin is [n_seqs, 2048, 2560] bfloat16 row-major.
    offset = seq_idx * SEQ_LEN * HIDDEN * 2  # bf16 = 2 bytes
    count = n_tokens * HIDDEN
    with open(HF_HIDDEN, "rb") as f:
        f.seek(offset)
        raw = np.frombuffer(f.read(count * 2), dtype=np.uint16)
    # Convert bf16 (stored as uint16) to float32.
    raw32 = raw.astype(np.uint32) << 16
    f32 = raw32.view(np.float32).copy().reshape(n_tokens, HIDDEN)
    return f32


def load_tokens(seq_idx: int, n_tokens: int) -> list[int]:
    # tokens.bin is uint16 packed token IDs, [n_total_tokens] flat.
    offset = seq_idx * SEQ_LEN * 4
    with open(TOKENS_BIN, "rb") as f:
        f.seek(offset)
        ids = np.frombuffer(f.read(n_tokens * 4), dtype=np.uint32)
    return ids.astype(np.int64).tolist()


def gguf_hidden_from_ids(ids: list[int], tensor_name: str) -> np.ndarray:
    tf = "/tmp/diag_v2_tokens.bin"
    np.array(ids, dtype=np.uint32).tofile(tf)
    out = "/tmp/diag_v2.bin"
    r = subprocess.run(
        [HIDDEN_DUMP, "-m", GGUF, "-p", "placeholder",
         "--tokens-file", tf, "--tensor", tensor_name, "--dump-out", out, "-t", "16"],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"dump failed: {r.stderr.decode()[:500]}")
    return np.fromfile(out, dtype=np.float32).reshape(-1, HIDDEN)


def cosine_rows(a, b):
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    return (a*b).sum(1) / (np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1) + 1e-9)


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    for seq_idx in range(N_PROBE_SEQS):
        print(f"\n===== seq {seq_idx}, first {N_PROBE_TOKENS} tokens =====")
        ids = load_tokens(seq_idx, N_PROBE_TOKENS)
        hf = load_hf_seq(seq_idx, N_PROBE_TOKENS)
        text = tok.decode(ids[:12], skip_special_tokens=False)
        print(f"  HF hidden shape: {hf.shape}  (from cache)")
        print(f"  first 12 tokens decoded: {text!r}")

        for tname in ["l_out-29", "norm"]:
            try:
                g = gguf_hidden_from_ids(ids, tname)
            except Exception as e:
                print(f"  [{tname}] dump error: {e}")
                continue
            n = min(len(hf), len(g))
            cos = cosine_rows(hf[:n], g[:n])
            mse = ((hf[:n]-g[:n])**2).mean(1)
            nh = np.linalg.norm(hf[:n],axis=1); ng = np.linalg.norm(g[:n],axis=1)
            print(f"  [{tname}] shape={g.shape}  aligned_n={n}")
            print(f"    cos:   min={cos.min():.4f} mean={cos.mean():.4f} max={cos.max():.4f}")
            print(f"    mse:   min={mse.min():.4f} mean={mse.mean():.4f} max={mse.max():.4f}")
            print(f"    norm:  hf_mean={nh.mean():.3f} gguf_mean={ng.mean():.3f} ratio={ng.mean()/nh.mean():.3f}")


if __name__ == "__main__":
    main()
