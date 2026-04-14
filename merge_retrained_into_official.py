"""Graft retrained Medusa heads from a PyTorch checkpoint onto Microsoft's
official BitNet GGUF.

Supersedes `merge_medusa_into_official.py`, which copied head bytes from an
older Medusa GGUF (I2_S-quantized heads trained on HF hidden states).

This script:
1. Reads official base GGUF bytes as-is (verified working with llama-cli).
2. Loads retrained heads from `checkpoints/<run>/medusa_heads_stepN.pt`.
3. Transposes them Python [in, out] -> GGML [out, in].
4. Writes them as F16 tensors with names `medusa.head.{k}.layer.{l}.{w_in,w_out}.weight`.
5. Emits the merged GGUF.

Usage:
    python3 merge_retrained_into_official.py \
        --ckpt checkpoints/full_gguf/medusa_heads_step2000.pt \
        --out ggml-model-i2_s-medusa-v2.gguf
"""
import argparse
import struct
from pathlib import Path

import numpy as np
import torch

OFFICIAL = Path("/home/cpinchington/models/official-bitnet/ggml-model-i2_s.gguf")
ALIGN = 32

GGML_TYPE_F16 = 1
GGML_TYPE_F32 = 0


def read_gguf(path: Path):
    raw = path.read_bytes()
    p = 4
    version, n_tensors, n_kv = struct.unpack_from('<IQQ', raw, p); p += 20

    def rstr(p):
        l, = struct.unpack_from('<Q', raw, p); p += 8
        return raw[p:p+l].decode('utf-8','replace'), p+l

    def raw_val(vt, p):
        start = p
        if vt in (0,1,7): p += 1
        elif vt in (2,3): p += 2
        elif vt in (4,5,6): p += 4
        elif vt in (10,11,12): p += 8
        elif vt == 8:
            l, = struct.unpack_from('<Q', raw, p); p += 8 + l
        elif vt == 9:
            at, = struct.unpack_from('<I', raw, p); p += 4
            n,  = struct.unpack_from('<Q', raw, p); p += 8
            for _ in range(n):
                _, p = raw_val(at, p)
        return raw[start:p], p

    kvs = []
    for _ in range(n_kv):
        k, p = rstr(p)
        vt, = struct.unpack_from('<I', raw, p); p += 4
        blob, p = raw_val(vt, p)
        kvs.append((k, vt, blob))

    tensors = []
    for _ in range(n_tensors):
        name, p = rstr(p)
        nd, = struct.unpack_from('<I', raw, p); p += 4
        dims = struct.unpack_from(f'<{nd}Q', raw, p); p += 8*nd
        dt, = struct.unpack_from('<I', raw, p); p += 4
        off, = struct.unpack_from('<Q', raw, p); p += 8
        tensors.append((name, list(dims), dt, off))

    ds = p
    if ds % ALIGN: ds += ALIGN - (ds % ALIGN)
    return raw, kvs, tensors, ds


def encode_string_kv(k, value_bytes, vt):
    kb = k.encode('utf-8')
    out = struct.pack('<Q', len(kb)) + kb
    out += struct.pack('<I', vt)
    out += value_bytes
    return out


def kv_uint32(k, v):
    kb = k.encode('utf-8')
    return struct.pack('<Q', len(kb)) + kb + struct.pack('<I', 4) + struct.pack('<I', v)


def kv_string(k, s):
    kb = k.encode('utf-8')
    sb = s.encode('utf-8')
    return (struct.pack('<Q', len(kb)) + kb + struct.pack('<I', 8) +
            struct.pack('<Q', len(sb)) + sb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--author", default="Parrish Corcoran")
    ap.add_argument("--note",
                    default=("MedusaBitNet heads retrained on GGUF-native hidden "
                             "states (post-final-RMSNorm from bitnet.cpp's "
                             "llama-hidden-dump). Parrish Corcoran, 2026."))
    args = ap.parse_args()

    print(f"[merge] loading official base GGUF: {OFFICIAL}")
    raw_off, off_kvs, off_tensors, off_ds = read_gguf(OFFICIAL)

    print(f"[merge] loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    heads = ckpt['heads']
    # w_in/w_out shape: [num_layers, num_heads, H, H]
    w_in = heads['w_in']
    w_out = heads['w_out']
    num_layers, num_heads, H, H2 = w_in.shape
    assert H == H2
    print(f"[merge] heads: layers={num_layers} heads={num_heads} H={H}")

    # Build new tensor payloads. Transpose each [H_in, H_out] -> [H_out, H_in]
    # because Python einsum uses [input, output] but ggml mul_mat expects
    # [output, input] in row-major layout.
    new_tensors = []  # list of (name, dims, dtype, bytes)
    for k in range(num_heads):
        for l in range(num_layers):
            for which, src in (("w_in", w_in), ("w_out", w_out)):
                name = f"medusa.head.{k}.layer.{l}.{which}.weight"
                arr = src[l, k].T.contiguous().to(torch.float16).numpy()
                # ggml stores shape as (ne[0]=cols, ne[1]=rows) for 2D tensors:
                # ne[0] is innermost; for a [H_out, H_in] matrix the innermost
                # stride is H_in, so ne[0]=H_in, ne[1]=H_out. But we transposed
                # from [H_in, H_out] to [H_out, H_in], so .shape is (H_out, H_in)
                # -> ne[0]=H_in (innermost), ne[1]=H_out.
                dims = [arr.shape[1], arr.shape[0]]  # [ne0, ne1]
                data = arr.tobytes()
                new_tensors.append((name, dims, GGML_TYPE_F16, data))
                print(f"[merge]   + {name}  dims={dims}  dtype=F16  bytes={len(data)}")

    # Collect original base tensor payloads
    def tensor_bytes(tensors, src_ds, src_raw, idx):
        t = tensors[idx]
        next_off = tensors[idx+1][3] if idx+1 < len(tensors) else (len(src_raw) - src_ds)
        sz = next_off - t[3]
        return src_raw[src_ds + t[3] : src_ds + t[3] + sz]

    payloads = []
    for i, t in enumerate(off_tensors):
        payloads.append((t[0], t[1], t[2], tensor_bytes(off_tensors, off_ds, raw_off, i)))
    payloads.extend(new_tensors)

    # Build KVs: official + medusa.*
    kv_blob = bytearray()
    for k, vt, blob in off_kvs:
        kv_blob += encode_string_kv(k, blob, vt)
    kv_blob += kv_uint32("medusa.n_heads", int(num_heads))
    kv_blob += kv_uint32("medusa.n_layers_per_head", int(num_layers))
    kv_blob += kv_uint32("medusa.hidden_size", int(H))
    kv_blob += kv_string("medusa.author", args.author)
    kv_blob += kv_string("medusa.note", args.note)
    n_kv_final = len(off_kvs) + 5

    # Header
    hdr = bytearray()
    hdr += b'GGUF'
    hdr += struct.pack('<IQQ', 3, len(payloads), n_kv_final)
    hdr += kv_blob

    # Tensor info with placeholder offsets (rebuilt after we know layout)
    def build_tinfo(payloads, offsets):
        buf = bytearray()
        for (name, dims, dtype, _data), off in zip(payloads, offsets):
            nb = name.encode('utf-8')
            buf += struct.pack('<Q', len(nb)) + nb
            buf += struct.pack('<I', len(dims))
            buf += struct.pack(f'<{len(dims)}Q', *dims)
            buf += struct.pack('<I', dtype)
            buf += struct.pack('<Q', off)
        return buf

    placeholder_tinfo = build_tinfo(payloads, [0]*len(payloads))
    hdr_with_tinfo = bytes(hdr) + bytes(placeholder_tinfo)
    data_start = len(hdr_with_tinfo)
    if data_start % ALIGN: data_start += ALIGN - (data_start % ALIGN)

    # Compute each tensor's final offset (relative to data_start)
    offsets = []
    cursor = 0
    for (_, _, _, data) in payloads:
        if cursor % ALIGN: cursor += ALIGN - (cursor % ALIGN)
        offsets.append(cursor)
        cursor += len(data)

    final_tinfo = build_tinfo(payloads, offsets)
    assert len(final_tinfo) == len(placeholder_tinfo), "tinfo size changed"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[merge] writing {out_path}")
    with open(out_path, 'wb') as f:
        f.write(hdr)
        f.write(final_tinfo)
        pad = data_start - (len(hdr) + len(final_tinfo))
        if pad > 0:
            f.write(b'\x00' * pad)
        cursor = 0
        for (_, _, _, data), off in zip(payloads, offsets):
            if cursor < off:
                f.write(b'\x00' * (off - cursor))
                cursor = off
            f.write(data)
            cursor += len(data)

    size = out_path.stat().st_size
    print(f"[merge] done: {size:,} bytes ({size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
