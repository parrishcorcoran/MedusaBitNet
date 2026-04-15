"""Merge only the first K heads from a 4-head checkpoint into the official GGUF."""
import argparse, torch
from merge_retrained_into_official import read_gguf, OFFICIAL, ALIGN, GGML_TYPE_F16
from merge_retrained_into_official import kv_uint32, kv_string, encode_string_kv
import struct
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, required=True, help="number of heads to keep")
    args = ap.parse_args()

    print(f"[merge-k] k={args.k}")
    raw_off, off_kvs, off_tensors, off_ds = read_gguf(OFFICIAL)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    w_in = ckpt['heads']['w_in']  # [L, K, H, H]
    w_out = ckpt['heads']['w_out']
    num_layers, num_heads_avail, H, _ = w_in.shape
    assert args.k <= num_heads_avail, f"ckpt has {num_heads_avail} heads, need {args.k}"

    new_tensors = []
    for k in range(args.k):
        for l in range(num_layers):
            for which, src in (("w_in", w_in), ("w_out", w_out)):
                name = f"medusa.head.{k}.layer.{l}.{which}.weight"
                arr = src[l, k].T.contiguous().to(torch.float16).numpy()
                new_tensors.append((name, [arr.shape[1], arr.shape[0]], GGML_TYPE_F16, arr.tobytes()))

    def tbytes(tensors, ds, raw, idx):
        t = tensors[idx]
        next_off = tensors[idx+1][3] if idx+1 < len(tensors) else (len(raw) - ds)
        return raw[ds + t[3] : ds + t[3] + (next_off - t[3])]

    payloads = [(t[0], t[1], t[2], tbytes(off_tensors, off_ds, raw_off, i))
                for i, t in enumerate(off_tensors)] + new_tensors

    kv_blob = bytearray()
    for k, vt, blob in off_kvs:
        kv_blob += encode_string_kv(k, blob, vt)
    kv_blob += kv_uint32("medusa.n_heads", args.k)
    kv_blob += kv_uint32("medusa.n_layers_per_head", num_layers)
    kv_blob += kv_uint32("medusa.hidden_size", H)
    kv_blob += kv_string("medusa.author", "Parrish Corcoran")
    kv_blob += kv_string("medusa.note", f"MedusaBitNet {args.k}-head variant for linear speculation benchmarks. Retrained on GGUF result_norm, shift=k+2.")
    n_kv_final = len(off_kvs) + 5

    hdr = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(payloads), n_kv_final) + bytes(kv_blob))

    def build_tinfo(pays, offsets):
        buf = bytearray()
        for (name, dims, dtype, _), off in zip(pays, offsets):
            nb = name.encode('utf-8')
            buf += struct.pack('<Q', len(nb)) + nb
            buf += struct.pack('<I', len(dims)) + struct.pack(f'<{len(dims)}Q', *dims)
            buf += struct.pack('<I', dtype) + struct.pack('<Q', off)
        return buf

    placeholder = build_tinfo(payloads, [0]*len(payloads))
    data_start = len(hdr) + len(placeholder)
    if data_start % ALIGN: data_start += ALIGN - (data_start % ALIGN)

    offsets, cursor = [], 0
    for (_, _, _, data) in payloads:
        if cursor % ALIGN: cursor += ALIGN - (cursor % ALIGN)
        offsets.append(cursor); cursor += len(data)

    final_tinfo = build_tinfo(payloads, offsets)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        f.write(hdr); f.write(final_tinfo)
        pad = data_start - (len(hdr) + len(final_tinfo))
        if pad > 0: f.write(b'\x00' * pad)
        cursor = 0
        for (_, _, _, data), off in zip(payloads, offsets):
            if cursor < off: f.write(b'\x00' * (off - cursor)); cursor = off
            f.write(data); cursor += len(data)
    print(f"[merge-k] wrote {out} ({out.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
