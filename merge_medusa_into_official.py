"""Graft Medusa head tensors + KVs onto Microsoft's working official GGUF.

This sidesteps the llama-quantize I2_S format divergence between stock
bitnet.cpp and Microsoft's internal tooling. The official base is proven
to produce coherent output; our Medusa heads are stand-alone tensors
with their own .scale siblings, so they can ride on top untouched.
"""
import struct
from pathlib import Path

OFFICIAL = Path("/home/cpinchington/models/official-bitnet/ggml-model-i2_s.gguf")
MEDUSA   = Path("/home/cpinchington/MedusaBitNet/models/bitnet-b1.58-2B-4T/ggml-model-i2_s-medusa.gguf")
OUT      = Path("/home/cpinchington/MedusaBitNet/models/bitnet-b1.58-2B-4T/ggml-model-i2_s-medusa-official.gguf")
ALIGN    = 32


def read_gguf(path):
    raw = path.read_bytes()
    p = 0
    assert raw[:4] == b'GGUF'
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

    kvs = []  # preserve order
    for _ in range(n_kv):
        k, p = rstr(p)
        vt, = struct.unpack_from('<I', raw, p); p += 4
        blob, p = raw_val(vt, p)
        kvs.append((k, vt, blob))

    tensors = []  # name, dims, dtype, data_offset
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


def tensor_byte_size(dims, dtype, next_off, this_off):
    return next_off - this_off


def write_string(buf, s):
    b = s.encode('utf-8')
    buf += struct.pack('<Q', len(b)) + b
    return buf


def main():
    off_raw, off_kvs, off_tensors, off_ds = read_gguf(OFFICIAL)
    med_raw, med_kvs, med_tensors, med_ds = read_gguf(MEDUSA)

    # Pick tensors: all official base, plus Medusa heads (name starts with 'medusa.')
    # AND their corresponding .scale tensors.
    off_names = {t[0] for t in off_tensors}
    medusa_extras = [t for t in med_tensors if t[0] not in off_names and
                     (t[0].startswith('medusa.') or
                      (t[0].endswith('.scale') and t[0].startswith('medusa.')))]
    # Simpler: any tensor whose name starts with 'medusa.'
    medusa_extras = [t for t in med_tensors if t[0].startswith('medusa.')]
    print(f"Official base tensors: {len(off_tensors)}")
    print(f"Medusa extras to graft: {len(medusa_extras)}")
    for t in medusa_extras:
        print(f"  {t[0]}  dims={t[1]}  dtype={t[2]}")

    # KVs: start with official, then append medusa.* entries from medusa GGUF
    medusa_kv_keep = [(k,vt,blob) for (k,vt,blob) in med_kvs if k.startswith('medusa.')]
    print(f"Medusa KVs to graft: {[k for k,_,_ in medusa_kv_keep]}")

    final_kvs = list(off_kvs) + medusa_kv_keep
    final_tensors = list(off_tensors) + medusa_extras

    # --- Build header ---
    hdr = bytearray()
    hdr += b'GGUF'
    hdr += struct.pack('<IQQ', 3, len(final_tensors), len(final_kvs))

    for (k, vt, blob) in final_kvs:
        kb = k.encode('utf-8')
        hdr += struct.pack('<Q', len(kb)) + kb
        hdr += struct.pack('<I', vt)
        hdr += blob

    # Reserve tensor info slots; we'll fill offsets in pass 2 after we know them.
    tinfo_start = len(hdr)
    # First compute tensor sizes from source files
    def source_tensor_bytes(tensors, src_ds, src_raw, idx):
        t = tensors[idx]
        next_off = tensors[idx+1][3] if idx+1 < len(tensors) else (len(src_raw) - src_ds)
        sz = next_off - t[3]
        return src_raw[src_ds + t[3] : src_ds + t[3] + sz]

    # Gather payloads in the same order as final_tensors
    payloads = []  # list of (name, dims, dtype, bytes)
    for i, t in enumerate(off_tensors):
        payloads.append((t[0], t[1], t[2], source_tensor_bytes(off_tensors, off_ds, off_raw, i)))
    med_idx_by_name = {t[0]: i for i,t in enumerate(med_tensors)}
    for t in medusa_extras:
        i = med_idx_by_name[t[0]]
        payloads.append((t[0], t[1], t[2], source_tensor_bytes(med_tensors, med_ds, med_raw, i)))

    # Write tensor info (names/dims/dtype/offset) — we need to know data layout first.
    # Reserve space by building tinfo with placeholder offsets, compute alignment, then
    # patch offsets.
    tinfo = bytearray()
    for (name, dims, dtype, data) in payloads:
        nb = name.encode('utf-8')
        tinfo += struct.pack('<Q', len(nb)) + nb
        tinfo += struct.pack('<I', len(dims))
        tinfo += struct.pack(f'<{len(dims)}Q', *dims)
        tinfo += struct.pack('<I', dtype)
        tinfo += struct.pack('<Q', 0)  # placeholder
    hdr += tinfo

    # Align data start
    data_start = len(hdr)
    if data_start % ALIGN: data_start += ALIGN - (data_start % ALIGN)
    pad = data_start - len(hdr)
    hdr += b'\x00' * pad

    # Now compute each tensor's offset (relative to data_start) respecting alignment.
    offsets = []
    cursor = 0
    for (name, dims, dtype, data) in payloads:
        if cursor % ALIGN: cursor += ALIGN - (cursor % ALIGN)
        offsets.append(cursor)
        cursor += len(data)

    # Patch offsets in tinfo region
    # Re-walk the header's tinfo to find the offset slots. Easier: rebuild tinfo with offsets.
    new_tinfo = bytearray()
    for (name, dims, dtype, data), off in zip(payloads, offsets):
        nb = name.encode('utf-8')
        new_tinfo += struct.pack('<Q', len(nb)) + nb
        new_tinfo += struct.pack('<I', len(dims))
        new_tinfo += struct.pack(f'<{len(dims)}Q', *dims)
        new_tinfo += struct.pack('<I', dtype)
        new_tinfo += struct.pack('<Q', off)
    hdr[tinfo_start : tinfo_start + len(tinfo)] = new_tinfo

    # Rebuild alignment padding (tinfo size unchanged — same count of tensors)
    assert len(new_tinfo) == len(tinfo), "tinfo size changed"

    # Write file: header + aligned data
    with open(OUT, 'wb') as f:
        f.write(hdr)
        cursor = 0
        for (name, dims, dtype, data), off in zip(payloads, offsets):
            if cursor < off:
                f.write(b'\x00' * (off - cursor))
                cursor = off
            f.write(data)
            cursor += len(data)

    print(f"\nWrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == '__main__':
    main()
