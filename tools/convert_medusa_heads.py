"""
Convert trained Medusa head checkpoints (PyTorch .pt) into a GGUF file that
bitnet.cpp can load alongside the BitNet backbone tensors.

Part of the MedusaBitNet project by Parrish Corcoran — first integration of
Medusa speculative decoding with Microsoft's BitNet b1.58 ternary-weight LLM.

Input:
    --backbone_gguf   Path to the existing BitNet GGUF (backbone weights).
    --heads_ckpt      Path to medusa_heads_stepN.pt produced by train.py.
    --out_gguf        Path to write the merged GGUF.

Output GGUF contents:
    - All tensors and metadata from the input backbone GGUF (copied verbatim).
    - New Medusa tensors, one per (layer, head):
        medusa.head.{k}.layer.{l}.w_in.weight   [H, H] F16
        medusa.head.{k}.layer.{l}.w_out.weight  [H, H] F16
    - New metadata keys:
        medusa.n_heads                 u32
        medusa.n_layers_per_head       u32
        medusa.hidden_size             u32

Tensor naming rationale: bitnet.cpp's loader looks up tensors by exact name
strings, so we pick a convention that's unambiguous and groups per-head
weights together for easy iteration on the C++ side.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

import gguf  # noqa: E402  (pip install gguf)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone_gguf", required=True, help="input BitNet backbone GGUF")
    p.add_argument("--heads_ckpt", required=True, help="trained Medusa heads .pt")
    p.add_argument("--out_gguf", required=True, help="output merged GGUF path")
    p.add_argument("--dtype", choices=["f16", "f32"], default="f16",
                   help="precision to store Medusa head weights in")
    return p.parse_args()


def _torch_to_np(t: torch.Tensor, dtype: str) -> np.ndarray:
    t = t.detach().contiguous()
    if dtype == "f16":
        return t.to(torch.float16).cpu().numpy()
    return t.to(torch.float32).cpu().numpy()


def main():
    args = parse_args()

    ckpt = torch.load(args.heads_ckpt, map_location="cpu", weights_only=False)
    heads_state = ckpt["heads"]
    # MedusaHeads.state_dict() keys: "w_in", "w_out"
    # Each of shape [num_layers_per_head, num_heads, hidden, hidden]
    w_in = heads_state["w_in"]
    w_out = heads_state["w_out"]
    assert w_in.ndim == 4 and w_out.ndim == 4, (
        f"expected 4D w_in/w_out, got {w_in.shape}, {w_out.shape}"
    )
    num_layers_per_head, num_heads, hidden, hidden2 = w_in.shape
    assert hidden == hidden2, f"w_in is not square: {w_in.shape}"
    assert tuple(w_out.shape) == tuple(w_in.shape), (
        f"w_in/w_out shape mismatch: {w_in.shape} vs {w_out.shape}"
    )
    print(f"[convert] heads: L={num_layers_per_head} K={num_heads} H={hidden}")

    reader = gguf.GGUFReader(args.backbone_gguf)
    arch = reader.fields.get("general.architecture")
    arch_name = bytes(arch.parts[arch.data[0]]).decode() if arch else "llama"
    print(f"[convert] backbone arch={arch_name}  tensors={len(reader.tensors)}")

    os.makedirs(os.path.dirname(args.out_gguf) or ".", exist_ok=True)
    writer = gguf.GGUFWriter(args.out_gguf, arch_name)

    # ---- copy all metadata fields from input -------------------------------
    skipped_fields = {
        "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
    }
    for key, field in reader.fields.items():
        if key in skipped_fields:
            continue
        try:
            value = _extract_field_value(field)
        except Exception as e:
            print(f"[convert] skipping metadata {key}: {e}")
            continue
        _add_field(writer, key, value, field.types)

    # ---- add Medusa metadata -----------------------------------------------
    writer.add_uint32("medusa.n_heads", num_heads)
    writer.add_uint32("medusa.n_layers_per_head", num_layers_per_head)
    writer.add_uint32("medusa.hidden_size", hidden)
    # Easter-egg author stamp: embedded in the GGUF itself, hidden unless
    # you dump the metadata. Travels with every copy of the model file.
    writer.add_string("medusa.author", "Parrish Corcoran")
    writer.add_string(
        "medusa.note",
        "First integration of Medusa speculative decoding with "
        "BitNet b1.58 ternary-weight inference. Built on the HP Z8 G4.",
    )

    # ---- copy all backbone tensors -----------------------------------------
    for t in reader.tensors:
        data = np.array(t.data)
        writer.add_tensor(t.name, data, raw_dtype=t.tensor_type)

    # ---- add Medusa tensors ------------------------------------------------
    for k in range(num_heads):
        for l in range(num_layers_per_head):
            for which, src in (("w_in", w_in), ("w_out", w_out)):
                name = f"medusa.head.{k}.layer.{l}.{which}.weight"
                arr = _torch_to_np(src[l, k], args.dtype)
                writer.add_tensor(name, arr)
                print(f"[convert]   + {name}  shape={arr.shape}  dtype={arr.dtype}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[convert] wrote {args.out_gguf}")


def _extract_field_value(field):
    """Reconstruct a python value from a GGUFReader field for re-emission."""
    if len(field.types) == 1 and field.types[0] == gguf.GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")
    if len(field.types) == 1:
        vt = field.types[0]
        if vt == gguf.GGUFValueType.BOOL:
            return bool(field.parts[field.data[0]][0])
        return field.parts[field.data[0]].tolist()[0]
    # Array type: [ARRAY, element_type]
    if field.types[0] == gguf.GGUFValueType.ARRAY:
        elem_t = field.types[1]
        if elem_t == gguf.GGUFValueType.STRING:
            return [bytes(field.parts[i]).decode("utf-8", errors="replace") for i in field.data]
        return [field.parts[i].tolist()[0] for i in field.data]
    raise ValueError(f"unhandled field types: {field.types}")


def _add_field(writer, key, value, types):
    vt = types[0]
    if vt == gguf.GGUFValueType.STRING:
        writer.add_string(key, value)
    elif vt == gguf.GGUFValueType.BOOL:
        writer.add_bool(key, value)
    elif vt == gguf.GGUFValueType.UINT32:
        writer.add_uint32(key, int(value))
    elif vt == gguf.GGUFValueType.INT32:
        writer.add_int32(key, int(value))
    elif vt == gguf.GGUFValueType.UINT64:
        writer.add_uint64(key, int(value))
    elif vt == gguf.GGUFValueType.INT64:
        writer.add_int64(key, int(value))
    elif vt == gguf.GGUFValueType.FLOAT32:
        writer.add_float32(key, float(value))
    elif vt == gguf.GGUFValueType.FLOAT64:
        writer.add_float64(key, float(value))
    elif vt == gguf.GGUFValueType.ARRAY:
        elem_t = types[1]
        if elem_t == gguf.GGUFValueType.STRING:
            writer.add_array(key, value)
        else:
            writer.add_array(key, value)
    else:
        raise ValueError(f"unhandled type for {key}: {types}")


if __name__ == "__main__":
    main()
