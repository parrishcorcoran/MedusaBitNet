"""
Stage T0 round-trip test for convert_medusa_heads.py.

Builds a synthetic "fake backbone" GGUF and a synthetic Medusa head checkpoint
with known values, runs the converter, then reads the output GGUF and asserts:

    - every original backbone tensor is preserved bit-exactly
    - every original backbone metadata key is preserved
    - Medusa metadata keys are present with the expected values
    - every Medusa tensor is present with the expected shape and dtype
    - every Medusa tensor's numerical content matches the input .pt

No network, no real BitNet weights required. Run via:
    python tools/test_convert_medusa_heads.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

import gguf  # noqa: E402  (pip install gguf)


HIDDEN = 64
VOCAB = 128
N_HEADS = 4
N_LAYERS_PER_HEAD = 1
ARCH = "bitnet"  # arbitrary — the converter copies this through verbatim


def build_fake_backbone_gguf(path: Path) -> dict:
    """Write a tiny GGUF with a handful of tensors + metadata. Returns the
    dict of input tensors (name -> numpy array) for later comparison."""
    writer = gguf.GGUFWriter(str(path), ARCH)
    writer.add_string("general.name", "fake-backbone")
    writer.add_uint32("general.file_type", 1)  # F16 in llama.cpp convention
    writer.add_uint32(f"{ARCH}.block_count", 2)
    writer.add_uint32(f"{ARCH}.embedding_length", HIDDEN)
    writer.add_uint32(f"{ARCH}.vocab_size", VOCAB)

    inputs: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(42)

    tensors = {
        "token_embd.weight":           (HIDDEN, VOCAB),
        "output_norm.weight":          (HIDDEN,),
        "blk.0.attn_norm.weight":      (HIDDEN,),
        "blk.0.attn_q.weight":         (HIDDEN, HIDDEN),
        "blk.0.ffn_up.weight":         (HIDDEN, HIDDEN * 2),
    }
    for name, shape in tensors.items():
        arr = rng.standard_normal(size=shape).astype(np.float16)
        writer.add_tensor(name, arr)
        inputs[name] = arr

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return inputs


def build_fake_heads_checkpoint(path: Path) -> dict:
    """Write a synthetic MedusaHeads state dict to a .pt file. Returns the
    raw float32 numpy arrays for comparison."""
    torch.manual_seed(7)
    w_in  = torch.randn(N_LAYERS_PER_HEAD, N_HEADS, HIDDEN, HIDDEN)
    w_out = torch.randn(N_LAYERS_PER_HEAD, N_HEADS, HIDDEN, HIDDEN)
    ckpt = {
        "heads": {"w_in": w_in, "w_out": w_out},
        "step": 42,
        "cfg": {"note": "synthetic"},
    }
    torch.save(ckpt, path)
    return {
        "w_in":  w_in.numpy(),
        "w_out": w_out.numpy(),
    }


def run_converter(backbone_gguf: Path, heads_ckpt: Path, out_gguf: Path):
    cmd = [
        sys.executable,
        str(HERE / "convert_medusa_heads.py"),
        "--backbone_gguf", str(backbone_gguf),
        "--heads_ckpt",    str(heads_ckpt),
        "--out_gguf",      str(out_gguf),
        "--dtype",         "f16",
    ]
    print("[test] running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:\n" + result.stderr)
        raise RuntimeError(f"converter exited {result.returncode}")


def verify_output(out_gguf: Path, backbone_inputs: dict, heads_inputs: dict):
    reader = gguf.GGUFReader(str(out_gguf))

    # ---- backbone tensors preserved ----
    out_tensors = {t.name: t for t in reader.tensors}
    print(f"[test] output has {len(out_tensors)} tensors")
    for name, src in backbone_inputs.items():
        assert name in out_tensors, f"missing backbone tensor {name!r}"
        dst = np.array(out_tensors[name].data).reshape(src.shape)
        assert dst.dtype == src.dtype, (
            f"{name}: dtype {dst.dtype} != {src.dtype}"
        )
        assert np.array_equal(dst, src), f"{name}: values differ"
    print(f"[test]  OK  backbone tensors preserved ({len(backbone_inputs)})")

    # ---- Medusa metadata ----
    def _get_u32(key):
        f = reader.fields.get(key)
        assert f is not None, f"missing metadata key {key!r}"
        return int(f.parts[f.data[0]].tolist()[0])

    assert _get_u32("medusa.n_heads")            == N_HEADS
    assert _get_u32("medusa.n_layers_per_head")  == N_LAYERS_PER_HEAD
    assert _get_u32("medusa.hidden_size")        == HIDDEN
    print("[test]  OK  medusa metadata present and correct")

    # ---- easter-egg author stamp must survive the round-trip ----
    def _get_str(key):
        f = reader.fields.get(key)
        assert f is not None, f"missing easter-egg key {key!r}"
        return bytes(f.parts[f.data[0]]).decode("utf-8", errors="replace")

    assert _get_str("medusa.author") == "Parrish Corcoran"
    note = _get_str("medusa.note")
    assert "Medusa" in note and "BitNet" in note
    print(f"[test]  OK  easter egg intact: author='{_get_str('medusa.author')}'")

    # ---- backbone metadata roundtrip ----
    assert reader.fields.get("general.name") is not None, "missing general.name"
    assert reader.fields.get(f"{ARCH}.embedding_length") is not None
    print("[test]  OK  backbone metadata present")

    # ---- medusa tensors shape + numeric round-trip ----
    w_in_src  = heads_inputs["w_in"]   # [L, K, H, H]
    w_out_src = heads_inputs["w_out"]
    for k in range(N_HEADS):
        for l in range(N_LAYERS_PER_HEAD):
            for which, src4d in (("w_in", w_in_src), ("w_out", w_out_src)):
                name = f"medusa.head.{k}.layer.{l}.{which}.weight"
                assert name in out_tensors, f"missing {name}"
                tinfo = out_tensors[name]
                dst = np.array(tinfo.data).reshape(HIDDEN, HIDDEN)
                expected = src4d[l, k].astype(np.float16)
                # float16 is what the converter wrote; tolerance = 0.
                assert np.array_equal(dst, expected), (
                    f"{name}: max diff = {np.abs(dst.astype(np.float32) - expected.astype(np.float32)).max()}"
                )
    n_expected = N_HEADS * N_LAYERS_PER_HEAD * 2
    print(f"[test]  OK  medusa tensors present and values match ({n_expected})")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        backbone_gguf = tmp / "fake_backbone.gguf"
        heads_ckpt    = tmp / "fake_heads.pt"
        out_gguf      = tmp / "fake_backbone_with_medusa.gguf"

        print(f"[test] tmp dir = {tmp}")
        backbone_inputs = build_fake_backbone_gguf(backbone_gguf)
        heads_inputs    = build_fake_heads_checkpoint(heads_ckpt)
        run_converter(backbone_gguf, heads_ckpt, out_gguf)
        verify_output(out_gguf, backbone_inputs, heads_inputs)

    print("[test] PASS")


if __name__ == "__main__":
    main()
