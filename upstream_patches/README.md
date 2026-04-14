# Upstream patches

Changes to `bitnet.cpp` and its vendored `llama.cpp` that are required to reproduce the
MedusaBitNet C++ inference path. These are carried as patches rather than pushed upstream
because `bitnet.cpp`'s origin is Microsoft's repo and `llama.cpp` is a third-party fork.

## What's here

- `01-bitnet.cpp-converter-arch-fix.patch` — fixes `convert-hf-to-gguf-bitnet.py` to emit
  `general.architecture = "bitnet-b1.58"` (the correct `LLM_ARCH_BITNET_B158` compute graph)
  instead of the default `"bitnet"` (which silently routes to the wrong compute graph and
  produces degenerate output). Also writes `rope.dimension_count`, `pad_token_id`,
  `add_bos_token`, and `token_scores` to match the official Microsoft GGUF.

- `02-llama.cpp-bitnet_b158-enum.patch` — adds the missing `MODEL_ARCH.BITNET_B158` enum
  and arch name mapping and tensor list to `gguf-py/gguf/constants.py`. Required by patch 01.

- `03-llama.cpp-hidden-dump-cmake.patch` + `hidden-dump.cpp` — adds a new `llama-hidden-dump`
  binary that captures a named tensor (e.g. `l_out-29`, `norm`) per token during prefill.
  This is the correct way to extract layer-N hidden states from the GGUF inference path —
  the previous approach via `llama-embedding` returned the pooled sentence embedding, not
  the per-token residual stream, and produced garbage training targets.

## How to apply

From inside `bitnet.cpp/`:

```bash
# Patch 01 targets bitnet.cpp itself
git apply /path/to/MedusaBitNet/upstream_patches/01-bitnet.cpp-converter-arch-fix.patch

# Patches 02 + 03 target the vendored llama.cpp submodule
cd 3rdparty/llama.cpp
git apply /path/to/MedusaBitNet/upstream_patches/02-llama.cpp-bitnet_b158-enum.patch
git apply /path/to/MedusaBitNet/upstream_patches/03-llama.cpp-hidden-dump-cmake.patch
cp /path/to/MedusaBitNet/upstream_patches/hidden-dump.cpp examples/eval-callback/
cd ../..

# Rebuild
cd build && cmake .. && cmake --build . --target llama-cli llama-quantize llama-hidden-dump llama-medusa
```

## Why this exists

These changes unblock the C++/GGUF deployment path for MedusaBitNet. Before the converter
patch, our produced GGUFs loaded with the wrong compute graph and generated repetitive
garbage. See the repo's main `README.md` (section "C++ deployment") for the full story.
