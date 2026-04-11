# Setup guide (fresh machine)

This guide takes a freshly-cloned `MedusaBitNet` checkout from zero to a
running `llama-medusa` binary on a new machine. Aimed at AMD Strix Halo
(Zen 5 + RDNA 3.5 iGPU + unified LPDDR5x), but the CPU-only path works
anywhere modern.

## 0. Prerequisites

System packages:

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip cmake build-essential numactl
```

For the iGPU training path on Strix Halo you'll also need ROCm (≥ 6.3 for
RDNA 3.5). Follow AMD's ROCm installation guide for your distro — it's too
platform-specific to script here.

## 1. Clone this repo plus bitnet.cpp

```bash
git clone https://github.com/parrishcorcoran/MedusaBitNet.git
cd MedusaBitNet

# bitnet.cpp lives alongside as a sibling clone (not a submodule) so we can
# keep our C++ edits as a patch file rather than a forked submodule.
cd ..
git clone https://github.com/microsoft/BitNet.git bitnet.cpp
cd bitnet.cpp
git submodule update --init --recursive --depth 1
```

## 2. Apply the Medusa C++ patch

```bash
cd bitnet.cpp/3rdparty/llama.cpp
git apply ../../../MedusaBitNet/patches/medusa-llama-cpp.patch
```

The patch touches `include/llama.h`, `src/llama.cpp`, `examples/CMakeLists.txt`
and adds a new `examples/medusa/` directory. If it fails to apply, the
pinned llama.cpp submodule may have drifted from what we expect — check
the submodule commit and compare with `bitnet.cpp/3rdparty/llama.cpp`
commit `1f86f058` which is what we built against.

## 3. Python environment

```bash
cd ../../..      # back to the parent dir that holds MedusaBitNet/ and bitnet.cpp/
cd MedusaBitNet
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# CPU Torch (default, works on any machine):
pip install torch --index-url https://download.pytorch.org/whl/cpu
# ... OR, for Strix Halo iGPU (ROCm 6.3+):
#     pip install torch --index-url https://download.pytorch.org/whl/rocm6.3

pip install intel_extension_for_pytorch  # only used on CPU for ipex.optimize
pip install -r requirements.txt
pip install gguf                         # needed by the converter tool
```

## 4. Build bitnet.cpp + llama-medusa

```bash
cd ../bitnet.cpp

# Copy a pre-tuned LUT kernel header. Strix Halo is x86_64 so use the TL2
# variant from whichever preset matches your target model.
cp preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl2.h include/bitnet-lut-kernels.h

# Configure and build. TL2=OFF gives a plain llama.cpp build (no custom
# BitNet GEMM kernels); flip to ON once you've verified the plain build
# works and you want the optimized ternary GEMM.
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBITNET_X86_TL2=OFF \
      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build --target llama-medusa -j

# Quick smoke: should print the usage line
./build/bin/llama-medusa --help
```

## 5. Download a BitNet backbone

Two compatible backbones are known to work:

- `1bitLLM/bitnet_b1_58-3B` — the "original" BitNet, has preset LUT kernels
  in this repo, converter supports it.
- `microsoft/BitNet-b1.58-2B-4T` — newer microsoft release used during the
  Z8 training run. Currently requires extending `convert-hf-to-gguf-bitnet.py`
  to support the `BitNetForCausalLM` architecture.

```bash
cd ../MedusaBitNet
source .venv/bin/activate

# Pick one:
huggingface-cli download 1bitLLM/bitnet_b1_58-3B \
    --local-dir ../bitnet.cpp/models/bitnet_b1_58-3B
# OR
huggingface-cli download microsoft/BitNet-b1.58-2B-4T \
    --local-dir ../bitnet.cpp/models/BitNet-b1.58-2B-4T
```

Convert to GGUF (f32 is sufficient for training; you can quantize later):

```bash
cd ../bitnet.cpp
python utils/convert-hf-to-gguf-bitnet.py models/bitnet_b1_58-3B --outtype f32
# produces models/bitnet_b1_58-3B/ggml-model-f32.gguf
```

## 6. Tokenize the training corpus

```bash
cd ../MedusaBitNet
source .venv/bin/activate
python -c "from dataset import build_token_bin, PackingConfig; \
           build_token_bin(PackingConfig(tokenizer_name_or_path='microsoft/bitnet-b1.58-2B-4T'))"
# produces data/tokens.bin
```

## 7. Cache backbone hidden states

This is the expensive step on CPU. On Strix Halo's iGPU it should be
dramatically faster (single-digit minutes vs. the 11.5 hours the Z8 took).

```bash
# CPU (dual socket). Takes ~11.5 hours on a Xeon Platinum 8260 with 16 cores/socket.
./run_cache_parallel.sh

# OR: single-machine, single-device. On Strix Halo iGPU this is preferred.
# (Requires adding --device cuda support to cache_hidden.py — TODO.)
python cache_hidden.py --start 0 --end -1 \
    --out data/hidden.bin \
    --lm_head_out data/lm_head.pt
```

Outputs: `data/hidden.bin` (~20 GB for 2023 sequences of length 2048) and
`data/lm_head.pt` (~627 MB tied LM head).

## 8. Train Medusa heads

Minutes-to-days depending on device. CPU default:

```bash
python train.py \
    --cached_hidden_path data/hidden.bin \
    --cached_lm_head_path data/lm_head.pt \
    --max_steps 2000 \
    --log_every 10
```

Strix Halo iGPU (if ROCm is set up):

```bash
python train.py \
    --cached_hidden_path data/hidden.bin \
    --cached_lm_head_path data/lm_head.pt \
    --max_steps 2000 \
    --log_every 10 \
    --device cuda      # ROCm uses the CUDA API name
```

Expected signals (validated on Z8 CPU at step 40):
- Loss drops from ~9.8 to ~5.5 within 40 steps
- acc@1 stays around the backbone's own next-token accuracy (~0.65)
- acc@2 climbs from ~0.05 to ~0.25 by step 40, toward ~0.5 by step 100-150

Checkpoints go to `checkpoints/medusa_heads_stepN.pt`.

## 9. Convert to GGUF

```bash
python tools/convert_medusa_heads.py \
    --backbone_gguf ../bitnet.cpp/models/bitnet_b1_58-3B/ggml-model-f32.gguf \
    --heads_ckpt    checkpoints/medusa_heads_step2000.pt \
    --out_gguf      ../bitnet.cpp/models/bitnet_b1_58-3B/ggml-model-f32-medusa.gguf
```

The output GGUF carries the original backbone tensors plus
`medusa.head.{k}.layer.{l}.{w_in,w_out}.weight` and the metadata keys
`medusa.n_heads`, `medusa.n_layers_per_head`, `medusa.hidden_size`,
plus an author-stamp easter egg (`medusa.author`, `medusa.note`).

## 10. Run Medusa speculative decoding

```bash
cd ../bitnet.cpp
./build/bin/llama-medusa \
    -m models/bitnet_b1_58-3B/ggml-model-f32-medusa.gguf \
    -p "The capital of France" \
    -n 64
```

Expected: output text identical to vanilla greedy decoding on the same
prompt, plus an `[medusa] steps=... accepted_speculations=... mean_accept_per_step=...`
line reporting how often the speculated continuations were accepted.

## Troubleshooting

**`AttributeError: 'memmap' object has no attribute 'newbyteorder'`** — your
`gguf` package is too old for numpy 2.x. Fix:
```bash
pip install --upgrade gguf
```

**`__m256i has not been declared`** during the bitnet.cpp build — the LUT
kernel header was copied but the compiler isn't getting AVX2 flags. Either
pass `-DBITNET_X86_TL2=OFF` (disables the ternary kernels; uses plain
llama.cpp instead) or make sure `-march=native` reaches the `ggml-bitnet-*`
translation units.

**`expected 296, got 288 tensors`** when loading a Medusa GGUF — you're
loading via an architecture branch that doesn't have the Medusa loader
code. Check that `LLM_ARCH_BITNET` and `LLM_ARCH_BITNET_B158` both have the
optional Medusa loader block from the patch.
