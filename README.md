# MedusaBitNet

Train [Medusa](https://github.com/FasterDecoding/Medusa) speculative-decoding heads on top of a **frozen [BitNet b1.58](https://arxiv.org/abs/2402.17764)** backbone — **on CPU**, with aggressive AVX-512 fusion via Intel Extension for PyTorch (IPEX).

Designed and tuned for an **HP Z8 G4** workstation:
- Dual Intel Xeon (Cascade Lake / Skylake-SP) with AVX-512
- ~400 GB system RAM
- A low-end 4 GB display GPU that **must not** be touched by PyTorch

> **Why this exists.** BitNet b1.58 makes CPU inference viable; Medusa adds `k` extra heads that predict future tokens in parallel for speculative decoding. Training only the heads (the backbone is frozen) is small enough to fit comfortably on a beefy dual-Xeon box, with no GPU required.

---

## Repository layout

```
MedusaBitNet/
├── model.py              # Frozen BitNet + k vectorized Medusa heads
├── dataset.py            # mmap'd uint32 token-packed dataset
├── train.py              # CPU-only training loop (IPEX + torch.compile)
├── requirements.txt
├── run_z8_training.sh    # NUMA-pinned launcher for the Z8 G4
└── README.md
```

---

## Key design choices

### 1. Strictly CPU — no accidental GPU usage
The Z8 G4 ships with a tiny display GPU. If PyTorch initializes CUDA on it, training falls off a cliff. `train.py` sets

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

**before any torch import**, and `run_z8_training.sh` re-exports the same variable as a belt-and-suspenders guard.

### 2. Vectorized Medusa heads — no Python `for` loops in the hot path
Naïvely, `k` Medusa heads become a Python loop over `k` separate `nn.Linear`s. On CPU that's catastrophic — every iteration pays Python/dispatch overhead.

Instead, `MedusaHeads` (see [`model.py`](model.py)) **stacks all `k` heads into a single parameter** of shape `[num_layers, k, H, H]` and executes the residual block with a single fused `torch.einsum("bthi,hio->btho", ...)`. The final vocab projection is likewise one fused einsum producing `[B, T, k, V]`. No per-head Python iteration.

### 3. AVX-512 fusion via IPEX + `torch.compile`
```python
import intel_extension_for_pytorch as ipex
model, optim = ipex.optimize(model, optimizer=optim, dtype=torch.bfloat16, inplace=True)
model = torch.compile(model, backend="ipex")
```
`ipex.optimize` folds ops into AVX-512 bf16 kernels, and `torch.compile` with the `ipex` backend JITs the full training graph. If the backend isn't registered in your IPEX build, the code falls back to eager mode with a printed warning.

### 4. NUMA pinning via `numactl`
The Z8 G4 is a **dual-socket** box. Cross-socket traffic (UPI latency + cache ping-pong) wrecks bf16 GEMM throughput. `run_z8_training.sh` pins the process to **socket 0** for both CPU scheduling and memory allocation:

```bash
exec numactl --cpunodebind=0 --membind=0 python train.py "$@"
```

To use socket 1 instead, swap both flags to `1`. Confirm your node layout with `numactl --hardware`.

### 5. Memory-mapped token packing
`dataset.py` does a one-shot tokenization pass into a single `uint32` `.bin` file, then slices fixed-length sequences via `np.memmap`. No Python work in the data path at train time.

### 6. Only the heads are trainable
The backbone runs under `torch.no_grad()` and every backbone parameter has `requires_grad=False`. Checkpoints save **only** `model.heads.state_dict()` — they're small.

---

## Installation (HP Z8 G4, Ubuntu)

```bash
# 1) Clone
git clone https://github.com/parrishcorcoran/MedusaBitNet.git
cd MedusaBitNet

# 2) (Recommended) create a fresh venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3) Install PyTorch CPU + IPEX that are version-matched
#    IPEX MUST match your torch minor version — check
#    https://intel.github.io/intel-extension-for-pytorch/ for the latest pair.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install intel_extension_for_pytorch

# 4) Everything else
pip install -r requirements.txt

# 5) System utility (for the launcher script)
sudo apt-get install -y numactl

# 6) (Optional) W&B login
wandb login
```

### Sanity check that IPEX sees AVX-512
```bash
python - <<'PY'
import torch, intel_extension_for_pytorch as ipex
print("torch:", torch.__version__)
print("ipex :", ipex.__version__)
print("bf16 supported:", torch.cpu._is_avx512_bf16_supported() if hasattr(torch.cpu, "_is_avx512_bf16_supported") else "check ipex verbose")
PY
```

---

## Quick start

```bash
# Default: 4 heads, seq_len 2048, batch 1 × 16 grad accum, Alpaca dataset
./run_z8_training.sh --max_steps 2000
```

First run will tokenize the dataset into `data/tokens.bin` (cached; reused on subsequent runs). Checkpoints land in `checkpoints/medusa_heads_step*.pt`.

### Common flags
```bash
./run_z8_training.sh \
    --backbone microsoft/bitnet-b1.58-2B-4T \
    --dataset_name tatsu-lab/alpaca \
    --num_heads 4 \
    --num_layers_per_head 1 \
    --seq_len 2048 \
    --batch_size 1 \
    --grad_accum_steps 16 \
    --lr 1e-3 \
    --warmup_steps 50 \
    --max_steps 2000 \
    --log_every 10 \
    --ckpt_every 500 \
    --wandb_project medusa-bitnet \
    --wandb_run_name z8-run-01
```

All flags come directly from the `TrainConfig` dataclass in `train.py` — add/rename fields there and they're automatically exposed on the CLI.

---

## The Medusa loss

Head `i` predicts token at position `t + i + 1` given the hidden state at position `t`:

- **Head 1** is the standard next-token LM objective.
- **Head 2** predicts two tokens ahead.
- **Head `k`** predicts `k` tokens ahead.

For each head we shift the targets, truncate the tail that has no valid label, and average unweighted cross-entropy across heads. Top-1 accuracy per head is logged to W&B as `top1_acc_head_{i}`.

During training you should see `acc@1` rise fastest (it's the easiest objective), with `acc@2`, `acc@3`, `acc@4` following behind at progressively lower plateaus. That gap is exactly what Medusa exploits at inference time.

---

## Tuning knobs for the Z8 G4

The launcher auto-detects cores per socket via `lscpu` and sets:
```
OMP_NUM_THREADS = CORES_PER_SOCKET
KMP_AFFINITY    = granularity=fine,compact,1,0
KMP_BLOCKTIME   = 1
DNNL_PRIMITIVE_CACHE_CAPACITY = 1024
```

If you want to experiment:
- **Lower `OMP_NUM_THREADS`** (try `cores/2`) if you see contention — hyperthreading often hurts bf16 GEMM.
- **Bump `seq_len`** toward 4096 if RAM allows; the heads' cost is linear in `T`, and longer sequences amortize backbone cost better.
- **Increase `grad_accum_steps`** before increasing `batch_size` — RAM footprint of the backbone activations is the bottleneck.

---

## Troubleshooting

**`AutoModelForCausalLM` can't load the BitNet checkpoint.**
Some BitNet releases ship custom `modeling_*.py`. Add `trust_remote_code=True` to the `from_pretrained(...)` call in `model.py`.

**`torch.compile(backend="ipex")` raises `BackendCompilerFailed`.**
Your IPEX build doesn't register the inductor backend. The training loop catches this and runs eager — performance drops but training still works. Upgrading to a matched `torch` / `ipex` pair usually fixes it.

**Display GPU still being used.**
Check `nvidia-smi` — if `python` appears, `CUDA_VISIBLE_DEVICES` wasn't empty at import time. Make sure nothing in your shell rc file re-exports it.

**Cross-socket slowdowns.**
Run `numactl --hardware` and confirm both `node 0` and `node 1` exist. If you're accidentally using both, double-check that `numactl` is actually on the `exec` line of `run_z8_training.sh`.

**Dataset tokenization is slow.**
It's single-pass and one-shot — the `.bin` is reused on subsequent runs. If you want parallel tokenization, swap the loop in `dataset.build_token_bin` for `datasets.map(..., num_proc=N)`.

---

## Using the trained heads

Checkpoints contain **only the head weights**:
```python
import torch
from model import MedusaBitNet, MedusaConfig

model = MedusaBitNet(MedusaConfig(backbone_name_or_path="microsoft/bitnet-b1.58-2B-4T"))
ckpt = torch.load("checkpoints/medusa_heads_step2000.pt", map_location="cpu")
model.heads.load_state_dict(ckpt["heads"])
model.eval()
```

From here you can wire the heads into any Medusa-style tree-attention decoding loop. The heads expose `[B, T, k, V]` logits; pick the top candidates from each head, run them as a draft tree through the backbone, and verify.

---

## License

MIT — see `LICENSE` (add one if you plan to publish).

## Acknowledgements

- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — Ma et al., 2024
- [Medusa](https://arxiv.org/abs/2401.10774) — Cai et al., 2024
- [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch)
