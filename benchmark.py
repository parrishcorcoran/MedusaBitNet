"""
Smoke-test / benchmark for MedusaBitNet on the Z8 G4.

Runs a handful of forward+backward passes with the real model (frozen BitNet
backbone + Medusa heads, IPEX-optimized, bf16) on **random token inputs** —
no dataset download, no tokenization. The goal is to answer:

    1. Does the pipeline actually run end-to-end without hanging?
    2. How many seconds per step should I expect?
    3. What does a full `--max_steps N` run translate to in wall-clock hours?

Usage (via the NUMA launcher):
    ./run_z8_training.sh --benchmark                      # defaults: 10 steps
    ./run_z8_training.sh --benchmark --bench_steps 20 --seq_len 1024

Or directly (skips NUMA pinning — not recommended on the Z8):
    python benchmark.py --bench_steps 10
"""

# ---- CPU-only guardrail: MUST be first, before any torch import ----------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# --------------------------------------------------------------------------

import argparse
import time

import torch
import torch.nn.functional as F

import intel_extension_for_pytorch as ipex  # noqa: F401

from model import MedusaBitNet, MedusaConfig
from train import medusa_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default="microsoft/bitnet-b1.58-2B-4T")
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers_per_head", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--bench_steps", type=int, default=10,
                   help="Number of timed forward+backward steps.")
    p.add_argument("--warmup_steps", type=int, default=2,
                   help="Untimed warmup passes (JIT compile, kernel dispatch).")
    p.add_argument("--projected_steps", type=int, default=2000,
                   help="For the 'full run' ETA at the end.")
    p.add_argument("--grad_accum_steps", type=int, default=16,
                   help="Used only for the ETA projection, not the benchmark itself.")
    p.add_argument("--no_compile", action="store_true",
                   help="Skip torch.compile — useful to isolate IPEX-only perf.")
    return p.parse_args()


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    if seconds < 86400:
        return f"{seconds/3600:.2f}h"
    return f"{seconds/86400:.2f}d"


def main():
    args = parse_args()
    torch.manual_seed(0)

    print("=" * 72)
    print("  MedusaBitNet smoke-test / benchmark")
    print("=" * 72)
    print(f"  backbone        : {args.backbone}")
    print(f"  num_heads (k)   : {args.num_heads}")
    print(f"  seq_len         : {args.seq_len}")
    print(f"  batch_size      : {args.batch_size}")
    print(f"  bench_steps     : {args.bench_steps}  (+ {args.warmup_steps} warmup)")
    print(f"  torch threads   : {torch.get_num_threads()}")
    print(f"  CUDA visible    : {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}")
    print("-" * 72)

    # ---- Load model ---------------------------------------------------------
    t_load = time.time()
    print("[bench] loading backbone + heads (this is the slow part the first time)...")
    cfg = MedusaConfig(
        backbone_name_or_path=args.backbone,
        num_heads=args.num_heads,
        num_layers_per_head=args.num_layers_per_head,
        dtype=torch.bfloat16,
    )
    model = MedusaBitNet(cfg)
    model.train()
    vocab_size = model.backbone.config.vocab_size
    hidden_size = model.backbone.config.hidden_size

    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[bench] hidden_size={hidden_size}  vocab_size={vocab_size}")
    print(f"[bench] trainable params: {n_trainable/1e6:.2f}M "
          f"/ total params: {n_total/1e6:.2f}M "
          f"({100*n_trainable/n_total:.3f}%)")
    print(f"[bench] model load took {fmt_duration(time.time()-t_load)}")

    # ---- Optimizer + IPEX + torch.compile ----------------------------------
    optim = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3, betas=(0.9, 0.95))
    t_opt = time.time()
    model, optim = ipex.optimize(
        model, optimizer=optim, dtype=torch.bfloat16, inplace=True,
    )
    print(f"[bench] ipex.optimize done in {fmt_duration(time.time()-t_opt)}")

    if not args.no_compile:
        try:
            t_c = time.time()
            model = torch.compile(model, backend="ipex")
            print(f"[bench] torch.compile(backend='ipex') registered "
                  f"(actual graph compile happens on first step) "
                  f"[{fmt_duration(time.time()-t_c)}]")
        except Exception as e:
            print(f"[bench] torch.compile unavailable, eager mode: {e}")
    else:
        print("[bench] torch.compile disabled via --no_compile")

    # ---- Synthetic input ---------------------------------------------------
    # Random token ids in [0, vocab_size). Shape matches what the dataset
    # would produce: inputs = [B, T], targets = [B, T+1].
    batch = torch.randint(
        low=0, high=vocab_size,
        size=(args.batch_size, args.seq_len + 1),
        dtype=torch.int64,
    )
    inputs = batch[:, :-1].contiguous()
    targets = batch

    # ---- Warmup (untimed) ---------------------------------------------------
    print("-" * 72)
    print(f"[bench] warmup ({args.warmup_steps} step(s); first step includes graph compile)...")
    for i in range(args.warmup_steps):
        t_w = time.time()
        with torch.cpu.amp.autocast(dtype=torch.bfloat16):
            logits = model(inputs)
            loss, _ = medusa_loss(logits, targets, args.num_heads)
        loss.backward()
        optim.zero_grad(set_to_none=True)
        print(f"[bench]   warmup step {i+1}/{args.warmup_steps}: "
              f"{fmt_duration(time.time()-t_w)}")

    # ---- Timed steps --------------------------------------------------------
    print("-" * 72)
    print(f"[bench] timing {args.bench_steps} forward+backward+step passes...")
    per_step = []
    fwd_times = []
    bwd_times = []
    last_loss = None
    last_accs = None

    for i in range(args.bench_steps):
        t0 = time.time()

        t_f = time.time()
        with torch.cpu.amp.autocast(dtype=torch.bfloat16):
            logits = model(inputs)
            loss, head_accs = medusa_loss(logits, targets, args.num_heads)
        fwd_times.append(time.time() - t_f)

        t_b = time.time()
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        bwd_times.append(time.time() - t_b)

        dt = time.time() - t0
        per_step.append(dt)
        last_loss = loss.item()
        last_accs = head_accs
        print(f"[bench]   step {i+1:2d}/{args.bench_steps}: "
              f"total={dt*1000:7.1f}ms  fwd={fwd_times[-1]*1000:6.1f}ms  "
              f"bwd+opt={bwd_times[-1]*1000:6.1f}ms  loss={last_loss:.3f}")

    # ---- Summary ------------------------------------------------------------
    def stats(xs):
        xs_sorted = sorted(xs)
        mean = sum(xs) / len(xs)
        median = xs_sorted[len(xs) // 2]
        return mean, median, min(xs), max(xs)

    mean, median, mn, mx = stats(per_step)
    fmean, fmed, _, _ = stats(fwd_times)
    bmean, bmed, _, _ = stats(bwd_times)

    print("=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print(f"  per-step (total): mean={mean*1000:.1f}ms  median={median*1000:.1f}ms  "
          f"min={mn*1000:.1f}ms  max={mx*1000:.1f}ms")
    print(f"  forward only   : mean={fmean*1000:.1f}ms  median={fmed*1000:.1f}ms")
    print(f"  backward+opt   : mean={bmean*1000:.1f}ms  median={bmed*1000:.1f}ms")
    print(f"  final loss     : {last_loss:.4f}")
    if last_accs:
        print(f"  head accs      : " + "  ".join(
            f"acc@{i+1}={a:.3f}" for i, a in enumerate(last_accs)
        ))
        print(f"                   (random-init heads + random tokens → "
              f"acc ≈ 1/V ≈ {1/vocab_size:.2e})")

    # ETA projection. One optimizer step in train.py consumes
    # `grad_accum_steps` micro-batches, so we multiply accordingly.
    micro_per_opt = args.grad_accum_steps
    est_opt_step = median * micro_per_opt
    total_seconds = est_opt_step * args.projected_steps

    print("-" * 72)
    print("  ETA projection for full training run")
    print("-" * 72)
    print(f"  assumed: grad_accum_steps={micro_per_opt}, max_steps={args.projected_steps}")
    print(f"  → one optimizer step  ≈ {fmt_duration(est_opt_step)}")
    print(f"  → {args.projected_steps} optimizer steps ≈ {fmt_duration(total_seconds)}")
    print()
    print("  Rough scaling hints:")
    print(f"    - halve seq_len (1024)  → ~{fmt_duration(total_seconds/2)} (roughly)")
    print(f"    - double seq_len (4096) → ~{fmt_duration(total_seconds*2)} (roughly)")
    print(f"    - halve grad_accum (8)  → ~{fmt_duration(total_seconds/2)}")
    print("=" * 72)

    if mean > 60:
        print("[bench] WARNING: >60s per micro-step. Check:")
        print("         - is numactl actually pinning the process? (`numactl --show`)")
        print("         - is another process using the same socket?")
        print("         - is OMP_NUM_THREADS sane? (`echo $OMP_NUM_THREADS`)")
        print("         - did torch.compile actually take the ipex backend?")
    else:
        print("[bench] looks healthy. Safe to launch the full training run.")


if __name__ == "__main__":
    main()
