"""
One-time backbone hidden-state cache for Medusa training.

The BitNet backbone is frozen: every training step currently reruns a 2B
forward on identical inputs. This script runs the backbone once over a
range of sequences in `data/tokens.bin` and writes the final hidden states
as raw bf16 to an output file. Training can then skip the backbone entirely
and read features from disk.

Layout of the output file (raw, no header):
    [num_samples_in_range, seq_len, hidden_size] bfloat16, row-major.

The script is designed to be launched twice in parallel — once pinned to
each NUMA node — with disjoint [--start, --end) ranges.
"""

import os
# Don't blank CUDA_VISIBLE_DEVICES — allow --device cuda for iGPU.

import argparse
import time

import numpy as np
import torch
try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
except ImportError:
    ipex = None
    HAS_IPEX = False
from transformers import AutoModelForCausalLM

from dataset import PackedTokenDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="microsoft/bitnet-b1.58-2B-4T")
    p.add_argument("--bin_path", default="data/tokens.bin")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1, help="exclusive; -1 = all")
    p.add_argument("--out", required=True)
    p.add_argument("--lm_head_out", default="",
                   help="if set, also save backbone's lm_head weight to this path")
    p.add_argument("--device", default="cpu",
                   help="cpu or cuda (ROCm iGPU via HIP)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)

    dataset = PackedTokenDataset(args.bin_path, args.seq_len)
    n_total = len(dataset)
    start = args.start
    end = n_total if args.end < 0 else min(args.end, n_total)
    n = end - start
    assert n > 0, f"empty range [{start}, {end})"

    print(f"[cache] loading backbone {args.backbone}", flush=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.backbone,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    backbone.eval()
    for p_ in backbone.parameters():
        p_.requires_grad_(False)

    device = torch.device(args.device)
    if device.type != "cpu":
        backbone = backbone.to(device)
        print(f"[cache] backbone moved to {device}", flush=True)
    elif HAS_IPEX:
        try:
            backbone = ipex.optimize(backbone, dtype=torch.bfloat16, inplace=True)
            print("[cache] ipex.optimize applied", flush=True)
        except Exception as e:
            print(f"[cache] ipex.optimize unavailable: {e}", flush=True)

    hidden_size = backbone.config.hidden_size
    print(f"[cache] range=[{start},{end}) n={n} seq_len={args.seq_len} "
          f"hidden={hidden_size} bs={args.batch_size}", flush=True)

    if args.lm_head_out:
        lm_w = backbone.get_output_embeddings().weight.detach().to(torch.bfloat16).contiguous()
        os.makedirs(os.path.dirname(args.lm_head_out) or ".", exist_ok=True)
        torch.save(lm_w, args.lm_head_out)
        print(f"[cache] wrote lm_head weight {tuple(lm_w.shape)} -> {args.lm_head_out}",
              flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    bytes_per_seq = args.seq_len * hidden_size * 2  # bf16 = 2 bytes
    total_bytes = bytes_per_seq * n
    print(f"[cache] output size: {total_bytes / 1e9:.2f} GB -> {args.out}", flush=True)

    t0 = time.time()
    n_done = 0
    with open(args.out, "wb") as f:
        with torch.inference_mode():
            for batch_start in range(start, end, args.batch_size):
                batch_end = min(batch_start + args.batch_size, end)
                rows = [dataset[i][: args.seq_len] for i in range(batch_start, batch_end)]
                input_ids = torch.stack(rows, dim=0).to(device)  # [B, T] int64

                out = backbone(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden = out.hidden_states[-1].to(torch.bfloat16).contiguous()  # [B, T, H]
                # numpy has no bf16 dtype — reinterpret the raw bytes as uint16
                # (same 2-byte width) and write sequentially.
                raw = hidden.view(-1).cpu().view(torch.int16).numpy().view(np.uint16)
                raw.tofile(f)

                n_done += (batch_end - batch_start)
                if True:
                    dt = time.time() - t0
                    rate = n_done / max(dt, 1e-6)
                    eta = (n - n_done) / max(rate, 1e-6)
                    print(
                        f"[cache] {n_done}/{n} seq  "
                        f"{rate:.2f} seq/s  elapsed={dt:.0f}s  eta={eta:.0f}s",
                        flush=True,
                    )

    dt = time.time() - t0
    print(f"[cache] done: {n} seqs in {dt:.0f}s ({n/max(dt,1e-6):.2f} seq/s) -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
