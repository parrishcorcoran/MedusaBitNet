"""
Medusa-on-BitNet CPU training loop for the HP Z8 G4.

Hardware guardrails (important — the Z8 has a tiny display GPU we must NOT
touch):
    - `CUDA_VISIBLE_DEVICES=""` is set *before* torch is imported so CUDA
      never initializes.
    - IPEX wraps the model for AVX-512 fusion, and `torch.compile(..., backend="ipex")`
      JIT-fuses the training graph.
    - Outer CPU/memory pinning to a single NUMA node is handled by
      `run_z8_training.sh` via `numactl`.

Medusa loss:
    Head i predicts token t+i+1 (head 0 is the standard next-token LM objective;
    heads 1..k-1 predict further into the future). For each head we shift the
    targets accordingly, mask out the tail that doesn't have a valid label,
    and take cross-entropy. We log top-1 accuracy per head to W&B.
"""

# ---- CPU-only guardrail: MUST be first, before any torch import ----------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
# --------------------------------------------------------------------------

import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import intel_extension_for_pytorch as ipex  # noqa: F401  (AVX-512 fusion)

from dataset import PackedTokenDataset, PackingConfig, build_token_bin, collate_packed
from model import MedusaBitNet, MedusaConfig


@dataclass
class TrainConfig:
    # Data
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_split: str = "train"
    bin_path: str = "data/tokens.bin"
    seq_len: int = 2048
    batch_size: int = 1
    grad_accum_steps: int = 16
    num_workers: int = 2

    # Model
    backbone: str = "microsoft/bitnet-b1.58-2B-4T"
    num_heads: int = 4
    num_layers_per_head: int = 1

    # Optim
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_steps: int = 2000
    warmup_steps: int = 50
    grad_clip: float = 1.0

    # Logging / ckpt
    log_every: int = 10
    wandb_project: str = "medusa-bitnet"
    wandb_run_name: str | None = None
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 500


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    for field in TrainConfig.__dataclass_fields__.values():
        p.add_argument(f"--{field.name}", type=type(field.default) if field.default is not None else str,
                       default=field.default)
    ns = p.parse_args()
    return TrainConfig(**vars(ns))


def medusa_loss(
    medusa_logits: torch.Tensor,   # [B, T, k, V]
    targets: torch.Tensor,         # [B, T+1] — inputs||one-more
    num_heads: int,
) -> tuple[torch.Tensor, list[float]]:
    """
    Cross-entropy across k heads + per-head top-1 accuracy.

    Head i should predict token at position (t + i + 1) given hidden state
    at position t. We compute each head's loss over the valid subset of
    positions [0 .. T - i - 1) and average (unweighted) across heads.
    """
    B, T_plus_1 = targets.shape
    T = medusa_logits.shape[1]
    assert T_plus_1 == T + 1, f"targets must be seq_len+1 long, got {T_plus_1} vs T={T}"

    total_loss = torch.zeros((), dtype=medusa_logits.dtype, device=medusa_logits.device)
    accuracies: list[float] = []

    for i in range(num_heads):
        shift = i + 1
        # Valid positions: we need targets[:, t + shift] to exist, so t < T - shift + 1.
        valid_len = T - shift + 1
        if valid_len <= 0:
            accuracies.append(0.0)
            continue
        logits_i = medusa_logits[:, :valid_len, i, :]            # [B, valid_len, V]
        targets_i = targets[:, shift : shift + valid_len]        # [B, valid_len]

        loss_i = F.cross_entropy(
            logits_i.reshape(-1, logits_i.size(-1)).float(),
            targets_i.reshape(-1),
        )
        total_loss = total_loss + loss_i

        with torch.no_grad():
            preds = logits_i.argmax(dim=-1)
            acc = (preds == targets_i).float().mean().item()
            accuracies.append(acc)

    total_loss = total_loss / num_heads
    return total_loss, accuracies


def warmup_cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main():
    cfg = parse_args()
    torch.manual_seed(0)

    # ---- W&B (optional — degrades gracefully if not installed/logged in) --
    try:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=vars(cfg))
        use_wandb = True
    except Exception as e:
        print(f"[train] wandb disabled ({e})")
        use_wandb = False

    # ---- Data ---------------------------------------------------------------
    pack_cfg = PackingConfig(
        dataset_name=cfg.dataset_name,
        dataset_split=cfg.dataset_split,
        seq_len=cfg.seq_len,
        bin_path=cfg.bin_path,
        tokenizer_name_or_path=cfg.backbone,
    )
    build_token_bin(pack_cfg)
    dataset = PackedTokenDataset(cfg.bin_path, cfg.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_packed,
        pin_memory=False,  # CPU-only: pin_memory is meaningless and wastes RAM
        drop_last=True,
    )

    # ---- Model --------------------------------------------------------------
    model_cfg = MedusaConfig(
        backbone_name_or_path=cfg.backbone,
        num_heads=cfg.num_heads,
        num_layers_per_head=cfg.num_layers_per_head,
        dtype=torch.bfloat16,
    )
    model = MedusaBitNet(model_cfg)
    model.train()  # heads train; backbone stays in eval via no_grad context

    # ---- Optimizer (only heads are trainable) ------------------------------
    optim = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # ---- IPEX + torch.compile for AVX-512 fusion ---------------------------
    # IPEX folds ops (e.g. linear+silu) into AVX-512 kernels; torch.compile
    # with the ipex backend then JITs the full training graph.
    model, optim = ipex.optimize(
        model, optimizer=optim, dtype=torch.bfloat16, inplace=True,
    )
    try:
        model = torch.compile(model, backend="ipex")
        print("[train] torch.compile(backend='ipex') enabled")
    except Exception as e:
        print(f"[train] torch.compile unavailable, running eager: {e}")

    scheduler = CosineAnnealingLR(optim, T_max=max(1, cfg.max_steps - cfg.warmup_steps))

    # ---- Training loop ------------------------------------------------------
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    step = 0
    accum = 0
    t0 = time.time()
    loss_accum = 0.0
    optim.zero_grad(set_to_none=True)
    data_iter = iter(loader)

    while step < cfg.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # batch: [B, seq_len+1] int64
        inputs = batch[:, :-1].contiguous()   # [B, T]
        targets = batch                       # [B, T+1]

        with torch.cpu.amp.autocast(dtype=torch.bfloat16):
            medusa_logits = model(inputs)     # [B, T, k, V]
            loss, head_accs = medusa_loss(medusa_logits, targets, cfg.num_heads)
            loss = loss / cfg.grad_accum_steps

        loss.backward()
        loss_accum += loss.item() * cfg.grad_accum_steps
        accum += 1

        if accum < cfg.grad_accum_steps:
            continue

        # ---- optimizer step --------------------------------------------
        lr_now = warmup_cosine_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.lr)
        for g in optim.param_groups:
            g["lr"] = lr_now

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)
        optim.step()
        optim.zero_grad(set_to_none=True)
        if step >= cfg.warmup_steps:
            scheduler.step()

        step += 1
        accum = 0

        if step % cfg.log_every == 0:
            dt = time.time() - t0
            avg_loss = loss_accum / (cfg.log_every * cfg.grad_accum_steps)
            msg = (
                f"step {step:5d} | loss {avg_loss:.4f} | lr {lr_now:.2e} | "
                f"{dt/cfg.log_every:.2f}s/step | "
                + " ".join(f"acc@{i+1}={a:.3f}" for i, a in enumerate(head_accs))
            )
            print(msg, flush=True)
            if use_wandb:
                log = {"loss": avg_loss, "lr": lr_now, "step": step, "sec_per_step": dt / cfg.log_every}
                for i, a in enumerate(head_accs):
                    log[f"top1_acc_head_{i+1}"] = a
                wandb.log(log, step=step)
            loss_accum = 0.0
            t0 = time.time()

        if step % cfg.ckpt_every == 0:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"medusa_heads_step{step}.pt")
            # Only save the trainable head weights — the backbone is frozen.
            torch.save({"heads": model.heads.state_dict(), "step": step, "cfg": vars(cfg)}, ckpt_path)
            print(f"[train] saved {ckpt_path}")

    print("[train] done")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
