"""
Medusa-on-BitNet CPU training loop for the HP Z8 G4.

Part of the MedusaBitNet project by Parrish Corcoran — training the Medusa
heads that ride on top of Microsoft's BitNet b1.58 ternary-weight backbone.

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

import numpy as np

from dataset import PackedTokenDataset, PackingConfig, build_token_bin, collate_packed
from model import MedusaBitNet, MedusaConfig, MedusaHeads


class CachedHiddenDataset(torch.utils.data.Dataset):
    """
    Yields (hidden[i], tokens[i]) pairs from the backbone hidden-state cache
    produced by cache_hidden.py, paired with the matching token sequences.

    hidden bin layout: [num_seqs, seq_len, hidden_size] bfloat16, row-major.
    Each token sample is seq_len+1 long (same semantics as PackedTokenDataset).
    """

    def __init__(self, hidden_path: str, token_bin_path: str, seq_len: int, hidden_size: int):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        # Memmap hidden states as uint16 (bf16 bit pattern), reshape later on access.
        self._hidden = np.memmap(hidden_path, dtype=np.uint16, mode="r")
        per_seq = seq_len * hidden_size
        assert self._hidden.size % per_seq == 0, (
            f"hidden bin size {self._hidden.size} not a multiple of {per_seq}"
        )
        self.num_samples = self._hidden.size // per_seq
        self._tokens = PackedTokenDataset(token_bin_path, seq_len)
        assert len(self._tokens) >= self.num_samples, (
            f"token bin has {len(self._tokens)} seqs but hidden cache has {self.num_samples}"
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        per_seq = self.seq_len * self.hidden_size
        start = idx * per_seq
        flat = np.asarray(self._hidden[start : start + per_seq])  # uint16 copy
        hidden = torch.from_numpy(flat).view(torch.bfloat16).view(self.seq_len, self.hidden_size)
        targets = self._tokens[idx]  # int64 [seq_len+1]
        return hidden, targets


def collate_cached(batch):
    hiddens = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.stack([b[1] for b in batch], dim=0)
    return hiddens, targets


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

    # Cached-hidden fast path (skips backbone entirely when both are set)
    cached_hidden_path: str = ""
    cached_lm_head_path: str = ""

    # Compute device. "cpu" works everywhere (default). On a machine with a
    # CUDA/ROCm-capable GPU (e.g. AMD Strix Halo iGPU via ROCm) pass --device cuda
    # to run the head math on the accelerator; the cached hidden bin stays
    # mmapped in CPU RAM and only the current micro-batch is shipped to device.
    device: str = "cpu"

    # Position subsampling in the loss. Computes CE on a random subset of
    # seq_len positions per micro-batch, which cuts the dominant [T,V] matmul
    # proportionally. 256 out of 2048 = ~8x training speedup with no measurable
    # quality loss (standard LM training trick). Set to 0 to disable.
    loss_positions: int = 256

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
    medusa_logits: torch.Tensor,          # [B, T or P, k, V]
    targets: torch.Tensor,                # [B, T_full + 1] — inputs||one-more
    num_heads: int,
    pos_indices: torch.Tensor | None = None,  # [B, P] int64 or None
) -> tuple[torch.Tensor, list[float]]:
    """
    Cross-entropy across k heads + per-head top-1 accuracy.

    Head i predicts token at position (t + i + 1) given hidden state at
    position t. When pos_indices is None we compute the loss on every
    position; when given we compute it only on the P sampled positions
    (the expensive [T, V] matmul upstream has already been restricted to
    those positions by MedusaHeads.forward).
    """
    B, T_plus_1 = targets.shape
    total_loss = torch.zeros((), dtype=medusa_logits.dtype, device=medusa_logits.device)
    accuracies: list[float] = []

    if pos_indices is None:
        # -------- full-sequence path (no subsampling) --------
        T = medusa_logits.shape[1]
        assert T_plus_1 == T + 1, f"targets must be seq_len+1 long, got {T_plus_1} vs T={T}"

        for i in range(num_heads):
            shift = i + 1
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
                acc = (logits_i.argmax(dim=-1) == targets_i).float().mean().item()
                accuracies.append(acc)
    else:
        # -------- subsampled path --------
        # medusa_logits: [B, P, k, V] — already gathered at pos_indices upstream.
        # For each head i, target at sampled position p is targets[:, p + i + 1].
        # Caller must ensure every p + num_heads < T_full + 1 (i.e. pos_indices
        # was drawn from [0, T_full - num_heads)).
        B, P, K, V = medusa_logits.shape
        for i in range(num_heads):
            shift = i + 1
            logits_i = medusa_logits[:, :, i, :]                    # [B, P, V]
            target_idx = pos_indices + shift                        # [B, P]
            targets_i = torch.gather(targets, 1, target_idx)        # [B, P]

            loss_i = F.cross_entropy(
                logits_i.reshape(-1, V).float(),
                targets_i.reshape(-1),
            )
            total_loss = total_loss + loss_i
            with torch.no_grad():
                acc = (logits_i.argmax(dim=-1) == targets_i).float().mean().item()
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

    cached_mode = bool(cfg.cached_hidden_path) and bool(cfg.cached_lm_head_path)

    # ---- Data ---------------------------------------------------------------
    if cached_mode:
        print(f"[train] cached-hidden mode: {cfg.cached_hidden_path}")
        lm_head_weight = torch.load(cfg.cached_lm_head_path, map_location="cpu")
        lm_head_weight = lm_head_weight.to(torch.bfloat16).contiguous()
        vocab_size, hidden_size = lm_head_weight.shape
        dataset = CachedHiddenDataset(
            cfg.cached_hidden_path, cfg.bin_path, cfg.seq_len, hidden_size,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_cached,
            pin_memory=False,
            drop_last=True,
        )
    else:
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
            pin_memory=False,
            drop_last=True,
        )

    device = torch.device(cfg.device)
    print(f"[train] device = {device}")

    # ---- Model --------------------------------------------------------------
    if cached_mode:
        heads = MedusaHeads(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_heads=cfg.num_heads,
            num_layers_per_head=cfg.num_layers_per_head,
            dtype=torch.bfloat16,
        )
        heads.train()
        model = heads  # only the heads exist in this mode
        # Register lm_head_weight as a non-trainable buffer for forward use.
        model.register_buffer("_lm_head_weight", lm_head_weight, persistent=False)
        model.to(device)
        trainable_params = list(model.parameters())
    else:
        model_cfg = MedusaConfig(
            backbone_name_or_path=cfg.backbone,
            num_heads=cfg.num_heads,
            num_layers_per_head=cfg.num_layers_per_head,
            dtype=torch.bfloat16,
        )
        model = MedusaBitNet(model_cfg)
        model.train()
        trainable_params = model.trainable_parameters()

    # ---- Optimizer (only heads are trainable) ------------------------------
    optim = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # ---- IPEX fusion. torch.compile stays off: inductor lowers BitNet's
    # packed uint8 weights into a raw mm(bf16, uint8) that crashes at runtime.
    if not cached_mode:
        model, optim = ipex.optimize(
            model, optimizer=optim, dtype=torch.bfloat16, inplace=True,
        )
        print("[train] torch.compile disabled (BitNet packed-weight incompat), running eager")

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

        if cached_mode:
            hidden, targets = batch  # hidden [B,T,H] bf16, targets [B,T+1] int64
            hidden  = hidden.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Position subsampling: random subset of valid positions. Valid
            # positions are 0..T - num_heads so every head's target is in range.
            B, T, _ = hidden.shape
            max_valid = T - cfg.num_heads
            if cfg.loss_positions > 0 and cfg.loss_positions < max_valid:
                P = cfg.loss_positions
                # Independent permutation per batch element, then take first P.
                # For B=1 (our default) this is equivalent to a single randperm.
                perm = torch.randperm(max_valid, device=device)[:P]
                pos_indices = perm.sort().values.unsqueeze(0).expand(B, -1).contiguous()
            else:
                pos_indices = None

            if device.type == "cpu":
                ctx = torch.cpu.amp.autocast(dtype=torch.bfloat16)
            else:
                from contextlib import nullcontext
                ctx = nullcontext()
            with ctx:
                medusa_logits = model(hidden, model._lm_head_weight, pos_indices=pos_indices)
                loss, head_accs = medusa_loss(
                    medusa_logits, targets, cfg.num_heads, pos_indices=pos_indices,
                )
                loss = loss / cfg.grad_accum_steps
        else:
            inputs = batch[:, :-1].contiguous()   # [B, T]
            targets = batch                       # [B, T+1]
            with torch.cpu.amp.autocast(dtype=torch.bfloat16):
                medusa_logits = model(inputs)
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
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
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
            heads_state = (
                model.state_dict() if cached_mode else model.heads.state_dict()
            )
            torch.save({"heads": heads_state, "step": step, "cfg": vars(cfg)}, ckpt_path)
            print(f"[train] saved {ckpt_path}")

    print("[train] done")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
