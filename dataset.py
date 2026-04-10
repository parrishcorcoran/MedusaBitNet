"""
Memory-mapped token-packed dataset for Medusa training.

Pipeline:
    1) Stream an instruction dataset from HuggingFace `datasets`.
    2) Apply the tokenizer's chat template (falling back to a simple format
       if the tokenizer doesn't define one).
    3) Write the resulting token stream into a single uint32 .bin file on disk.
    4) At train time, mmap that .bin file and slice fixed-size sequences out
       of it. No Python-level per-sample work in the hot path.

We use uint32 because modern LLM vocabs can exceed 65k (uint16 max) but
comfortably fit in 32 bits, and uint32 is a native numpy dtype that mmaps well.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class PackingConfig:
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_split: str = "train"
    text_field: Optional[str] = None     # if None, use chat template on messages
    seq_len: int = 2048
    bin_path: str = "data/tokens.bin"
    tokenizer_name_or_path: str = "microsoft/bitnet-b1.58-2B-4T"


def _format_example(example: dict, tokenizer, text_field: Optional[str]) -> str:
    """Render one dataset row into a plain string ready to tokenize."""
    if text_field is not None and text_field in example:
        return example[text_field]

    # Alpaca-style fields — most common instruction datasets.
    if "instruction" in example:
        instr = example.get("instruction", "")
        inp = example.get("input", "")
        out = example.get("output", "")
        user = f"{instr}\n\n{inp}" if inp else instr
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": out},
        ]
    elif "messages" in example:
        messages = example["messages"]
    else:
        # Last-ditch: stringify the whole row.
        return str(example)

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def build_token_bin(cfg: PackingConfig) -> str:
    """One-shot tokenizer pass: write a single uint32 .bin of packed tokens."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    bin_path = Path(cfg.bin_path)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    if bin_path.exists():
        print(f"[dataset] reusing existing token bin at {bin_path}")
        return str(bin_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name_or_path, use_fast=True)
    eos_id = tokenizer.eos_token_id or 0

    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    print(f"[dataset] tokenizing {len(ds)} rows from {cfg.dataset_name}")

    # Write in chunks so we never hold the full tokenized corpus in RAM.
    chunk_buf: list[np.ndarray] = []
    total_tokens = 0
    with open(bin_path, "wb") as f:
        for i, example in enumerate(ds):
            text = _format_example(example, tokenizer, cfg.text_field)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            ids.append(eos_id)
            chunk_buf.append(np.asarray(ids, dtype=np.uint32))
            if len(chunk_buf) >= 1000:
                np.concatenate(chunk_buf).tofile(f)
                total_tokens += sum(len(a) for a in chunk_buf)
                chunk_buf.clear()
            if (i + 1) % 10000 == 0:
                print(f"[dataset]   {i+1} rows, {total_tokens} tokens written")
        if chunk_buf:
            np.concatenate(chunk_buf).tofile(f)
            total_tokens += sum(len(a) for a in chunk_buf)

    print(f"[dataset] done: {total_tokens} tokens at {bin_path}")
    return str(bin_path)


class PackedTokenDataset(Dataset):
    """
    Fixed-length slices of a packed token bin, read via numpy memmap.

    Each __getitem__ returns `seq_len + 1` tokens; the training loop uses the
    first `seq_len` as inputs and the last `seq_len` as the base (t+1) targets,
    from which the t+i shifts for the Medusa heads are derived.
    """

    def __init__(self, bin_path: str, seq_len: int):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(
                f"Token bin not found at {bin_path}. Run build_token_bin() first."
            )
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r")
        # We need seq_len+1 tokens (inputs + one more for the base target shift).
        self.num_samples = (len(self.data) - 1) // seq_len
        if self.num_samples <= 0:
            raise ValueError(
                f"Token bin too small ({len(self.data)} tokens) for seq_len={seq_len}"
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        # Copy out of the memmap into an owned int64 tensor (torch can't index
        # embeddings with uint32).
        chunk = np.asarray(self.data[start:end], dtype=np.int64)
        return torch.from_numpy(chunk)


def collate_packed(batch: list[torch.Tensor]) -> torch.Tensor:
    """Stack fixed-length samples into a single [B, seq_len+1] tensor."""
    return torch.stack(batch, dim=0)
