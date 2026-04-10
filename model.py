"""
MedusaBitNet model: frozen BitNet b1.58 backbone + k Medusa heads.

Design notes:
- Backbone is loaded via HuggingFace `transformers` and frozen. We expose
  `output_hidden_states=True` and use the final hidden state as the feature
  that each Medusa head consumes.
- Each Medusa head is a small residual block (Linear -> SiLU -> Linear) that
  produces logits over the vocab, tied to the backbone's LM head weight by
  default (configurable).
- The forward pass is fully vectorized across the k heads: we stack the k
  heads into a single batched weight and do one matmul, rather than looping
  over heads in Python. Same story for the final projection to logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


@dataclass
class MedusaConfig:
    backbone_name_or_path: str = "microsoft/bitnet-b1.58-2B-4T"
    num_heads: int = 4                 # k: how many future tokens to predict
    num_layers_per_head: int = 1       # residual blocks per head
    tie_lm_head: bool = True           # share vocab projection with backbone
    dtype: torch.dtype = torch.bfloat16


class MedusaResidualBlock(nn.Module):
    """Linear -> SiLU -> Linear with a residual connection."""

    def __init__(self, hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.w_in = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.w_out = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        # Zero-init the out projection so each head starts as identity.
        nn.init.zeros_(self.w_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.w_out(F.silu(self.w_in(x)))


class MedusaHeads(nn.Module):
    """
    k Medusa heads, vectorized.

    Instead of storing k independent `nn.Linear` modules and looping over them
    in the forward pass, we stack the weights into a single parameter of shape
    [k, hidden, hidden] and use `torch.einsum` so that all k residual blocks
    run as one fused matmul. The same applies to the final vocab projection.
    """

    def __init__(self, hidden_size: int, vocab_size: int, num_heads: int,
                 num_layers_per_head: int, dtype: torch.dtype):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers_per_head = num_layers_per_head
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Stacked residual block weights: [num_layers, k, hidden, hidden]
        self.w_in = nn.Parameter(
            torch.empty(num_layers_per_head, num_heads, hidden_size, hidden_size, dtype=dtype)
        )
        self.w_out = nn.Parameter(
            torch.zeros(num_layers_per_head, num_heads, hidden_size, hidden_size, dtype=dtype)
        )
        for layer in range(num_layers_per_head):
            for head in range(num_heads):
                nn.init.kaiming_uniform_(self.w_in[layer, head], a=5 ** 0.5)
        # w_out stays zero-initialized (identity start).

    def forward(self, hidden: torch.Tensor, lm_head_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, T, H] final hidden state from the frozen backbone.
            lm_head_weight: [V, H] vocab projection (shared with backbone).

        Returns:
            logits: [B, T, k, V] — one set of vocab logits per head position.
        """
        # Broadcast hidden across the k heads: [B, T, 1, H] -> [B, T, k, H]
        x = hidden.unsqueeze(2).expand(-1, -1, self.num_heads, -1)

        # Vectorized residual blocks. einsum contract: "bthi,lhio->bltho" would
        # mix layers incorrectly, so we iterate over *layers* (usually 1) but
        # keep the k-head dimension fused inside the einsum.
        for layer in range(self.num_layers_per_head):
            hidden_proj = torch.einsum("bthi,hio->btho", x, self.w_in[layer])
            hidden_proj = F.silu(hidden_proj)
            hidden_proj = torch.einsum("bthi,hio->btho", hidden_proj, self.w_out[layer])
            x = x + hidden_proj

        # Single fused projection to vocab for all k heads.
        # [B, T, k, H] @ [H, V] -> [B, T, k, V]
        logits = torch.einsum("bthi,iv->bthv", x, lm_head_weight.t())
        return logits


class MedusaBitNet(nn.Module):
    """Frozen BitNet backbone with k Medusa heads on top."""

    def __init__(self, config: MedusaConfig):
        super().__init__()
        self.config = config

        backbone = AutoModelForCausalLM.from_pretrained(
            config.backbone_name_or_path,
            torch_dtype=config.dtype,
            low_cpu_mem_usage=True,
        )
        # Freeze every backbone parameter. Medusa only trains the heads.
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        self.backbone = backbone

        hidden_size = backbone.config.hidden_size
        vocab_size = backbone.config.vocab_size

        self.heads = MedusaHeads(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_heads=config.num_heads,
            num_layers_per_head=config.num_layers_per_head,
            dtype=config.dtype,
        )

        if config.tie_lm_head:
            # Keep a reference — don't copy — so we always use the backbone's weight.
            self._lm_head_weight_ref = self.backbone.get_output_embeddings().weight
        else:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=config.dtype)
            self._lm_head_weight_ref = self.lm_head.weight

    @property
    def lm_head_weight(self) -> torch.Tensor:
        return self._lm_head_weight_ref

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns medusa logits of shape [B, T, k, V].

        The backbone runs under `torch.no_grad()` since it's frozen — this
        saves a lot of memory and compute on CPU.
        """
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[-1]  # [B, T, H]

        # Heads are the only trainable part.
        medusa_logits = self.heads(hidden, self.lm_head_weight)
        return medusa_logits

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
