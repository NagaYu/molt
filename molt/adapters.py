"""Architecture adapter: uniform access to a causal-LM's internals.

Molt has to reach *inside* a HuggingFace model to do three things that the
public API does not expose:

1. read the per-layer KV geometry (``n_kv_heads``, ``head_dim``, depth),
2. capture the hidden state entering a chosen layer while generating,
3. re-run **only a contiguous range of decoder layers** over a batch of
   positions, writing native KV into a fresh cache.

(3) is the mechanism behind KVTransplant's *selective top-k recompute*.  It is
verified bit-exact against a full forward pass in
``tests/test_kv_transplant.py::test_partial_layer_recompute_is_exact``.

Claims supported by this module
-------------------------------
* **low migration cost**: running ``k`` layers instead of ``L`` layers over the
  prefix is the entire reason a transplant is cheaper than a re-prefill.  This
  adapter is what makes the partial run possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from ._compat import (build_causal_mask, empty_cache_for,
                      has_causal_mask_builder, run_decoder_layer)


@dataclass(frozen=True)
class ModelGeometry:
    """Static KV geometry of one model, i.e. the shape of its cache."""

    n_layers: int
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int

    @property
    def kv_width(self) -> int:
        """Flattened per-token KV width of one layer (``n_kv_heads*head_dim``)."""
        return self.n_kv_heads * self.head_dim

    def bytes_per_token(self, dtype: torch.dtype) -> int:
        """K and V, all layers, one token."""
        return 2 * self.n_layers * self.kv_width * torch.finfo(dtype).bits // 8


class LMAdapter:
    """Uniform façade over a decoder-only causal LM.

    Only ``LlamaModel``-shaped stacks (Llama, Qwen2, Qwen3, Mistral, SmolLM …)
    support the partial-layer path; anything else degrades gracefully to
    ``supports_partial_recompute == False`` and Molt falls back to
    ``recompute_top_k = 0``.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.config = model.config
        self.backbone = self._find_backbone(model)
        self.geometry = self._read_geometry(model.config)

    # -- discovery ---------------------------------------------------------
    @staticmethod
    def _find_backbone(model: nn.Module) -> Optional[nn.Module]:
        for attr in ("model", "transformer", "gpt_neox"):
            sub = getattr(model, attr, None)
            if sub is not None and (hasattr(sub, "layers") or hasattr(sub, "h")):
                return sub
        return None

    @staticmethod
    def _read_geometry(cfg) -> ModelGeometry:
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd")
        n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head")
        n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer")
        n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads
        head_dim = getattr(cfg, "head_dim", None) or (hidden // n_heads)
        return ModelGeometry(
            n_layers=int(n_layers),
            hidden_size=int(hidden),
            n_heads=int(n_heads),
            n_kv_heads=int(n_kv),
            head_dim=int(head_dim),
            vocab_size=int(getattr(cfg, "vocab_size")),
        )

    @property
    def layers(self) -> Optional[Sequence[nn.Module]]:
        if self.backbone is None:
            return None
        return getattr(self.backbone, "layers", None) or getattr(self.backbone, "h", None)

    @property
    def supports_partial_recompute(self) -> bool:
        """True when we can re-run an arbitrary layer range by hand."""
        if self.backbone is None or not has_causal_mask_builder():
            return False
        if not hasattr(self.backbone, "rotary_emb") or not hasattr(self.backbone, "layers"):
            return False
        return getattr(self.backbone, "has_sliding_layers", False) is False

    # -- hidden-state capture ---------------------------------------------
    def capture_hook(self, layer_idx: int, sink: Callable[[torch.Tensor], None]):
        """Register a pre-hook that hands the input hidden state of ``layer_idx``
        to ``sink``.

        Used by :class:`molt.runtime.TierSession` to keep a rolling trace of the
        hidden states at the recompute boundary.  Without this trace a transplant
        can only project KV (``recompute_top_k == 0``); with it we can rebuild the
        destination model's *native* KV for its top layers at a cost of ``k/L`` of
        a full prefill.
        """
        layers = self.layers
        if layers is None or not (0 <= layer_idx < len(layers)):
            raise IndexError(f"layer {layer_idx} out of range for depth {self.geometry.n_layers}")

        def _hook(_mod, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            if h is not None:
                sink(h.detach())
            return None

        return layers[layer_idx].register_forward_pre_hook(_hook, with_kwargs=True)

    # -- partial layer range ----------------------------------------------
    @torch.no_grad()
    def run_layer_range(
        self,
        hidden: torch.Tensor,
        start: int,
        end: int,
        cache: Optional[DynamicCache] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, DynamicCache]:
        """Run decoder layers ``[start, end)`` over ``hidden`` and fill ``cache``.

        Parameters
        ----------
        hidden:
            ``[B, T, hidden_size]`` — the hidden state *entering* layer ``start``.
        cache:
            A cache with layers ``[start, end)`` **uninitialised**.  A fresh
            ``DynamicCache`` is created when omitted.

        Returns the hidden state leaving layer ``end-1`` (not normalised) and the
        cache now holding native destination-model KV for that range.

        This is the load-bearing routine for the *low migration cost* claim: cost
        scales with ``(end-start)/n_layers`` of a full prefill.
        """
        if not self.supports_partial_recompute:
            raise RuntimeError(
                f"{type(self.model).__name__} does not support partial layer recompute"
            )
        layers = self.layers
        B, T, _ = hidden.shape
        device = hidden.device
        cache = cache if cache is not None else empty_cache_for(self.config)

        cache_position = torch.arange(position_offset, position_offset + T, device=device)
        position_ids = cache_position.unsqueeze(0).expand(B, -1)
        mask = build_causal_mask(self.config, hidden, position_ids,
                                 cache_position=cache_position)
        pos_emb = self.backbone.rotary_emb(hidden, position_ids)

        h = hidden
        for i in range(start, end):
            h = run_decoder_layer(layers[i], h, attention_mask=mask,
                                  position_ids=position_ids, past_key_values=cache,
                                  position_embeddings=pos_emb,
                                  cache_position=cache_position)
        return h, cache

    # -- misc --------------------------------------------------------------
    def weight_bytes(self) -> int:
        total = 0
        for p in self.model.parameters():
            total += p.numel() * p.element_size()
        for b in self.model.buffers():
            total += b.numel() * b.element_size()
        return total

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


def flops_per_token_prefill(geom: ModelGeometry, intermediate_size: int) -> float:
    """Rough forward FLOPs for one token through one full model.

    ``2 * params_in_matmuls`` — good enough to compare a ``k``-layer partial
    recompute against a full ``L``-layer prefill, which is the number the
    *low migration cost* claim actually rests on.
    """
    h = geom.hidden_size
    per_layer = (
        h * geom.n_heads * geom.head_dim          # q_proj
        + 2 * h * geom.n_kv_heads * geom.head_dim  # k_proj, v_proj
        + geom.n_heads * geom.head_dim * h         # o_proj
        + 3 * h * intermediate_size                # gate, up, down
    )
    return 2.0 * per_layer * geom.n_layers


def flops_layer_range(geom: ModelGeometry, intermediate_size: int, n_layers: int) -> float:
    """FLOPs for a partial (``n_layers``-deep) forward of one token."""
    if geom.n_layers == 0:
        return 0.0
    return flops_per_token_prefill(geom, intermediate_size) * (n_layers / geom.n_layers)
