"""A thin, inspectable wrapper around ``past_key_values``.

HuggingFace's :class:`~transformers.cache_utils.DynamicCache` is the object we
must surgically edit in order to migrate a live generation between models.  The
wrapper here keeps the cache as plain tensors plus the metadata a transplant
needs (which tier produced it, which token ids it covers, where the hidden-state
trace was tapped), and converts back and forth without copying when possible.

Claims supported by this module
-------------------------------
* **no-stall**: the cache survives a tier switch as data, so the decode loop
  never has to rewind to a prompt boundary.
* **low migration cost**: :meth:`MoltCache.nbytes` gives the byte volume actually
  moved during a transplant, the denominator of the cost table in the README.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers.cache_utils import DynamicCache

from ._compat import cache_pairs, make_dynamic_cache
from .adapters import ModelGeometry

KVPair = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class CacheMeta:
    """Provenance of a cache: what produced it and what it covers."""

    tier_name: str
    geometry: ModelGeometry
    token_ids: List[int] = field(default_factory=list)
    #: layer indices (in the producing model) at which hidden states were tapped.
    #: One entry per outgoing migration route that wants a top-k recompute.
    trace_layers: List[int] = field(default_factory=list)
    #: number of prompt tokens (the rest are generated)
    n_prompt_tokens: int = 0
    #: absolute RoPE position of cache entry 0.  Non-zero only after a
    #: sliding-window :meth:`MoltCache.crop`; the transplant must un-rotate the
    #: source keys at their *original* angles, not at 0..T-1.
    pos_offset: int = 0

    def clone(self) -> "CacheMeta":
        return CacheMeta(
            tier_name=self.tier_name,
            geometry=self.geometry,
            token_ids=list(self.token_ids),
            trace_layers=list(self.trace_layers),
            n_prompt_tokens=self.n_prompt_tokens,
            pos_offset=self.pos_offset,
        )


class MoltCache:
    """Per-layer ``(K, V)`` tensors of shape ``[B, n_kv_heads, T, head_dim]``.

    Also carries an optional *hidden-state trace* — the hidden states that
    entered ``meta.trace_layer`` for every position — which is what lets
    KVTransplant recompute the destination model's top-k layers natively instead
    of projecting them.
    """

    __slots__ = ("layers", "meta", "hidden_traces")

    def __init__(
        self,
        layers: Sequence[KVPair],
        meta: CacheMeta,
        hidden_traces: Optional[Dict[int, torch.Tensor]] = None,
    ):
        self.layers: List[KVPair] = [(k, v) for k, v in layers]
        self.meta = meta
        #: ``{layer_idx: [B, T, hidden_size]}`` — hidden states entering that layer
        self.hidden_traces: Dict[int, torch.Tensor] = dict(hidden_traces or {})

    # -- construction ------------------------------------------------------
    @classmethod
    def from_hf(
        cls,
        cache: DynamicCache,
        meta: CacheMeta,
        hidden_traces: Optional[Dict[int, torch.Tensor]] = None,
        clone: bool = False,
    ) -> "MoltCache":
        pairs = cache_pairs(cache)
        if clone:
            pairs = [(k.clone(), v.clone()) for k, v in pairs]
        return cls(pairs, meta, hidden_traces)

    @classmethod
    def empty(cls, meta: CacheMeta) -> "MoltCache":
        return cls([], meta, None)

    def to_hf(self, config=None) -> DynamicCache:
        """Materialise a ``DynamicCache`` the destination model can decode with."""
        return make_dynamic_cache(self.layers)

    # -- introspection -----------------------------------------------------
    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def seq_len(self) -> int:
        return 0 if not self.layers else int(self.layers[0][0].shape[-2])

    @property
    def batch_size(self) -> int:
        return 1 if not self.layers else int(self.layers[0][0].shape[0])

    @property
    def device(self) -> torch.device:
        return self.layers[0][0].device if self.layers else torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return self.layers[0][0].dtype if self.layers else torch.float32

    def nbytes(self, include_trace: bool = True) -> int:
        """Total bytes held.  Reported as ``bytes_moved`` by a transplant."""
        total = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                    for k, v in self.layers)
        if include_trace:
            total += self.trace_bytes()
        return total

    def trace_bytes(self) -> int:
        """Bytes spent on the hidden-state traces — the *overhead* Molt pays for
        the cheap top-k recompute path.  Reported honestly in the memory table."""
        return sum(t.numel() * t.element_size() for t in self.hidden_traces.values())

    def describe(self) -> str:
        return (
            f"MoltCache(tier={self.meta.tier_name}, layers={self.n_layers}, T={self.seq_len}, "
            f"kv_heads={self.meta.geometry.n_kv_heads}, head_dim={self.meta.geometry.head_dim}, "
            f"{self.nbytes()/2**20:.2f} MiB)"
        )

    # -- editing -----------------------------------------------------------
    def clone(self) -> "MoltCache":
        return MoltCache(
            [(k.clone(), v.clone()) for k, v in self.layers],
            self.meta.clone(),
            {i: t.clone() for i, t in self.hidden_traces.items()},
        )

    def crop(self, max_len: int) -> "MoltCache":
        """Keep the **most recent** ``max_len`` positions (sliding-window carry).

        Used when ``TransplantConfig.max_carry_tokens`` bounds how much history
        follows a generation across a migration under acute pressure.

        Note this changes the *absolute* positions the surviving tokens occupy;
        the transplant re-applies RoPE at the positions the destination model
        will actually use, so a cropped cache stays self-consistent.
        """
        if max_len is None or self.seq_len <= max_len:
            return self
        drop = self.seq_len - max_len
        keep = slice(drop, self.seq_len)
        layers = [(k[..., keep, :].contiguous(), v[..., keep, :].contiguous())
                  for k, v in self.layers]
        meta = self.meta.clone()
        meta.token_ids = meta.token_ids[-max_len:]
        meta.n_prompt_tokens = max(0, meta.n_prompt_tokens - drop)
        meta.pos_offset = self.meta.pos_offset + drop
        traces = {i: t[:, keep, :].contiguous() for i, t in self.hidden_traces.items()}
        return MoltCache(layers, meta, traces)

    def truncate(self, new_len: int) -> "MoltCache":
        """Keep the **first** ``new_len`` positions, dropping the tail.

        Used at a migration point when the last cached position's logits belong
        to the outgoing model: the destination re-derives that one token itself,
        which costs a single decode step and gives it one natively-computed
        position to anchor on.
        """
        if new_len >= self.seq_len:
            return self
        keep = slice(0, new_len)
        layers = [(k[..., keep, :].contiguous(), v[..., keep, :].contiguous())
                  for k, v in self.layers]
        meta = self.meta.clone()
        meta.token_ids = meta.token_ids[:new_len]
        meta.n_prompt_tokens = min(meta.n_prompt_tokens, new_len)
        traces = {i: t[:, keep, :].contiguous() for i, t in self.hidden_traces.items()}
        return MoltCache(layers, meta, traces)

    def append_trace(self, layer_idx: int, hidden: torch.Tensor) -> None:
        """Extend the trace for ``layer_idx`` by this step's hidden states."""
        h = hidden.detach()
        prev = self.hidden_traces.get(layer_idx)
        self.hidden_traces[layer_idx] = h.clone() if prev is None else torch.cat([prev, h], dim=1)

    def trace_for(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.hidden_traces.get(layer_idx)

    def truncate_traces(self, seq_len: int) -> None:
        for i, t in list(self.hidden_traces.items()):
            if t.shape[1] > seq_len:
                self.hidden_traces[i] = t[:, -seq_len:, :]

    def to(self, device=None, dtype=None) -> "MoltCache":
        layers = [(k.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype))
                  for k, v in self.layers]
        traces = {i: t.to(device=device, dtype=dtype) for i, t in self.hidden_traces.items()}
        return MoltCache(layers, self.meta.clone(), traces)

    def free(self) -> None:
        """Drop tensor references so the allocator can reclaim them promptly."""
        self.layers = []
        self.hidden_traces = {}


def stack_kv(cache: MoltCache) -> torch.Tensor:
    """``[n_layers, B, T, n_kv_heads*head_dim*2]`` view used by the projector."""
    out = []
    for k, v in cache.layers:
        B, H, T, D = k.shape
        out.append(torch.cat([k.permute(0, 2, 1, 3).reshape(B, T, H * D),
                              v.permute(0, 2, 1, 3).reshape(B, T, H * D)], dim=-1))
    return torch.stack(out, dim=0)
