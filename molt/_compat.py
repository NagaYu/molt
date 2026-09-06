"""Compatibility shims for the two HuggingFace ``transformers`` generations.

Molt reaches inside ``transformers`` further than most code does — it builds
caches by hand and runs individual decoder layers — so it is exposed to the
private-ish surface that changed between v4 and v5:

===========================  ==========================  =========================
                             transformers 4.x            transformers 5.x
===========================  ==========================  =========================
build a cache from tensors   ``from_legacy_cache(tuple)`` ``DynamicCache(ddp_cache_data=…)``
causal-mask kwarg            ``input_embeds``            ``inputs_embeds``
``cache_position``           required by mask + layer    removed from both
===========================  ==========================  =========================

Rather than pinning to v4 (which would ship a library that breaks on the next
``pip install``) or hard-switching on ``__version__`` (which breaks again on the
next rename), each shim **inspects the callable it is about to use** and passes
only the arguments that exist.  The behaviour is verified identical on both
generations by the test-suite, which is version-agnostic for the same reason.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import transformers
from transformers.cache_utils import DynamicCache

TRANSFORMERS_MAJOR = int(transformers.__version__.split(".")[0])

try:  # transformers >= 4.53
    from transformers.masking_utils import create_causal_mask as _create_causal_mask
except Exception:  # pragma: no cover - very old transformers
    _create_causal_mask = None

_MASK_PARAMS = set(inspect.signature(_create_causal_mask).parameters) \
    if _create_causal_mask is not None else set()
_LAYER_PARAMS_CACHE: Dict[type, set] = {}


def has_causal_mask_builder() -> bool:
    return _create_causal_mask is not None


def make_dynamic_cache(layers: Sequence[Tuple[torch.Tensor, torch.Tensor]]
                       ) -> DynamicCache:
    """Build a ``DynamicCache`` holding the given per-layer ``(K, V)`` tensors."""
    data = tuple((k, v) for k, v in layers)
    if hasattr(DynamicCache, "from_legacy_cache"):        # 4.x
        return DynamicCache.from_legacy_cache(data)
    return DynamicCache(ddp_cache_data=data)              # 5.x


def empty_cache_for(config) -> DynamicCache:
    """A cache with the right number of *uninitialised* layers.

    Used as the destination of a partial layer-range recompute: layers outside
    the recomputed range must stay empty so ``update()`` writes at position 0.
    """
    return DynamicCache(config=config)


def cache_pairs(cache: DynamicCache) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Per-layer ``(K, V)`` of a cache, raising if any layer is uninitialised."""
    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i, layer in enumerate(cache.layers):
        k, v = layer.keys, layer.values
        if k is None or v is None:
            raise ValueError(f"cache layer {i} is uninitialised")
        pairs.append((k, v))
    return pairs


def build_causal_mask(config, hidden: torch.Tensor, position_ids: torch.Tensor,
                      cache_position: Optional[torch.Tensor] = None,
                      past_key_values=None, attention_mask=None):
    """Version-agnostic causal mask for a hand-rolled layer loop."""
    if _create_causal_mask is None:                        # pragma: no cover
        return None
    kwargs: Dict[str, Any] = dict(
        config=config, attention_mask=attention_mask,
        past_key_values=past_key_values, position_ids=position_ids)
    kwargs["inputs_embeds" if "inputs_embeds" in _MASK_PARAMS else "input_embeds"] = hidden
    if "cache_position" in _MASK_PARAMS and cache_position is not None:
        kwargs["cache_position"] = cache_position
    return _create_causal_mask(**kwargs)


def run_decoder_layer(layer, hidden: torch.Tensor, *, attention_mask,
                      position_ids, past_key_values, position_embeddings,
                      cache_position: Optional[torch.Tensor] = None):
    """Call one decoder layer, passing only the kwargs its signature accepts."""
    params = _LAYER_PARAMS_CACHE.get(type(layer))
    if params is None:
        params = set(inspect.signature(type(layer).forward).parameters)
        _LAYER_PARAMS_CACHE[type(layer)] = params
    kwargs: Dict[str, Any] = dict(
        attention_mask=attention_mask, position_ids=position_ids,
        past_key_values=past_key_values, use_cache=True,
        position_embeddings=position_embeddings)
    if "cache_position" in params and cache_position is not None:
        kwargs["cache_position"] = cache_position
    out = layer(hidden, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def from_pretrained_kwargs(dtype: torch.dtype) -> Dict[str, Any]:
    """The dtype kwarg ``AutoModelForCausalLM.from_pretrained`` currently wants.

    ``torch_dtype`` was renamed to ``dtype``; passing the wrong one is either a
    ``TypeError`` or — worse, on some versions — a silently ignored argument
    that loads the model in the checkpoint's own precision and quietly doubles
    the footprint this project is trying to measure.
    """
    from transformers import AutoModelForCausalLM

    try:
        params = set(inspect.signature(AutoModelForCausalLM.from_pretrained).parameters)
    except (TypeError, ValueError):  # pragma: no cover
        params = set()
    if "dtype" in params or TRANSFORMERS_MAJOR >= 5:
        return {"dtype": dtype}
    return {"torch_dtype": dtype}
