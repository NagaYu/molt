"""Removing and re-applying rotary position embeddings on cached keys.

mlx-lm's Qwen2 ``Attention`` applies RoPE to keys *before* writing them into the
``KVCache`` (verified in :func:`selftest` against the model's own code path), so a
cached key carries its absolute position in its rotation.  A position-independent
linear map between two models' caches therefore has to see keys with RoPE
removed, and the destination's RoPE re-applied afterwards.  This is the standard
treatment in prior cross-model KV transfer (Heo et al., arXiv:2608.03893, strip
RoPE from keys before the ridge map) and is **not** a contribution of this work.

Implementation note: the inverse rotation is computed with the *same* kernel as
the forward one, ``mx.fast.rope``, by negating ``scale`` (angle = scale·pos·θ),
so forward and inverse agree to float precision by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class RopeSpec:
    """Everything needed to (un)rotate one model's keys. A few bytes — this is
    all of the *source* model a transplant needs once its weights are gone."""

    dims: int
    base: float
    traditional: bool
    scale: float = 1.0

    @classmethod
    def from_model(cls, model) -> "RopeSpec":
        rope = model.model.layers[0].self_attn.rope
        dims = getattr(rope, "dims", None) or getattr(rope, "_dims")
        base = getattr(rope, "base", None)
        if base is None:
            base = getattr(rope, "_base")
        scale = getattr(rope, "scale", None)
        if scale is None:
            scale = getattr(rope, "_scale", 1.0)
        trad = getattr(rope, "traditional", None)
        if trad is None:
            trad = getattr(rope, "_traditional", False)
        return cls(int(dims), float(base), bool(trad), float(scale))


def apply(keys: mx.array, spec: RopeSpec, offset: int = 0) -> mx.array:
    """Rotate ``keys`` ``[B, H, T, D]`` as if they sat at positions offset..offset+T-1."""
    return mx.fast.rope(keys, spec.dims, traditional=spec.traditional,
                        base=spec.base, scale=spec.scale, offset=offset)


def remove(keys: mx.array, spec: RopeSpec, offset: int = 0) -> mx.array:
    """Exact inverse of :func:`apply` (rotation by −θ)."""
    return mx.fast.rope(keys, spec.dims, traditional=spec.traditional,
                        base=spec.base, scale=-spec.scale, offset=offset)


def selftest() -> None:
    from .tinymodels import make, BIG

    mx.random.seed(0)
    spec = RopeSpec(dims=32, base=10000.0, traditional=False)
    k = mx.random.normal((1, 2, 37, 32))
    for off in (0, 11):
        err = mx.abs(remove(apply(k, spec, off), spec, off) - k).max().item()
        assert err < 1e-4, (off, err)
    print(f"  remove(apply(k)) == k  (max err {err:.1e})")

    # The cached keys of a real mlx-lm Qwen2 are exactly rope(k_proj(x)).
    from mlx_lm.models.cache import make_prompt_cache
    m = make(BIG, 3)
    ids = mx.array([[5, 17, 99, 3, 42, 7, 8]])
    cache = make_prompt_cache(m)
    m(ids, cache=cache)
    attn = m.model.layers[0].self_attn
    h = m.model.layers[0].input_layernorm(m.model.embed_tokens(ids))
    raw = attn.k_proj(h).reshape(1, 7, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    cached = cache[0].state[0]
    s = RopeSpec.from_model(m)
    assert mx.abs(apply(raw, s) - cached).max().item() < 1e-4
    assert mx.abs(remove(cached, s) - raw).max().item() < 1e-4
    print(f"  mlx-lm caches post-RoPE keys; remove() recovers k_proj output "
          f"(spec: dims={s.dims} base={s.base:g} traditional={s.traditional})")
    print("rope: ok")


if __name__ == "__main__":
    selftest()
