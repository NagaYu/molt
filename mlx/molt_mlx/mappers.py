"""Closed-form ridge KV mappers — the transfer itself, which is **prior work**.

One class, two *selection policies*, so the comparison between them isolates
exactly one variable:

``r2_topk``  (Heo et al., arXiv:2608.03893 — the baseline, implemented faithfully)
    For every target layer, rank source layers by head-averaged single-source R²
    (averaged over RoPE-stripped keys and values) and keep the top ``k``.  The
    input to target head ``h`` is the concatenation of **all** KV heads of those
    ``k`` layers; separate K and V maps per (target layer, target head); bias
    recovered by centring; ``λ = 0.01``.

``depth_linear``  (molt v0.1's compact projector, kept as an ablation)
    One source layer per target layer chosen by linear relative depth; same
    ridge, same features.  ``k = 1`` by construction.

Because every target head in a layer shares the same selected source layers, the
per-head maps are solved together as one multi-output system per target layer —
algebraically identical to separate per-head solves (each output column is an
independent regression on the same ``X``).

Nothing in this module is claimed as a contribution.  What molt adds is *when*
and *under what memory* this transfer runs; see ``schedule.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx

from . import rope as R
from .kvcache import KV


@dataclass(frozen=True)
class Geometry:
    n_layers: int
    n_kv_heads: int
    head_dim: int

    @property
    def width(self) -> int:
        """Per-layer flattened K (or V) width: n_kv_heads * head_dim."""
        return self.n_kv_heads * self.head_dim

    @classmethod
    def from_model(cls, model) -> "Geometry":
        a = model.model.layers[0].self_attn
        return cls(len(model.model.layers), a.n_kv_heads,
                   model.model.layers[0].self_attn.k_proj.weight.shape[0] // a.n_kv_heads)


@dataclass
class LayerMap:
    """Maps for one target layer: K and V, each ``[k*width_s, width_t]`` + bias."""

    sources: List[int]
    Wk: mx.array
    bk: mx.array
    Wv: mx.array
    bv: mx.array


@dataclass
class KVMapper:
    """A fitted mapper for one ordered (source → target) model pair."""

    src: Geometry
    dst: Geometry
    src_rope: R.RopeSpec
    dst_rope: R.RopeSpec
    policy: str
    k: int
    layers: List[LayerMap]
    meta: Dict = field(default_factory=dict)

    # -- size -----------------------------------------------------------
    def nbytes(self) -> int:
        return sum(m.Wk.nbytes + m.bk.nbytes + m.Wv.nbytes + m.bv.nbytes
                   for m in self.layers)

    def n_params(self) -> int:
        return sum(m.Wk.size + m.bk.size + m.Wv.size + m.bv.size for m in self.layers)

    # -- apply ----------------------------------------------------------
    def apply(self, src_pairs: Sequence[KV], offset: int = 0,
              out_dtype: Optional[mx.Dtype] = None) -> List[KV]:
        """Map a whole source cache to the target's geometry.

        ``src_pairs[l] = (K, V)`` with ``K`` post-RoPE ``[B, n_kv_s, T, d_s]``.
        Keys are de-rotated at positions ``offset..offset+T-1`` with the source
        RoPE, mapped, and re-rotated at the same positions with the target RoPE.
        """
        B, _, T, _ = src_pairs[0][0].shape
        dt = out_dtype or src_pairs[0][0].dtype
        flat_k: Dict[int, mx.array] = {}
        flat_v: Dict[int, mx.array] = {}

        def fk(s):
            if s not in flat_k:
                k = R.remove(src_pairs[s][0].astype(mx.float32), self.src_rope, offset)
                flat_k[s] = k.transpose(0, 2, 1, 3).reshape(B, T, self.src.width)
            return flat_k[s]

        def fv(s):
            if s not in flat_v:
                v = src_pairs[s][1].astype(mx.float32)
                flat_v[s] = v.transpose(0, 2, 1, 3).reshape(B, T, self.src.width)
            return flat_v[s]

        out: List[KV] = []
        H, D = self.dst.n_kv_heads, self.dst.head_dim
        for lm in self.layers:
            xk = mx.concatenate([fk(s) for s in lm.sources], axis=-1)
            xv = mx.concatenate([fv(s) for s in lm.sources], axis=-1)
            yk = xk @ lm.Wk.astype(mx.float32) + lm.bk.astype(mx.float32)
            yv = xv @ lm.Wv.astype(mx.float32) + lm.bv.astype(mx.float32)
            k = yk.reshape(B, T, H, D).transpose(0, 2, 1, 3)
            v = yv.reshape(B, T, H, D).transpose(0, 2, 1, 3)
            k = R.apply(k, self.dst_rope, offset)
            out.append((k.astype(dt), v.astype(dt)))
        return out

    # -- persistence ----------------------------------------------------
    def save(self, path: str, dtype: mx.Dtype = mx.float16) -> None:
        arrays: Dict[str, mx.array] = {}
        for l, m in enumerate(self.layers):
            arrays[f"layer{l}.Wk"] = m.Wk.astype(dtype)
            arrays[f"layer{l}.bk"] = m.bk.astype(dtype)
            arrays[f"layer{l}.Wv"] = m.Wv.astype(dtype)
            arrays[f"layer{l}.bv"] = m.bv.astype(dtype)
        meta = dict(
            format="molt-kv-mapper/1", policy=self.policy, k=str(self.k),
            src=json.dumps(self.src.__dict__), dst=json.dumps(self.dst.__dict__),
            src_rope=json.dumps(self.src_rope.__dict__),
            dst_rope=json.dumps(self.dst_rope.__dict__),
            sources=json.dumps([m.sources for m in self.layers]),
            extra=json.dumps(self.meta))
        mx.save_safetensors(path, arrays, metadata=meta)

    @classmethod
    def load(cls, path: str) -> "KVMapper":
        arrays, meta = mx.load(path, return_metadata=True)
        src = Geometry(**json.loads(meta["src"]))
        dst = Geometry(**json.loads(meta["dst"]))
        sources = json.loads(meta["sources"])
        layers = [LayerMap(sources[l], arrays[f"layer{l}.Wk"], arrays[f"layer{l}.bk"],
                           arrays[f"layer{l}.Wv"], arrays[f"layer{l}.bv"])
                  for l in range(dst.n_layers)]
        return cls(src, dst, R.RopeSpec(**json.loads(meta["src_rope"])),
                   R.RopeSpec(**json.loads(meta["dst_rope"])), meta["policy"],
                   int(meta["k"]), layers, json.loads(meta.get("extra", "{}")))


def depth_linear_sources(n_src: int, n_dst: int) -> List[List[int]]:
    """molt v0.1's selection: one source layer per target layer, by relative depth."""
    if n_dst == 1:
        return [[n_src - 1]]
    return [[min(n_src - 1, int(round(i * (n_src - 1) / (n_dst - 1))))]
            for i in range(n_dst)]


def selftest() -> None:
    import os
    import tempfile

    from .tinymodels import make, BIG, SMALL

    big, small = make(BIG, 1), make(SMALL, 2)
    gs, gd = Geometry.from_model(big), Geometry.from_model(small)
    assert (gs.head_dim, gd.head_dim) == (32, 16), (gs, gd)
    rs, rd = R.RopeSpec.from_model(big), R.RopeSpec.from_model(small)
    mx.random.seed(0)
    layers = [LayerMap([s], mx.random.normal((gs.width, gd.width)) * 0.05,
                       mx.zeros((gd.width,)), mx.random.normal((gs.width, gd.width)) * 0.05,
                       mx.zeros((gd.width,)))
              for [s] in depth_linear_sources(gs.n_layers, gd.n_layers)]
    m = KVMapper(gs, gd, rs, rd, "depth_linear", 1, layers)
    src = [(mx.random.normal((1, 2, 9, 32)), mx.random.normal((1, 2, 9, 32)))
           for _ in range(gs.n_layers)]
    out = m.apply(src, offset=5)
    assert len(out) == gd.n_layers and out[0][0].shape == (1, 2, 9, 16)
    p = os.path.join(tempfile.mkdtemp(), "m.safetensors")
    m.save(p, dtype=mx.float32)
    m2 = KVMapper.load(p)
    out2 = m2.apply(src, offset=5)
    err = max(mx.abs(a[0] - b[0]).max().item() for a, b in zip(out, out2))
    assert err < 1e-5, err
    print(f"  apply() maps 6x[1,2,9,32] -> 4x[1,2,9,16]; save/load round-trips "
          f"(err {err:.1e}); {m.n_params()} params")
    print("mappers: ok")


if __name__ == "__main__":
    selftest()
