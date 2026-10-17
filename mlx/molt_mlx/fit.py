"""Fitting KV mappers in closed form from sufficient statistics.

This is the calibration recipe of Heo et al. (arXiv:2608.03893), reproduced as
faithfully as their text allows, so that condition (C) of the benchmark is the
prior method and not a weakened copy of it:

* features: per source layer, all KV heads flattened; keys with the source RoPE
  removed, values as-is.  Targets: the destination's own keys (RoPE removed) and
  values, per destination layer and head.
* centred ridge, ``(XᵀX + λI)⁻¹ XᵀY`` with ``λ = 0.01``, bias ``b = ȳ − x̄W``.
* covariance accumulated in float32, solved in float64.
* positions sub-sampled with stride 4 from 1 024-token sequences.
* top-k source layers per destination layer by head-averaged single-source R²
  (mean of the K and V R²), a *fixed* ranking, ``k`` swept.

Interpretation we had to make (stated in the paper's Methods): the text writes
``XᵀX + λI`` without saying whether ``XᵀX`` is normalised.  On ~128 K rows an
unnormalised Gram makes ``λ = 0.01`` a no-op, and their reported collapse at
``λ = 1`` could not happen, so we read ``XᵀX`` as the centred covariance (÷N).
``fit.selftest`` and ``bench/fit_mappers.py --lam`` let either reading be run.

Everything a mapper or an R² needs is a function of the first and second moments,
so one pass over the data supports every ``k`` and every ``λ``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

from . import rope as R
from .mappers import Geometry, KVMapper, LayerMap, depth_linear_sources

HEO_LAMBDA = 0.01
HEO_STRIDE = 4


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def features(model, ids: mx.array, stride: int = HEO_STRIDE,
             start: int = 0) -> Tuple[mx.array, mx.array]:
    """Run ``model`` on ``ids`` ``[1, T]``; return de-roped K and V features.

    Shapes ``[n, L*W]`` with ``n = len(range(start, T, stride))`` and the layer
    blocks laid out in layer order, heads in head order — the same layout
    :meth:`KVMapper.apply` concatenates.
    """
    spec = R.RopeSpec.from_model(model)
    cache = make_prompt_cache(model)
    model(ids, cache=cache)
    ks, vs = [], []
    for c in cache:
        k, v = c.state
        k = R.remove(k.astype(mx.float32), spec, 0)
        B, H, T, D = k.shape
        ks.append(k.transpose(0, 2, 1, 3).reshape(T, H * D)[start::stride])
        vs.append(v.astype(mx.float32).transpose(0, 2, 1, 3).reshape(T, H * D)[start::stride])
    xk, xv = mx.concatenate(ks, axis=-1), mx.concatenate(vs, axis=-1)
    mx.eval(xk, xv)
    return xk, xv


# --------------------------------------------------------------------------
# sufficient statistics
# --------------------------------------------------------------------------
class _Moments:
    """Shifted first/second moments of X, cross with Y, and diag of Y².

    Shifting by the first batch's mean before accumulating keeps float32 sums
    well conditioned when features have large means (Qwen's k_proj has a bias).
    """

    def __init__(self):
        self.n = 0
        self.mx_ = self.my_ = None

    def update(self, X: mx.array, Y: mx.array) -> None:
        if self.n == 0:
            self.mx_, self.my_ = X.mean(0), Y.mean(0)
            Fx, Fy = X.shape[1], Y.shape[1]
            self.sx = mx.zeros((Fx,)); self.sy = mx.zeros((Fy,))
            self.xx = mx.zeros((Fx, Fx)); self.xy = mx.zeros((Fx, Fy))
            self.yy = mx.zeros((Fy,))
        Xc, Yc = X - self.mx_, Y - self.my_
        self.sx = self.sx + Xc.sum(0)
        self.sy = self.sy + Yc.sum(0)
        self.xx = self.xx + Xc.T @ Xc
        self.xy = self.xy + Xc.T @ Yc
        self.yy = self.yy + (Yc * Yc).sum(0)
        self.n += X.shape[0]
        mx.eval(self.sx, self.sy, self.xx, self.xy, self.yy)

    def finalize(self) -> Dict[str, np.ndarray]:
        """Centred covariances.  The two big matrices stay float32 (as in Heo et
        al.: "float32 covariance"); the selected block is promoted to float64 only
        inside the solve."""
        n = float(self.n)
        dx = np.asarray(self.sx, dtype=np.float64) / n
        dy = np.asarray(self.sy, dtype=np.float64) / n
        cxx = np.array(self.xx, dtype=np.float32)
        cxx /= n
        cxx -= np.outer(dx, dx).astype(np.float32)
        cxx += cxx.T.copy(); cxx *= 0.5
        cxy = np.array(self.xy, dtype=np.float32)
        cxy /= n
        cxy -= np.outer(dx, dy).astype(np.float32)
        vy = np.asarray(self.yy, dtype=np.float64) / n - dy * dy
        return dict(n=self.n, cxx=cxx, cxy=cxy, vy=vy,
                    mean_x=np.asarray(self.mx_, dtype=np.float64) + dx,
                    mean_y=np.asarray(self.my_, dtype=np.float64) + dy)


@dataclass
class PairStats:
    """Moments for one ordered (source → destination) pair, K and V separately."""

    src: Geometry
    dst: Geometry
    src_rope: R.RopeSpec
    dst_rope: R.RopeSpec
    k: _Moments = field(default_factory=_Moments)
    v: _Moments = field(default_factory=_Moments)
    _fin: Optional[Tuple[str, Dict[str, np.ndarray]]] = None

    def update(self, src_feats: Tuple[mx.array, mx.array],
               dst_feats: Tuple[mx.array, mx.array]) -> None:
        self.k.update(src_feats[0], dst_feats[0])
        self.v.update(src_feats[1], dst_feats[1])
        self._fin = None

    @property
    def n(self) -> int:
        return self.k.n

    def final(self, kv: str) -> Dict[str, np.ndarray]:
        """Finalised moments for ``kv`` ('k' or 'v').  Only one is held at a time:
        K and V are solved in separate passes to halve peak memory."""
        if self._fin is None or self._fin[0] != kv:
            self._fin = None
            self._fin = (kv, getattr(self, kv).finalize())
        return self._fin[1]

    # raw float32 moments: written right after collection so that the models can
    # be unloaded before the float64 finalisation (keeps the fit inside 16 GB)
    _RAW = ("mx_", "my_", "sx", "sy", "xx", "xy", "yy")

    def save_raw(self, path: str) -> None:
        arrays = {f"{kv}.{a}": getattr(getattr(self, kv), a) for kv in ("k", "v") for a in self._RAW}
        meta = dict(n=str(self.n), geom=repr((self.src.__dict__, self.dst.__dict__,
                                              self.src_rope.__dict__, self.dst_rope.__dict__)))
        mx.save_safetensors(path, arrays, metadata=meta)

    @classmethod
    def load_raw(cls, path: str) -> "PairStats":
        import ast
        arrays, meta = mx.load(path, return_metadata=True)
        s, d, sr, dr = ast.literal_eval(meta["geom"])
        ps = cls(Geometry(**s), Geometry(**d), R.RopeSpec(**sr), R.RopeSpec(**dr))
        for kv in ("k", "v"):
            m = getattr(ps, kv)
            for a in cls._RAW:
                setattr(m, a, arrays[f"{kv}.{a}"])
            m.n = int(meta["n"])
        return ps

def collect(models: Dict[str, object], pairs: Sequence[Tuple[str, str]],
            sequences: Iterable[Sequence[int]], stride: int = HEO_STRIDE,
            batch_rows: int = 1024, progress=None,
            stats: Optional[Dict[Tuple[str, str], "PairStats"]] = None, start: int = 0,
            checkpoint=None) -> Dict[Tuple[str, str], PairStats]:
    """One pass over ``sequences``; every model runs each sequence once and every
    requested pair's moments are updated from the shared features.

    ``stats``/``start`` resume an interrupted pass; ``checkpoint(i, stats)`` is
    called after each flush with the number of sequences consumed so far."""
    stats = stats or {p: PairStats(Geometry.from_model(models[p[0]]), Geometry.from_model(models[p[1]]),
                                   R.RopeSpec.from_model(models[p[0]]), R.RopeSpec.from_model(models[p[1]]))
                      for p in pairs}
    used = sorted({m for p in pairs for m in p})
    buf: Dict[str, List[Tuple[mx.array, mx.array]]] = {m: [] for m in used}
    rows = 0

    def flush():
        nonlocal rows
        if rows == 0:
            return
        cat = {m: (mx.concatenate([a for a, _ in buf[m]]), mx.concatenate([b for _, b in buf[m]]))
               for m in used}
        for (s, d), st in stats.items():
            st.update(cat[s], cat[d])
        for m in used:
            buf[m].clear()
        rows = 0

    for i, seq in enumerate(sequences):
        if i < start:
            continue
        ids = mx.array([list(seq)])
        for m in used:
            buf[m].append(features(models[m], ids, stride))
        rows += buf[used[0]][-1][0].shape[0]
        if rows >= batch_rows:
            flush()
            if checkpoint:
                checkpoint(i + 1, stats)
        if progress:
            progress(i)
    flush()
    return stats


# --------------------------------------------------------------------------
# solving
# --------------------------------------------------------------------------
def _cols(layers: Sequence[int], width: int):
    """Column selector for the concatenation of ``layers``: a slice when the
    layers are contiguous (no copy), else an index array."""
    layers = list(layers)
    if layers == list(range(layers[0], layers[-1] + 1)):
        return slice(layers[0] * width, (layers[-1] + 1) * width)
    return np.concatenate([np.arange(l * width, (l + 1) * width) for l in layers])


def _sub(M: np.ndarray, r, c) -> np.ndarray:
    """``M[r, c]`` for slice/index selectors (a view when both are slices)."""
    if isinstance(r, slice) or isinstance(c, slice):
        return M[r, c]
    return M[np.ix_(r, c)]


class _Factor:
    """float64 Cholesky of ``A + λI`` for one selected column set."""

    def __init__(self, f: Dict[str, np.ndarray], xi, lam: float):
        import scipy.linalg as sl
        self.A = _sub(f["cxx"], xi, xi)                      # float32 view or copy
        reg = self.A.astype(np.float64)
        reg[np.diag_indices_from(reg)] += lam
        try:
            self.c = sl.cho_factor(reg, lower=True, overwrite_a=True, check_finite=False)
            self.kind = "chol"
        except np.linalg.LinAlgError:
            self.c, self.kind = reg, "lstsq"

    def solve(self, B: np.ndarray) -> np.ndarray:
        import scipy.linalg as sl
        B = B.astype(np.float64)
        if self.kind == "chol":
            return sl.cho_solve(self.c, B, check_finite=False)
        return np.linalg.lstsq(self.c, B, rcond=None)[0]


def _ridge(f: Dict[str, np.ndarray], xi, yi, lam: float, fac: Optional[_Factor] = None):
    """Centred ridge; returns W, b, per-dim residual variance, per-dim variance."""
    fac = fac or _Factor(f, xi, lam)
    B = _sub(f["cxy"], xi, yi)
    W = fac.solve(B)
    b = f["mean_y"][yi] - f["mean_x"][xi] @ W
    vy = f["vy"][yi]
    AW = (fac.A @ W.astype(np.float32)).astype(np.float64)
    res = vy - 2.0 * np.einsum("ij,ij->j", W, B) + np.einsum("ij,ij->j", W, AW)
    return W, b, res, vy


def _head_r2(res: np.ndarray, vy: np.ndarray, n_heads: int) -> np.ndarray:
    """Head-wise R² = 1 − Σres/Σvar over each head's dims; shape [..., n_heads]."""
    r = res.reshape(*res.shape[:-1], n_heads, -1).sum(-1)
    v = vy.reshape(*vy.shape[:-1], n_heads, -1).sum(-1)
    return 1.0 - r / np.maximum(v, 1e-30)


def single_source_r2(stats: PairStats, lam: float = HEO_LAMBDA) -> np.ndarray:
    """``[L_dst, L_src]`` head-averaged single-source R², mean of K and V."""
    Ws, Wt, Hd = stats.src.width, stats.dst.width, stats.dst.n_kv_heads
    Lt = stats.dst.n_layers
    out = np.zeros((Lt, stats.src.n_layers))
    yi_all = slice(0, Lt * Wt)
    for kv in ("k", "v"):
        f = stats.final(kv)
        for s in range(stats.src.n_layers):
            _, _, res, vy = _ridge(f, _cols([s], Ws), yi_all, lam)
            out[:, s] += _head_r2(res.reshape(Lt, Wt), vy.reshape(Lt, Wt), Hd).mean(-1) / 2.0
    return out


def heo_sources(r2: np.ndarray, k: int) -> List[List[int]]:
    """Top-``k`` source layers per destination layer from a fixed R² ranking,
    returned in ascending layer order (the concatenation order)."""
    k = min(k, r2.shape[1])
    return [sorted(np.argsort(-row, kind="stable")[:k].tolist()) for row in r2]


@dataclass
class FitReport:
    """In-sample, head-averaged R² of the fitted mapper (per destination layer)."""

    r2_k: np.ndarray
    r2_v: np.ndarray

    @property
    def mean(self) -> float:
        return float((self.r2_k.mean() + self.r2_v.mean()) / 2)


def solve_many(stats: PairStats, configs: Dict[str, List[List[int]]], lam: float = HEO_LAMBDA,
               policies: Optional[Dict[str, str]] = None, meta: Optional[dict] = None
               ) -> Dict[str, Tuple[KVMapper, FitReport]]:
    """Solve several source-selection configs in one K pass and one V pass.
    Destination layers that share a source set share one factorisation."""
    Ws, Wt, Hd = stats.src.width, stats.dst.width, stats.dst.n_kv_heads
    parts: Dict[str, Dict[str, list]] = {n: {"k": [], "v": []} for n in configs}
    for kv in ("k", "v"):
        f = stats.final(kv)
        factors: Dict[Tuple[int, ...], _Factor] = {}
        for name, sources in configs.items():
            for t, S in enumerate(sources):
                key = tuple(S)
                if key not in factors:
                    if len(factors) >= 4:
                        factors.clear()
                    factors[key] = _Factor(f, _cols(S, Ws), lam)
                yi = slice(t * Wt, (t + 1) * Wt)
                W, b, res, vy = _ridge(f, _cols(S, Ws), yi, lam, factors[key])
                parts[name][kv].append((W.astype(np.float32), b.astype(np.float32),
                                        float(_head_r2(res, vy, Hd).mean())))
        factors.clear()
    out = {}
    for name, sources in configs.items():
        layers = [LayerMap(list(S), mx.array(kp[0]), mx.array(kp[1]), mx.array(vp[0]), mx.array(vp[1]))
                  for S, kp, vp in zip(sources, parts[name]["k"], parts[name]["v"])]
        pol = (policies or {}).get(name, "r2_topk")
        m = KVMapper(stats.src, stats.dst, stats.src_rope, stats.dst_rope, pol,
                     max(len(S) for S in sources), layers, dict(meta or {}, lam=lam, n_rows=stats.n))
        out[name] = (m, FitReport(np.array([p[2] for p in parts[name]["k"]]),
                                  np.array([p[2] for p in parts[name]["v"]])))
    return out


def solve(stats: PairStats, sources: List[List[int]], lam: float = HEO_LAMBDA,
          policy: str = "r2_topk", meta: Optional[dict] = None) -> Tuple[KVMapper, FitReport]:
    return solve_many(stats, {"m": sources}, lam, {"m": policy}, meta)["m"]


def fit(stats: PairStats, policy: str, k: Optional[int] = None,
        lam: float = HEO_LAMBDA, r2: Optional[np.ndarray] = None,
        meta: Optional[dict] = None) -> Tuple[KVMapper, FitReport]:
    """``policy='r2_topk'`` (Heo et al.) or ``'depth_linear'`` (molt v0.1)."""
    if policy == "r2_topk":
        r2 = single_source_r2(stats, lam) if r2 is None else r2
        S = heo_sources(r2, k or stats.src.n_layers)
    elif policy == "depth_linear":
        S = depth_linear_sources(stats.src.n_layers, stats.dst.n_layers)
    else:
        raise ValueError(policy)
    return solve(stats, S, lam, policy, meta)


# --------------------------------------------------------------------------
# empirical evaluation on held-out text
# --------------------------------------------------------------------------
def heldout_r2(mapper: KVMapper, src_model, dst_model,
               sequences: Iterable[Sequence[int]]) -> FitReport:
    """Apply ``mapper`` to real source caches and score against the destination's
    own caches (RoPE removed on both sides): head-averaged R² per layer."""
    L, H, D = mapper.dst.n_layers, mapper.dst.n_kv_heads, mapper.dst.head_dim
    res = {kv: np.zeros((L, H)) for kv in "kv"}
    s1 = {kv: np.zeros((L, H, D)) for kv in "kv"}
    s2 = {kv: np.zeros((L, H, D)) for kv in "kv"}
    n = 0
    for seq in sequences:
        ids = mx.array([list(seq)])
        cs, cd = make_prompt_cache(src_model), make_prompt_cache(dst_model)
        src_model(ids, cache=cs)
        dst_model(ids, cache=cd)
        mapped = mapper.apply([c.state for c in cs], offset=0, out_dtype=mx.float32)
        for l, c in enumerate(cd):
            tk, tv = c.state
            pk, pv = mapped[l]
            for kv, p, t in (("k", R.remove(pk, mapper.dst_rope),
                              R.remove(tk.astype(mx.float32), mapper.dst_rope)),
                             ("v", pv, tv.astype(mx.float32))):
                p = np.array(p[0], dtype=np.float64)   # [H, T, D]
                t = np.array(t[0], dtype=np.float64)
                res[kv][l] += ((p - t) ** 2).sum(axis=(1, 2))
                s1[kv][l] += t.sum(axis=1)
                s2[kv][l] += (t * t).sum(axis=1)
        n += ids.shape[1]
    out = {}
    for kv in "kv":
        tot = (s2[kv] - s1[kv] ** 2 / n).sum(-1)
        out[kv] = (1.0 - res[kv] / np.maximum(tot, 1e-30)).mean(-1)
    return FitReport(out["k"], out["v"])


def selftest() -> None:
    from .tinymodels import make, BIG, SMALL

    big, small = make(BIG, 1), make(SMALL, 2)
    rng = np.random.default_rng(0)
    seqs = [rng.integers(0, 256, size=64).tolist() for _ in range(12)]

    # (1) exact recovery: a destination that IS a linear map of the source
    xk, xv = features(big, mx.array([seqs[0]]), stride=1)
    Wtrue = mx.array(rng.normal(size=(xk.shape[1], 40)).astype(np.float32) * 0.1)
    st = PairStats(Geometry(6, 2, 32), Geometry(1, 2, 20), R.RopeSpec(32, 1e4, False),
                   R.RopeSpec(20, 1e4, False))
    for s in seqs:
        a, b = features(big, mx.array([s]), stride=1)
        st.update((a, b), (a @ Wtrue + 3.0, b @ Wtrue - 1.0))
    m, rep = solve(st, [list(range(6))], lam=1e-8)
    err = mx.abs(m.layers[0].Wk - Wtrue).max().item()
    assert rep.mean > 0.999999 and err < 2e-2, (rep.mean, err)
    assert abs(m.layers[0].bk.mean().item() - 3.0) < 1e-2
    print(f"  recovers a planted linear map: R²={rep.mean:.6f}, max |ΔW|={err:.1e}, bias ok")

    # (2) stats-derived R² == R² of actually applying the mapper (stride 1, in-sample)
    stats = collect({"big": big, "small": small}, [("big", "small")], seqs, stride=1)[("big", "small")]
    r2 = single_source_r2(stats)
    assert r2.shape == (4, 6)
    for k in (1, 2, 6):
        mk, rk = fit(stats, "r2_topk", k, r2=r2)
        emp = heldout_r2(mk, big, small, seqs)
        d = max(np.abs(emp.r2_k - rk.r2_k).max(), np.abs(emp.r2_v - rk.r2_v).max())
        assert d < 2e-3, (k, d)
        print(f"  k={k}: stats R²={rk.mean:.4f}  empirical R²={emp.mean:.4f}  (|Δ|={d:.1e})")
    m1, r1 = fit(stats, "r2_topk", 1, r2=r2)
    m6, r6 = fit(stats, "r2_topk", 6, r2=r2)
    assert r6.mean >= r1.mean - 1e-9, "in-sample R² must not drop as k grows"
    mdl, rdl = fit(stats, "depth_linear")
    assert all(len(l.sources) == 1 for l in mdl.layers)
    print(f"  depth_linear R²={rdl.mean:.4f}; top-1 by R² picks {[l.sources[0] for l in m1.layers]}")
    print("fit: ok")


if __name__ == "__main__":
    selftest()
