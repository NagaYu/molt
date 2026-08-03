"""Learned linear maps that carry a KV cache across a tier boundary.

This is requirement (i) of KVTransplant — "a trained linear projection for the
case where hidden size / head count differ" — plus the machinery needed to make
such a projection *well posed*.

The subtlety nobody can skip
----------------------------
HuggingFace caches keys **after** RoPE.  Two rungs with different ``head_dim``
(or different ``rope_theta``) rotate by different angles, so a single
position-independent matrix cannot map one cache onto the other: the required
map would depend on the token's absolute position.

So the projection sandwiches the learned matrix between an *un-rotation* and a
*re-rotation*::

    K_src  --(inverse RoPE @ src angles)-->  K̄_src
           --(learned W_k, per layer)------>  K̄_dst
           --(forward RoPE @ dst angles)--->  K_dst

``V`` is never rotated and is projected directly.  Turning the sandwich off
(``rope_realign=False``) is an ablation the benchmark reports, and it is
dramatically worse — which is the empirical evidence that the sandwich matters.

Fitting is closed-form ridge regression on a small calibration corpus (a few
hundred tokens is enough), so building a projector takes seconds and is
deterministic — no SGD, no hyper-parameter search.

Claims supported by this module
-------------------------------
* **continuity**: a well-fit projector is what keeps the incoming model's
  distribution close to the outgoing one at the switch point.
* **low migration cost**: the projection is ``O(L · T · d_src · d_dst)`` — two
  small matmuls per layer — versus a full re-prefill's ``O(L · T · d²)`` plus
  attention.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .adapters import LMAdapter, ModelGeometry


# --------------------------------------------------------------------------
# RoPE helpers
# --------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``k``: ``[B, H, T, D]``; ``cos``/``sin``: ``[B, T, D]``."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return k * cos + rotate_half(k) * sin


def unapply_rope(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Exact inverse of :func:`apply_rope` (rotation by ``-θ``)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return k * cos - rotate_half(k) * sin


def rope_tables(adapter: LMAdapter, positions: torch.Tensor, ref: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` of ``adapter``'s rotary embedding at ``positions``.

    ``positions`` is ``[B, T]``; ``ref`` only supplies dtype/device.
    """
    backbone = adapter.backbone
    if backbone is None or not hasattr(backbone, "rotary_emb"):
        raise RuntimeError("model has no rotary_emb; RoPE re-alignment unavailable")
    return backbone.rotary_emb(ref, positions)


# --------------------------------------------------------------------------
# linear maps
# --------------------------------------------------------------------------


class LinearMap(nn.Module):
    """A fitted affine map with three flavours.

    ``identity``
        shapes match and no correction is wanted (ablation baseline).
    ``diag``
        per-channel gain + bias.  This is the *quantisation scale re-alignment*
        of KVTransplant requirement (ii): crossing an fp→int8 boundary leaves
        geometry untouched and only perturbs per-channel scale/offset.
    ``dense``
        full ``[in, out]`` matrix — needed when ``head_dim`` or ``n_kv_heads``
        change (requirement i).
    """

    def __init__(self, in_dim: int, out_dim: int, kind: str = "dense"):
        super().__init__()
        self.in_dim, self.out_dim, self.kind = in_dim, out_dim, kind
        if kind == "identity":
            if in_dim != out_dim:
                raise ValueError("identity map requires equal dims")
        elif kind == "diag":
            if in_dim != out_dim:
                raise ValueError("diag map requires equal dims")
            self.register_buffer("gain", torch.ones(out_dim))
            self.register_buffer("bias", torch.zeros(out_dim))
        elif kind == "dense":
            self.register_buffer("weight", torch.zeros(in_dim, out_dim))
            self.register_buffer("bias", torch.zeros(out_dim))
            with torch.no_grad():
                n = min(in_dim, out_dim)
                self.weight[:n, :n] = torch.eye(n)
        else:
            raise ValueError(f"unknown map kind {kind!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[..., in_dim]`` -> ``[..., out_dim]``."""
        if self.kind == "identity":
            return x
        if self.kind == "diag":
            return x * self.gain.to(x.dtype) + self.bias.to(x.dtype)
        w = self.weight.to(x.dtype)
        return torch.matmul(x, w) + self.bias.to(x.dtype)

    def flops(self, n_tokens: int) -> float:
        if self.kind == "identity":
            return 0.0
        if self.kind == "diag":
            return 2.0 * n_tokens * self.out_dim
        return 2.0 * n_tokens * self.in_dim * self.out_dim

    @torch.no_grad()
    def residual(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Relative RMS error of this map on ``(X, Y)`` — usable on held-out data."""
        return _rel_rms(Y.to(torch.float64) - self(X.to(self.dtype_of())).to(torch.float64),
                       Y.to(torch.float64))

    def dtype_of(self) -> torch.dtype:
        for b in self.buffers():
            return b.dtype
        return torch.float32

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor, ridge: float = 1e-3) -> float:
        """Closed-form least squares.  Returns relative residual RMS.

        ``X``: ``[N, in_dim]``, ``Y``: ``[N, out_dim]``.
        """
        X = X.to(torch.float64)
        Y = Y.to(torch.float64)
        if self.kind == "identity":
            return _rel_rms(Y - X, Y)
        if self.kind == "diag":
            # per-channel a*x + b, solved independently for each channel
            xm, ym = X.mean(0), Y.mean(0)
            xc, yc = X - xm, Y - ym
            var = (xc * xc).mean(0).clamp_min(1e-12)
            gain = (xc * yc).mean(0) / var
            bias = ym - gain * xm
            self.gain.copy_(gain.to(self.gain.dtype))
            self.bias.copy_(bias.to(self.bias.dtype))
            return _rel_rms(Y - (X * gain + bias), Y)
        # dense with intercept
        N = X.shape[0]
        Xa = torch.cat([X, torch.ones(N, 1, dtype=X.dtype)], dim=1)
        A = Xa.T @ Xa
        A += ridge * torch.eye(A.shape[0], dtype=A.dtype) * (A.diagonal().mean().clamp_min(1e-12))
        B = Xa.T @ Y
        try:
            W = torch.linalg.solve(A, B)
        except Exception:
            W = torch.linalg.lstsq(A, B).solution
        self.weight.copy_(W[:-1].to(self.weight.dtype))
        self.bias.copy_(W[-1].to(self.bias.dtype))
        return _rel_rms(Y - (X @ W[:-1] + W[-1]), Y)


def _rel_rms(resid: torch.Tensor, target: torch.Tensor) -> float:
    num = resid.to(torch.float64).pow(2).mean().sqrt()
    den = target.to(torch.float64).pow(2).mean().sqrt().clamp_min(1e-12)
    return float(num / den)


# --------------------------------------------------------------------------
# depth remapping
# --------------------------------------------------------------------------


def build_layer_map(n_src: int, n_dst: int, mode: str = "linear") -> List[int]:
    """For each destination layer, which source layer feeds it.

    ``linear`` stretches relative depth, which preserves the "early layers carry
    lexical structure / late layers carry semantics" ordering that makes a
    cross-depth transplant work at all.
    """
    if n_dst == 1:
        return [n_src - 1]
    if mode == "linear":
        return [min(n_src - 1, int(round(i * (n_src - 1) / (n_dst - 1)))) for i in range(n_dst)]
    if mode == "truncate":
        return [min(i, n_src - 1) for i in range(n_dst)]
    raise ValueError(f"unknown layer-map mode {mode!r}")


# --------------------------------------------------------------------------
# the projector
# --------------------------------------------------------------------------


@dataclass
class ProjectorMeta:
    src_tier: str
    dst_tier: str
    src_geom: ModelGeometry
    dst_geom: ModelGeometry
    layer_map: List[int]
    kind: str                # "identity" | "diag" | "dense"
    rope_realign: bool
    trace_src_layer: Optional[int] = None
    trace_dst_layer: Optional[int] = None
    #: relative RMS residual **on the data the map was fitted to**
    fit_residual: Dict[str, float] = field(default_factory=dict)
    #: relative RMS residual on **held-out** windows.  This is the number that
    #: says whether the map generalises; a dense 257x128 map fitted on a few
    #: hundred tokens can drive ``fit_residual`` near zero while being useless.
    val_residual: Dict[str, float] = field(default_factory=dict)
    n_calib_tokens: int = 0
    n_val_tokens: int = 0

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["src_geom"] = self.src_geom.__dict__
        d["dst_geom"] = self.dst_geom.__dict__
        return d


class KVProjector(nn.Module):
    """Maps a whole :class:`~molt.kv_cache.MoltCache` from one rung to another."""

    def __init__(self, meta: ProjectorMeta):
        super().__init__()
        self.meta = meta
        sg, dg = meta.src_geom, meta.dst_geom
        kind = meta.kind
        self.k_maps = nn.ModuleList(
            [LinearMap(sg.kv_width, dg.kv_width, kind) for _ in range(dg.n_layers)])
        self.v_maps = nn.ModuleList(
            [LinearMap(sg.kv_width, dg.kv_width, kind) for _ in range(dg.n_layers)])
        self._stack_cache: Dict[tuple, tuple] = {}
        self.hidden_map: Optional[LinearMap] = None
        if meta.trace_src_layer is not None and meta.trace_dst_layer is not None:
            h_kind = "dense" if sg.hidden_size != dg.hidden_size else (
                "diag" if kind != "identity" else "identity")
            self.hidden_map = LinearMap(sg.hidden_size, dg.hidden_size, h_kind)

    # -- geometry helpers --------------------------------------------------
    @property
    def layer_map(self) -> List[int]:
        return self.meta.layer_map

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        """``[B, H, T, D]`` -> ``[B, T, H*D]``."""
        B, H, T, D = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, T, H * D)

    @staticmethod
    def _unflatten(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
        """``[B, T, H*D]`` -> ``[B, H, T, D]``."""
        B, T, _ = x.shape
        return x.reshape(B, T, n_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    # -- the actual projection --------------------------------------------
    @torch.no_grad()
    def project_layer(
        self, k: torch.Tensor, v: torch.Tensor, dst_layer: int,
        src_cos: Optional[torch.Tensor] = None, src_sin: Optional[torch.Tensor] = None,
        dst_cos: Optional[torch.Tensor] = None, dst_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dg = self.meta.dst_geom
        # The RoPE sandwich is applied iff the caller supplied angle tables;
        # it must match how the maps were *fitted* (``meta.rope_realign``).
        if src_cos is not None:
            k = unapply_rope(k, src_cos, src_sin)
        kf = self._flatten(k)
        vf = self._flatten(v)
        kf = self.k_maps[dst_layer](kf)
        vf = self.v_maps[dst_layer](vf)
        k_out = self._unflatten(kf, dg.n_kv_heads, dg.head_dim)
        v_out = self._unflatten(vf, dg.n_kv_heads, dg.head_dim)
        if dst_cos is not None:
            k_out = apply_rope(k_out, dst_cos, dst_sin)
        return k_out, v_out

    # -- batched projection ------------------------------------------------
    def _stack_maps(self, maps: nn.ModuleList, n: int, device, dtype):
        """Stack ``n`` per-layer maps into one batched operator.

        A 28-layer transplant done layer-by-layer is ~170 small kernel launches;
        as one ``bmm`` it is four.  On short prefixes and small models that
        launch overhead *is* the migration cost, and it would otherwise mask the
        FLOPs advantage the technique actually has.
        """
        key = (id(maps), n, str(device), str(dtype))
        cached = self._stack_cache.get(key)
        if cached is not None:
            return cached
        kind = self.meta.kind
        if kind == "dense":
            W = torch.stack([maps[i].weight for i in range(n)]).to(device=device, dtype=dtype)
            b = torch.stack([maps[i].bias for i in range(n)]).to(device=device, dtype=dtype)
        elif kind == "diag":
            W = torch.stack([maps[i].gain for i in range(n)]).to(device=device, dtype=dtype)
            b = torch.stack([maps[i].bias for i in range(n)]).to(device=device, dtype=dtype)
        else:
            W = b = None
        out = (kind, W, b)
        self._stack_cache[key] = out
        return out

    @staticmethod
    def _rope_batched(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                      inverse: bool) -> torch.Tensor:
        """``x``: ``[L, B, H, T, D]``; ``cos``/``sin``: ``[B, T, D]``."""
        c = cos[None, :, None, :, :]
        s = sin[None, :, None, :, :]
        return x * c + rotate_half(x) * (-s if inverse else s)

    @torch.no_grad()
    def project_range(
        self, layers: List[Tuple[torch.Tensor, torch.Tensor]], n_dst: int,
        src_cos=None, src_sin=None, dst_cos=None, dst_sin=None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Project destination layers ``[0, n_dst)`` in one batched pass."""
        if n_dst <= 0:
            return []
        dg = self.meta.dst_geom
        lm = self.meta.layer_map
        ref = layers[0][0]
        device, dtype = ref.device, ref.dtype

        idx = [min(lm[l], len(layers) - 1) for l in range(n_dst)]
        K = torch.stack([layers[i][0] for i in idx])       # [L, B, Hs, T, Ds]
        V = torch.stack([layers[i][1] for i in idx])
        if src_cos is not None:
            K = self._rope_batched(K, src_cos.to(dtype), src_sin.to(dtype), inverse=True)

        L, B, Hs, T, Ds = K.shape
        Kf = K.permute(0, 1, 3, 2, 4).reshape(L, B * T, Hs * Ds)
        Vf = V.permute(0, 1, 3, 2, 4).reshape(L, B * T, Hs * Ds)

        kind, Wk, bk = self._stack_maps(self.k_maps, n_dst, device, dtype)
        _, Wv, bv = self._stack_maps(self.v_maps, n_dst, device, dtype)
        if kind == "dense":
            Kf = torch.baddbmm(bk.unsqueeze(1), Kf, Wk)
            Vf = torch.baddbmm(bv.unsqueeze(1), Vf, Wv)
        elif kind == "diag":
            Kf = Kf * Wk.unsqueeze(1) + bk.unsqueeze(1)
            Vf = Vf * Wv.unsqueeze(1) + bv.unsqueeze(1)

        Ko = Kf.reshape(L, B, T, dg.n_kv_heads, dg.head_dim).permute(0, 1, 3, 2, 4)
        Vo = Vf.reshape(L, B, T, dg.n_kv_heads, dg.head_dim).permute(0, 1, 3, 2, 4)
        if dst_cos is not None:
            Ko = self._rope_batched(Ko, dst_cos.to(dtype), dst_sin.to(dtype), inverse=False)
        return [(Ko[l].contiguous(), Vo[l].contiguous()) for l in range(L)]

    @torch.no_grad()
    def project_hidden(self, h: torch.Tensor) -> torch.Tensor:
        if self.hidden_map is None:
            raise RuntimeError("this projector has no hidden-state map "
                               "(fit it with trace layers to enable top-k recompute)")
        return self.hidden_map(h)

    def flops(self, n_tokens: int) -> float:
        total = sum(m.flops(n_tokens) for m in self.k_maps)
        total += sum(m.flops(n_tokens) for m in self.v_maps)
        if self.hidden_map is not None:
            total += self.hidden_map.flops(n_tokens)
        return total

    # -- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"meta": self.meta.to_dict(), "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "KVProjector":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        m = blob["meta"]
        meta = ProjectorMeta(
            src_tier=m["src_tier"], dst_tier=m["dst_tier"],
            src_geom=ModelGeometry(**m["src_geom"]), dst_geom=ModelGeometry(**m["dst_geom"]),
            layer_map=list(m["layer_map"]), kind=m["kind"], rope_realign=m["rope_realign"],
            trace_src_layer=m.get("trace_src_layer"), trace_dst_layer=m.get("trace_dst_layer"),
            fit_residual=m.get("fit_residual", {}),
            val_residual=m.get("val_residual", {}),
            n_calib_tokens=m.get("n_calib_tokens", 0),
            n_val_tokens=m.get("n_val_tokens", 0),
        )
        proj = cls(meta)
        proj.load_state_dict(blob["state"])
        return proj

    @staticmethod
    def route_filename(src_tier: str, dst_tier: str, ladder_name: str) -> str:
        return f"{ladder_name}__{src_tier}__to__{dst_tier}.pt"


@torch.no_grad()
def rebase_rope(cache, adapter: LMAdapter, new_start: int = 0):
    """Re-rotate a cropped cache so its keys sit at positions ``[new_start, …)``.

    After a sliding-window crop the surviving keys still carry the RoPE angles of
    their *original* absolute positions, but the model will index the shortened
    cache from zero — every relative distance would be off by the number of
    dropped tokens.  Un-rotating at the old angles and re-rotating at the new
    ones fixes that exactly (this is the same position-shift trick that makes
    streaming attention work).

    Mutates and returns ``cache``.
    """
    off = cache.meta.pos_offset
    if off == new_start or not cache.layers:
        return cache
    T = cache.seq_len
    device = cache.device
    ref = cache.layers[0][0].reshape(-1)[:1]
    old_pos = torch.arange(off, off + T, device=device).unsqueeze(0)
    new_pos = torch.arange(new_start, new_start + T, device=device).unsqueeze(0)
    old_cos, old_sin = rope_tables(adapter, old_pos, ref)
    new_cos, new_sin = rope_tables(adapter, new_pos, ref)
    out = []
    for k, v in cache.layers:
        k = unapply_rope(k, old_cos, old_sin)
        k = apply_rope(k, new_cos, new_sin)
        out.append((k, v))
    cache.layers = out
    cache.meta.pos_offset = new_start
    return cache


def make_identity_projector(src_geom: ModelGeometry, dst_geom: ModelGeometry,
                            src_tier: str, dst_tier: str,
                            trace_src_layer: Optional[int] = None,
                            trace_dst_layer: Optional[int] = None,
                            rope_realign: Optional[bool] = None) -> KVProjector:
    """Untrained projector: identity when shapes match, else truncated identity.

    Used as the *fallback* when no calibrated projector exists for a route, and
    as the ``no-projection`` ablation arm of the benchmark.
    """
    same = (src_geom.kv_width == dst_geom.kv_width)
    kind = "identity" if same else "dense"
    meta = ProjectorMeta(
        src_tier=src_tier, dst_tier=dst_tier, src_geom=src_geom, dst_geom=dst_geom,
        layer_map=build_layer_map(src_geom.n_layers, dst_geom.n_layers),
        kind=kind, rope_realign=(not same) if rope_realign is None else bool(rope_realign),
        trace_src_layer=trace_src_layer, trace_dst_layer=trace_dst_layer,
    )
    return KVProjector(meta)
