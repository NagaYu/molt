"""Molt core #1 — **KVTransplant**: move a live KV cache between tier rungs.

Given a cache produced by rung *A* over ``T`` tokens, produce a cache that rung
*B* can keep decoding from, without ever re-reading the prompt.

Three mechanisms, matching the three requirements in the brief:

(i) **learned linear projection** — when ``head_dim`` / ``n_kv_heads`` / depth
    differ, a per-destination-layer affine map (fitted offline by ridge
    regression, see :mod:`molt.fit_projector`) carries K and V across, wrapped
    in a RoPE un-rotate / re-rotate sandwich so the map is position-independent.

(ii) **quantisation scale re-alignment** — when the rungs share an architecture
    but differ in weight precision, geometry is untouched and only per-channel
    scale/offset drifts; a diagonal map corrects it.  This is the ``diag``
    flavour of :class:`~molt.projector.LinearMap`.

(iii) **selective top-k recompute** — the destination's final ``k`` layers are
    recomputed *natively* from a projected hidden state instead of being
    projected.  DroidSpeak recomputes a layer range for an identically-shaped
    sibling; here the same idea is made to work across different sizes by
    projecting the hidden state at the boundary.  Cost scales with ``k/L``.

Every call returns a :class:`TransplantReport` with wall-clock milliseconds
broken down by phase, estimated FLOPs, and bytes moved — the evidence for the
*low migration cost* claim.

Claims supported by this module
-------------------------------
* **low migration cost**: transplant is measured against the re-prefill it
  replaces (:meth:`KVTransplant.reprefill` implements condition C's fallback so
  both are measured by identical instrumentation).
* **no-stall**: the returned cache is immediately decodable, so the generation
  loop resumes on the very next token.
* **continuity**: projection + scale re-alignment are what keep the incoming
  model's distribution near the outgoing one.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from .adapters import (LMAdapter, ModelGeometry, flops_layer_range,
                       flops_per_token_prefill)
from .config import TierSpec, TransplantConfig
from .kv_cache import CacheMeta, MoltCache
from .metrics import MB, Stopwatch
from .model_zoo import LoadedTier
from .projector import KVProjector, rope_tables


# --------------------------------------------------------------------------
# a source that does not need to stay resident
# --------------------------------------------------------------------------


@dataclass
class TierRef:
    """Everything a transplant needs to know about its **source**.

    The key observation for an on-device setting: a transplant needs the source
    model's *RoPE parameters* (a few kilobytes of ``inv_freq``) but **not** its
    weights.  So the outgoing model can be evicted *before* the incoming one is
    loaded, and the two never have to be resident at the same time.  Without
    this, every migration would need headroom for both rungs at once — which is
    precisely the memory the system does not have at the moment pressure hits.

    Claims supported
    ----------------
    * **zero-kill**: bounds peak memory during a migration to
      ``max(src_weights, dst_weights) + caches`` instead of their sum.
    """

    name: str
    model_id: str
    quant: str
    geometry: ModelGeometry
    intermediate_size: int
    #: deep-copied rotary embedding module, or None when RoPE is not needed
    rope: Optional[torch.nn.Module] = None

    @classmethod
    def from_loaded(cls, lt: LoadedTier, snapshot_rope: bool = False) -> "TierRef":
        rope = getattr(lt.adapter.backbone, "rotary_emb", None) if lt.adapter.backbone else None
        if rope is not None and snapshot_rope:
            rope = copy.deepcopy(rope)
        return cls(name=lt.spec.name, model_id=lt.spec.model_id, quant=lt.spec.quant,
                   geometry=lt.geometry, intermediate_size=lt.intermediate_size, rope=rope)

    def rope_tables(self, positions: torch.Tensor, ref: torch.Tensor):
        if self.rope is None:
            raise RuntimeError(f"tier {self.name} has no RoPE snapshot")
        return self.rope(ref, positions)

    def rope_bytes(self) -> int:
        if self.rope is None:
            return 0
        return sum(b.numel() * b.element_size() for b in self.rope.buffers())


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------


@dataclass
class TransplantReport:
    """Full cost accounting of one cache migration."""

    src_tier: str
    dst_tier: str
    method: str                # "molt" | "reprefill" | "drop" | "identity"
    n_tokens: int
    wall_ms: float = 0.0
    rope_ms: float = 0.0
    project_ms: float = 0.0
    recompute_ms: float = 0.0
    assemble_ms: float = 0.0
    model_load_ms: float = 0.0
    flops_projection: float = 0.0
    flops_recompute: float = 0.0
    #: what a full re-prefill of the same prefix on the destination would cost
    flops_reprefill_equiv: float = 0.0
    bytes_in: int = 0
    bytes_out: int = 0
    n_projected_layers: int = 0
    n_recomputed_layers: int = 0
    used_projection: bool = False
    used_scale_realign: bool = False
    used_rope_realign: bool = False
    degraded_reason: str = ""

    @property
    def flops_total(self) -> float:
        return self.flops_projection + self.flops_recompute

    @property
    def flops_saving(self) -> float:
        """Fraction of a re-prefill's FLOPs avoided (1.0 = free)."""
        if self.flops_reprefill_equiv <= 0:
            return 0.0
        return 1.0 - (self.flops_total / self.flops_reprefill_equiv)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["flops_total"] = self.flops_total
        d["flops_saving"] = self.flops_saving
        d["mb_in"] = self.bytes_in / MB
        d["mb_out"] = self.bytes_out / MB
        return d

    def __str__(self) -> str:
        return (f"[transplant {self.src_tier}->{self.dst_tier} {self.method}] "
                f"T={self.n_tokens} wall={self.wall_ms:.1f}ms "
                f"(proj {self.project_ms:.1f} + recompute {self.recompute_ms:.1f}) "
                f"flops={self.flops_total:.2e} ({self.flops_saving*100:.0f}% saved) "
                f"{self.bytes_in/MB:.1f}->{self.bytes_out/MB:.1f} MiB")


# --------------------------------------------------------------------------
# the operator
# --------------------------------------------------------------------------


class KVTransplant:
    """Stateless operator turning a source cache into a destination cache."""

    def __init__(self, cfg: TransplantConfig):
        self.cfg = cfg

    # -- planning ----------------------------------------------------------
    def plan(self, src: "TierRef", dst: LoadedTier,
             projector: Optional[KVProjector]) -> Dict[str, Any]:
        """Decide which mechanisms apply to this route, before touching data."""
        sg, dg = src.geometry, dst.geometry
        same_model = src.model_id == dst.spec.model_id
        same_shape = (sg.kv_width == dg.kv_width and sg.n_layers == dg.n_layers
                      and sg.head_dim == dg.head_dim)
        crossing_quant = (src.quant != dst.spec.quant)
        k = self.cfg.top_k_for(dg.n_layers)
        boundary_dst = dg.n_layers - k
        boundary_src = sg.n_layers - self.cfg.top_k_for(sg.n_layers)
        can_recompute = (
            k > 0
            and dst.adapter.supports_partial_recompute
            and projector is not None
            and projector.hidden_map is not None
        )
        # Same weights, different precision -> geometry is untouched and only
        # per-channel scale drifts: requirement (ii).  Different weights ->
        # a dense learned map is needed: requirement (i).
        return dict(
            same_model=same_model,
            same_shape=same_shape,
            identity=same_model and not crossing_quant,
            crossing_quant=crossing_quant,
            k=k,
            boundary_dst=boundary_dst,
            boundary_src=boundary_src,
            can_recompute=can_recompute,
            need_projection=not (same_model and same_shape),
            need_scale_realign=crossing_quant and same_model and same_shape,
            # keys of two *different* models are rotated into different bases
            # (different head_dim and/or rope_theta), so the sandwich is needed
            # whenever the map is not a same-model rescale.
            rope_realign=self.cfg.use_rope_realign and not (same_model and same_shape),
        )

    # -- main entry point --------------------------------------------------
    @torch.no_grad()
    def transplant(
        self,
        cache: MoltCache,
        src: "TierRef",
        dst: LoadedTier,
        projector: Optional[KVProjector],
        device: Optional[torch.device] = None,
    ) -> Tuple[MoltCache, TransplantReport]:
        """Carry ``cache`` from rung ``src`` to rung ``dst``.

        ``src`` is a :class:`TierRef`, not a live model — the source weights may
        already have been evicted by the time this runs.

        Demonstrates **low migration cost** (the returned report's ``wall_ms`` is
        what the benchmark compares against a re-prefill) and **no-stall** (the
        result is immediately decodable, so the next token follows the switch by
        one ordinary decode step).
        """
        device = device or cache.device
        sg, dg = src.geometry, dst.geometry
        plan = self.plan(src, dst, projector)
        T = cache.seq_len
        rep = TransplantReport(
            src_tier=src.name, dst_tier=dst.spec.name, method="molt", n_tokens=T,
            bytes_in=cache.nbytes(),
            flops_reprefill_equiv=flops_per_token_prefill(dg, dst.intermediate_size) * T,
        )

        if self.cfg.max_carry_tokens is not None:
            cache = cache.crop(self.cfg.max_carry_tokens)
            T = cache.seq_len
            rep.n_tokens = T
            rep.flops_reprefill_equiv = flops_per_token_prefill(dg, dst.intermediate_size) * T

        with Stopwatch(device) as total_sw:
            # ---- 0. degenerate route: same weights, same precision ---------
            if plan["identity"] and cache.meta.pos_offset == 0:
                out = MoltCache(
                    [(k, v) for k, v in cache.layers],
                    CacheMeta(dst.spec.name, dg, list(cache.meta.token_ids),
                              list(cache.meta.trace_layers), cache.meta.n_prompt_tokens,
                              cache.meta.pos_offset),
                    dict(cache.hidden_traces),
                )
                rep.method = "identity"
                rep.bytes_out = out.nbytes()
                rep.n_projected_layers = dg.n_layers
                return out, rep

            # ``use_projection=False`` is the ablation for requirement (i): fall
            # back to the truncated-identity map a naive "just reshape the
            # tensors" implementation would use.  The flag has to be honoured
            # *here*, at the point of use — a config field that nothing reads
            # would make the ablation arm silently identical to the full system.
            if projector is not None and not self.cfg.use_projection:
                projector = None
                rep.degraded_reason = "use_projection=False (ablation)"

            if projector is None:
                if self.cfg.fallback_to_reprefill:
                    rep.degraded_reason = "no projector for route"
                    raise NeedsReprefill(rep.degraded_reason)
                from .projector import make_identity_projector

                projector = make_identity_projector(
                    sg, dg, src.name, dst.spec.name,
                    rope_realign=self.cfg.use_rope_realign and not plan["same_model"])
                if not rep.degraded_reason:
                    rep.degraded_reason = "untrained projector (identity fallback)"

            projector = projector.to(device)
            # The sandwich must match how the maps were fitted.  A cropped cache
            # additionally *requires* re-rotation: its keys carry angles for
            # absolute positions [pos_offset, pos_offset+T) while the
            # destination will index the cache from 0.
            do_rope = projector.meta.rope_realign or cache.meta.pos_offset != 0

            # ---- 1. RoPE tables at the positions this prefix occupies -------
            # Source keys were rotated at their *absolute* positions
            # [pos_offset, pos_offset+T); the destination will index this cache
            # from 0, so it must be re-rotated at [0, T).
            src_cos = src_sin = dst_cos = dst_sin = None
            if do_rope:
                with Stopwatch(device) as sw:
                    off = cache.meta.pos_offset
                    src_pos = torch.arange(off, off + T, device=device).unsqueeze(0)
                    dst_pos = torch.arange(T, device=device).unsqueeze(0)
                    ref_s = cache.layers[0][0].reshape(-1)[:1]
                    src_cos, src_sin = src.rope_tables(src_pos, ref_s)
                    dst_cos, dst_sin = rope_tables(dst.adapter, dst_pos, ref_s)
                rep.rope_ms = sw.ms
                rep.used_rope_realign = True

            # ---- 2. project every destination layer -------------------------
            layer_map = projector.layer_map
            if len(layer_map) != dg.n_layers:
                raise ValueError(
                    f"projector layer_map has {len(layer_map)} entries but destination "
                    f"{dst.spec.name} has {dg.n_layers} layers")
            # Re-derive this *after* the projector has been resolved: the plan
            # was made against the registry's projector, but the identity
            # fallback (and the use_projection=False ablation) has no hidden-state
            # map, and asking it for one raises — which previously sent the whole
            # migration down the re-prefill exception path and made the ablation
            # silently measure condition C instead of itself.
            can_recompute = (plan["k"] > 0
                             and dst.adapter.supports_partial_recompute
                             and projector.hidden_map is not None)
            k_recompute = plan["k"] if can_recompute else 0
            src_trace = cache.trace_for(plan["boundary_src"])
            if k_recompute > 0 and (src_trace is None or src_trace.shape[1] != T):
                # trace unavailable or stale (e.g. right after a previous
                # transplant): degrade to pure projection rather than stalling.
                have = "none" if src_trace is None else f"len {src_trace.shape[1]} != {T}"
                k_recompute = 0
                rep.degraded_reason = (rep.degraded_reason + "; " if rep.degraded_reason else "") + \
                    f"no usable hidden trace at src layer {plan['boundary_src']} ({have})"
            boundary = dg.n_layers - k_recompute

            out_dtype = dg_dtype(dst)
            with Stopwatch(device) as sw:
                src_layers = [(k.to(device=device, dtype=out_dtype),
                               v.to(device=device, dtype=out_dtype))
                              for k, v in cache.layers]
                projected: List[Tuple[torch.Tensor, torch.Tensor]] = \
                    projector.project_range(src_layers, boundary,
                                            src_cos, src_sin, dst_cos, dst_sin)
                projected += [(None, None)] * (dg.n_layers - boundary)
            rep.project_ms = sw.ms
            rep.n_projected_layers = boundary
            rep.used_projection = projector.meta.kind == "dense"
            rep.used_scale_realign = projector.meta.kind == "diag"
            rep.flops_projection = sum(
                projector.k_maps[l].flops(T) + projector.v_maps[l].flops(T)
                for l in range(boundary))

            # ---- 3. selective top-k native recompute -------------------------
            new_traces: Dict[int, torch.Tensor] = {}
            if k_recompute > 0:
                with Stopwatch(device) as sw:
                    h_src = src_trace.to(device)
                    h_dst = projector.project_hidden(h_src).to(out_dtype)
                    rc = DynamicCache(config=dst.model.config)
                    _, rc = dst.adapter.run_layer_range(
                        h_dst, boundary, dg.n_layers, cache=rc, position_offset=0)
                    for l in range(boundary, dg.n_layers):
                        projected[l] = (rc.layers[l].keys, rc.layers[l].values)
                    # the destination can immediately serve its own outgoing
                    # migrations from this boundary
                    new_traces[boundary] = h_dst
                rep.recompute_ms = sw.ms
                rep.n_recomputed_layers = k_recompute
                rep.flops_recompute = (
                    flops_layer_range(dg, dst.intermediate_size, k_recompute) * T
                    + projector.hidden_map.flops(T)
                )
            elif (src_trace is not None and src_trace.shape[1] == T
                  and projector.hidden_map is not None):
                # Even when this hop does not recompute, project the trace
                # forward so the *next* hop still can.  One small matmul stops a
                # single degraded migration from disabling the cheap path for
                # the rest of the generation.
                new_traces[plan["boundary_dst"]] = (
                    projector.project_hidden(src_trace.to(device)).to(out_dtype))

            # ---- 4. assemble --------------------------------------------------
            with Stopwatch(device) as sw:
                if any(k is None for k, _ in projected):
                    missing = [i for i, (k, _) in enumerate(projected) if k is None]
                    raise RuntimeError(f"transplant left layers {missing} unfilled")
                meta = CacheMeta(
                    tier_name=dst.spec.name, geometry=dg,
                    token_ids=list(cache.meta.token_ids),
                    trace_layers=sorted(new_traces),
                    n_prompt_tokens=cache.meta.n_prompt_tokens,
                    pos_offset=0,  # keys were re-rotated to start at position 0
                )
                out = MoltCache(projected, meta, new_traces)
            rep.assemble_ms = sw.ms
            rep.bytes_out = out.nbytes()

        rep.wall_ms = total_sw.ms
        return out, rep

    # -- condition C's operator -------------------------------------------
    @torch.no_grad()
    def reprefill(
        self,
        token_ids: List[int],
        dst: LoadedTier,
        device: torch.device,
        n_prompt_tokens: int = 0,
        trace_layer: Optional[int] = None,
        src_tier: str = "?",
    ) -> Tuple[MoltCache, TransplantReport]:
        """Throw the cache away and re-read the whole prefix on ``dst``.

        This is the baseline the *low migration cost* claim is measured against
        (condition **C**, restart-on-pressure).  It is deliberately implemented
        here, with the same stopwatch, so the comparison is apples-to-apples.
        """
        dg = dst.geometry
        rep = TransplantReport(
            src_tier=src_tier, dst_tier=dst.spec.name, method="reprefill",
            n_tokens=len(token_ids),
            flops_reprefill_equiv=flops_per_token_prefill(dg, dst.intermediate_size) * len(token_ids),
        )
        ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        traces: Dict[int, torch.Tensor] = {}
        handles = []
        if trace_layer is not None and trace_layer < dg.n_layers:
            handles.append(dst.adapter.capture_hook(
                trace_layer, lambda h, _t=traces, _l=trace_layer: _t.__setitem__(_l, h)))
        try:
            with Stopwatch(device) as sw:
                out = dst.model(ids, use_cache=True)
        finally:
            for h in handles:
                h.remove()
        rep.wall_ms = rep.recompute_ms = sw.ms
        rep.flops_recompute = rep.flops_reprefill_equiv
        rep.n_recomputed_layers = dg.n_layers
        meta = CacheMeta(dst.spec.name, dg, list(token_ids), sorted(traces), n_prompt_tokens)
        cache = MoltCache.from_hf(out.past_key_values, meta, traces)
        rep.bytes_out = cache.nbytes()
        return cache, rep


class NeedsReprefill(RuntimeError):
    """Raised when a route cannot be transplanted and must fall back."""


def dg_dtype(tier: LoadedTier) -> torch.dtype:
    for p in tier.model.parameters():
        return p.dtype
    return torch.float32


# --------------------------------------------------------------------------
# route registry
# --------------------------------------------------------------------------


class ProjectorRegistry:
    """Finds the trained projector for a ``(src, dst)`` route, or None.

    Missing projectors are not fatal: the transplant degrades to an untrained
    identity/zero-padded map and says so in ``TransplantReport.degraded_reason``,
    which the benchmark surfaces rather than hiding.
    """

    def __init__(self, directory: str, ladder_name: str, device: torch.device):
        self.directory = directory
        self.ladder_name = ladder_name
        self.device = device
        self._cache: Dict[Tuple[str, str], Optional[KVProjector]] = {}

    def get(self, src_tier: str, dst_tier: str) -> Optional[KVProjector]:
        import os

        key = (src_tier, dst_tier)
        if key in self._cache:
            return self._cache[key]
        path = os.path.join(self.directory,
                            KVProjector.route_filename(src_tier, dst_tier, self.ladder_name))
        proj = None
        if os.path.exists(path):
            proj = KVProjector.load(path, map_location=str(self.device)).to(self.device)
        self._cache[key] = proj
        return proj

    def put(self, projector: KVProjector) -> None:
        self._cache[(projector.meta.src_tier, projector.meta.dst_tier)] = projector

    def missing_routes(self, ladder) -> List[Tuple[str, str]]:
        return [r for r in ladder.routes() if self.get(*r) is None]
