"""Fit the KV projections of one migration route on a small calibration set.

Closed-form ridge regression, no SGD: for every destination layer we solve

    min_W  || [K̄_src(layer_map[l]) , 1] · W  -  K̄_dst(l) ||²  +  λ||W||²

where ``K̄`` denotes keys with RoPE removed (see :mod:`molt.projector`).  ``V`` is
fitted the same way without the un-rotation.  A separate map is fitted for the
hidden state at the recompute boundary, which is what enables requirement (iii).

A dense map from a 256-wide to a 128-wide cache carries 257x128 unknowns *per
layer*, so it is not as data-cheap as it looks: fitted on a few hundred tokens it
drives the training residual near zero and generalises not at all.  Fitting
therefore holds out whole windows and reports :attr:`ProjectorMeta.val_residual`
alongside the training residual — the held-out number is the one that means
anything, and it is what ``mean_residual`` returns by default.

Claims supported by this module
-------------------------------
* **continuity**: ``ProjectorMeta.val_residual`` reports the *held-out* relative
  RMS error per layer; the benchmark shows that low residual ⇒ low distribution
  jump at a migration.
* **low migration cost**: fitting is offline and takes seconds; at runtime the
  route costs two small matmuls per layer.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .config import TransplantConfig
from .model_zoo import LoadedTier
from .projector import (KVProjector, ProjectorMeta, build_layer_map,
                        rope_tables, unapply_rope)

# A tiny, self-contained calibration corpus.  Deliberately generic prose: the
# projector should learn the *geometry* of the two caches, not the topic.
DEFAULT_CALIB_TEXTS: List[str] = [
    "The engineer opened the report and began reading the summary of last quarter's results.",
    "A small river runs through the valley, and in spring the meadows are covered with flowers.",
    "To solve the equation, first isolate the variable on the left-hand side, then divide both sides.",
    "She explained that the experiment had failed three times before producing a usable measurement.",
    "The library closes at six on weekdays, but it stays open until nine on Saturday evenings.",
    "In distributed systems, consistency, availability and partition tolerance cannot all be maximised.",
    "He packed a notebook, two pencils, a folding map and a thermos of coffee for the long trip.",
    "The compiler emits a warning whenever an unused variable shadows one from an outer scope.",
    "Historians disagree about the causes, but most accept that trade routes played a central role.",
    "Add the flour gradually, stirring constantly, until the mixture thickens and pulls from the sides.",
    "The telescope was pointed at a faint smudge of light that turned out to be a distant galaxy.",
    "Before deploying to production, run the integration tests and check the memory profile carefully.",
    "Children were playing in the courtyard while their parents talked quietly near the fountain.",
    "A cache is useful only when the cost of recomputation is higher than the cost of storage.",
    "The novel begins in a coastal town during a rainy autumn and ends many years later abroad.",
    "Measurements were repeated ten times, and the standard deviation was reported alongside the mean.",
    "Please review the attached document and let me know whether the schedule still looks feasible.",
    "Photosynthesis converts light energy into chemical energy stored in the bonds of glucose molecules.",
    "The train was delayed by forty minutes, so he waited on the platform and read the newspaper.",
    "When memory pressure rises, the operating system begins reclaiming pages from inactive processes.",
    "彼は窓の外を眺めながら、これからの計画についてゆっくりと考えを巡らせていた。",
    "この装置は電力消費を抑えつつ、従来と同等の性能を維持することを目的として設計されている。",
    "会議の前に資料をまとめ、要点を三つに絞って説明できるように準備しておいてください。",
    "山の頂上からは、朝もやに包まれた町並みと、その向こうに広がる海が見渡せた。",
]


def _flatten_kv(x: torch.Tensor) -> torch.Tensor:
    """``[B, H, T, D]`` -> ``[B*T, H*D]``."""
    B, H, T, D = x.shape
    return x.permute(0, 2, 1, 3).reshape(B * T, H * D)


@torch.no_grad()
def _collect(tier: LoadedTier, ids: torch.Tensor, boundary: int, derotate: bool
             ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor]]:
    """Run one model over ``ids`` and return per-layer (K̄, V) plus boundary hidden."""
    trace: Dict[int, torch.Tensor] = {}
    handles = []
    if boundary is not None and 0 <= boundary < tier.geometry.n_layers:
        handles.append(tier.adapter.capture_hook(
            boundary, lambda h, _t=trace: _t.__setitem__("h", h)))
    try:
        out = tier.model(ids, use_cache=True)
    finally:
        for h in handles:
            h.remove()

    T = ids.shape[1]
    keys, vals = [], []
    cos = sin = None
    if derotate:
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        ref = out.past_key_values.layers[0].keys.reshape(-1)[:1]
        cos, sin = rope_tables(tier.adapter, pos, ref)
    for layer in out.past_key_values.layers:
        k = layer.keys.float()
        if derotate:
            k = unapply_rope(k, cos.float(), sin.float())
        keys.append(_flatten_kv(k).cpu())
        vals.append(_flatten_kv(layer.values.float()).cpu())
    h = trace.get("h")
    h = None if h is None else h.float().reshape(-1, tier.geometry.hidden_size).cpu()
    del out
    return keys, vals, h


def _tokenize_corpus(tokenizer, texts, device, max_tokens_per_text, max_total_tokens):
    """Turn the calibration corpus into a fixed list of id tensors.

    The corpus is concatenated and re-cut into fixed-length windows rather than
    used one sentence per batch.  Two reasons:

    * a dense map from a 256-wide to a 128-wide cache has 257x128 unknowns per
      layer, so a few hundred tokens would be an under-determined fit;
    * short isolated sentences under-represent long-range attention, and the
      cache being projected comes from *long* prefixes.

    Windows are taken with a stride shorter than the window so that positions
    are seen at several offsets — which is exactly the invariance the RoPE
    sandwich is supposed to give the learned matrix.
    """
    joined = "\n\n".join(texts)
    ids = tokenizer(joined, return_tensors="pt")["input_ids"][0]
    n = int(ids.numel())
    if n < 8:
        return [], 0
    win = max(8, min(max_tokens_per_text, n))
    stride = max(1, win // 2)
    batches, total, start, guard = [], 0, 0, 0
    while total < max_total_tokens and guard < 10_000:
        guard += 1
        if start + win > n:
            start = 0 if win >= n else (start + win - n) % max(1, n - win)
        chunk = ids[start:start + win]
        if chunk.numel() < 8:
            break
        batches.append(chunk.unsqueeze(0).to(device))
        total += int(chunk.numel())
        start += stride
        if len(batches) > 512:
            break
    return batches, total


@torch.no_grad()
def _collect_all(tier: LoadedTier, batches, boundary, derotate):
    """Run one model over the whole corpus, returning per-layer stacked K̄/V/H."""
    n = tier.geometry.n_layers
    K: List[List[torch.Tensor]] = [[] for _ in range(n)]
    V: List[List[torch.Tensor]] = [[] for _ in range(n)]
    H: List[torch.Tensor] = []
    for ids in batches:
        ks, vs, h = _collect(tier, ids, boundary, derotate)
        for i in range(n):
            K[i].append(ks[i]); V[i].append(vs[i])
        if h is not None:
            H.append(h)
    return ([torch.cat(x, 0) for x in K],
            [torch.cat(x, 0) for x in V],
            torch.cat(H, 0) if H else None)


@torch.no_grad()
def fit_route(
    src_spec,
    dst_spec,
    zoo,
    tokenizer,
    cfg: TransplantConfig,
    device: torch.device,
    texts: Optional[Sequence[str]] = None,
    max_tokens_per_text: int = 128,
    max_total_tokens: int = 3072,
    ridge: float = 1e-3,
    verbose: bool = False,
    sequential: bool = True,
    val_frac: float = 0.2,
) -> KVProjector:
    """Fit and return the projector for ``src -> dst``.

    The maps are fitted on *de-rotated* keys so that the learned matrix is
    position-independent — the single most important detail for making a
    cross-``head_dim`` transplant work at all.

    With ``sequential=True`` (the default) the two rungs are visited in separate
    passes and the first is evicted before the second is loaded, so fitting a
    route never needs both models resident.  That matters: on the device this
    prototype targets, holding a 1.5B and a 0.5B model at once is most of the
    memory the whole experiment is about.
    """
    texts = list(texts or DEFAULT_CALIB_TEXTS)
    src = zoo.acquire(src_spec)
    sg = src.geometry
    dst_geom_probe = None
    same_model = src_spec.model_id == dst_spec.model_id

    batches, total = _tokenize_corpus(tokenizer, texts, device,
                                      max_tokens_per_text, max_total_tokens)
    if total == 0:
        zoo.release(src_spec)
        raise ValueError("calibration corpus produced no usable tokens")
    # Hold out whole windows, not rows: neighbouring rows inside one window share
    # a prefix, so a row-wise split would leak and the "validation" residual
    # would flatter the map exactly as much as the training one does.
    n_val = max(1, int(round(val_frac * len(batches)))) if len(batches) > 2 else 0
    val_batches = batches[-n_val:] if n_val else []
    fit_batches = batches[:-n_val] if n_val else batches
    n_val_tokens = sum(int(b.numel()) for b in val_batches)

    # We need the destination geometry before running the source pass so the
    # boundary layers and the map flavour are decided consistently.
    if zoo.known_geometry(dst_spec) is None:
        d_tmp = zoo.acquire(dst_spec)
        dst_geom_probe = d_tmp.geometry
        zoo.release(dst_spec, evict_if_unused=sequential)
    dg = dst_geom_probe or zoo.known_geometry(dst_spec)

    same_shape = (sg.kv_width == dg.kv_width and sg.n_layers == dg.n_layers
                  and sg.head_dim == dg.head_dim)
    kind = "diag" if (same_model and same_shape) else "dense"
    derotate = cfg.use_rope_realign and not (same_model and same_shape)
    if kind == "diag" and not cfg.use_scale_realign:
        kind = "identity"

    boundary_src = sg.n_layers - cfg.top_k_for(sg.n_layers)
    boundary_dst = dg.n_layers - cfg.top_k_for(dg.n_layers)
    want_hidden = cfg.top_k_for(dg.n_layers) > 0

    meta = ProjectorMeta(
        src_tier=src_spec.name, dst_tier=dst_spec.name, src_geom=sg, dst_geom=dg,
        layer_map=build_layer_map(sg.n_layers, dg.n_layers),
        kind=kind, rope_realign=derotate,
        trace_src_layer=boundary_src if want_hidden else None,
        trace_dst_layer=boundary_dst if want_hidden else None,
    )
    proj = KVProjector(meta)
    meta.n_calib_tokens = total - n_val_tokens
    meta.n_val_tokens = n_val_tokens

    # ---- pass 1: source --------------------------------------------------
    b_src = boundary_src if want_hidden else None
    K_src, V_src, H_src = _collect_all(src, fit_batches, b_src, derotate)
    VK_src, VV_src, VH_src = (_collect_all(src, val_batches, b_src, derotate)
                              if val_batches else (None, None, None))
    zoo.release(src_spec, evict_if_unused=sequential)
    del src

    # ---- pass 2: destination --------------------------------------------
    dst = zoo.acquire(dst_spec)
    b_dst = boundary_dst if want_hidden else None
    K_dst, V_dst, H_dst = _collect_all(dst, fit_batches, b_dst, derotate)
    VK_dst, VV_dst, VH_dst = (_collect_all(dst, val_batches, b_dst, derotate)
                              if val_batches else (None, None, None))
    zoo.release(dst_spec, evict_if_unused=sequential)
    del dst

    # ---- solve ----------------------------------------------------------
    residuals: Dict[str, float] = {}
    val: Dict[str, float] = {}
    for dst_l in range(dg.n_layers):
        src_l = meta.layer_map[dst_l]
        residuals[f"k{dst_l}"] = proj.k_maps[dst_l].fit(K_src[src_l], K_dst[dst_l], ridge)
        residuals[f"v{dst_l}"] = proj.v_maps[dst_l].fit(V_src[src_l], V_dst[dst_l], ridge)
        if val_batches:
            val[f"k{dst_l}"] = proj.k_maps[dst_l].residual(VK_src[src_l], VK_dst[dst_l])
            val[f"v{dst_l}"] = proj.v_maps[dst_l].residual(VV_src[src_l], VV_dst[dst_l])
        if verbose:
            print(f"  layer {dst_l:>2} <- {src_l:>2}: "
                  f"k_res={residuals[f'k{dst_l}']:.3f} v_res={residuals[f'v{dst_l}']:.3f}"
                  + (f"  |  val k={val[f'k{dst_l}']:.3f} v={val[f'v{dst_l}']:.3f}"
                     if val_batches else ""))

    if proj.hidden_map is not None and H_src is not None and H_dst is not None:
        residuals["hidden"] = proj.hidden_map.fit(H_src, H_dst, ridge)
        if VH_src is not None and VH_dst is not None:
            val["hidden"] = proj.hidden_map.residual(VH_src, VH_dst)
        if verbose:
            print(f"  hidden {boundary_src} -> {boundary_dst}: res={residuals['hidden']:.3f}")
    elif proj.hidden_map is not None:
        # no trace captured -> disable the recompute path rather than shipping
        # an untrained hidden map that would silently poison the top-k layers.
        proj.hidden_map = None
        meta.trace_src_layer = meta.trace_dst_layer = None

    meta.fit_residual = residuals
    meta.val_residual = val
    return proj


def mean_residual(proj: KVProjector, prefix: str = "", held_out: bool = True) -> float:
    """Mean relative residual.  Defaults to the **held-out** number.

    Training residual is reported too, but quoting it as if it measured map
    quality would be misleading — see :class:`~molt.projector.ProjectorMeta`.
    """
    src = proj.meta.val_residual if (held_out and proj.meta.val_residual) \
        else proj.meta.fit_residual
    vals = [v for k, v in src.items() if k.startswith(prefix) and k != "hidden"]
    if prefix == "hidden":
        vals = [v for k, v in src.items() if k == "hidden"]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def fit_all_routes(
    zoo, ladder, tokenizer, cfg: TransplantConfig, device: torch.device,
    out_dir: Optional[str] = None, texts: Optional[Sequence[str]] = None,
    verbose: bool = True, routes: Optional[Sequence[Tuple[str, str]]] = None,
    sequential: bool = True, max_total_tokens: int = 3072,
    ridge: float = 1e-3, val_frac: float = 0.2,
) -> Dict[Tuple[str, str], KVProjector]:
    """Fit every migration route of a ladder and (optionally) save them."""
    out_dir = out_dir or cfg.projector_dir
    result: Dict[Tuple[str, str], KVProjector] = {}
    for src_name, dst_name in (routes or ladder.routes()):
        s_spec, d_spec = ladder.by_name(src_name), ladder.by_name(dst_name)
        if s_spec.model_id == d_spec.model_id and s_spec.quant == d_spec.quant:
            continue
        if verbose:
            print(f"[fit] {src_name} -> {dst_name}")
        proj = fit_route(s_spec, d_spec, zoo, tokenizer, cfg, device,
                         texts=texts, verbose=False, sequential=sequential,
                         max_total_tokens=max_total_tokens, ridge=ridge,
                         val_frac=val_frac)
        result[(src_name, dst_name)] = proj
        if out_dir:
            path = os.path.join(out_dir, KVProjector.route_filename(src_name, dst_name, ladder.name))
            proj.save(path)
        if verbose:
            print(f"       kind={proj.meta.kind} rope={proj.meta.rope_realign} "
                  f"fit/val tokens={proj.meta.n_calib_tokens}/{proj.meta.n_val_tokens}\n"
                  f"       held-out  k={mean_residual(proj, 'k'):.3f} "
                  f"v={mean_residual(proj, 'v'):.3f} "
                  f"h={mean_residual(proj, 'hidden'):.3f}   "
                  f"(train k={mean_residual(proj, 'k', held_out=False):.3f} "
                  f"v={mean_residual(proj, 'v', held_out=False):.3f})")
    return result
