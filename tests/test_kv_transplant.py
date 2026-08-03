"""Core #1 — KVTransplant.

Claims under test
-----------------
* **low migration cost**  — transplanting beats re-prefilling, in both wall time
  and FLOPs, and the advantage grows with prefix length.
* **continuity**          — a transplanted cache produces a next-token
  distribution close to what the destination model would have produced natively.
* mechanism correctness   — the partial layer-range recompute is *bit-exact*, and
  the RoPE sandwich is an exact inverse.
"""

from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import DynamicCache

from molt.config import TransplantConfig
from molt.kv_cache import CacheMeta, MoltCache
from molt.kv_transplant import KVTransplant, TierRef
from molt.metrics import js_divergence, top1_agreement
from molt.model_zoo import ModelZoo
from molt.projector import (apply_rope, rebase_rope, rope_tables, unapply_rope)

from .conftest import RECOMPUTE_K, held_out_ids


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def build_cache(tier, ids, cfg: TransplantConfig):
    """Prefill ``tier`` on ``ids``, capturing the boundary hidden-state trace."""
    k = cfg.top_k_for(tier.geometry.n_layers)
    boundary = tier.geometry.n_layers - k
    trace, handle = {}, None
    if k > 0:
        handle = tier.adapter.capture_hook(boundary,
                                           lambda x: trace.__setitem__(boundary, x))
    try:
        with torch.no_grad():
            out = tier.model(ids, use_cache=True)
    finally:
        if handle is not None:
            handle.remove()
    meta = CacheMeta(tier.spec.name, tier.geometry, ids[0].tolist(),
                     sorted(trace), ids.shape[1])
    return MoltCache.from_hf(out.past_key_values, meta, trace), out.logits


# --------------------------------------------------------------------------
# mechanism correctness
# --------------------------------------------------------------------------


def test_partial_layer_recompute_is_exact(ladder, device):
    """Running layers ``[j, L)`` by hand reproduces a full forward exactly.

    This is the mechanism selective top-k recompute rests on: if it drifted,
    every "recomputed" layer would be quietly wrong and the *continuity* claim
    would be unfounded.
    """
    zoo = ModelZoo(device, torch.float32)
    tier = zoo.acquire(ladder.by_name("tier0"))
    L = tier.geometry.n_layers
    j = L - 3
    ids = torch.randint(0, 400, (1, 11))

    cap = {}
    h = tier.adapter.capture_hook(j, lambda x: cap.__setitem__("h", x))
    with torch.no_grad():
        ref = tier.model(ids, use_cache=True)
    h.remove()

    rc = DynamicCache(config=tier.model.config)
    _, rc = tier.adapter.run_layer_range(cap["h"], j, L, cache=rc)
    for i in range(j, L):
        assert torch.equal(rc.layers[i].keys, ref.past_key_values.layers[i].keys)
        assert torch.equal(rc.layers[i].values, ref.past_key_values.layers[i].values)
    zoo.evict_all()


def test_rope_sandwich_is_an_exact_inverse(ladder, device):
    """``unapply_rope`` really inverts ``apply_rope``.

    The learned projection is only position-independent if this holds; a sloppy
    inverse would silently make every cross-``head_dim`` transplant depend on
    where in the sequence a token happened to sit.
    """
    zoo = ModelZoo(device, torch.float32)
    tier = zoo.acquire(ladder.by_name("tier0"))
    g = tier.geometry
    k = torch.randn(1, g.n_kv_heads, 17, g.head_dim)
    pos = torch.arange(17).unsqueeze(0)
    cos, sin = rope_tables(tier.adapter, pos, k.reshape(-1)[:1])
    assert torch.allclose(unapply_rope(apply_rope(k, cos, sin), cos, sin), k, atol=1e-5)
    zoo.evict_all()


def test_rebase_rope_matches_a_natively_shorter_prefix(ladder, device):
    """Re-basing a cropped cache reproduces the model's own positions.

    After shedding context the surviving keys must behave as if they had always
    been at positions ``0..T'``; otherwise every relative distance is off by the
    number of dropped tokens and the generation degrades invisibly.
    """
    zoo = ModelZoo(device, torch.float32)
    tier = zoo.acquire(ladder.by_name("tier0"))
    cfg = TransplantConfig(recompute_top_k=RECOMPUTE_K)
    ids = torch.randint(0, 400, (1, 20))
    cache, _ = build_cache(tier, ids, cfg)

    drop = 6
    cropped = cache.crop(cache.seq_len - drop)
    assert cropped.meta.pos_offset == drop
    rebase_rope(cropped, tier.adapter, 0)
    assert cropped.meta.pos_offset == 0

    # the same suffix, prefilled natively from position 0
    native, _ = build_cache(tier, ids[:, drop:], cfg)
    # layer 0 keys are a pure function of the token and its position
    assert torch.allclose(cropped.layers[0][0], native.layers[0][0], atol=2e-3), \
        "re-based keys should match a natively short prefix at layer 0"
    zoo.evict_all()


# --------------------------------------------------------------------------
# the claims
# --------------------------------------------------------------------------


@pytest.mark.parametrize("route", [("tier0", "tier2"), ("tier0", "tier1"),
                                   ("tier2", "tier0")])
def test_transplant_preserves_the_distribution(ladder, device, projector_dir, tokenizer,
                                               route):
    """A transplanted cache decodes close to the destination's native cache.

    **continuity.**  The comparison is against the *destination model's own*
    prefill of the same prefix, so this measures the transplant's error and
    nothing else.
    """
    from molt.kv_transplant import ProjectorRegistry

    src_name, dst_name = route
    zoo = ModelZoo(device, torch.float32)
    cfg = TransplantConfig(recompute_top_k=RECOMPUTE_K, projector_dir=projector_dir)
    reg = ProjectorRegistry(projector_dir, ladder.name, device)

    src = zoo.acquire(ladder.by_name(src_name))
    ids = held_out_ids(tokenizer, 40)
    cache, _ = build_cache(src, ids, cfg)
    src_ref = TierRef.from_loaded(src, snapshot_rope=True)
    zoo.release(ladder.by_name(src_name), evict_if_unused=True)

    dst = zoo.acquire(ladder.by_name(dst_name))
    tp = KVTransplant(cfg)
    new, rep = tp.transplant(cache, src_ref, dst, reg.get(src_name, dst_name), device)

    assert new.seq_len == cache.seq_len
    assert new.n_layers == dst.geometry.n_layers
    assert new.layers[0][0].shape[-1] == dst.geometry.head_dim

    nxt = torch.tensor([[int(ids[0, -1])]])
    with torch.no_grad():
        got = dst.model(nxt, past_key_values=new.to_hf(), use_cache=True).logits[0, -1]
        native = dst.model(torch.cat([ids, nxt], 1)).logits[0, -1]

    assert torch.isfinite(got).all()
    jsd = js_divergence(got, native)
    assert jsd < 0.35, f"{src_name}->{dst_name}: transplant JSD {jsd:.4f} too large"
    zoo.evict_all()


def test_transplant_is_cheaper_than_reprefill(ladder, device, projector_dir):
    """Carrying the cache beats re-reading the prompt — the *low migration cost* claim.

    Checked on FLOPs (implementation-independent) **and** wall time (what a user
    feels), with the models kept warm so the comparison is about the KV work
    rather than about model loading, which both paths pay identically.
    """
    from molt.kv_transplant import ProjectorRegistry

    zoo = ModelZoo(device, torch.float32)
    cfg = TransplantConfig(recompute_top_k=RECOMPUTE_K, projector_dir=projector_dir)
    reg = ProjectorRegistry(projector_dir, ladder.name, device)
    src = zoo.acquire(ladder.by_name("tier0"))
    dst = zoo.acquire(ladder.by_name("tier2"))
    tp = KVTransplant(cfg)
    proj = reg.get("tier0", "tier2")

    ids = torch.randint(0, 400, (1, 320))
    cache, _ = build_cache(src, ids, cfg)
    src_ref = TierRef.from_loaded(src, snapshot_rope=True)

    # warm both paths so neither pays a first-call penalty
    tp.transplant(cache, src_ref, dst, proj, device)
    tp.reprefill(cache.meta.token_ids, dst, device)

    best_t = min(tp.transplant(cache, src_ref, dst, proj, device)[1].wall_ms
                 for _ in range(3))
    _, rep_t = tp.transplant(cache, src_ref, dst, proj, device)
    best_r = min(tp.reprefill(cache.meta.token_ids, dst, device)[1].wall_ms
                 for _ in range(3))

    assert rep_t.flops_total < rep_t.flops_reprefill_equiv, (
        f"transplant FLOPs {rep_t.flops_total:.3e} should be below a re-prefill's "
        f"{rep_t.flops_reprefill_equiv:.3e}")
    assert rep_t.flops_saving > 0.3, f"only {rep_t.flops_saving:.1%} of FLOPs saved"
    assert best_t < best_r, (
        f"transplant {best_t:.1f} ms should beat re-prefill {best_r:.1f} ms "
        f"at T={cache.seq_len}")
    zoo.evict_all()


def test_transplant_cost_grows_slower_than_reprefill(ladder, device, projector_dir):
    """The advantage widens with context length.

    Re-prefill pays attention's quadratic term; a transplant is linear in the
    number of carried tokens.  This is why the technique matters exactly where
    it is hardest to restart — long conversations.
    """
    from molt.kv_transplant import ProjectorRegistry

    zoo = ModelZoo(device, torch.float32)
    cfg = TransplantConfig(recompute_top_k=RECOMPUTE_K, projector_dir=projector_dir)
    reg = ProjectorRegistry(projector_dir, ladder.name, device)
    src, dst = zoo.acquire(ladder.by_name("tier0")), zoo.acquire(ladder.by_name("tier2"))
    tp, proj = KVTransplant(cfg), reg.get("tier0", "tier2")
    src_ref = TierRef.from_loaded(src, snapshot_rope=True)

    ratios = []
    for T in (64, 384):
        ids = torch.randint(0, 400, (1, T))
        cache, _ = build_cache(src, ids, cfg)
        tp.transplant(cache, src_ref, dst, proj, device)          # warm
        _, rt = tp.transplant(cache, src_ref, dst, proj, device)
        _, rr = tp.reprefill(cache.meta.token_ids, dst, device)
        ratios.append(rr.flops_reprefill_equiv / max(1.0, rt.flops_total))
    assert ratios[1] >= ratios[0] * 0.9, (
        f"FLOPs advantage should not shrink with length: {ratios}")
    zoo.evict_all()


def test_top_k_recompute_improves_on_pure_projection(ladder, device, projector_dir,
                                                     tokenizer):
    """Recomputing the destination's top layers is worth its cost — core #1(iii).

    Compared against the same transplant with ``recompute_top_k = 0``.  If this
    ever fails, the third mechanism is dead weight and should be removed rather
    than reported.
    """
    from molt.kv_transplant import ProjectorRegistry

    zoo = ModelZoo(device, torch.float32)
    reg = ProjectorRegistry(projector_dir, ladder.name, device)
    src = zoo.acquire(ladder.by_name("tier0"))
    dst = zoo.acquire(ladder.by_name("tier2"))
    proj = reg.get("tier0", "tier2")

    cfg_k = TransplantConfig(recompute_top_k=RECOMPUTE_K, projector_dir=projector_dir)
    cfg_0 = TransplantConfig(recompute_top_k=0, projector_dir=projector_dir)
    ids = held_out_ids(tokenizer, 48)
    cache, _ = build_cache(src, ids, cfg_k)
    src_ref = TierRef.from_loaded(src, snapshot_rope=True)
    nxt = torch.tensor([[int(ids[0, -1])]])
    with torch.no_grad():
        native = dst.model(torch.cat([ids, nxt], 1)).logits[0, -1]

    def jsd_for(cfg):
        new, rep = KVTransplant(cfg).transplant(cache, src_ref, dst, proj, device)
        with torch.no_grad():
            got = dst.model(nxt, past_key_values=new.to_hf(), use_cache=True).logits[0, -1]
        return js_divergence(got, native), rep

    jsd_k, rep_k = jsd_for(cfg_k)
    jsd_0, rep_0 = jsd_for(cfg_0)
    assert rep_k.n_recomputed_layers == RECOMPUTE_K
    assert rep_0.n_recomputed_layers == 0
    assert jsd_k <= jsd_0 + 1e-6, (
        f"top-k recompute ({jsd_k:.5f}) should not be worse than pure projection "
        f"({jsd_0:.5f})")
    zoo.evict_all()


def _kv_rel_error(transplanted: MoltCache, native: MoltCache) -> float:
    """Mean relative RMS error between two caches, over all layers."""
    errs = []
    for (kt, vt), (kn, vn) in zip(transplanted.layers, native.layers):
        for a, b in ((kt, kn), (vt, vn)):
            num = (a.float() - b.float()).pow(2).mean().sqrt()
            den = b.float().pow(2).mean().sqrt().clamp_min(1e-9)
            errs.append(float(num / den))
    return sum(errs) / len(errs)


def test_fitted_projector_beats_the_naive_map(ladder, device, projector_dir, tokenizer):
    """The learned map is doing real work — core #1(i).

    Measured **in cache space**, against the destination model's own KV for the
    same prefix.  That is the quantity the projector is fitted to reproduce, and
    it is far more sensitive than a downstream logit comparison — on a randomly
    initialised model the next-token distribution barely depends on the context,
    so a logit-space check here would pass for *any* map and prove nothing.

    The control is the truncated-identity fallback with no RoPE sandwich: exactly
    what a naive "just reshape the tensors" implementation would produce.
    """
    from molt.kv_transplant import ProjectorRegistry
    from molt.projector import make_identity_projector

    zoo = ModelZoo(device, torch.float32)
    cfg = TransplantConfig(recompute_top_k=0, projector_dir=projector_dir)
    reg = ProjectorRegistry(projector_dir, ladder.name, device)
    src, dst = zoo.acquire(ladder.by_name("tier0")), zoo.acquire(ladder.by_name("tier2"))
    ids = held_out_ids(tokenizer, 48)
    cache, _ = build_cache(src, ids, cfg)
    native, _ = build_cache(dst, ids, cfg)
    src_ref = TierRef.from_loaded(src, snapshot_rope=True)

    def err_with(proj):
        new, _ = KVTransplant(cfg).transplant(cache, src_ref, dst, proj, device)
        return _kv_rel_error(new, native)

    fitted = err_with(reg.get("tier0", "tier2"))
    naive = err_with(make_identity_projector(src.geometry, dst.geometry, "tier0",
                                             "tier2", rope_realign=False))
    assert fitted < naive, (
        f"fitted projector KV error {fitted:.4f} should beat the naive map {naive:.4f}")
    assert fitted < 0.75, f"fitted projector KV error {fitted:.4f} is implausibly high"
    zoo.evict_all()


def test_rope_sandwich_matters_for_cross_head_dim_routes(ladder, device, projector_dir,
                                                         tokenizer):
    """Fitting *without* the RoPE sandwich produces a worse map.

    This is the empirical justification for the un-rotate / re-rotate step: a
    position-dependent target cannot be matched by a position-independent
    matrix, so the residual of the no-sandwich fit must be larger.
    """
    from molt.fit_projector import fit_route, mean_residual

    zoo = ModelZoo(device, torch.float32)
    s, d = ladder.by_name("tier0"), ladder.by_name("tier2")
    from .conftest import CALIB_TEXTS

    with_rope = fit_route(s, d, zoo, tokenizer,
                          TransplantConfig(recompute_top_k=0, use_rope_realign=True),
                          device, texts=CALIB_TEXTS)
    without = fit_route(s, d, zoo, tokenizer,
                        TransplantConfig(recompute_top_k=0, use_rope_realign=False),
                        device, texts=CALIB_TEXTS)
    a, b = mean_residual(with_rope, "k"), mean_residual(without, "k")
    assert a < b, (f"key-map residual with the RoPE sandwich ({a:.3f}) should be "
                   f"below the residual without it ({b:.3f})")
    zoo.evict_all()


def test_source_weights_are_not_needed_for_a_transplant(ladder, device, projector_dir):
    """A transplant works after the source model has been evicted.

    **zero-kill.**  If the source had to stay resident, every migration would
    need headroom for two rungs at once — precisely the memory that is missing
    when pressure hits.
    """
    from molt.kv_transplant import ProjectorRegistry

    zoo = ModelZoo(device, torch.float32)
    cfg = TransplantConfig(recompute_top_k=RECOMPUTE_K, projector_dir=projector_dir)
    reg = ProjectorRegistry(projector_dir, ladder.name, device)
    spec = ladder.by_name("tier0")
    src = zoo.acquire(spec)
    ids = torch.randint(0, 400, (1, 32))
    cache, _ = build_cache(src, ids, cfg)
    ref = TierRef.from_loaded(src, snapshot_rope=True)

    zoo.release(spec, evict_if_unused=True)
    assert not zoo.is_resident(spec), "source should be gone"
    assert ref.rope_bytes() < 64 * 1024, "the RoPE snapshot must be tiny"

    dst = zoo.acquire(ladder.by_name("tier2"))
    new, rep = KVTransplant(cfg).transplant(cache, ref, dst, reg.get("tier0", "tier2"),
                                            device)
    assert new.seq_len == cache.seq_len and rep.wall_ms >= 0
    zoo.evict_all()
