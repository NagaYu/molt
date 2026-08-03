"""Unit tests for the supporting machinery: cache, quantisation, metrics, pressure.

These are the invariants the four core claims are computed *from*.  If
:func:`js_divergence` is wrong, the continuity claim is unfalsifiable; if
:class:`MoltCache` accounting is wrong, the zero-kill claim is unfalsifiable.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from molt.adapters import ModelGeometry, flops_layer_range, flops_per_token_prefill
from molt.config import TransplantConfig, get_ladder
from molt.kv_cache import CacheMeta, MoltCache
from molt.metrics import (DiscontinuityProbe, LatencyRecorder, TokenRecord,
                          distinct_n, js_divergence, kl_divergence,
                          repetition_rate, top1_agreement)
from molt.model_zoo import ModelZoo
from molt.pressure import (PressureEvent, PressureTrace, TraceReplaySource,
                           builtin_traces, load_trace)
from molt.quantization import (dequantize_int4, dequantize_int8, quantize_int4,
                               quantize_int8, quantize_model_)


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def _cache(n_layers=4, T=10, H=2, D=8, hidden=16):
    geom = ModelGeometry(n_layers, hidden, 4, H, D, 100)
    layers = [(torch.randn(1, H, T, D), torch.randn(1, H, T, D)) for _ in range(n_layers)]
    meta = CacheMeta("t", geom, list(range(T)), [1], T)
    return MoltCache(layers, meta, {1: torch.randn(1, T, hidden)})


def test_cache_roundtrips_through_huggingface():
    c = _cache()
    hf = c.to_hf()
    assert hf.get_seq_length() == c.seq_len
    back = MoltCache.from_hf(hf, c.meta, c.hidden_traces)
    assert back.n_layers == c.n_layers
    for (a, b), (x, y) in zip(back.layers, c.layers):
        assert torch.equal(a, x) and torch.equal(b, y)


def test_crop_keeps_the_tail_and_records_the_offset():
    c = _cache(T=10)
    out = c.crop(4)
    assert out.seq_len == 4
    assert out.meta.pos_offset == 6, "cropping shifts absolute RoPE positions"
    assert out.meta.token_ids == [6, 7, 8, 9]
    assert torch.equal(out.layers[0][0], c.layers[0][0][..., 6:, :])
    assert out.hidden_traces[1].shape[1] == 4


def test_truncate_keeps_the_head():
    c = _cache(T=10)
    out = c.truncate(7)
    assert out.seq_len == 7 and out.meta.pos_offset == 0
    assert out.meta.token_ids == list(range(7))
    assert torch.equal(out.layers[0][0], c.layers[0][0][..., :7, :])


def test_byte_accounting_is_exact():
    c = _cache(n_layers=4, T=10, H=2, D=8, hidden=16)
    kv = 4 * 2 * (1 * 2 * 10 * 8) * 4          # layers * (k,v) * elems * fp32
    trace = 1 * 10 * 16 * 4
    assert c.nbytes(include_trace=False) == kv
    assert c.trace_bytes() == trace
    assert c.nbytes() == kv + trace


def test_geometry_bytes_per_token_matches_a_real_cache():
    c = _cache(n_layers=4, T=10, H=2, D=8)
    per_token = c.meta.geometry.bytes_per_token(torch.float32)
    assert per_token * 10 == c.nbytes(include_trace=False)


# --------------------------------------------------------------------------
# quantisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(64, 128), (17, 65)])
def test_int8_roundtrip_is_accurate(shape):
    w = torch.randn(*shape)
    q, s = quantize_int8(w)
    err = (w - dequantize_int8(q, s, torch.float32)).abs().max() / w.abs().max()
    assert q.dtype == torch.int8
    assert err < 0.02, f"int8 relative error {err:.4f}"


@pytest.mark.parametrize("shape", [(64, 128), (17, 100)])
def test_int4_roundtrip_is_accurate_and_packed(shape):
    w = torch.randn(*shape)
    packed, s, in_f = quantize_int4(w, group_size=32)
    assert packed.dtype == torch.uint8
    assert in_f == shape[1]
    out = dequantize_int4(packed, s, in_f, 32, torch.float32)
    assert out.shape == w.shape
    err = (w - out).abs().max() / w.abs().max()
    assert err < 0.15, f"int4 relative error {err:.4f}"


def test_quantising_a_model_shrinks_it_and_keeps_it_running(tmp_path):
    ladder = get_ladder("synthetic")
    zoo = ModelZoo(torch.device("cpu"), torch.float32)
    fp = zoo.acquire(ladder.by_name("tier0"))
    ids = torch.randint(0, 400, (1, 6))
    with torch.no_grad():
        ref = fp.model(ids).logits
    before = fp.mb

    q = zoo.acquire(ladder.by_name("tier1"))          # int8 sibling of tier0
    with torch.no_grad():
        got = q.model(ids).logits
    assert q.mb < before * 0.8, f"int8 rung is {q.mb:.1f} MiB vs fp {before:.1f} MiB"
    assert torch.isfinite(got).all()
    assert top1_agreement(ref[0, -1], got[0, -1]) == 1.0, (
        "int8 quantisation should not change the greedy token on a tiny model")
    assert q.kv_quant_stats is not None, "K/V quantisation stats should be collected"
    zoo.evict_all()


def test_lm_head_and_embeddings_are_not_quantised():
    """Tied weights would be corrupted for every rung sharing the tensor."""
    from molt.quantization import QuantLinear

    ladder = get_ladder("synthetic")
    zoo = ModelZoo(torch.device("cpu"), torch.float32)
    q = zoo.acquire(ladder.by_name("tier1"))
    assert not isinstance(q.model.lm_head, QuantLinear)
    zoo.evict_all()


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_divergences_behave():
    a = torch.randn(50)
    assert js_divergence(a, a) < 1e-9
    assert kl_divergence(a, a) < 1e-9
    b = torch.randn(50) * 5
    assert js_divergence(a, b) > 0
    assert abs(js_divergence(a, b) - js_divergence(b, a)) < 1e-9, "JSD is symmetric"
    assert js_divergence(a, b) <= math.log(2) + 1e-6, "JSD is bounded by ln 2"


def test_discontinuity_probe_isolates_the_migration():
    """``excess_jsd`` must be ~0 when the switch changes nothing."""
    probe = DiscontinuityProbe(window=4)
    torch.manual_seed(0)
    base = torch.randn(64) * 2
    for t in range(30):
        probe.observe(base + torch.randn(64) * 0.05)
        if t == 12:
            probe.mark_migration(t)
    assert abs(probe.excess_jsd()) < 0.01, probe.summary()

    probe2 = DiscontinuityProbe(window=4)
    other = torch.randn(64) * 2
    for t in range(30):
        src = base if t < 12 else other
        probe2.observe(src + torch.randn(64) * 0.05)
        if t == 12:
            probe2.mark_migration(t)
    assert probe2.excess_jsd() > probe.excess_jsd() + 0.05, (
        "a real switch must register as excess divergence")


def test_latency_recorder_percentiles_and_stalls():
    rec = LatencyRecorder()
    t = 0.0
    for i, dur in enumerate([0.01] * 20 + [0.5] + [0.01] * 9):
        rec.add(TokenRecord(i, t, t + dur, "tier0", i))
        t += dur
    assert rec.max_itl_ms == pytest.approx(500, rel=0.01)
    assert rec.pct_itl_ms(50) == pytest.approx(10, rel=0.01)
    stalls = rec.stalls(threshold_ms=100)
    assert len(stalls) == 1 and stalls[0][0] == 20
    assert rec.stall_time_ms(100) == pytest.approx(400, rel=0.01)


def test_repetition_and_distinct():
    assert repetition_rate([1, 2, 3, 1, 2, 3, 1, 2, 3], 3) > 0.5
    assert repetition_rate(list(range(20)), 3) == 0.0
    assert distinct_n([1, 1, 1, 1], 2) < 0.5


def test_flops_model_scales_with_depth():
    g = ModelGeometry(24, 896, 14, 2, 64, 151936)
    full = flops_per_token_prefill(g, 4864)
    part = flops_layer_range(g, 4864, 6)
    assert part == pytest.approx(full * 6 / 24, rel=1e-6)


# --------------------------------------------------------------------------
# pressure
# --------------------------------------------------------------------------


def test_trace_lookup_is_a_step_function():
    tr = PressureTrace("t", "", 10.0, [PressureEvent(0, 100, "a"),
                                       PressureEvent(5, 40, "b")])
    assert tr.budget_at(0) == 100 and tr.budget_at(4.9) == 100
    assert tr.budget_at(5) == 40 and tr.budget_at(99) == 40
    assert tr.label_at(6) == "b"
    assert tr.min_budget_mb == 40 and tr.max_budget_mb == 100


def test_trace_json_roundtrip(tmp_path):
    tr = builtin_traces(1000.0)["sawtooth"]
    p = tmp_path / "t.json"
    tr.save(str(p))
    back = PressureTrace.load(str(p))
    assert back.name == tr.name and len(back.events) == len(tr.events)
    assert back.budget_at(7) == tr.budget_at(7)


def test_shipped_traces_are_loadable_and_bite():
    """Every shipped trace must actually go below what the top rung needs."""
    import glob
    import os

    files = glob.glob("benchmarks/pressure_traces/*.json")
    assert files, "no pressure traces shipped"
    for f in files:
        tr = PressureTrace.load(f)
        assert tr.duration_s > 0 and tr.events
        if tr.name != "flat":
            assert tr.min_budget_mb < tr.max_budget_mb, f"{tr.name} never varies"


def test_virtual_clock_is_deterministic():
    tr = builtin_traces(1000.0)["spike_mid_answer"]
    src = TraceReplaySource(tr, virtual_step_s=1.0)
    src.start()
    seen = []
    for _ in range(20):
        seen.append(src.budget_mb())
        src.tick()
    src2 = TraceReplaySource(tr, virtual_step_s=1.0)
    src2.start()
    again = []
    for _ in range(20):
        again.append(src2.budget_mb())
        src2.tick()
    assert seen == again
    assert len(set(seen)) > 1, "the trace should change during 20 steps"
