"""Core #3 — MigrationCalibration.

Claims under test
-----------------
* **continuity** — blending across the seam reduces the distribution jump at a
  migration relative to switching cold, and the reduction is measured with the
  same probe the benchmark reports (``excess_jsd``).
* the cheap mode (``frozen_ref``) works without a second resident model, so
  continuity never has to be bought with memory during a pressure event.
"""

from __future__ import annotations

import math

import pytest
import torch

from molt.calibration import (MigrationCalibrator, blend_weight, entropy_of,
                              mix_logits, temperature_for_entropy)
from molt.config import CalibrationConfig, MigrationConfig
from molt.metrics import js_divergence
from molt.runtime import GenerationRequest, Policy

from .conftest import PROMPT


# --------------------------------------------------------------------------
# the pieces
# --------------------------------------------------------------------------


@pytest.mark.parametrize("schedule", ["linear", "cosine", "exp"])
def test_blend_schedules_start_at_one_and_reach_zero(schedule):
    n = 8
    w = [blend_weight(i, n, schedule) for i in range(n)]
    assert w[0] > 0.5, w
    assert abs(w[-1]) < 1e-9, w
    assert all(a >= b - 1e-12 for a, b in zip(w, w[1:])), f"not monotone: {w}"


def test_mixture_is_a_distribution_and_interpolates():
    a = torch.randn(64) * 3
    b = torch.randn(64) * 3
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = torch.softmax(mix_logits(a, b, w), dim=-1)
        assert abs(float(p.sum()) - 1.0) < 1e-5
    assert torch.allclose(torch.softmax(mix_logits(a, b, 0.0), -1),
                          torch.softmax(a, -1), atol=1e-6)
    assert torch.allclose(torch.softmax(mix_logits(a, b, 1.0), -1),
                          torch.softmax(b, -1), atol=1e-6)
    # the mixture sits between the two endpoints
    mid = mix_logits(a, b, 0.5)
    assert js_divergence(mid, a) < js_divergence(b, a) + 1e-9


def test_temperature_search_hits_the_target_entropy():
    z = torch.randn(2048) * 2.0
    for target in (2.0, 4.0, 6.0):
        tau = temperature_for_entropy(z, target)
        got = entropy_of(z / tau)
        assert abs(got - target) < 0.05, f"target {target}, got {got} at tau {tau}"


def test_dual_mode_degrades_when_memory_is_critical():
    """Continuity must never *require* two resident models under pressure."""
    cal = MigrationCalibrator(CalibrationConfig(mode="dual", critical_mode="frozen_ref"))
    assert cal.effective_mode(0.4, critical=False) == "dual"
    assert cal.effective_mode(1.2, critical=True) == "frozen_ref"
    cal.begin(pressure=1.2, critical=True)
    assert not cal.needs_old_model, "critical blending must not pin the old model"


def test_frozen_ref_moves_entropy_towards_the_outgoing_model():
    cal = MigrationCalibrator(CalibrationConfig(mode="frozen_ref", blend_steps=4))
    torch.manual_seed(0)
    old = torch.randn(512) * 0.5          # high entropy
    new = torch.randn(512) * 4.0          # low entropy
    for _ in range(8):
        cal.observe(old)
    cal.begin(pressure=0.9, critical=False)
    out = cal.apply(new)
    h_target, h_raw, h_out = entropy_of(old), entropy_of(new), entropy_of(out)
    assert h_raw < h_out <= h_target + 1e-3, (
        f"blended entropy {h_out:.3f} should sit between the incoming model's "
        f"{h_raw:.3f} and the outgoing model's {h_target:.3f}")


def test_none_mode_is_a_true_passthrough():
    cal = MigrationCalibrator(CalibrationConfig(mode="none"))
    cal.begin(pressure=1.0)
    z = torch.randn(32)
    assert torch.equal(cal.apply(z), z)
    assert not cal.state.active


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


def _run(rt, ladder, n=40, hi=(5, 40)):
    state = {"i": 0}
    req = GenerationRequest("t", PROMPT, max_new_tokens=n)
    gen = rt.make_generation(
        req, Policy.MOLT, ladder.top,
        lambda: 0.95 if hi[0] <= state["i"] < hi[1] else 0.15,
        lambda: float("inf"))
    return rt.run(gen, tick=lambda: state.__setitem__("i", state["i"] + 1))


def _simulated_streams(n_steps=40, migrate_at=16, vocab=256, seed=0):
    """Two model "voices" over the same token positions, with a hard switch.

    Deliberately synthetic: the randomly-initialised ladder used elsewhere in
    this suite produces next-token distributions that barely depend on the cache,
    so an end-to-end run there has *no* discontinuity to smooth and could not
    distinguish a working calibrator from a no-op.  Here the jump is known by
    construction, which is what makes the comparison meaningful.
    """
    g = torch.Generator().manual_seed(seed)
    base_a = torch.randn(vocab, generator=g) * 3.0
    base_b = torch.randn(vocab, generator=g) * 3.0
    drift = 0.12
    old, new = [], []
    for t in range(n_steps):
        jitter_a = torch.randn(vocab, generator=g) * drift
        jitter_b = torch.randn(vocab, generator=g) * drift
        old.append(base_a + jitter_a)
        new.append(base_b + jitter_b)
    return old, new, migrate_at


def _excess_jsd_for(mode, blend_steps=6, **kw):
    """Replay the simulated streams through the calibrator + the real probe."""
    from molt.metrics import DiscontinuityProbe

    old, new, m = _simulated_streams(**kw)
    cal = MigrationCalibrator(CalibrationConfig(mode=mode, blend_steps=blend_steps,
                                                schedule="cosine"))
    probe = DiscontinuityProbe(window=blend_steps + 2)
    for t in range(len(old)):
        if t < m:
            emitted = old[t]
            cal.observe(emitted)
        else:
            if t == m:
                cal.begin(pressure=0.9, critical=False)
                probe.mark_migration(t)
            emitted = cal.apply(new[t], old[t] if mode == "dual" else None)
            cal.advance()
            cal.observe(new[t])
        probe.observe(emitted)
    return probe.excess_jsd(), probe.peak_jsd_at_migration()


@pytest.mark.parametrize("mode", ["dual", "frozen_ref"])
def test_calibration_reduces_the_distribution_jump(mode):
    """**continuity.**  Blended migrations are smoother than cold ones.

    ``excess_jsd`` is the step-to-step divergence around the switch *minus* the
    stream's own steady-state drift, so a merely noisier arm does not win by
    accident.  Compared against ``mode="none"`` — the ablation the README reports.
    """
    on_x, on_peak = _excess_jsd_for(mode)
    off_x, off_peak = _excess_jsd_for("none")
    assert off_x > 0.05, f"the control arm must actually jump (got {off_x:.4f})"
    assert on_x < off_x, (
        f"calibration={mode} excess JSD {on_x:.5f} should be below "
        f"the uncalibrated {off_x:.5f}")
    assert on_peak < off_peak, (
        f"peak jump at the seam: calibrated {on_peak:.5f} vs raw {off_peak:.5f}")


def test_longer_blends_are_smoother():
    """More blend steps buy more continuity — the knob does what it says."""
    xs = [_excess_jsd_for("dual", blend_steps=b)[0] for b in (2, 6, 12)]
    assert xs[0] > xs[1] > xs[2], f"excess JSD should fall with blend length: {xs}"


def test_cosine_schedule_beats_a_hard_cut_at_the_seam():
    """The schedule matters, not just the presence of a blend."""
    none_peak = _excess_jsd_for("none")[1]
    dual_peak = _excess_jsd_for("dual", blend_steps=8)[1]
    assert dual_peak < none_peak * 0.8, (
        f"cosine blend peak {dual_peak:.5f} vs hard cut {none_peak:.5f}")


def test_calibration_window_is_recorded_and_bounded(runtime_factory, ladder):
    """The blend really runs, for the configured number of tokens, and stops."""
    rt = runtime_factory(
        calibration=CalibrationConfig(mode="dual", blend_steps=5),
        migration=MigrationConfig(down_threshold=0.85, cooldown_steps=8,
                                  up_shift_patience=99),
        max_new_tokens=36)
    res = _run(rt, ladder, n=36)
    blending = [t.blending for t in res.latency.tokens]
    assert any(blending), "no token was emitted during a blend window"
    # a blend never outlives its budget
    run_len, longest = 0, 0
    for b in blending:
        run_len = run_len + 1 if b else 0
        longest = max(longest, run_len)
    assert longest <= 5 + 1, f"blend ran for {longest} tokens, budget was 5"


def test_dual_blend_releases_the_second_model(runtime_factory, ladder):
    """The outgoing model is not kept alive past the blend window."""
    rt = runtime_factory(
        calibration=CalibrationConfig(mode="dual", blend_steps=4),
        migration=MigrationConfig(down_threshold=0.85, cooldown_steps=10,
                                  up_shift_patience=99),
        max_new_tokens=30)
    res = _run(rt, ladder, n=30)
    assert res.migrations
    resident_after = [t.resident_mb for t in res.latency.tokens[-5:]]
    peak = max(t.resident_mb for t in res.latency.tokens)
    assert min(resident_after) < peak, (
        "footprint should fall back after the blend window closes")
