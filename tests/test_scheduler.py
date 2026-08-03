"""Core #4 — QoSScheduler.

Claims under test
-----------------
* **zero-kill** — under every shipped pressure trace, with three tenants sharing
  one budget, nothing is terminated *and* the budget is never exceeded.  Both
  halves matter: surviving by ignoring the budget would prove nothing.
* QoS ordering — background work degrades before foreground work.
* symmetry — parked and demoted work recovers when the budget does.
"""

from __future__ import annotations

import pytest
import torch

from molt.config import CalibrationConfig, MigrationConfig
from molt.metrics import EventLog
from molt.pressure import (PressureEvent, PressureTrace, TraceReplaySource,
                           builtin_traces)
from molt.runtime import GenerationRequest, Policy
from molt.scheduler import AppSpec, QoSScheduler, SchedulerConfig

from .conftest import PROMPT, make_config

TRACE_STEP = 0.6


def _apps(n_tokens=24):
    return [
        AppSpec(GenerationRequest("fg-chat", PROMPT, max_new_tokens=n_tokens,
                                  priority=0, kind="chat"), Policy.MOLT, "tier0", 0),
        AppSpec(GenerationRequest("bg-1", PROMPT + " one", max_new_tokens=n_tokens,
                                  priority=1, kind="batch"), Policy.MOLT, "tier0", 1),
        AppSpec(GenerationRequest("bg-2", PROMPT + " two", max_new_tokens=n_tokens,
                                  priority=2, kind="batch"), Policy.MOLT, "tier0", 2),
    ]


def _run(rt, ladder, trace, n_tokens=24, scfg=None):
    src = TraceReplaySource(trace, virtual_step_s=TRACE_STEP)
    sched = QoSScheduler(rt.cfg, rt, src,
                         scfg or SchedulerConfig(safety_margin=0.05))
    for a in _apps(n_tokens):
        sched.submit(a)
    return sched.run()


@pytest.fixture
def scheduler_runtime(runtime_factory, ladder):
    rt = runtime_factory(max_new_tokens=24,
                         calibration=CalibrationConfig(mode="dual", blend_steps=3),
                         migration=MigrationConfig(down_threshold=0.85,
                                                   up_threshold=0.5,
                                                   cooldown_steps=3,
                                                   up_shift_patience=6))
    # let the scheduler budget against real sizes and real cache geometry
    for spec in ladder:
        rt.zoo.measure_footprint(spec)
    rt.zoo.evict_all()
    return rt


@pytest.mark.parametrize("trace_name", ["spike_mid_answer", "sawtooth", "staircase"])
def test_zero_kills_and_no_overshoot(scheduler_runtime, ladder, trace_name):
    """The headline invariant, on every shipped trace."""
    top_mb = scheduler_runtime.zoo.known_mb(ladder.top)
    trace = builtin_traces(top_tier_mb=top_mb)[trace_name]
    rep = _run(scheduler_runtime, ladder, trace)

    assert rep.kills == 0, f"{trace_name}: {rep.kills} forced terminations"
    assert rep.max_overshoot_mb <= 1e-6, (
        f"{trace_name}: budget exceeded by {rep.max_overshoot_mb:.2f} MiB")
    assert rep.results, "no app produced a result"
    for app_id, res in rep.results.items():
        assert not res.killed, f"{trace_name}/{app_id}: {res.kill_reason}"
        assert res.token_ids, f"{trace_name}/{app_id}: produced nothing"


def test_all_tenants_finish_under_a_spike(scheduler_runtime, ladder):
    """Degrading is not the same as dropping: everyone still completes."""
    top_mb = scheduler_runtime.zoo.known_mb(ladder.top)
    trace = builtin_traces(top_tier_mb=top_mb)["spike_mid_answer"]
    rep = _run(scheduler_runtime, ladder, trace, n_tokens=20)
    assert set(rep.results) == {"fg-chat", "bg-1", "bg-2"}
    for app_id, res in rep.results.items():
        assert len(res.token_ids) == 20, (
            f"{app_id} produced {len(res.token_ids)}/20 tokens")


def test_background_work_degrades_before_the_foreground(scheduler_runtime, ladder):
    """The QoS ordering is the policy, not an accident of iteration order."""
    top_mb = scheduler_runtime.zoo.known_mb(ladder.top)
    trace = builtin_traces(top_tier_mb=top_mb)["staircase"]
    rep = _run(scheduler_runtime, ladder, trace, n_tokens=20)

    pauses = [a for a in rep.actions if a["action"] == "pause"]
    if pauses:
        first = pauses[0]
        assert first["app"] != "fg-chat" or first.get("last_resort"), (
            "the foreground was parked before any background job")

    fg = rep.results["fg-chat"]
    bgs = [rep.results[k] for k in ("bg-1", "bg-2")]
    fg_top = fg.summary()["frac_tokens_on_top_tier"]
    bg_top = sum(b.summary()["frac_tokens_on_top_tier"] for b in bgs) / len(bgs)
    assert fg_top >= bg_top - 1e-9, (
        f"foreground spent {fg_top:.2f} of its tokens on the top rung, "
        f"background averaged {bg_top:.2f}")


def test_recovery_is_symmetric(scheduler_runtime, ladder):
    """After the spike passes, parked or demoted work comes back."""
    top_mb = scheduler_runtime.zoo.known_mb(ladder.top)
    trace = builtin_traces(top_tier_mb=top_mb)["spike_mid_answer"]
    rep = _run(scheduler_runtime, ladder, trace, n_tokens=28)
    s = rep.summary()
    recovered = s["n_resumes"] + s["n_promotions"]
    degraded = s["n_pauses"] + s["n_demotions"]
    if degraded:
        assert recovered > 0, (
            f"{degraded} degradations but no recovery — elasticity is one-way")


def test_admission_control_defers_instead_of_overcommitting(scheduler_runtime, ladder):
    """A request that does not fit waits; it is never admitted into an OOM."""
    top_mb = scheduler_runtime.zoo.known_mb(ladder.top)
    tight = PressureTrace(
        "tight", "budget below two rungs for a long while", 90.0,
        [PressureEvent(0.0, top_mb * 0.35, "already squeezed"),
         PressureEvent(60.0, top_mb * 3.0, "released")])
    rep = _run(scheduler_runtime, ladder, tight, n_tokens=12)
    assert rep.kills == 0
    assert rep.max_overshoot_mb <= 1e-6
    assert rep.summary()["n_deferred_admissions"] > 0, (
        "nothing was deferred — the budget was not actually binding")


def test_no_deadlock_when_nothing_fits(scheduler_runtime, ladder):
    """Even a budget below the cheapest rung + full contexts makes progress.

    Context shedding is the last lever; refusing to run forever is a failure too.
    """
    bottom_mb = scheduler_runtime.zoo.known_mb(ladder.bottom)
    brutal = PressureTrace(
        "brutal", "barely above the cheapest rung", 120.0,
        [PressureEvent(0.0, bottom_mb * 1.6, "brutal")])
    rep = _run(scheduler_runtime, ladder, brutal, n_tokens=10,
               scfg=SchedulerConfig(safety_margin=0.02, max_rounds=4000,
                                    max_defer_rounds=8))
    assert rep.kills == 0
    produced = sum(len(r.token_ids) for r in rep.results.values())
    assert produced > 0, "the device made no progress at all"


def test_scheduler_is_the_only_migration_authority(scheduler_runtime, ladder):
    """Tenants do not self-demote behind the scheduler's back.

    Three tenants each reacting to *global* pressure would each load a
    destination rung while the others still hold the source; the transient sum is
    what breaks the invariant.
    """
    top_mb = scheduler_runtime.zoo.known_mb(ladder.top)
    trace = builtin_traces(top_tier_mb=top_mb)["sawtooth"]
    src = TraceReplaySource(trace, virtual_step_s=TRACE_STEP)
    sched = QoSScheduler(scheduler_runtime.cfg, scheduler_runtime, src,
                         SchedulerConfig(safety_margin=0.05))
    for a in _apps(16):
        sched.submit(a)
    sched._admit()
    assert sched.active, "nothing was admitted"
    assert all(not g.self_migrate for g in sched.active.values())
    for g in list(sched.active.values()):
        g.close()
