"""Core #2 — MidStreamMigration, and the runtime loop that carries it.

Claims under test
-----------------
* **no-stall**   — the token stream continues across a tier change; the step in
  which a migration happens still emits a token, and every token index is
  accounted for.
* **zero-kill**  — a Molt generation survives a pressure trace that reclaims a
  static-large one.
* bidirectionality and anti-thrash — up-shifts happen when pressure recedes, and
  the controller does not oscillate.
"""

from __future__ import annotations

import pytest
import torch

from molt.config import (CalibrationConfig, MigrationConfig, TransplantConfig)
from molt.migration import (MigrationController, MigrationDecision,
                            RequestBoundaryController)
from molt.runtime import GenerationRequest, OOMKilled, Policy

from .conftest import PROMPT, RECOMPUTE_K, make_config


# --------------------------------------------------------------------------
# the policy, in isolation (no models involved)
# --------------------------------------------------------------------------


def test_controller_is_bidirectional(ladder):
    cfg = MigrationConfig(down_threshold=0.8, up_threshold=0.5,
                          cooldown_steps=0, up_shift_patience=3)
    c = MigrationController(ladder, cfg)
    down = c.decide(ladder.top, pressure=0.95)
    assert down and down.direction == "down"

    c.note_migration(down, ladder.top.name, 0, 0.0)
    for _ in range(5):
        c.observe(0.2)
        c.tick()
    up = c.decide(ladder.bottom, pressure=0.2)
    assert up and up.direction == "up", up.reason


def test_controller_respects_hysteresis_and_cooldown(ladder):
    cfg = MigrationConfig(down_threshold=0.9, up_threshold=0.4,
                          cooldown_steps=5, up_shift_patience=3)
    c = MigrationController(ladder, cfg)
    assert not c.decide(ladder.top, pressure=0.7), "inside the band: no move"

    d = c.decide(ladder.top, pressure=0.95)
    c.note_migration(d, ladder.top.name, 0, 0.0)
    c.tick()
    # cooldown blocks a *non-critical* follow-up …
    assert not c.decide(ladder[1], pressure=0.92 - 0.0)
    # … but never blocks survival
    crit = c.decide(ladder[1], pressure=1.4)
    assert crit and crit.critical, "cooldown must not outrank an over-budget reading"


def test_controller_descends_multiple_rungs_when_one_is_not_enough(ladder):
    """A single pressure event can remove more memory than one rung's worth."""
    cfg = MigrationConfig(down_threshold=0.8, cooldown_steps=0)
    only_bottom_fits = lambda t: t.name == ladder.bottom.name
    c = MigrationController(ladder, cfg, affordable=only_bottom_fits)
    d = c.decide(ladder.top, pressure=1.5)
    assert d and d.target.name == ladder.bottom.name, d.reason


def test_scheduler_command_overrides_the_pressure_policy(ladder):
    c = MigrationController(ladder, MigrationConfig(cooldown_steps=999))
    d = c.decide(ladder.top, pressure=0.0, forced_target=ladder.bottom,
                 forced_reason="qos")
    assert d and d.forced and d.target.name == ladder.bottom.name


def test_request_boundary_controller_cannot_act_mid_stream(ladder):
    """The measurement behind "request-boundary switching is not enough"."""
    c = RequestBoundaryController(ladder, MigrationConfig(down_threshold=0.5))
    assert c.decide(ladder.top, pressure=0.99), "may choose at the boundary"
    c.tick()
    assert not c.decide(ladder.top, pressure=0.99), "must be stuck mid-stream"


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def _pressure_window(state, lo, hi, high=0.95, low=0.15):
    return lambda: high if lo <= state["i"] < hi else low


def test_generation_continues_across_migrations(runtime_factory, ladder):
    """**no-stall.**  Tokens keep coming, and the tier really did change.

    The assertion that matters is not merely "it did not crash": every token
    index from 0..N-1 is present, the stream visits more than one rung, and the
    token produced *at* the migration index exists.
    """
    rt = runtime_factory(max_new_tokens=36)
    state = {"i": 0}
    req = GenerationRequest("t", PROMPT, max_new_tokens=36)
    gen = rt.make_generation(req, Policy.MOLT, ladder.top,
                             _pressure_window(state, 5, 22),
                             lambda: float("inf"))
    res = rt.run(gen, tick=lambda: state.__setitem__("i", state["i"] + 1))

    assert not res.killed
    assert len(res.token_ids) == 36, "the stream must not be truncated"
    assert len(res.tier_per_token) == 36
    assert len(set(res.tier_per_token)) > 1, "no migration happened at all"
    assert res.migrations, "no migration recorded"
    for m in res.migrations:
        idx = m["token_index"]
        assert 0 <= idx < 36
        assert res.tier_per_token[idx] == m["to_tier"], (
            "the token emitted in the migrating step must come from the new rung")
    assert [t.index for t in res.latency.tokens] == list(range(36))


def test_up_shift_returns_to_the_top_rung(runtime_factory, ladder):
    """Elasticity is symmetric: pressure recedes, quality comes back."""
    cfg = make_config(ladder, "artifacts/proj_test",
                      migration=MigrationConfig(down_threshold=0.85, up_threshold=0.5,
                                                cooldown_steps=2, up_shift_patience=4),
                      max_new_tokens=48)
    cfg.transplant.projector_dir = None  # replaced below
    rt = runtime_factory(max_new_tokens=48,
                         migration=MigrationConfig(down_threshold=0.85, up_threshold=0.5,
                                                   cooldown_steps=2, up_shift_patience=4))
    state = {"i": 0}
    req = GenerationRequest("t", PROMPT, max_new_tokens=48)
    gen = rt.make_generation(req, Policy.MOLT, ladder.top,
                             _pressure_window(state, 4, 18),
                             lambda: float("inf"))
    res = rt.run(gen, tick=lambda: state.__setitem__("i", state["i"] + 1))
    dirs = [m["direction"] for m in res.migrations]
    assert "down" in dirs and "up" in dirs, dirs
    assert res.tier_per_token[-1] == ladder.top.name, (
        f"should have climbed back to {ladder.top.name}, ended on "
        f"{res.tier_per_token[-1]}")


def test_static_large_dies_where_molt_survives(runtime_factory, ladder):
    """**zero-kill**, stated as a contrast rather than in isolation.

    Same budget, same prompt, same trace: condition A is reclaimed, condition D
    finishes.  A test that only checked "Molt survives" could pass with a budget
    that was never actually binding.
    """
    def run(policy, start):
        rt = runtime_factory(max_new_tokens=30)
        state = {"i": 0}
        top_mb = rt.zoo.measure_footprint(ladder.top)
        budget = lambda: (top_mb * 0.5) if state["i"] >= 6 else (top_mb * 4)
        req = GenerationRequest("t", PROMPT, max_new_tokens=30)
        gen = rt.make_generation(req, policy, start,
                                 lambda: rt.broker.pressure(budget()), budget)
        return rt.run(gen, tick=lambda: state.__setitem__("i", state["i"] + 1))

    a = run(Policy.STATIC, ladder.top)
    d = run(Policy.MOLT, ladder.top)
    assert a.killed, "the budget was not actually binding — condition A survived"
    assert not d.killed, f"Molt was killed: {d.kill_reason}"
    assert len(d.token_ids) == 30


def test_no_kills_under_every_builtin_trace(runtime_factory, ladder):
    """**zero-kill** under the shipped pressure traces, not a hand-picked one."""
    from molt.pressure import TraceReplaySource, builtin_traces

    rt0 = runtime_factory()
    top_mb = rt0.zoo.measure_footprint(ladder.top)
    for name, trace in builtin_traces(top_tier_mb=top_mb).items():
        rt = runtime_factory(max_new_tokens=28)
        src = TraceReplaySource(trace, virtual_step_s=1.5)
        src.start()
        req = GenerationRequest(f"t-{name}", PROMPT, max_new_tokens=28)
        gen = rt.make_generation(req, Policy.MOLT, ladder.top,
                                 lambda: rt.broker.pressure(src.budget_mb()),
                                 src.budget_mb)
        res = rt.run(gen, tick=src.tick)
        src.stop()
        assert not res.killed, f"trace {name}: {res.kill_reason}"
        assert len(res.token_ids) == 28, f"trace {name}: stream truncated"


def test_restart_policy_pays_a_bigger_stall_than_molt(runtime_factory, ladder):
    """**low migration cost**, measured through the runtime rather than the operator.

    Both conditions migrate at the same token under the same trace; the only
    difference is whether the cache is carried or thrown away.
    """
    def run(policy):
        rt = runtime_factory(max_new_tokens=30)
        for spec in ladder:
            rt.zoo.acquire(spec)          # warm: isolate KV work from model loads
        state = {"i": 0}
        req = GenerationRequest("t", PROMPT, max_new_tokens=30)
        gen = rt.make_generation(req, policy, ladder.top,
                                 _pressure_window(state, 6, 30), lambda: float("inf"))
        res = rt.run(gen, tick=lambda: state.__setitem__("i", state["i"] + 1))
        for spec in ladder:
            rt.zoo.release(spec)
        return res

    molt, restart = run(Policy.MOLT), run(Policy.RESTART)
    assert molt.migrations and restart.migrations
    m_ms = sum(t.wall_ms for t in molt.transplants)
    r_ms = sum(t.wall_ms for t in restart.transplants)
    assert m_ms < r_ms, f"molt {m_ms:.1f} ms vs restart {r_ms:.1f} ms"
    assert molt.latency.max_itl_ms < restart.latency.max_itl_ms, (
        f"worst token: molt {molt.latency.max_itl_ms:.1f} ms vs "
        f"restart {restart.latency.max_itl_ms:.1f} ms")


def test_context_is_shed_rather_than_the_process(runtime_factory, ladder):
    """Last-resort degradation: history is dropped, the generation is not.

    Under a budget below what even the smallest rung plus a full cache needs,
    the runtime crops the oldest positions instead of being reclaimed.
    """
    rt = runtime_factory(max_new_tokens=24)
    bottom_mb = rt.zoo.measure_footprint(ladder.bottom)
    per_token = rt.zoo.kv_mb_per_token(ladder.bottom) or 0.01
    # room for the cheapest rung plus ~64 tokens of context: a long prompt must
    # be trimmed to fit, but the target is reachable
    tight = bottom_mb + per_token * 64
    state = {"i": 0}
    budget = lambda: tight if state["i"] >= 4 else 10_000.0
    req = GenerationRequest("t", PROMPT * 3, max_new_tokens=24)
    gen = rt.make_generation(req, Policy.MOLT, ladder.top,
                             lambda: rt.broker.pressure(budget()), budget)
    res = rt.run(gen, tick=lambda: state.__setitem__("i", state["i"] + 1))
    assert not res.killed, res.kill_reason
    assert len(res.token_ids) == 24
    assert rt.log.of_kind("context_shed"), "expected the context-shedding lever to fire"
