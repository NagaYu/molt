"""Molt core #4 — **QoSScheduler**: several apps, one memory budget, no kills.

Three pseudo-apps share a device: one foreground conversation (priority 0) and
two background batch jobs (priorities 1 and 2).  The budget moves underneath
them.  The scheduler's contract is a hard invariant:

    **at no point may the tracked footprint exceed the budget, and no
    generation may be terminated to achieve that.**

It has four levers, applied in order of increasing harm, and it applies them
*before* stepping anyone — never after a violation:

1. **admission control** — a queued request waits until its rung fits.
2. **demotion** — the lowest-priority running generation is commanded down a
   rung.  Thanks to core #2 this is a mid-stream migration, so the demoted app
   keeps producing tokens; it does not restart and does not stall.
3. **pause** — the lowest-priority generation gives up its *weights* but keeps
   its KV cache, and resumes with full context when the budget recovers.
4. **demote the foreground too** — last resort, and still not a kill.

The ordering is the QoS policy: background work degrades before foreground work,
and *everything* degrades before anything dies.

Claims supported by this module
-------------------------------
* **zero-kill**: :meth:`QoSScheduler.enforce` is the proof obligation;
  ``result.kills == 0`` under every trace is the headline number.
* **no-stall**: demotion is preferred over pausing precisely because core #1+#2
  make it free of a restart, so a background job keeps streaming while shrinking.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .config import MoltConfig, TierLadder, TierSpec
from .metrics import EventLog, MemoryProbe, now
from .pressure import PressureSource
from .runtime import (Generation, GenerationRequest, GenerationResult,
                      MemoryBroker, MoltRuntime, OOMKilled, Policy)


@dataclass
class SchedulerConfig:
    """Policy knobs of the QoS layer."""

    #: fraction of the budget the scheduler refuses to allocate, absorbing
    #: transient allocator overshoot during a forward pass
    safety_margin: float = 0.10
    #: background job of priority p runs one step every ``1 << p`` rounds
    background_step_divisor: int = 2
    #: estimated KV MiB per token, per app (refined online from real caches)
    kv_mb_per_token_hint: float = 0.02
    #: allow pausing background jobs (lever 3)
    allow_pause: bool = True
    #: allow demoting the foreground app (lever 4)
    allow_foreground_demotion: bool = True
    #: climb back up when the budget recovers (elasticity must be symmetric)
    allow_promotion: bool = True
    #: only promote into this fraction of the budget, so a promotion does not
    #: immediately trigger the demotion that undoes it
    promotion_headroom: float = 0.75
    #: rounds a generation must be settled before it may be promoted
    promotion_cooldown: int = 12
    #: only resume a parked job into this fraction of the budget (hysteresis)
    resume_headroom: float = 0.80
    #: minimum rounds a job stays parked before it may be resumed
    min_pause_rounds: int = 6
    #: keep at most this many rounds before giving up (runaway guard)
    max_rounds: int = 100_000
    #: if nothing is running and nothing can be admitted for this many rounds,
    #: force the highest-priority request in on the cheapest rung (shedding
    #: context if necessary).  Deferring forever is a failure mode too — the
    #: no-kill invariant is about not losing work, not about doing none.
    max_defer_rounds: int = 24
    #: rounds to keep waiting when the budget is below even the cheapest rung.
    #: Waiting is correct: pressure traces recover, and parked work still holds
    #: its context.  This is only a runaway guard.
    max_stall_rounds: int = 400
    #: stop as soon as the pressure trace is exhausted
    stop_on_trace_end: bool = False


@dataclass
class AppSpec:
    """One pseudo-app: a request plus its QoS class."""

    request: GenerationRequest
    policy: Policy = Policy.MOLT
    start_tier: Optional[str] = None
    #: lower is more important; 0 = foreground conversation
    priority: int = 0

    @property
    def app_id(self) -> str:
        return self.request.app_id


@dataclass
class SchedulerReport:
    results: Dict[str, GenerationResult] = field(default_factory=dict)
    kills: int = 0
    admissions: List[dict] = field(default_factory=list)
    actions: List[dict] = field(default_factory=list)
    #: highest footprint observed at a *step boundary* — the number the no-kill
    #: invariant is about
    peak_mb: float = 0.0
    #: highest footprint observed at any instant, including the moment during a
    #: grouped demotion when two rungs are briefly resident.  Reported
    #: separately rather than hidden, because on a real device this transient is
    #: what an aggressive jetsam would see.
    peak_transient_mb: float = 0.0
    min_budget_mb: float = float("inf")
    rounds: int = 0
    #: rounds spent waiting because the budget was below the cheapest rung
    stalled_rounds: int = 0
    #: ``(elapsed_s, budget_mb, usage_before_enforce_mb, usage_after_enforce_mb)``
    budget_series: List[Tuple[float, float, float, float]] = field(default_factory=list)

    @property
    def zero_kill(self) -> bool:
        return self.kills == 0

    @property
    def max_overshoot_mb(self) -> float:
        """``max_t (usage(t) - budget(t))`` — the real invariant.

        Comparing a global peak against a global minimum budget would be
        meaningless: they happen at different instants.  What matters is whether
        usage ever exceeded the budget *at the same time*.
        """
        if not self.budget_series:
            return float("nan")
        return max(post - budget for _, budget, _pre, post in self.budget_series)

    def summary(self) -> Dict[str, Any]:
        return dict(
            kills=self.kills, zero_kill=self.zero_kill, rounds=self.rounds,
            peak_mb=self.peak_mb, peak_transient_mb=self.peak_transient_mb,
            min_budget_mb=self.min_budget_mb,
            max_overshoot_mb=self.max_overshoot_mb,
            min_headroom_mb=-self.max_overshoot_mb,
            n_demotions=sum(1 for a in self.actions if a["action"] == "demote"),
            n_promotions=sum(1 for a in self.actions if a["action"] == "promote"),
            n_context_sheds=sum(1 for a in self.actions if a["action"] == "shed_context"),
            stalled_rounds=self.stalled_rounds,
            n_pauses=sum(1 for a in self.actions if a["action"] == "pause"),
            n_resumes=sum(1 for a in self.actions if a["action"] == "resume"),
            n_deferred_admissions=sum(1 for a in self.admissions if not a["admitted"]),
            apps={k: v.summary() for k, v in self.results.items()},
        )


class QoSScheduler:
    """Round-robin, priority-weighted, budget-enforcing multi-tenant driver."""

    def __init__(
        self,
        cfg: MoltConfig,
        runtime: MoltRuntime,
        pressure: PressureSource,
        sched_cfg: Optional[SchedulerConfig] = None,
    ):
        self.cfg = cfg
        self.rt = runtime
        self.pressure = pressure
        self.scfg = sched_cfg or SchedulerConfig()
        self.ladder: TierLadder = cfg.ladder
        self.log: EventLog = runtime.log
        self.broker: MemoryBroker = runtime.broker
        self.report = SchedulerReport()

        self.pending: List[AppSpec] = []
        self.active: Dict[str, Generation] = {}
        self.specs: Dict[str, AppSpec] = {}
        self._prompt_lens: Dict[str, int] = {}
        self._round = 0
        self._starved_rounds = 0
        self._stalled_rounds = 0
        self._paused_at: Dict[str, int] = {}

    # -- budget helpers ----------------------------------------------------
    def budget_mb(self) -> float:
        return self.pressure.budget_mb()

    def usable_mb(self) -> float:
        """Budget minus the safety margin — what the scheduler will hand out."""
        return self.budget_mb() * (1.0 - self.scfg.safety_margin)

    def pressure_level(self) -> float:
        """0..1+ — usage relative to the usable budget."""
        u = self.usable_mb()
        return float("inf") if u <= 0 else self.broker.usage_mb() / u

    def _est_cache_mb(self, tier: TierSpec, n_tokens: int) -> float:
        per_token = self.rt.zoo.kv_mb_per_token(tier)
        if per_token is None:
            per_token = self.scfg.kv_mb_per_token_hint
        return per_token * n_tokens

    def _prompt_len(self, spec: AppSpec) -> int:
        """Token length of a request's prompt — the dominant term in its cache."""
        cached = self._prompt_lens.get(spec.app_id)
        if cached is None:
            try:
                cached = len(self.rt.tok(spec.request.prompt)["input_ids"][0])
            except Exception:
                cached = max(1, len(spec.request.prompt) // 4)
            self._prompt_lens[spec.app_id] = cached
        return cached

    # -- submission --------------------------------------------------------
    def submit(self, spec: AppSpec) -> None:
        self.specs[spec.app_id] = spec
        self.pending.append(spec)

    def _start_tier_for(self, spec: AppSpec) -> Optional[TierSpec]:
        """Best rung that fits right now — this is *admission control*.

        Returning ``None`` means "not now"; the request waits rather than being
        admitted into a footprint that would breach the budget.  Deferring is
        the only lever that can be applied *before* any memory is committed.
        """
        preferred = (self.ladder.by_name(spec.start_tier) if spec.start_tier
                     else self.ladder.top)
        budget = self.usable_mb()
        # the prompt dominates the cache: a 500-token prompt costs an order of
        # magnitude more than the 40 tokens the request will emit
        n_tokens = self._prompt_len(spec) + spec.request.max_new_tokens
        candidates = [t for t in self.ladder if t.tier >= preferred.tier]
        for tier in candidates:
            need = self.rt.zoo.incremental_mb(tier) + self._est_cache_mb(tier, n_tokens)
            if self.broker.would_fit(need, budget):
                return tier
        return None

    def _admit(self) -> None:
        forced_in = None
        if self.pending and not self.active:
            self._starved_rounds += 1
            if self._starved_rounds >= self.scfg.max_defer_rounds:
                forced_in = min(self.pending, key=lambda s: s.priority).app_id
                self.log.log("admission_forced", app=forced_in,
                             starved_rounds=self._starved_rounds,
                             budget_mb=self.usable_mb())
                self._starved_rounds = 0
        else:
            self._starved_rounds = 0

        still: List[AppSpec] = []
        for spec in self.pending:
            tier = self._start_tier_for(spec)
            if tier is None and spec.app_id == forced_in:
                tier = self.ladder.bottom      # last resort: cheapest rung
            if tier is None:
                self.report.admissions.append(dict(
                    app=spec.app_id, admitted=False, round=self._round,
                    budget_mb=self.usable_mb(), usage_mb=self.broker.usage_mb()))
                self.log.log("admission_deferred", app=spec.app_id,
                             budget_mb=self.usable_mb(), usage_mb=self.broker.usage_mb())
                still.append(spec)
                continue
            gen = self.rt.make_generation(
                spec.request, spec.policy, tier,
                pressure_fn=self.pressure_level, budget_fn=self.usable_mb,
                enforce_oom=True, self_migrate=False)
            try:
                gen.prefill()
            except OOMKilled as exc:
                # Should be unreachable: admission already proved it fits.
                self.report.kills += 1
                self.log.log("oom_kill", app=spec.app_id, phase="prefill",
                             usage_mb=exc.usage_mb, budget_mb=exc.budget_mb)
                gen.killed, gen.done, gen.kill_reason = True, True, str(exc)
                self.report.results[spec.app_id] = gen.result()
                gen.close()
                continue
            self.active[spec.app_id] = gen
            self.report.admissions.append(dict(
                app=spec.app_id, admitted=True, round=self._round, tier=tier.name,
                budget_mb=self.usable_mb(), usage_mb=self.broker.usage_mb()))
            self.log.log("admitted", app=spec.app_id, tier=tier.name,
                         priority=spec.priority)
        self.pending = still

    # -- the invariant -----------------------------------------------------
    def _by_priority(self, reverse: bool = True) -> List[Generation]:
        """Running generations, least important first when ``reverse``."""
        gens = list(self.active.values())
        gens.sort(key=lambda g: self.specs[g.req.app_id].priority, reverse=reverse)
        return gens

    def _act(self, action: str, gen: Generation, **detail) -> None:
        rec = dict(action=action, app=gen.req.app_id, round=self._round,
                   usage_mb=self.broker.usage_mb(), budget_mb=self.usable_mb(), **detail)
        self.report.actions.append(rec)
        self.log.log(f"qos_{action}", **rec)

    def _demotable(self, gen: Generation) -> bool:
        if gen.paused or gen.done or gen.cache is None:
            return False
        if gen.policy is Policy.STATIC:
            return False
        if (self.specs[gen.req.app_id].priority == 0
                and not self.scfg.allow_foreground_demotion):
            return False
        return self.ladder.next_down(gen.tier) is not None

    def _demote_group(self) -> bool:
        """Demote every generation sitting on the most expensive resident rung.

        Grouped on purpose: moving *one* of three apps off a shared rung frees
        nothing (the rung stays resident for the other two) while adding the
        destination rung's weights — usage would go **up**.  Demoting the whole
        group retires the expensive rung in one go, so the transient peak is
        bounded by ``old_rung + new_rung`` rather than growing with the number
        of tenants.
        """
        candidates = [g for g in self._by_priority(reverse=True) if self._demotable(g)]
        if not candidates:
            return False
        worst_tier = min(g.tier.tier for g in candidates)
        group = [g for g in candidates if g.tier.tier == worst_tier]
        # keep QoS order within the group: background demotes first
        for gen in group:
            nxt = self.ladder.next_down(gen.tier)
            if nxt is None:
                continue
            gen.forced_target = nxt
            gen.forced_reason = (f"QoS demotion: usage {self.broker.usage_mb():.0f} MiB "
                                 f"> budget {self.usable_mb():.0f} MiB")
            self._act("demote", gen, from_tier=gen.tier.name, to_tier=nxt.name,
                      priority=self.specs[gen.req.app_id].priority)
            gen.apply_forced_migration()   # synchronous: the budget cannot wait
        return True

    def enforce(self) -> None:
        """Bring usage under budget using the four levers, in order.

        This is the **zero-kill** proof obligation.  It runs *before* any
        generation steps and, crucially, applies each lever **synchronously** —
        a queued demotion that only takes effect on some other app's turn is not
        an enforcement mechanism, it is a hope.
        """
        guard = 0
        while self.broker.usage_mb() > self.usable_mb() and guard < 64:
            guard += 1

            # ---- lever 2: demote (background first, whole rung at a time) ---
            if self._demote_group():
                continue

            # ---- lever 3: park background work -----------------------------
            if self.scfg.allow_pause:
                parked = False
                for gen in self._by_priority(reverse=True):
                    if gen.paused or self.specs[gen.req.app_id].priority == 0:
                        continue
                    freed = gen.pause()
                    self._paused_at[gen.req.app_id] = self._round
                    self._act("pause", gen, freed_mb=freed, tier=gen.tier.name)
                    parked = True
                    break
                if parked:
                    continue

                # ---- lever 4: park the foreground too --------------------
                # Still not a kill: the conversation keeps its cache and resumes
                # with full context.  Only reached when the device genuinely
                # cannot hold the cheapest rung of anything.
                for gen in self._by_priority(reverse=False):
                    if gen.paused:
                        continue
                    freed = gen.pause()
                    self._paused_at[gen.req.app_id] = self._round
                    self._act("pause", gen, freed_mb=freed, tier=gen.tier.name,
                              last_resort=True)
                    parked = True
                    break
                if parked:
                    continue

            # ---- lever 5: shed context ---------------------------------
            # Everything is parked and the *caches* alone are over budget.
            # Give up the oldest history, least important tenant first.  This is
            # the last lever before the invariant would be broken, and it is
            # still recoverable work rather than lost work.
            over = self.broker.usage_mb() - self.usable_mb()
            shed = False
            for gen in self._by_priority(reverse=True):
                if over <= 0:
                    break
                before = self.broker.usage_mb()
                if gen.shed_context(over):
                    self._act("shed_context", gen, tier=gen.tier.name,
                              freed_mb=before - self.broker.usage_mb())
                    over = self.broker.usage_mb() - self.usable_mb()
                    shed = True
            if shed:
                continue
            break
        if guard >= 64:  # pragma: no cover - runaway guard
            self.log.log("enforce_gave_up", usage_mb=self.broker.usage_mb(),
                         budget_mb=self.usable_mb())

    def _maybe_promote(self) -> None:
        """Give slack back to the most important tenant first.

        The scheduler owns *both* directions of elasticity in multi-tenant mode.
        A demotion that is never undone would make one pressure spike a
        permanent quality tax on the device, so once the budget recovers the
        foreground conversation climbs back before the batch jobs do.
        """
        if not self.scfg.allow_promotion:
            return
        budget = self.usable_mb() * self.scfg.promotion_headroom
        for gen in self._by_priority(reverse=False):      # foreground first
            if gen.paused or gen.done or gen.cache is None or gen.forced_target:
                continue
            if self._round - gen.controller.state.last_migration_step < \
                    self.scfg.promotion_cooldown:
                continue
            better = [t for t in self.ladder.richer_than(gen.tier)
                      if self.broker.would_fit(self.rt.zoo.incremental_mb(t), budget)]
            if not better:
                continue
            target = better[0]
            gen.forced_target = target
            gen.forced_reason = "QoS promotion: budget recovered"
            self._act("promote", gen, from_tier=gen.tier.name, to_tier=target.name,
                      priority=self.specs[gen.req.app_id].priority)
            gen.apply_forced_migration()
            return   # one promotion per round: re-measure before doing more

    def _maybe_resume(self) -> None:
        """Bring paused work back when the budget allows — the *up* direction.

        Elasticity has to be symmetric, otherwise a single transient spike
        permanently degrades the device.
        """
        # Resume into *headroom*, not into the last free byte.  Resuming the
        # instant a rung nominally fits makes the very next enforce() pause it
        # again: the two levers oscillate at one round per cycle and the tenant
        # makes no progress.  Requiring slack is the scheduler's hysteresis.
        budget = self.usable_mb() * self.scfg.resume_headroom
        for gen in self._by_priority(reverse=False):
            if not gen.paused:
                continue
            if self._round - self._paused_at.get(gen.req.app_id, -10**9) \
                    < self.scfg.min_pause_rounds:
                continue
            need = self.rt.zoo.incremental_mb(gen.tier)
            if self.broker.would_fit(need, budget):
                gen.resume()
                self._act("resume", gen, tier=gen.tier.name)
                budget = self.usable_mb() * self.scfg.resume_headroom

    # -- the loop ----------------------------------------------------------
    def _should_step(self, gen: Generation) -> bool:
        if gen.paused or gen.done:
            return False
        prio = self.specs[gen.req.app_id].priority
        if prio == 0:
            return True
        divisor = max(1, self.scfg.background_step_divisor ** prio)
        return (self._round % divisor) == 0

    def run(self) -> SchedulerReport:
        self.pressure.start()
        try:
            while self._round < self.scfg.max_rounds:
                self._round += 1
                self.report.rounds = self._round

                budget = self.usable_mb()
                self.report.min_budget_mb = min(self.report.min_budget_mb, budget)
                usage_pre = self.broker.usage_mb()

                self.enforce()
                self._maybe_resume()
                self._maybe_promote()
                self._admit()
                self.report.budget_series.append(
                    (self.pressure.elapsed(), budget, usage_pre, self.broker.usage_mb()))

                if not self.active and not self.pending:
                    break
                if self.scfg.stop_on_trace_end and self.pressure.finished():
                    break

                progressed = False
                for app_id, gen in list(self.active.items()):
                    if not self._should_step(gen):
                        continue
                    progressed = True
                    try:
                        gen.step()
                    except OOMKilled as exc:  # pragma: no cover - invariant breach
                        gen.killed, gen.done, gen.kill_reason = True, True, str(exc)
                        self.log.log("oom_kill", app=app_id, usage_mb=exc.usage_mb,
                                     budget_mb=exc.budget_mb)
                    if gen.killed:
                        self.report.kills += 1
                    if gen.done:
                        self.report.results[app_id] = gen.result()
                        gen.close()
                        del self.active[app_id]

                if not progressed and not self.pending and self.active:
                    # No-one moved.  That is only a *deadlock* if everybody is
                    # paused — otherwise it is just a gap in the weighted
                    # round-robin and the next round will serve someone.
                    if all(g.paused for g in self.active.values()):
                        if self._break_deadlock():
                            self._stalled_rounds = 0
                        else:
                            # The budget is currently below even the cheapest
                            # rung: no arrangement of tenants can run.  The
                            # right answer is to *wait* — traces recover, and a
                            # service that discards parked work to avoid idling
                            # has broken the no-kill contract for no reason.
                            self._stalled_rounds += 1
                            if self._stalled_rounds >= self.scfg.max_stall_rounds:
                                self.log.log("gave_up_waiting",
                                             stalled_rounds=self._stalled_rounds,
                                             budget_mb=self.usable_mb())
                                break
                self._tick_pressure()
                self.rt.memory.sample(self.broker.usage_mb())
                self.report.peak_mb = max(self.report.peak_mb, self.broker.usage_mb())
        finally:
            self.pressure.stop()
            for app_id, gen in list(self.active.items()):
                self.report.results[app_id] = gen.result()
                gen.close()
            self.active.clear()
        self.report.peak_transient_mb = max(self.report.peak_mb, self.broker.peak_mb)
        self.report.stalled_rounds = self._stalled_rounds
        return self.report

    def _tick_pressure(self) -> None:
        """Advance a virtual-clock pressure source, if it uses one.

        Virtual time makes a scheduler run reproducible: the same trace bites at
        the same *round* on a fast and a slow machine, so the zero-kill test is
        not a race against the wall clock.
        """
        tick = getattr(self.pressure, "tick", None)
        if callable(tick):
            tick()

    def _break_deadlock(self) -> bool:
        """Everything is paused.  Resume the most important job on the cheapest
        rung that fits, rather than stalling forever or killing anyone."""
        budget = self.usable_mb()
        cheapest = self.ladder.bottom
        for gen in self._by_priority(reverse=False):
            if not gen.paused:
                continue
            for tier in sorted(self.ladder, key=lambda t: -t.tier):
                if self.broker.would_fit(self.rt.zoo.incremental_mb(tier), budget):
                    gen.resume(target=tier)
                    self._act("resume", gen, tier=gen.tier.name, deadlock_break=True)
                    return True

        # Nothing fits even on the cheapest rung: the *caches* are what is too
        # big.  Shed the oldest context from the least important tenants until
        # the most important one can run again.  Terminating is never the answer.
        need = (self.broker.usage_mb() + self.rt.zoo.incremental_mb(cheapest)) - budget
        for gen in self._by_priority(reverse=True):
            if need <= 0:
                break
            before = self.broker.usage_mb()
            if gen.shed_context(need):
                self._act("shed_context", gen, freed_mb=before - self.broker.usage_mb(),
                          tier=gen.tier.name)
                need = (self.broker.usage_mb()
                        + self.rt.zoo.incremental_mb(cheapest)) - budget
        for gen in self._by_priority(reverse=False):
            if gen.paused and self.broker.would_fit(
                    self.rt.zoo.incremental_mb(cheapest), budget):
                gen.resume(target=cheapest)
                self._act("resume", gen, tier=gen.tier.name, deadlock_break=True,
                          after_shed=True)
                return True

        self.log.log("deadlock", budget_mb=budget, usage_mb=self.broker.usage_mb())
        return False
