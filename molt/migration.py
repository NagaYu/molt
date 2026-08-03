"""Molt core #2 — **MidStreamMigration**: change tier between two tokens.

The decision layer.  It answers one question every decode step: *given the
current memory pressure, should this generation still be running on this rung?*
— and it is deliberately **bidirectional**: when pressure recedes the generation
climbs back up, so a transient spike does not condemn a long answer to the
smallest model for the rest of its life.

The mechanics of actually moving (transplant, blend, resume) live in
:mod:`molt.runtime`; this module only produces :class:`MigrationDecision`
objects, which makes the policy trivially testable without loading a model.

Anti-thrash is three mechanisms, not one:

* **hysteresis** — separate ``down_threshold`` / ``up_threshold``;
* **cooldown** — a floor on steps between two migrations;
* **patience** — an up-shift needs the low-pressure reading to persist.

Claims supported by this module
-------------------------------
* **no-stall**: the decision is taken *inside* the token loop, so the response
  to pressure is one token away rather than one request away.  This is the
  precise sense in which request-boundary switching is insufficient — see the
  README figure ``request_boundary_gap``.
* **zero-kill**: the controller is also driven by the QoS scheduler's commands,
  so a forced demotion always has a place to land.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .config import MigrationConfig, TierLadder, TierSpec


@dataclass
class MigrationDecision:
    """What the controller wants to happen before the next token."""

    target: Optional[TierSpec]
    reason: str
    direction: str = "none"   # "down" | "up" | "none"
    forced: bool = False      # issued by the QoS scheduler rather than by pressure
    critical: bool = False    # pressure is high enough to forbid dual-model blending

    def __bool__(self) -> bool:
        return self.target is not None


@dataclass
class MigrationState:
    """Per-generation bookkeeping used to damp oscillation."""

    step: int = 0
    last_migration_step: int = -10_000
    low_pressure_run: int = 0
    high_pressure_run: int = 0
    history: List[dict] = field(default_factory=list)

    @property
    def n_migrations(self) -> int:
        return len(self.history)


class MigrationController:
    """Pressure-driven, hysteretic, bidirectional tier policy."""

    def __init__(self, ladder: TierLadder, cfg: MigrationConfig,
                 affordable: Optional[Callable[[TierSpec], bool]] = None):
        self.ladder = ladder
        self.cfg = cfg
        self.state = MigrationState()
        #: injected by the runtime: "would this rung fit in the current budget?"
        #: Without it the controller can only ever step one rung at a time, which
        #: is not enough when a single pressure event removes more memory than
        #: one rung's worth — the generation would be reclaimed while politely
        #: descending one step per cooldown period.
        self.affordable = affordable or (lambda _t: True)

    # -- helpers -----------------------------------------------------------
    def _cooling_down(self) -> bool:
        return (self.state.step - self.state.last_migration_step) < self.cfg.cooldown_steps

    def observe(self, pressure: float) -> None:
        """Update the run-length counters that ``patience`` is based on."""
        if pressure <= self.cfg.up_threshold:
            self.state.low_pressure_run += 1
            self.state.high_pressure_run = 0
        elif pressure >= self.cfg.down_threshold:
            self.state.high_pressure_run += 1
            self.state.low_pressure_run = 0
        else:
            self.state.low_pressure_run = 0
            self.state.high_pressure_run = 0

    # -- the policy --------------------------------------------------------
    def decide(
        self,
        current: TierSpec,
        pressure: float,
        forced_target: Optional[TierSpec] = None,
        forced_reason: str = "",
    ) -> MigrationDecision:
        """Return the tier this generation should be on for the next token.

        ``forced_target`` comes from :class:`~molt.scheduler.QoSScheduler` and
        overrides the pressure policy — the scheduler owns the global no-kill
        invariant and this controller must not fight it.
        """
        critical = pressure >= 1.0 or pressure >= self.cfg.down_threshold + 0.1

        if forced_target is not None and forced_target.name != current.name:
            direction = "down" if forced_target.tier > current.tier else "up"
            return MigrationDecision(forced_target, forced_reason or "scheduler command",
                                     direction, forced=True, critical=critical)

        if not self.cfg.enabled:
            return MigrationDecision(None, "disabled")
        # Cooldown damps oscillation, but it must never outrank survival: when
        # pressure is already over budget the cheaper rung cannot wait.
        if self._cooling_down() and not critical:
            return MigrationDecision(None, "cooldown")

        # ---- down-shift: pressure is high -------------------------------
        if pressure >= self.cfg.down_threshold:
            cheaper = self.ladder.cheaper_than(current)
            if not cheaper:
                return MigrationDecision(None, "already on the cheapest rung")
            # best rung that actually fits; if nothing fits, the cheapest one.
            target = next((t for t in cheaper if self.affordable(t)), cheaper[-1])
            hops = len([t for t in cheaper if t.tier <= target.tier])
            return MigrationDecision(
                target,
                f"pressure {pressure:.2f} >= {self.cfg.down_threshold:.2f}"
                + (f" (descending {hops} rungs at once)" if hops > 1 else ""),
                "down", critical=critical)

        # ---- up-shift: pressure has been low for a while ----------------
        if (self.cfg.allow_up_shift
                and pressure <= self.cfg.up_threshold
                and self.state.low_pressure_run >= self.cfg.up_shift_patience):
            richer = [t for t in self.ladder.richer_than(current) if self.affordable(t)]
            if not richer:
                return MigrationDecision(None, "no better rung is affordable")
            return MigrationDecision(
                richer[0],
                f"pressure {pressure:.2f} <= {self.cfg.up_threshold:.2f} for "
                f"{self.state.low_pressure_run} steps",
                "up")

        return MigrationDecision(None, "within hysteresis band")

    # -- bookkeeping -------------------------------------------------------
    def note_migration(self, decision: MigrationDecision, from_tier: str,
                       token_index: int, wall_ms: float) -> None:
        self.state.last_migration_step = self.state.step
        self.state.low_pressure_run = 0
        self.state.high_pressure_run = 0
        self.state.history.append(dict(
            step=self.state.step, token_index=token_index, direction=decision.direction,
            from_tier=from_tier, to_tier=decision.target.name if decision.target else None,
            reason=decision.reason, forced=decision.forced, wall_ms=wall_ms,
        ))

    def tick(self) -> None:
        self.state.step += 1

    def reset(self) -> None:
        self.state = MigrationState()

    def summary(self) -> dict:
        return dict(
            n_migrations=self.state.n_migrations,
            n_down=sum(1 for h in self.state.history if h["direction"] == "down"),
            n_up=sum(1 for h in self.state.history if h["direction"] == "up"),
            history=self.state.history,
        )


# --------------------------------------------------------------------------
# request-boundary baseline (the straw man the README figure knocks down)
# --------------------------------------------------------------------------


class RequestBoundaryController(MigrationController):
    """Only allowed to change tier *between* requests.

    This is what every conventional elastic-serving stack does, and it exists
    here so the README's "why request-boundary switching is not enough" figure
    is a measurement rather than an assertion: under a pressure spike that
    arrives mid-answer, this controller cannot act until the answer is over —
    by which time the process has either been killed (condition A) or has spent
    the whole answer on the wrong rung (condition B).
    """

    def decide(self, current, pressure, forced_target=None, forced_reason="") -> MigrationDecision:
        if self.state.step == 0:
            return super().decide(current, pressure, forced_target, forced_reason)
        return MigrationDecision(None, "request-boundary policy: cannot switch mid-stream")
