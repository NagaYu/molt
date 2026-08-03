"""The four benchmark conditions, as data.

Each condition is a policy + a starting rung + a few switches, so that the same
runtime code path serves all of them and no condition gets an unfair advantage
from a different implementation.

============  =====================================================================
**A** static-large   always tier0.  Under pressure the process is reclaimed: this
                     condition exists to produce non-zero *forced terminations*.
**B** static-small   always the cheapest rung.  Survives everything, and is the
                     quality floor Molt has to beat.
**C** restart        switches rungs under pressure but throws the cache away and
                     re-prefills — the "obvious" implementation of elasticity,
                     and the source of the long stalls in the hero figure.
**D** molt           switches rungs *and transplants the cache*, with calibration.
============  =====================================================================

Two extra arms make the ablations honest:

``D-nocal``     Molt without logit blending — isolates core #3.
``D-reqbound``  Molt's machinery, but only allowed to switch between requests —
                the measurement behind "why request-boundary switching is not
                enough".
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from molt.config import (CalibrationConfig, MigrationConfig, MoltConfig,
                         TransplantConfig)
from molt.runtime import Policy


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    policy: Policy
    start_tier: str
    #: mid-stream migration allowed at all?
    request_boundary: bool = False
    calibration_mode: Optional[str] = None      # override cfg.calibration.mode
    transplant_overrides: Dict[str, Any] = field(default_factory=dict)
    #: A is the only condition allowed to die; every other condition asserting a
    #: kill is a bug, not a result.
    expect_kills: bool = False
    description: str = ""

    def apply(self, cfg: MoltConfig) -> MoltConfig:
        cal = cfg.calibration
        if self.calibration_mode is not None:
            cal = replace(cal, mode=self.calibration_mode)
        tp = cfg.transplant
        if self.transplant_overrides:
            tp = replace(tp, **self.transplant_overrides)
        mig = cfg.migration
        if self.policy is Policy.STATIC:
            mig = replace(mig, enabled=False)
        return replace(cfg, calibration=cal, transplant=tp, migration=mig)


CONDITIONS: Dict[str, Condition] = {
    "A": Condition(
        key="A", label="A · Static-large", policy=Policy.STATIC, start_tier="tier0",
        expect_kills=True,
        description="Always the best model.  Highest quality until the OS reclaims it."),
    "B": Condition(
        key="B", label="B · Static-small", policy=Policy.STATIC, start_tier="tier2",
        description="Always the smallest model.  Survives, but never better than its floor."),
    "C": Condition(
        key="C", label="C · Restart-on-pressure", policy=Policy.RESTART, start_tier="tier0",
        description="Switches rungs under pressure, discards the KV cache, re-prefills."),
    "D": Condition(
        key="D", label="D · Molt", policy=Policy.MOLT, start_tier="tier0",
        description="Switches rungs, transplants the KV cache, calibrates the seam."),
    # ---- ablations -------------------------------------------------------
    "D-nocal": Condition(
        key="D-nocal", label="D⁻ · Molt, no calibration", policy=Policy.MOLT,
        start_tier="tier0", calibration_mode="none",
        description="Ablation for core #3: same transplant, no logit blending."),
    "D-noproj": Condition(
        key="D-noproj", label="D⁻ · Molt, no learned projection", policy=Policy.MOLT,
        start_tier="tier0",
        transplant_overrides=dict(use_projection=False, use_rope_realign=False),
        description="Ablation for core #1(i): truncated-identity map, no RoPE sandwich. "
                    "Note this necessarily also disables the top-k recompute, which "
                    "depends on a *learned* hidden-state map — so read it against "
                    "D-norecompute rather than against D alone."),
    "D-norecompute": Condition(
        key="D-norecompute", label="D⁻ · Molt, no top-k recompute", policy=Policy.MOLT,
        start_tier="tier0", transplant_overrides=dict(recompute_top_k=0),
        description="Ablation for core #1(iii): pure projection, nothing recomputed."),
    "D-reqbound": Condition(
        key="D-reqbound", label="D⁻ · Molt, request-boundary only", policy=Policy.MOLT,
        start_tier="tier0", request_boundary=True,
        description="Molt's machinery restricted to switching between requests — the "
                    "baseline that shows why mid-stream migration is necessary."),
}

MAIN_CONDITIONS = ["A", "B", "C", "D"]
ABLATIONS = ["D-nocal", "D-noproj", "D-norecompute", "D-reqbound"]


def get_conditions(keys: Optional[List[str]] = None) -> List[Condition]:
    keys = keys or MAIN_CONDITIONS
    out = []
    for k in keys:
        if k not in CONDITIONS:
            raise KeyError(f"unknown condition {k!r}; have {sorted(CONDITIONS)}")
        out.append(CONDITIONS[k])
    return out
