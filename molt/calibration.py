"""Molt core #3 — **MigrationCalibration**: smooth the seam between two models.

A transplant hands the incoming model a cache it did not build.  Even a good
projection leaves a step change in the *output distribution*: entropy jumps,
the top-1 token flips, and the classic failure mode follows — the text starts
repeating or the register changes mid-sentence.

Calibration spends a short window right after the switch making the emitted
distribution move continuously instead of teleporting.

Three modes, all implemented so the ablation in the README is real:

``dual``
    The outgoing model is kept alive for ``blend_steps`` tokens and decodes in
    lock-step.  The emitted distribution is a mixture
    ``(1-w_i)·p_new + w_i·p_old`` with ``w_i`` annealed to zero.  Highest
    fidelity, but two models are momentarily resident — which is exactly what a
    memory-pressure event cannot always afford, so it is *not* always the right
    choice.

``frozen_ref``
    No second model.  The outgoing model's recent **entropy** is recorded before
    it is released, and the incoming model's logits are temperature-rescaled so
    entropy interpolates from the old value to its own natural value.  Costs one
    bisection over a scalar per token; costs zero extra memory.

``none``
    Ablation baseline: switch and hope.

Claims supported by this module
-------------------------------
* **continuity**: :class:`MigrationCalibrator` is the component the
  ``excess JSD`` and ``repetition rate`` columns of the benchmark isolate; the
  ``none`` arm is what it is measured against.
* **zero-kill**: ``frozen_ref`` exists so that continuity never *requires*
  holding two models at once under critical pressure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .config import CalibrationConfig


# --------------------------------------------------------------------------
# schedules
# --------------------------------------------------------------------------


def blend_weight(step: int, total: int, schedule: str = "cosine", w0: float = 1.0) -> float:
    """Weight given to the *outgoing* model at blend step ``step`` (0-based).

    All schedules start at ``w0`` and reach 0 at ``step == total``.  Cosine is
    the default because its derivative vanishes at both ends, which is what
    keeps step-to-step JSD flat rather than merely small on average.
    """
    if total <= 0:
        return 0.0
    x = min(1.0, max(0.0, (step + 1) / total))
    if schedule == "linear":
        f = 1.0 - x
    elif schedule == "cosine":
        f = 0.5 * (1.0 + math.cos(math.pi * x))
    elif schedule == "exp":
        f = math.exp(-3.0 * x) - math.exp(-3.0)
        f /= (1.0 - math.exp(-3.0))
    else:
        raise ValueError(f"unknown blend schedule {schedule!r}")
    return w0 * f


# --------------------------------------------------------------------------
# distribution utilities
# --------------------------------------------------------------------------


def mix_logits(new_logits: torch.Tensor, old_logits: torch.Tensor, w_old: float
               ) -> torch.Tensor:
    """Mixture **in probability space**, returned as log-probabilities.

    Mixing probabilities (rather than averaging logits) keeps the result a
    genuine distribution and preserves the outgoing model's confident modes,
    which is what stops the first post-switch token from derailing a sentence.
    """
    if w_old <= 0.0:
        return new_logits
    if w_old >= 1.0:
        return old_logits
    p_new = torch.softmax(new_logits.float(), dim=-1)
    p_old = torch.softmax(old_logits.float(), dim=-1)
    p = (1.0 - w_old) * p_new + w_old * p_old
    return p.clamp_min(1e-12).log()


def entropy_of(logits: torch.Tensor) -> float:
    p = torch.softmax(logits.float().reshape(-1), dim=-1)
    return float(-(p * p.clamp_min(1e-12).log()).sum())


def temperature_for_entropy(logits: torch.Tensor, target_entropy: float,
                            iters: int = 12, lo: float = 0.25, hi: float = 4.0) -> float:
    """Find ``τ`` with ``H(softmax(logits/τ)) ≈ target_entropy`` by bisection.

    Entropy is monotone increasing in ``τ``, so 12 bisection steps pin it to
    ~0.04% of the bracket.  This is the whole cost of ``frozen_ref`` mode.
    """
    z = logits.float().reshape(-1)
    def H(t: float) -> float:
        p = torch.softmax(z / t, dim=-1)
        return float(-(p * p.clamp_min(1e-12).log()).sum())
    if target_entropy <= H(lo):
        return lo
    if target_entropy >= H(hi):
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if H(mid) < target_entropy:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# the calibrator
# --------------------------------------------------------------------------


@dataclass
class BlendState:
    """Live state of one blend window."""

    active: bool = False
    mode: str = "none"
    step: int = 0
    total: int = 0
    schedule: str = "cosine"
    target_entropy: float = 0.0
    weights: List[float] = field(default_factory=list)
    #: JSD between the two models at the handoff instant (dual mode only)
    handoff_jsd: Optional[float] = None

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.step)


class MigrationCalibrator:
    """Applies the post-migration blend and reports what it did.

    Usage from the decode loop::

        cal.begin(pressure=p, old_entropy=...)      # at the switch
        logits = cal.apply(new_logits, old_logits)  # every subsequent token
        cal.advance()

    ``old_logits`` is ``None`` outside ``dual`` mode.
    """

    def __init__(self, cfg: CalibrationConfig):
        self.cfg = cfg
        self.state = BlendState()
        #: rolling entropy of the outgoing model, used by ``frozen_ref``
        self._entropy_window: List[float] = []
        self.entropy_window_size = 8
        #: telemetry
        self.blend_events: List[dict] = []

    # -- observing the outgoing model -------------------------------------
    def observe(self, logits: torch.Tensor) -> None:
        """Record the *current* model's output entropy.

        Called on every ordinary decode step so that, at the moment pressure
        forces a switch, the outgoing model's typical entropy is already known
        and ``frozen_ref`` needs nothing extra from it.
        """
        self._entropy_window.append(entropy_of(logits))
        if len(self._entropy_window) > self.entropy_window_size:
            self._entropy_window.pop(0)

    @property
    def reference_entropy(self) -> float:
        if not self._entropy_window:
            return 0.0
        return sum(self._entropy_window) / len(self._entropy_window)

    # -- lifecycle ---------------------------------------------------------
    def effective_mode(self, pressure: float, critical: bool) -> str:
        """``dual`` degrades to ``frozen_ref`` when memory cannot hold two models.

        This is the concrete link between core #3 and the *zero-kill* invariant:
        continuity must never be bought with a second resident model at the
        exact moment the system is short of memory.
        """
        if self.cfg.mode == "none":
            return "none"
        if self.cfg.mode == "dual" and critical:
            return self.cfg.critical_mode
        return self.cfg.mode

    def begin(self, pressure: float, critical: bool = False,
              handoff_jsd: Optional[float] = None) -> BlendState:
        """Open a blend window at a migration point."""
        mode = self.effective_mode(pressure, critical)
        total = self.cfg.critical_blend_steps if critical else self.cfg.blend_steps
        if mode == "none":
            total = 0
        self.state = BlendState(
            active=total > 0, mode=mode, step=0, total=total,
            schedule=self.cfg.schedule, target_entropy=self.reference_entropy,
            weights=[blend_weight(i, total, self.cfg.schedule) for i in range(total)],
            handoff_jsd=handoff_jsd,
        )
        self.blend_events.append(dict(mode=mode, steps=total, critical=critical,
                                      pressure=pressure,
                                      target_entropy=self.state.target_entropy))
        return self.state

    @property
    def needs_old_model(self) -> bool:
        """True while the outgoing model must stay resident (``dual`` only)."""
        return self.state.active and self.state.mode == "dual" and self.state.remaining > 0

    def apply(self, new_logits: torch.Tensor,
              old_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the distribution to actually sample from at this step."""
        st = self.state
        if not st.active or st.step >= st.total:
            return new_logits
        w = st.weights[st.step] if st.step < len(st.weights) else 0.0
        if st.mode == "dual" and old_logits is not None:
            return mix_logits(new_logits, old_logits, w)
        if st.mode == "frozen_ref":
            # interpolate entropy from the outgoing model's level to the
            # incoming model's own, instead of stepping there at once.
            own = entropy_of(new_logits)
            target = w * st.target_entropy + (1.0 - w) * own
            tau = temperature_for_entropy(new_logits, target)
            return new_logits.float() / tau
        return new_logits

    def advance(self) -> None:
        if self.state.active:
            self.state.step += 1
            if self.state.step >= self.state.total:
                self.state.active = False

    def reset(self) -> None:
        self.state = BlendState()
        self._entropy_window.clear()

    def summary(self) -> dict:
        return dict(mode=self.cfg.mode, blend_steps=self.cfg.blend_steps,
                    schedule=self.cfg.schedule, n_blends=len(self.blend_events),
                    events=self.blend_events)
