"""Measurement plumbing: latency, memory, and distribution-discontinuity.

Every number in the README's benchmark table is produced here.  Each metric is
annotated with the claim it is evidence for so the reader can trace a figure
back to a definition.

Claims supported by this module
-------------------------------
* **no-stall**: :class:`LatencyRecorder` records *per-token* inter-token latency,
  so a re-prefill stall shows up as a single fat spike rather than being
  averaged away.
* **low migration cost**: :class:`TransplantReport` totals are summed per run.
* **continuity**: :func:`js_divergence` / :func:`discontinuity_score` quantify
  the distribution jump at the migration point.
* **zero-kill**: :class:`MemoryProbe` tracks peak footprint against the budget
  the scheduler was given.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

MB = 1024.0 * 1024.0


# --------------------------------------------------------------------------
# clocks
# --------------------------------------------------------------------------


def now() -> float:
    return time.perf_counter()


def sync(device: torch.device) -> None:
    """Make timings meaningful on async backends."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class Stopwatch:
    """Context manager returning elapsed milliseconds."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device
        self.ms = 0.0

    def __enter__(self) -> "Stopwatch":
        if self.device is not None:
            sync(self.device)
        self._t0 = now()
        return self

    def __exit__(self, *exc) -> None:
        if self.device is not None:
            sync(self.device)
        self.ms = (now() - self._t0) * 1000.0
        return None


# --------------------------------------------------------------------------
# latency
# --------------------------------------------------------------------------


@dataclass
class TokenRecord:
    """One decoded token's telemetry."""

    index: int
    t_start: float
    t_end: float
    tier: str
    token_id: int
    #: wall-ms attributable to a migration that happened *before* this token
    migration_ms: float = 0.0
    #: pressure reading observed when this token was produced (0..1)
    pressure: float = 0.0
    #: whether the token was emitted while a calibration blend was active
    blending: bool = False
    resident_mb: float = 0.0
    #: wall-ms spent on *measurement* during this token (the handoff probe).
    #: Excluded from ``latency_ms``: instrumentation is not work the system does
    #: in production, and charging it to the migration would make the latency
    #: comparison a measurement of the measurement.
    instrument_ms: float = 0.0

    @property
    def latency_ms(self) -> float:
        return max(0.0, (self.t_end - self.t_start) * 1000.0 - self.instrument_ms)


@dataclass
class LatencyRecorder:
    """Collects TTFT and the inter-token-latency series of one generation."""

    t_request: float = field(default_factory=now)
    t_first_token: Optional[float] = None
    tokens: List[TokenRecord] = field(default_factory=list)

    def mark_first_token(self) -> None:
        if self.t_first_token is None:
            self.t_first_token = now()

    def add(self, rec: TokenRecord) -> None:
        self.mark_first_token()
        self.tokens.append(rec)

    # -- derived ----------------------------------------------------------
    @property
    def ttft_ms(self) -> float:
        """Time-to-first-token.  Condition C pays this **twice** (once per
        re-prefill); Molt pays it once."""
        if self.t_first_token is None:
            return float("nan")
        return (self.t_first_token - self.t_request) * 1000.0

    @property
    def itl_ms(self) -> List[float]:
        return [t.latency_ms for t in self.tokens]

    @property
    def max_itl_ms(self) -> float:
        return max(self.itl_ms) if self.tokens else float("nan")

    def pct_itl_ms(self, q: float) -> float:
        vals = sorted(self.itl_ms)
        if not vals:
            return float("nan")
        idx = min(len(vals) - 1, max(0, int(round(q / 100.0 * (len(vals) - 1)))))
        return vals[idx]

    def stalls(self, threshold_ms: float) -> List[Tuple[int, float]]:
        """Tokens whose latency exceeded ``threshold_ms`` — the *stall* events
        that the hero figure plots for condition C."""
        return [(t.index, t.latency_ms) for t in self.tokens if t.latency_ms > threshold_ms]

    def stall_time_ms(self, threshold_ms: float) -> float:
        return sum(l - threshold_ms for _, l in self.stalls(threshold_ms))

    def summary(self, stall_threshold_ms: float = 0.0) -> Dict[str, Any]:
        itl = self.itl_ms
        base = dict(
            ttft_ms=self.ttft_ms,
            n_tokens=len(self.tokens),
            mean_itl_ms=float(sum(itl) / len(itl)) if itl else float("nan"),
            max_itl_ms=self.max_itl_ms,
            p50_itl_ms=self.pct_itl_ms(50),
            p95_itl_ms=self.pct_itl_ms(95),
            p99_itl_ms=self.pct_itl_ms(99),
            total_ms=(self.tokens[-1].t_end - self.t_request) * 1000.0 if self.tokens else 0.0,
        )
        if stall_threshold_ms > 0:
            base["n_stalls"] = len(self.stalls(stall_threshold_ms))
            base["stall_time_ms"] = self.stall_time_ms(stall_threshold_ms)
        return base


# --------------------------------------------------------------------------
# distribution discontinuity  (Molt core #3's target metric)
# --------------------------------------------------------------------------


def _to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.detach().to(torch.float32).reshape(-1), dim=-1)


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """``KL(p || q)`` in nats between two next-token distributions."""
    p, q = _to_probs(p_logits), _to_probs(q_logits)
    m = p > 0
    return float((p[m] * (p[m].log() - q[m].clamp_min(1e-12).log())).sum())


def js_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """Symmetric Jensen–Shannon divergence in nats (0 = identical).

    This is the primary *continuity* metric: the jump between the distribution
    the outgoing model would have emitted and the one the incoming model
    actually emits at the migration boundary.
    """
    p, q = _to_probs(p_logits), _to_probs(q_logits)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float((a[mask] * (a[mask].log() - b[mask].clamp_min(1e-12).log())).sum())
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def top1_agreement(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    return float(p_logits.reshape(-1).argmax() == q_logits.reshape(-1).argmax())


@dataclass
class DiscontinuityProbe:
    """Measures how much a migration disturbs the emitted distribution.

    **The primary metric is** :attr:`handoff_jsd`: at the switch point, the
    outgoing model is asked for its next-token distribution over the *same*
    position the incoming model is about to answer, and the two are compared
    directly.  That isolates the effect of the migration.

    ``excess_jsd`` (step-to-step drift around the switch, minus steady-state
    drift) is **also recorded but is not a usable continuity metric on real
    text**, and saying so is part of the result: consecutive positions in natural
    language predict genuinely different things, so ``JSD(p_t, p_{t-1})``
    saturates near ``ln 2 ≈ 0.693`` regardless of which model produced them.
    Measured steady-state values on Qwen2.5 were 0.65–0.69, i.e. the signal a
    migration adds is far below the floor.  It is kept in the output only so the
    reader can see why it was rejected.
    """

    window: int = 8
    jsd_series: List[float] = field(default_factory=list)
    migration_steps: List[int] = field(default_factory=list)
    #: JSD between the outgoing model's distribution and the one actually
    #: **emitted** at the switch position — i.e. after calibration.  This is the
    #: number a caller experiences.
    handoff_jsd: List[float] = field(default_factory=list)
    #: the same comparison against the incoming model's **raw** logits, before
    #: calibration: the transplant's own error, isolated from any smoothing.
    handoff_jsd_raw: List[float] = field(default_factory=list)
    _prev: Optional[torch.Tensor] = None

    def observe(self, logits: torch.Tensor) -> None:
        cur = logits.detach().to(torch.float32).reshape(-1).cpu()
        if self._prev is not None:
            self.jsd_series.append(js_divergence(self._prev, cur))
        else:
            self.jsd_series.append(0.0)
        self._prev = cur

    def mark_migration(self, step: int, jsd_at_handoff: Optional[float] = None) -> None:
        self.migration_steps.append(step)
        if jsd_at_handoff is not None:
            self.handoff_jsd.append(float(jsd_at_handoff))

    # -- derived ----------------------------------------------------------
    def steady_state_jsd(self) -> float:
        excluded = set()
        for m in self.migration_steps:
            excluded.update(range(max(0, m - 1), m + self.window))
        vals = [v for i, v in enumerate(self.jsd_series) if i not in excluded and i > 0]
        return float(sum(vals) / len(vals)) if vals else 0.0

    def migration_jsd(self) -> float:
        vals: List[float] = []
        for m in self.migration_steps:
            vals.extend(self.jsd_series[m:m + self.window])
        return float(sum(vals) / len(vals)) if vals else 0.0

    def excess_jsd(self) -> float:
        """The headline continuity number.  Lower is smoother."""
        if not self.migration_steps:
            return 0.0
        return self.migration_jsd() - self.steady_state_jsd()

    def peak_jsd_at_migration(self) -> float:
        vals = [self.jsd_series[m] for m in self.migration_steps if m < len(self.jsd_series)]
        return max(vals) if vals else 0.0

    def mean_handoff_jsd(self) -> float:
        """**The** continuity number: JSD between what the outgoing model would
        have emitted and what the incoming model actually emitted, at the same
        position.  0 = the switch was invisible in distribution space."""
        return (float(sum(self.handoff_jsd) / len(self.handoff_jsd))
                if self.handoff_jsd else float("nan"))

    def max_handoff_jsd(self) -> float:
        return max(self.handoff_jsd) if self.handoff_jsd else float("nan")

    def mean_handoff_jsd_raw(self) -> float:
        """The transplant's own distribution error, before calibration."""
        return (float(sum(self.handoff_jsd_raw) / len(self.handoff_jsd_raw))
                if self.handoff_jsd_raw else float("nan"))

    def summary(self) -> Dict[str, Any]:
        return dict(
            mean_handoff_jsd=self.mean_handoff_jsd(),
            mean_handoff_jsd_raw=self.mean_handoff_jsd_raw(),
            max_handoff_jsd=self.max_handoff_jsd(),
            n_handoff_samples=len(self.handoff_jsd),
            # retained for transparency; see the class docstring for why this
            # one does not work on real text
            steady_state_jsd=self.steady_state_jsd(),
            migration_jsd=self.migration_jsd(),
            excess_jsd=self.excess_jsd(),
            peak_jsd_at_migration=self.peak_jsd_at_migration(),
            n_migrations=len(self.migration_steps),
        )


def repetition_rate(token_ids: Sequence[int], n: int = 3) -> float:
    """Fraction of repeated n-grams — a cheap detector of the *style collapse*
    that an uncalibrated migration tends to trigger."""
    if len(token_ids) < n + 1:
        return 0.0
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def distinct_n(token_ids: Sequence[int], n: int = 2) -> float:
    if len(token_ids) < n:
        return 1.0
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    return len(set(grams)) / len(grams)


# --------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------


@dataclass
class MemorySample:
    t: float
    rss_mb: float
    torch_mb: float
    tracked_mb: float  # what Molt believes it has allocated (models + caches)


class MemoryProbe:
    """Samples process RSS + accelerator allocator + Molt's own accounting.

    ``tracked_mb`` is the number the scheduler budgets against; RSS is reported
    alongside so the reader can see the two agree.  The **zero-kill** claim is
    ``max(tracked_mb) <= budget`` at all times.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.samples: List[MemorySample] = []
        self._proc = psutil.Process(os.getpid()) if psutil else None
        self._t0 = now()

    def rss_mb(self) -> float:
        if self._proc is None:
            return float("nan")
        return self._proc.memory_info().rss / MB

    def torch_mb(self) -> float:
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated() / MB
        if self.device.type == "mps":
            try:
                return torch.mps.current_allocated_memory() / MB
            except Exception:
                return float("nan")
        return float("nan")

    def sample(self, tracked_mb: float = float("nan")) -> MemorySample:
        s = MemorySample(now() - self._t0, self.rss_mb(), self.torch_mb(), tracked_mb)
        self.samples.append(s)
        return s

    @property
    def peak_rss_mb(self) -> float:
        vals = [s.rss_mb for s in self.samples if not math.isnan(s.rss_mb)]
        return max(vals) if vals else float("nan")

    @property
    def peak_tracked_mb(self) -> float:
        vals = [s.tracked_mb for s in self.samples if not math.isnan(s.tracked_mb)]
        return max(vals) if vals else float("nan")

    def summary(self) -> Dict[str, Any]:
        return dict(peak_rss_mb=self.peak_rss_mb, peak_tracked_mb=self.peak_tracked_mb,
                    n_samples=len(self.samples))


def system_available_mb() -> float:
    if psutil is None:
        return float("nan")
    return psutil.virtual_memory().available / MB


# --------------------------------------------------------------------------
# event log
# --------------------------------------------------------------------------


@dataclass
class Event:
    t: float
    kind: str
    detail: Dict[str, Any] = field(default_factory=dict)


class EventLog:
    """Append-only timeline shared by runtime, scheduler and pressure source.

    The hero figure is literally a rendering of this log.
    """

    def __init__(self):
        self.t0 = now()
        self.events: List[Event] = []

    def log(self, kind: str, **detail) -> Event:
        e = Event(now() - self.t0, kind, detail)
        self.events.append(e)
        return e

    def of_kind(self, kind: str) -> List[Event]:
        return [e for e in self.events if e.kind == kind]

    def to_list(self) -> List[Dict[str, Any]]:
        return [dict(t=e.t, kind=e.kind, **e.detail) for e in self.events]

    def dump(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_list(), f, indent=2, default=str)


def dump_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, (torch.dtype, torch.device)):
        return str(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    return str(o)
