"""Memory-pressure sources: a replayable trace and a real allocating process.

Two implementations behind one interface:

:class:`TraceReplaySource`
    Replays a JSON trace of ``(time, budget_MB)`` breakpoints.  Deterministic
    and identical across the four benchmark conditions, which is what makes the
    comparison fair — every condition sees *the same* pressure timeline.

:class:`ProcessHogSource`
    Spawns a real child process (``python -m molt.pressure_hog``) that actually
    allocates and frees anonymous memory, and derives the budget from
    ``psutil.virtual_memory().available``.  Slower and non-deterministic, but it
    reproduces the OS-level signal the deterministic trace only models.  Use
    ``--real-hog`` in the benchmark.

Both expose ``budget_mb(t)``; *pressure* itself is ``used / budget`` and is
computed by whoever knows ``used`` — see :class:`~molt.scheduler.QoSScheduler`.

Claims supported by this module
-------------------------------
* **zero-kill**: the trace is the adversary.  A run "survives" only if its
  tracked footprint stayed under ``budget_mb`` at every instant of the trace.
* **no-stall**: the trace's spike timings are chosen to land *mid-answer*, which
  is the regime where request-boundary elasticity provably cannot help.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

MB = 1024.0 * 1024.0


# --------------------------------------------------------------------------
# traces
# --------------------------------------------------------------------------


@dataclass
class PressureEvent:
    """A breakpoint: from time ``t`` onward, the budget is ``budget_mb``."""

    t: float
    budget_mb: float
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PressureTrace:
    name: str
    description: str
    duration_s: float
    events: List[PressureEvent]
    #: how much the hog process should actually allocate to *cause* each budget
    #: level, when running with a real child process
    total_system_mb: float = 8192.0

    def __post_init__(self) -> None:
        self.events = sorted(
            [e if isinstance(e, PressureEvent) else PressureEvent(**e) for e in self.events],
            key=lambda e: e.t)
        if not self.events:
            raise ValueError("a pressure trace needs at least one event")
        if self.events[0].t > 0:
            self.events.insert(0, PressureEvent(0.0, self.events[0].budget_mb, "start"))

    def budget_at(self, t: float) -> float:
        cur = self.events[0].budget_mb
        for e in self.events:
            if e.t <= t:
                cur = e.budget_mb
            else:
                break
        return cur

    def label_at(self, t: float) -> str:
        cur = self.events[0].label
        for e in self.events:
            if e.t <= t:
                cur = e.label
            else:
                break
        return cur

    @property
    def min_budget_mb(self) -> float:
        return min(e.budget_mb for e in self.events)

    @property
    def max_budget_mb(self) -> float:
        return max(e.budget_mb for e in self.events)

    def scaled(self, factor: float) -> "PressureTrace":
        """Rescale all budgets — used to fit a trace to a different ladder."""
        return PressureTrace(
            name=f"{self.name}x{factor:g}", description=self.description,
            duration_s=self.duration_s,
            events=[PressureEvent(e.t, e.budget_mb * factor, e.label) for e in self.events],
            total_system_mb=self.total_system_mb * factor)

    def time_scaled(self, factor: float) -> "PressureTrace":
        """Stretch/compress the timeline (slow CPUs need longer traces)."""
        return PressureTrace(
            name=self.name, description=self.description,
            duration_s=self.duration_s * factor,
            events=[PressureEvent(e.t * factor, e.budget_mb, e.label) for e in self.events],
            total_system_mb=self.total_system_mb)

    def to_dict(self) -> Dict[str, Any]:
        return dict(name=self.name, description=self.description,
                    duration_s=self.duration_s, total_system_mb=self.total_system_mb,
                    events=[e.to_dict() for e in self.events])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PressureTrace":
        with open(path) as f:
            blob = json.load(f)
        return cls(name=blob["name"], description=blob.get("description", ""),
                   duration_s=float(blob["duration_s"]),
                   events=[PressureEvent(**e) for e in blob["events"]],
                   total_system_mb=float(blob.get("total_system_mb", 8192.0)))


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------


class PressureSource:
    """Interface: something that says how much memory we may use right now."""

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def elapsed(self) -> float:
        raise NotImplementedError

    def budget_mb(self) -> float:
        raise NotImplementedError

    def label(self) -> str:
        return ""

    def finished(self) -> bool:
        return False

    def __enter__(self) -> "PressureSource":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


class TraceReplaySource(PressureSource):
    """Deterministic replay of a :class:`PressureTrace` against the wall clock.

    ``time_scale > 1`` slows the trace down (useful when a CPU-only run decodes
    at two tokens per second and the interesting spike would otherwise pass
    before the first migration).  ``virtual_step_s`` switches to a virtual clock
    advanced once per decode step, which makes runs bit-reproducible at the cost
    of decoupling from real time.
    """

    def __init__(self, trace: PressureTrace, time_scale: float = 1.0,
                 virtual_step_s: Optional[float] = None):
        self.trace = trace if time_scale == 1.0 else trace.time_scaled(time_scale)
        self.virtual_step_s = virtual_step_s
        self._t0 = 0.0
        self._virtual_t = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._virtual_t = 0.0

    def tick(self, n_steps: int = 1) -> None:
        """Advance the virtual clock (no-op in wall-clock mode)."""
        if self.virtual_step_s is not None:
            self._virtual_t += self.virtual_step_s * n_steps

    def elapsed(self) -> float:
        if self.virtual_step_s is not None:
            return self._virtual_t
        return time.perf_counter() - self._t0

    def budget_mb(self) -> float:
        return self.trace.budget_at(self.elapsed())

    def label(self) -> str:
        return self.trace.label_at(self.elapsed())

    def finished(self) -> bool:
        return self.elapsed() >= self.trace.duration_s


class ProcessHogSource(PressureSource):
    """A real child process that allocates and frees memory on a schedule.

    The child touches every page it allocates, so the pressure is genuine rather
    than a virtual-size illusion.  The budget reported here is what is *actually*
    available to this process, floored so a pathological host does not make the
    benchmark unrunnable.
    """

    def __init__(self, trace: PressureTrace, floor_mb: float = 512.0,
                 headroom_mb: float = 512.0, python: Optional[str] = None):
        self.trace = trace
        self.floor_mb = floor_mb
        self.headroom_mb = headroom_mb
        self.python = python or sys.executable
        self.proc: Optional[subprocess.Popen] = None
        self._t0 = 0.0

    def _schedule(self) -> List[Dict[str, float]]:
        """Convert budget breakpoints into allocation targets for the child."""
        top = self.trace.max_budget_mb
        return [dict(t=e.t, alloc_mb=max(0.0, top - e.budget_mb)) for e in self.trace.events]

    def start(self) -> None:
        script = os.path.join(os.path.dirname(__file__), "pressure_hog.py")
        payload = json.dumps(dict(schedule=self._schedule(), duration_s=self.trace.duration_s))
        self.proc = subprocess.Popen(
            [self.python, script, payload],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.proc = None

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def budget_mb(self) -> float:
        if psutil is None:
            return self.trace.budget_at(self.elapsed())
        avail = psutil.virtual_memory().available / MB
        rss = psutil.Process(os.getpid()).memory_info().rss / MB
        return max(self.floor_mb, avail + rss - self.headroom_mb)

    def label(self) -> str:
        return self.trace.label_at(self.elapsed())

    def finished(self) -> bool:
        return self.elapsed() >= self.trace.duration_s


class ConstantSource(PressureSource):
    """Fixed budget — the control condition (no pressure at all)."""

    def __init__(self, budget_mb: float, duration_s: float = 1e9):
        self._budget = budget_mb
        self._duration = duration_s
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def budget_mb(self) -> float:
        return self._budget

    def finished(self) -> bool:
        return self.elapsed() >= self._duration


# --------------------------------------------------------------------------
# built-in traces
# --------------------------------------------------------------------------


def builtin_traces(top_tier_mb: float = 6200.0) -> Dict[str, PressureTrace]:
    """Traces sized relative to the top rung's footprint.

    ``top_tier_mb`` is the fp weight footprint of tier0, so a budget of
    ``0.6 * top_tier_mb`` genuinely cannot hold tier0 — the pressure is real
    rather than a threshold the code chose to believe.
    """
    T = top_tier_mb

    def ev(t, mult, label):
        return PressureEvent(t, round(T * mult, 1), label)

    return {
        "spike_mid_answer": PressureTrace(
            name="spike_mid_answer",
            description=("A camera app launches while the assistant is halfway through a long "
                         "answer, starts encoding video, then closes.  The squeeze arrives in "
                         "two steps so that every rung of the ladder is exercised in one run, "
                         "and it lands mid-stream on purpose: this is the case a "
                         "request-boundary policy structurally cannot serve."),
            duration_s=60.0, total_system_mb=T * 2.0,
            events=[ev(0, 2.0, "idle"), ev(8, 0.45, "camera app launches"),
                    ev(20, 0.34, "video encoding starts"), ev(34, 2.0, "camera closes"),
                    ev(60, 2.0, "idle")],
        ),
        "sawtooth": PressureTrace(
            name="sawtooth",
            description="Repeated allocate/free cycles: exercises anti-thrash (hysteresis, "
                        "cooldown, up-shift patience) and bidirectional control.",
            duration_s=60.0, total_system_mb=T * 2.0,
            events=[ev(0, 2.0, "idle"), ev(6, 0.5, "burst 1"), ev(14, 2.0, "release 1"),
                    ev(20, 0.5, "burst 2"), ev(28, 2.0, "release 2"),
                    ev(34, 0.5, "burst 3"), ev(42, 2.0, "release 3"), ev(60, 2.0, "idle")],
        ),
        "staircase": PressureTrace(
            name="staircase",
            description="Monotonically tightening budget: every rung of the ladder is used in "
                        "turn, ending below what even the smallest model plus its cache needs "
                        "if nothing is demoted.",
            duration_s=60.0, total_system_mb=T * 2.0,
            events=[ev(0, 2.0, "idle"), ev(10, 1.1, "app A"), ev(22, 0.62, "app B"),
                    ev(34, 0.40, "app C"), ev(46, 0.30, "app D"), ev(60, 0.30, "saturated")],
        ),
        "flat": PressureTrace(
            name="flat", description="Control: no pressure event at all.",
            duration_s=60.0, total_system_mb=T * 2.0,
            events=[ev(0, 2.0, "idle")],
        ),
    }


def load_trace(path_or_name: str, top_tier_mb: float = 6200.0) -> PressureTrace:
    if os.path.exists(path_or_name):
        return PressureTrace.load(path_or_name)
    traces = builtin_traces(top_tier_mb)
    if path_or_name in traces:
        return traces[path_or_name]
    raise KeyError(f"unknown trace {path_or_name!r}; have {sorted(traces)} or pass a JSON path")
