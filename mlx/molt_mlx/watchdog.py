"""Build, attach and read the external footprint watchdog (``watchdog.c``).

The watchdog is a separate C process that samples the target's
``phys_footprint`` at ~1 kHz and SIGKILLs it the moment it exceeds a cap — an
emulation of iOS jetsam's per-process limit on a Mac, where no such limit exists.

What this emulation is and is not:

* It **is** faithful to the documented rule (footprint over the limit ⇒ the
  process is terminated) and it is external, so it keeps sampling while the
  target is busy inside MLX — which is exactly when peaks happen.
* It is **not** iOS.  Real jetsam is event-driven inside the kernel and its
  limit is device- and entitlement-dependent; a 1 ms poll can miss a spike
  shorter than 1 ms.  Device numbers come from ``ios/``, not from here.

Claims supported: **survival under a memory limit** and **peak footprint during
migration**, both on macOS / Apple silicon.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

MiB = 1024 * 1024
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "watchdog.c")


def build(out_dir: Optional[str] = None) -> str:
    """Compile ``watchdog.c`` (cached by source mtime). Needs clang from the CLT."""
    out_dir = out_dir or os.path.join(tempfile.gettempdir(), "molt_watchdog")
    os.makedirs(out_dir, exist_ok=True)
    exe = os.path.join(out_dir, f"watchdog_{int(os.path.getmtime(_SRC))}")
    if not os.path.exists(exe):
        cc = shutil.which("clang") or shutil.which("cc")
        if not cc:
            raise RuntimeError("clang not found; install the Xcode Command Line Tools")
        subprocess.run([cc, "-O2", "-Wall", "-o", exe, _SRC], check=True)
    return exe


@dataclass
class WatchdogResult:
    peak_bytes: int
    killed: bool
    samples: int
    #: (t_us, footprint_bytes, phase) — the full timeline
    timeline: List[Tuple[int, int, str]] = field(default_factory=list)
    #: kernel ledger maximum (exact; cannot miss a spike between samples)
    lifetime_max_bytes: int = 0
    kill_reason: Optional[str] = None
    #: ri_energy_nj accumulated by the target while watched
    energy_nj: Optional[int] = None

    @property
    def peak_mib(self) -> float:
        """Exact peak: the kernel's lifetime max when available, else sampled."""
        return max(self.peak_bytes, self.lifetime_max_bytes) / MiB

    @property
    def sampled_peak_mib(self) -> float:
        return self.peak_bytes / MiB

    def phase_peaks_mib(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for _, fp, ph in self.timeline:
            if fp > out.get(ph, 0) * MiB:
                out[ph] = fp / MiB
        return out

    def killed_in_phase(self) -> Optional[str]:
        if not self.killed:
            return None
        # the last real sample before the KILLED marker
        for _, _, ph in reversed(self.timeline):
            if ph != "KILLED":
                return ph
        return None

    def max_gap_us(self) -> int:
        """Largest gap between consecutive samples — the blind spot of this run."""
        ts = [t for t, _, _ in self.timeline]
        return max((b - a for a, b in zip(ts, ts[1:])), default=0)


def parse(csv_path: str) -> WatchdogResult:
    timeline: List[Tuple[int, int, str]] = []
    peak, killed, extra = 0, False, {}
    with open(csv_path) as fh:
        for line in fh:
            if line.startswith("t_us"):
                continue
            if line.startswith("#"):
                for kv in line[1:].split():
                    k, _, v = kv.partition("=")
                    if k == "peak_bytes":
                        peak = int(v)
                    elif k == "killed":
                        killed = v == "1"
                    elif k in ("lifetime_max_bytes", "energy_nj"):
                        extra[k] = int(v)
                    elif k == "kill_reason":
                        extra[k] = v
                continue
            t, fp, ph = line.rstrip("\n").split(",", 2)
            timeline.append((int(t), int(fp), ph))
    if not peak and timeline:
        peak = max(fp for _, fp, _ in timeline)
    killed = killed or any(ph == "KILLED" for _, _, ph in timeline)
    return WatchdogResult(peak, killed, len(timeline), timeline,
                          extra.get("lifetime_max_bytes", peak), extra.get("kill_reason"),
                          extra.get("energy_nj"))


def run_under_watchdog(cmd: List[str], cap_mib: float, csv_path: str,
                       interval_us: int = 1000, env: Optional[dict] = None,
                       timeout_s: float = 1800.0) -> Tuple[int, WatchdogResult]:
    """Start ``cmd``, attach the watchdog before the target does any work, wait.

    The target must wait for the file named in ``$MOLT_WATCHDOG_READY`` before
    allocating anything, so no early allocation escapes measurement.
    Returns ``(target_returncode, result)``; a SIGKILL shows as ``-9``.
    """
    exe = build()
    ready = csv_path + ".ready"
    for p in (ready, csv_path + ".phase"):
        if os.path.exists(p):
            os.remove(p)
    env = dict(env or os.environ)
    env["MOLT_WATCHDOG_READY"] = ready
    env["MOLT_WATCHDOG_CSV"] = csv_path
    target = subprocess.Popen(cmd, env=env)
    wd = subprocess.Popen([exe, str(target.pid), str(cap_mib), str(interval_us),
                           csv_path, ready])
    try:
        rc = target.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        target.kill()
        rc = target.wait()
    wd.wait(timeout=30)
    return rc, parse(csv_path)


def wait_until_watched(timeout_s: float = 10.0) -> None:
    """Called by the *target*: block until the watchdog has attached."""
    ready = os.environ.get("MOLT_WATCHDOG_READY")
    if not ready:
        return
    t0 = time.perf_counter()
    while not os.path.exists(ready):
        if time.perf_counter() - t0 > timeout_s:
            raise RuntimeError("watchdog never attached")
        time.sleep(0.002)


_CHILD = r"""
import sys, time
sys.path.insert(0, %r)
from molt_mlx.watchdog import wait_until_watched
from molt_mlx.footprint import PhaseMarker
import os, mlx.core as mx
wait_until_watched()
mark = PhaseMarker(os.environ.get("MOLT_WATCHDOG_CSV"))
mark("alloc")
xs = []
for _ in range(int(sys.argv[1])):
    a = mx.ones((32, 1024, 1024), mx.float32); mx.eval(a); xs.append(a)   # +128 MiB
    time.sleep(0.03)
mark("done"); time.sleep(0.05)
"""


def selftest() -> None:
    tmp = tempfile.mkdtemp(prefix="wd_selftest_")
    script = os.path.join(tmp, "child.py")
    with open(script, "w") as fh:
        fh.write(_CHILD % os.path.dirname(_HERE))

    # 1) no cap: the watchdog must see ~4 x 128 MiB
    rc, res = run_under_watchdog([sys.executable, script, "4"], 0, os.path.join(tmp, "a.csv"))
    assert rc == 0 and not res.killed, (rc, res.killed)
    assert res.peak_mib > 450, f"peak {res.peak_mib:.0f} MiB should include 512 MiB of MLX arrays"
    assert "alloc" in res.phase_peaks_mib(), res.phase_peaks_mib()
    print(f"  uncapped: peak {res.peak_mib:.0f} MiB over {res.samples} samples, "
          f"max sample gap {res.max_gap_us()/1000:.1f} ms")

    # 2) cap at 300 MiB: the target must be SIGKILLed during 'alloc'
    rc, res = run_under_watchdog([sys.executable, script, "8"], 300, os.path.join(tmp, "b.csv"))
    assert res.killed and rc == -9, (rc, res.killed)
    assert res.killed_in_phase() == "alloc", res.killed_in_phase()
    print(f"  capped at 300 MiB: killed (rc={rc}) in phase '{res.killed_in_phase()}' "
          f"at {res.peak_mib:.0f} MiB")
    shutil.rmtree(tmp, ignore_errors=True)
    print("watchdog: ok")


if __name__ == "__main__":
    selftest()
