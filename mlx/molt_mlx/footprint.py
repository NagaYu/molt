"""Measuring memory the way the OS does — phys_footprint, not RSS.

The number iOS jetsam enforces (and Xcode's memory gauge shows) is the process
*physical footprint*.  On Apple silicon, MLX keeps tensors in Metal buffers that
**do not appear in RSS at all**: a 1 GiB MLX array moves RSS by ~0 MiB and the
footprint by ~1 GiB (measured; see ``selftest``).  A study that reported "peak
RSS" for an MLX model would be measuring the wrong thing by three orders of
magnitude.  Both are exposed here so that gap can be shown, but every limit and
every peak in this project is footprint.

Further measured facts this module encodes — **freeing in Python is not freeing
in the footprint** (MLX 0.31, macOS; see ``selftest``):

1. After ``del`` an MLX array's buffer stays in MLX's allocator cache, and a later
   allocation of a *different* size does not reuse it — so "drop the big model,
   load the small one" is co-residence.
2. ``mx.clear_cache()`` called right after ``del`` can free **nothing**: a buffer
   returns to the cache only when the GPU has completed the command buffer that
   last used it.  ``mx.synchronize()`` first, then clear.
3. Even then the OS reclaims asynchronously (~10^2 ms observed), so a load issued
   immediately still overlaps.

:func:`release_and_wait` (synchronize, clear, and poll until phys_footprint is
actually down) is the primitive that guarantees the memory is gone.

Claim supported: **non-co-resident migration** (both models never resident).
"""

from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass
from typing import Optional

MiB = 1024 * 1024
_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")


_V6_FIELDS = (
    "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
    "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
    "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
    "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
    "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
    "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
    "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
    "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
    "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
    "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles", "ri_billed_energy",
    "ri_serviced_energy", "ri_interval_max_phys_footprint", "ri_runnable_time",
    "ri_flags", "ri_user_ptime", "ri_system_ptime", "ri_pinstructions", "ri_pcycles",
    "ri_energy_nj", "ri_penergy_nj", "ri_secure_time_in_system",
    "ri_secure_ptime_in_system", "ri_neural_footprint", "ri_lifetime_max_neural_footprint",
    "ri_interval_max_neural_footprint", "ri_conclave_footprint", "ri_page_wait_time_mach",
    "ri_page_cache_hits")


class _RUsageV6(ctypes.Structure):
    # struct rusage_info_v6 from the macOS SDK's <sys/resource.h>, field for field;
    # the kernel fills the whole struct, so the layout must be complete.
    _fields_ = ([("ri_uuid", ctypes.c_uint8 * 16)] +
                [(n, ctypes.c_uint64) for n in _V6_FIELDS] +
                [("ri_reserved", ctypes.c_uint64 * 6)])


_RUSAGE_INFO_V6 = 6


def _rusage(pid: Optional[int] = None) -> _RUsageV6:
    ru = _RUsageV6()
    rc = _libproc.proc_pid_rusage(pid or os.getpid(), _RUSAGE_INFO_V6, ctypes.byref(ru))
    if rc != 0:
        raise OSError(ctypes.get_errno(), f"proc_pid_rusage({pid}) failed")
    return ru


def footprint_bytes(pid: Optional[int] = None) -> int:
    """Physical footprint — what jetsam enforces."""
    return int(_rusage(pid).ri_phys_footprint)


def lifetime_max_footprint_bytes(pid: Optional[int] = None) -> int:
    """Kernel-tracked maximum phys_footprint over the process lifetime.

    Maintained by the kernel ledger on every change, so — unlike any sampler —
    it cannot miss a short spike.  The peak-footprint guarantee is asserted on
    this number; the 1 kHz watchdog trace is kept for the per-phase picture."""
    return int(_rusage(pid).ri_lifetime_max_phys_footprint)


def energy_nj(pid: Optional[int] = None) -> int:
    """Per-process energy estimate from the kernel (``ri_energy_nj``), nJ.
    Which units it covers is measured, not assumed: see ``bench/energy.py``."""
    return int(_rusage(pid).ri_energy_nj)


def rss_bytes(pid: Optional[int] = None) -> int:
    """Resident set size — reported only to show how badly it undercounts MLX."""
    return int(_rusage(pid).ri_resident_size)


def mlx_memory() -> dict:
    import mlx.core as mx
    return dict(active=mx.get_active_memory(), cache=mx.get_cache_memory(),
                peak=mx.get_peak_memory())


@dataclass
class ReleaseReport:
    """Outcome of :func:`release_and_wait`."""

    before_bytes: int
    after_bytes: int
    waited_ms: float
    reached_target: bool

    @property
    def freed_mib(self) -> float:
        return (self.before_bytes - self.after_bytes) / MiB


def release_and_wait(target_bytes: int, timeout_s: float = 2.0,
                     poll_s: float = 0.002) -> ReleaseReport:
    """Return MLX's cached buffers to the OS and **block until the footprint is
    actually at or below ``target_bytes``** (or ``timeout_s`` elapses).

    This is the step that makes non-co-residence real.  The caller must have
    dropped every Python reference to the arrays it wants gone *before* calling.
    """
    import gc

    import mlx.core as mx

    before = footprint_bytes()
    gc.collect()
    mx.synchronize()          # buffers return to the cache only after the GPU is done
    mx.clear_cache()
    t0 = time.perf_counter()
    cur = footprint_bytes()
    while cur > target_bytes and time.perf_counter() - t0 < timeout_s:
        time.sleep(poll_s)
        mx.clear_cache()      # anything returned to the cache since the last call
        cur = footprint_bytes()
    return ReleaseReport(before, cur, (time.perf_counter() - t0) * 1000.0,
                         cur <= target_bytes)


class PhaseMarker:
    """Labels the current phase for the external watchdog.

    The watchdog re-reads ``<csv>.phase`` every ~2 ms, so every footprint sample
    is attributed to the phase that produced it ("load_dst", "transplant", …).
    A no-op when no watchdog is attached.
    """

    def __init__(self, csv_path: Optional[str]):
        self.path = f"{csv_path}.phase" if csv_path else None

    def __call__(self, label: str) -> None:
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(label + "\n")
        os.replace(tmp, self.path)        # atomic: the reader never sees half a label


def emulated_available_memory(cap_bytes: int) -> int:
    """macOS stand-in for iOS ``os_proc_available_memory()``.

    Apple documents that function as "the current memory limit minus the memory
    footprint of your app", returning 0 once the limit is exceeded.  macOS has no
    per-app limit, so the cap is supplied by the experiment; the arithmetic is the
    documented one.
    """
    return max(0, cap_bytes - footprint_bytes())


def selftest() -> None:
    """Demonstrates both measured facts this module is built on."""
    import mlx.core as mx

    base = footprint_bytes()
    a = mx.ones((64, 1024, 1024), dtype=mx.float32)   # 256 MiB
    mx.eval(a)
    grown_fp = footprint_bytes() - base
    grown_rss = rss_bytes()
    assert grown_fp > 200 * MiB, f"footprint should see the MLX array ({grown_fp/MiB:.0f} MiB)"
    print(f"  256 MiB MLX array: footprint +{grown_fp/MiB:.0f} MiB, RSS {grown_rss/MiB:.0f} MiB total")

    del a
    stuck = footprint_bytes() - base
    assert stuck > 200 * MiB, "without clear_cache the buffer should still be in the footprint"
    mx.clear_cache()                       # the call a developer would write …
    time.sleep(0.3)
    naive = footprint_bytes() - base       # … typically frees nothing (fact 2)
    rep = release_and_wait(base + 64 * MiB, timeout_s=3.0)
    assert rep.reached_target, f"release_and_wait did not reach target: {rep}"
    print(f"  after del: still +{stuck/MiB:.0f} MiB; clear_cache()+300 ms: +{naive/MiB:.0f} MiB; "
          f"release_and_wait freed {rep.freed_mib:.0f} MiB in {rep.waited_ms:.0f} ms")
    lm = lifetime_max_footprint_bytes()
    assert lm >= base + 200 * MiB, "kernel lifetime max must remember the 256 MiB peak"
    print(f"  kernel lifetime-max footprint {lm/MiB:.0f} MiB (remembers the peak after release)")
    assert emulated_available_memory(base + 512 * MiB) > 0
    print("footprint: ok")


if __name__ == "__main__":
    selftest()
