"""Energy without root: what can be measured, and what each number covers.

Measured on this machine (``selftest``; M2, macOS 27):

* ``ri_energy_nj`` from ``proc_pid_rusage`` is **per-process but CPU-only**:
  a busy CPU loop reads ~3 W, a busy MLX GPU matmul loop ~0.03 W.  It is exact
  attribution for the CPU share and blind to the GPU.
* IOReport's "Energy Model" group (``libIOReport``, no root) exposes a
  **system-wide "GPU Energy"** counter in nJ that tracks MLX work; its CPU
  channels read zero here, so CPU comes from ``ri_energy_nj`` instead.

So a run's energy is reported as two named parts, never summed silently:
``cpu_j`` (this process, exact) and ``gpu_j`` (whole GPU, minus an idle
baseline measured just before the run; only meaningful when nothing else uses
the GPU — the benchmark records the baseline so a reader can judge).  DRAM
energy is not available without ``powermetrics`` (root); ``bench/powermetrics.sh``
is the cross-check a reader with sudo can run.
"""

from __future__ import annotations

import ctypes
import time
from dataclasses import asdict, dataclass
from typing import Optional

from .footprint import energy_nj

_V = ctypes.c_void_p
_UTF8 = 0x08000100


class _IOReport:
    """Minimal ctypes binding: subscribe to the 'Energy Model' group, read the
    'GPU Energy' channel delta between two samples."""

    def __init__(self):
        cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        io = ctypes.CDLL("/usr/lib/libIOReport.dylib")
        cf.CFStringCreateWithCString.restype = _V
        cf.CFStringCreateWithCString.argtypes = [_V, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFDictionaryGetValue.restype = _V; cf.CFDictionaryGetValue.argtypes = [_V, _V]
        cf.CFDictionaryGetCount.restype = ctypes.c_long; cf.CFDictionaryGetCount.argtypes = [_V]
        cf.CFDictionaryCreateMutableCopy.restype = _V
        cf.CFDictionaryCreateMutableCopy.argtypes = [_V, ctypes.c_long, _V]
        cf.CFArrayGetCount.restype = ctypes.c_long; cf.CFArrayGetCount.argtypes = [_V]
        cf.CFArrayGetValueAtIndex.restype = _V; cf.CFArrayGetValueAtIndex.argtypes = [_V, ctypes.c_long]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFStringGetCString.argtypes = [_V, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        cf.CFRelease.argtypes = [_V]
        io.IOReportCopyChannelsInGroup.restype = _V
        io.IOReportCopyChannelsInGroup.argtypes = [_V, _V, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        io.IOReportCreateSubscription.restype = _V
        io.IOReportCreateSubscription.argtypes = [_V, _V, ctypes.POINTER(_V), ctypes.c_uint64, _V]
        io.IOReportCreateSamples.restype = _V; io.IOReportCreateSamples.argtypes = [_V, _V, _V]
        io.IOReportCreateSamplesDelta.restype = _V; io.IOReportCreateSamplesDelta.argtypes = [_V, _V, _V]
        io.IOReportChannelGetChannelName.restype = _V; io.IOReportChannelGetChannelName.argtypes = [_V]
        io.IOReportSimpleGetIntegerValue.restype = ctypes.c_int64
        io.IOReportSimpleGetIntegerValue.argtypes = [_V, ctypes.c_int32]
        self.cf, self.io = cf, io
        ch = io.IOReportCopyChannelsInGroup(self._s("Energy Model"), None, 0, 0, 0)
        if not ch:
            raise OSError("IOReport 'Energy Model' group unavailable")
        self._mch = cf.CFDictionaryCreateMutableCopy(None, cf.CFDictionaryGetCount(ch), ch)
        self._subbed = _V()
        self._sub = io.IOReportCreateSubscription(None, self._mch, ctypes.byref(self._subbed), 0, None)
        if not self._sub:
            raise OSError("IOReport subscription failed")
        self._key = self._s("IOReportChannels")

    def _s(self, s: str):
        return self.cf.CFStringCreateWithCString(None, s.encode(), _UTF8)

    def _name(self, ref) -> str:
        b = ctypes.create_string_buffer(128)
        self.cf.CFStringGetCString(ref, b, 128, _UTF8)
        return b.value.decode()

    def sample(self):
        return self.io.IOReportCreateSamples(self._sub, self._subbed, None)

    def gpu_nj(self, a, b) -> int:
        d = self.io.IOReportCreateSamplesDelta(a, b, None)
        arr = self.cf.CFDictionaryGetValue(d, self._key)
        total = 0
        for i in range(self.cf.CFArrayGetCount(arr)):
            c = self.cf.CFArrayGetValueAtIndex(arr, i)
            if self._name(self.io.IOReportChannelGetChannelName(c)) == "GPU Energy":
                total += self.io.IOReportSimpleGetIntegerValue(c, 0)
        self.cf.CFRelease(d)
        return total


_IOR: Optional[_IOReport] = None


def _ior() -> Optional[_IOReport]:
    global _IOR
    if _IOR is None:
        try:
            _IOR = _IOReport()
        except Exception:
            return None
    return _IOR


@dataclass
class EnergyReading:
    seconds: float
    cpu_j: float                 # this process, ri_energy_nj (CPU only)
    gpu_j: Optional[float]       # whole GPU, IOReport, minus baseline
    gpu_j_raw: Optional[float]   # whole GPU, IOReport, no subtraction
    gpu_baseline_w: Optional[float]

    def to_dict(self):
        return asdict(self)


class EnergyMeter:
    """``m = EnergyMeter(); m.baseline(); m.start(); ...; r = m.stop()``."""

    def __init__(self):
        self.ior = _ior()
        self.base_w: Optional[float] = None

    def baseline(self, seconds: float = 0.5) -> Optional[float]:
        if not self.ior:
            return None
        a = self.ior.sample(); time.sleep(seconds); b = self.ior.sample()
        self.base_w = self.ior.gpu_nj(a, b) / 1e9 / seconds
        return self.base_w

    def start(self) -> None:
        self.t0 = time.perf_counter()
        self.e0 = energy_nj()
        self.s0 = self.ior.sample() if self.ior else None

    def stop(self) -> EnergyReading:
        dt = time.perf_counter() - self.t0
        cpu = (energy_nj() - self.e0) / 1e9
        if self.ior:
            raw = self.ior.gpu_nj(self.s0, self.ior.sample()) / 1e9
            gpu = raw - (self.base_w or 0.0) * dt
            return EnergyReading(dt, cpu, gpu, raw, self.base_w)
        return EnergyReading(dt, cpu, None, None, None)


def selftest() -> None:
    import mlx.core as mx

    m = EnergyMeter()
    base = m.baseline(0.5)
    a = mx.random.normal((2048, 2048)); mx.eval(a)
    m.start()
    t = time.time()
    while time.time() - t < 1.5:
        b = a
        for _ in range(10):
            b = b @ a
        mx.eval(b)
    r = m.stop()
    m.start()
    x = 0
    t = time.time()
    while time.time() - t < 1.0:
        x += sum(i * i for i in range(20000))
    rc = m.stop()
    assert rc.cpu_j / rc.seconds > 0.3, f"CPU loop should register CPU energy ({rc})"
    if r.gpu_j_raw is not None:
        assert r.gpu_j_raw / r.seconds > (base or 0.0) + 0.2, f"GPU loop should raise GPU energy ({r})"
        print(f"  GPU matmul: GPU {r.gpu_j_raw / r.seconds:.2f} W (baseline {base:.2f} W), "
              f"process CPU {r.cpu_j / r.seconds:.2f} W")
    else:
        print("  IOReport unavailable: GPU energy will be reported as missing")
    print(f"  CPU loop: process CPU {rc.cpu_j / rc.seconds:.2f} W")
    print("energy: ok")


if __name__ == "__main__":
    selftest()
