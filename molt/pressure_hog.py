"""Child process that creates *real* memory pressure on a schedule.

Run as ``python -m molt.pressure_hog '<json>'`` where the JSON is
``{"schedule": [{"t": 0.0, "alloc_mb": 0}, ...], "duration_s": 60}``.

Every allocated page is written to, so the pages are resident rather than merely
reserved — otherwise the OS would hand out address space and the "pressure"
would be fictional.

Claims supported by this module
-------------------------------
* **zero-kill**: this is the antagonist.  The benchmark's headline result is that
  under an adversary that really takes memory away, no Molt process is killed
  and no generation stops.
"""

from __future__ import annotations

import ctypes
import json
import sys
import time
from typing import Dict, List

CHUNK_MB = 32


class Hog:
    """Holds a list of byte buffers whose total size tracks a target."""

    def __init__(self) -> None:
        self.chunks: List[bytearray] = []

    @property
    def mb(self) -> float:
        return len(self.chunks) * CHUNK_MB

    def grow_to(self, target_mb: float) -> None:
        while self.mb < target_mb:
            buf = bytearray(CHUNK_MB * 1024 * 1024)
            # touch one byte per 4 KiB page so the pages are actually faulted in
            for off in range(0, len(buf), 4096):
                buf[off] = 1
            self.chunks.append(buf)

    def shrink_to(self, target_mb: float) -> None:
        while self.chunks and self.mb > target_mb:
            self.chunks.pop()

    def set(self, target_mb: float) -> None:
        if target_mb > self.mb:
            self.grow_to(target_mb)
        elif target_mb < self.mb:
            self.shrink_to(target_mb)


def run(schedule: List[Dict[str, float]], duration_s: float, poll_s: float = 0.1) -> None:
    schedule = sorted(schedule, key=lambda s: s["t"])
    hog = Hog()
    t0 = time.perf_counter()
    while True:
        t = time.perf_counter() - t0
        if t >= duration_s:
            break
        target = 0.0
        for s in schedule:
            if s["t"] <= t:
                target = float(s["alloc_mb"])
            else:
                break
        hog.set(target)
        time.sleep(poll_s)
    hog.set(0.0)


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m molt.pressure_hog '<json>'", file=sys.stderr)
        return 2
    cfg = json.loads(argv[1])
    try:
        run(cfg["schedule"], float(cfg.get("duration_s", 30.0)))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
