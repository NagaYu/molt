"""Shared paths and a machine/environment record attached to every result file."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MLX_DIR = ROOT / "mlx"
if str(MLX_DIR) not in sys.path:
    sys.path.insert(0, str(MLX_DIR))

RESULTS = ROOT / "bench" / "results"
CACHE = Path(os.path.expanduser("~/.cache/molt"))
ARTIFACTS = ROOT / "artifacts" / "mlx"


def _sysctl(name: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except Exception:
        return "?"


def machine() -> dict:
    """What a reader needs to judge a timing: chip, RAM, OS, versions, and load
    (other processes on the machine are *not* stopped for these runs)."""
    import mlx.core as mx
    import mlx_lm
    return dict(
        chip=_sysctl("machdep.cpu.brand_string"),
        ram_gib=round(int(_sysctl("hw.memsize") or 0) / 2 ** 30, 1),
        os=f"macOS {platform.mac_ver()[0]}",
        python=platform.python_version(),
        mlx=mx.__version__, mlx_lm=mlx_lm.__version__,
        loadavg=[round(x, 2) for x in os.getloadavg()],
        time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
