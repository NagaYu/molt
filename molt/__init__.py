"""Molt — elastic on-device inference that changes models mid-generation.

    Molt lets a running generation shed its model the way a crab sheds its
    shell: the KV cache — the animal — walks across to the next size down
    without the token stream ever stopping.

Public surface
--------------
:class:`~molt.runtime.MoltRuntime`
    The elastic generation loop.  Policies ``static`` / ``restart`` / ``molt``
    reproduce benchmark conditions A/B, C and D respectively.
:class:`~molt.kv_transplant.KVTransplant`
    Core #1 — move a KV cache between rungs of a tier ladder.
:class:`~molt.migration.MigrationController`
    Core #2 — decide *when* to switch, in both directions.
:class:`~molt.calibration.MigrationCalibrator`
    Core #3 — blend logits across the switch so the distribution does not jump.
:class:`~molt.scheduler.QoSScheduler`
    Core #4 — multi-tenant admission/demotion with a hard no-kill invariant.
"""

from .config import (LADDERS, CalibrationConfig, MigrationConfig, MoltConfig,
                     TierLadder, TierSpec, TransplantConfig, get_ladder)
from .kv_cache import CacheMeta, MoltCache
from .kv_transplant import KVTransplant, TransplantReport

__all__ = [
    "MoltConfig", "TierSpec", "TierLadder", "TransplantConfig",
    "CalibrationConfig", "MigrationConfig", "get_ladder", "LADDERS",
    "MoltCache", "CacheMeta", "KVTransplant", "TransplantReport",
]

__version__ = "0.1.0"
