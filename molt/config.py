"""Tier ladders and global configuration for Molt.

A *tier ladder* is a list of sibling models ordered from best-quality/heaviest
(``tier == 0``) to cheapest (``tier == N-1``).  Molt's whole premise is that a
running generation can hop between rungs of this ladder *between two tokens*,
so every tier must share a tokenizer (identical vocabulary) even when hidden
size, head_dim and depth differ.

Claims supported by this module
-------------------------------
* **no-stall / zero-kill**: the ladder makes a cheaper rung always available, so
  the scheduler never has to choose between "OOM" and "stop generating".
* **low migration cost**: :class:`TierSpec` records the static footprint used by
  the scheduler to decide *which* rung fits the current memory budget without
  actually loading it first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence

import torch

# --------------------------------------------------------------------------
# device / dtype helpers
# --------------------------------------------------------------------------

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_device(spec: str = "auto") -> torch.device:
    """Pick a torch device.

    ``auto`` prefers CUDA, then MPS, then CPU.  The prototype's correctness
    target is CPU (per the project constraints); MPS/CUDA are opportunistic
    speedups only.
    """
    if spec and spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(spec: str, device: torch.device) -> torch.dtype:
    """Map a dtype name to a torch dtype, guarding unsupported combos."""
    if spec == "auto":
        if device.type == "cuda":
            return torch.bfloat16
        if device.type == "mps":
            return torch.float16
        return torch.float32
    dt = _DTYPES.get(spec.lower())
    if dt is None:
        raise ValueError(f"unknown dtype {spec!r}; choose from {sorted(_DTYPES)}")
    if device.type == "cpu" and dt is torch.float16:
        # fp16 matmul on CPU is emulated and extremely slow; refuse loudly.
        raise ValueError("float16 on CPU is not supported by Molt; use float32 or bfloat16")
    return dt


# --------------------------------------------------------------------------
# tiers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TierSpec:
    """One rung of the elastic ladder.

    Attributes
    ----------
    name:
        Stable identifier used in logs, transplant routes and projector files.
    model_id:
        HuggingFace repo id, a local path, or ``synthetic:<key>`` for the tiny
        randomly-initialised models used by the hermetic test-suite.
    quant:
        ``"none"``, ``"int8"`` or ``"int4"``.  Quantised rungs share the parent's
        architecture, which is what makes the *scale re-alignment* transplant
        path (KVTransplant requirement ii) meaningful.
    tier:
        0 = highest quality.  Larger is cheaper.
    quality_rank:
        Optional a-priori quality score (higher is better).  Defaults to
        ``-tier`` and is only used for reporting.
    """

    name: str
    model_id: str
    tier: int
    quant: str = "none"
    parent: Optional[str] = None  # name of the fp tier this was quantised from
    label: str = ""
    est_weight_mb: float = 0.0  # static footprint estimate, used before loading
    quality_rank: Optional[float] = None
    synthetic_config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.quant not in ("none", "int8", "int4"):
            raise ValueError(f"tier {self.name}: bad quant {self.quant!r}")
        if not self.label:
            object.__setattr__(self, "label", self.name)
        if self.quality_rank is None:
            object.__setattr__(self, "quality_rank", float(-self.tier))

    @property
    def is_synthetic(self) -> bool:
        return self.model_id.startswith("synthetic:")

    @property
    def cache_key(self) -> str:
        """Key under which the loaded model is shared between pseudo-apps."""
        return f"{self.model_id}|{self.quant}"


@dataclass
class TierLadder:
    """An ordered ladder plus the tokenizer every rung must agree on."""

    name: str
    tiers: List[TierSpec]
    tokenizer_id: str

    def __post_init__(self) -> None:
        self.tiers = sorted(self.tiers, key=lambda t: t.tier)
        seen = set()
        for t in self.tiers:
            if t.name in seen:
                raise ValueError(f"duplicate tier name {t.name!r}")
            seen.add(t.name)

    def __iter__(self):
        return iter(self.tiers)

    def __len__(self) -> int:
        return len(self.tiers)

    def __getitem__(self, key) -> TierSpec:
        if isinstance(key, int):
            return self.tiers[key]
        return self.by_name(key)

    def by_name(self, name: str) -> TierSpec:
        for t in self.tiers:
            if t.name == name:
                return t
        raise KeyError(f"no tier named {name!r} in ladder {self.name!r}")

    @property
    def top(self) -> TierSpec:
        return self.tiers[0]

    @property
    def bottom(self) -> TierSpec:
        return self.tiers[-1]

    def cheaper_than(self, spec: TierSpec) -> List[TierSpec]:
        return [t for t in self.tiers if t.tier > spec.tier]

    def richer_than(self, spec: TierSpec) -> List[TierSpec]:
        return [t for t in self.tiers if t.tier < spec.tier]

    def next_down(self, spec: TierSpec) -> Optional[TierSpec]:
        below = self.cheaper_than(spec)
        return below[0] if below else None

    def next_up(self, spec: TierSpec) -> Optional[TierSpec]:
        above = self.richer_than(spec)
        return above[-1] if above else None

    def routes(self) -> List[tuple]:
        """All ordered (src, dst) tier-name pairs that a migration may take."""
        out = []
        for a in self.tiers:
            for b in self.tiers:
                if a.name != b.name:
                    out.append((a.name, b.name))
        return out


# --------------------------------------------------------------------------
# built-in ladders
# --------------------------------------------------------------------------

# The reference ladder.  The project brief asks for Llama-3.2-3B / INT4-3B /
# Llama-3.2-1B; Llama is gated on the Hub, so the default ladder uses the
# ungated Qwen2.5 instruct family, which has the exact structural properties the
# experiment needs:
#
#   rung      hidden  layers  kv_heads  head_dim
#   1.5B       1536      28        2       128
#   1.5B-int8  1536      28        2       128   <- same shape: *scale* realign
#   0.5B        896      24        2        64   <- different depth AND head_dim
#
# i.e. one route exercises quantisation-scale re-alignment and the other
# exercises the learned cross-dimension projection + depth remapping.
QWEN_LADDER = TierLadder(
    name="qwen2.5-1.5b-ladder",
    tokenizer_id="Qwen/Qwen2.5-0.5B-Instruct",
    tiers=[
        TierSpec("tier0", "Qwen/Qwen2.5-1.5B-Instruct", tier=0, quant="none",
                 label="Qwen2.5-1.5B (fp)", est_weight_mb=6170.0),
        TierSpec("tier1", "Qwen/Qwen2.5-1.5B-Instruct", tier=1, quant="int8",
                 parent="tier0", label="Qwen2.5-1.5B (int8)", est_weight_mb=1750.0),
        TierSpec("tier2", "Qwen/Qwen2.5-0.5B-Instruct", tier=2, quant="none",
                 label="Qwen2.5-0.5B (fp)", est_weight_mb=1980.0),
    ],
)

# Optional 3B-topped ladder (needs ~6 GB of downloads).  Enable with
# ``--ladder qwen-3b``.
QWEN3B_LADDER = TierLadder(
    name="qwen2.5-3b-ladder",
    tokenizer_id="Qwen/Qwen2.5-0.5B-Instruct",
    tiers=[
        TierSpec("tier0", "Qwen/Qwen2.5-3B-Instruct", tier=0, quant="none",
                 label="Qwen2.5-3B (fp)", est_weight_mb=12400.0),
        TierSpec("tier1", "Qwen/Qwen2.5-3B-Instruct", tier=1, quant="int4",
                 parent="tier0", label="Qwen2.5-3B (int4)", est_weight_mb=2100.0),
        TierSpec("tier2", "Qwen/Qwen2.5-0.5B-Instruct", tier=2, quant="none",
                 label="Qwen2.5-0.5B (fp)", est_weight_mb=1980.0),
    ],
)

# Hermetic ladder for pytest: randomly-initialised Qwen2 models, no downloads,
# millisecond-scale forward passes, but *structurally* identical to the real
# thing (different depth AND different head_dim between tier0 and tier2).
SYNTHETIC_LADDER = TierLadder(
    name="synthetic-tiny",
    tokenizer_id="synthetic:tok",
    tiers=[
        TierSpec(
            "tier0", "synthetic:big", tier=0, quant="none", label="tiny-big",
            est_weight_mb=8.0,
            synthetic_config=dict(hidden_size=128, num_hidden_layers=8, num_attention_heads=4,
                                  num_key_value_heads=2, intermediate_size=256, vocab_size=512),
        ),
        TierSpec(
            "tier1", "synthetic:big", tier=1, quant="int8", parent="tier0", label="tiny-big-int8",
            est_weight_mb=3.0,
            synthetic_config=dict(hidden_size=128, num_hidden_layers=8, num_attention_heads=4,
                                  num_key_value_heads=2, intermediate_size=256, vocab_size=512),
        ),
        TierSpec(
            "tier2", "synthetic:small", tier=2, quant="none", label="tiny-small",
            est_weight_mb=3.0,
            synthetic_config=dict(hidden_size=64, num_hidden_layers=5, num_attention_heads=4,
                                  num_key_value_heads=2, intermediate_size=128, vocab_size=512),
        ),
    ],
)

LADDERS: Dict[str, TierLadder] = {
    "qwen": QWEN_LADDER,
    "qwen-3b": QWEN3B_LADDER,
    "synthetic": SYNTHETIC_LADDER,
}


def get_ladder(name: str) -> TierLadder:
    if name not in LADDERS:
        raise KeyError(f"unknown ladder {name!r}; choose from {sorted(LADDERS)}")
    return LADDERS[name]


# --------------------------------------------------------------------------
# runtime configuration
# --------------------------------------------------------------------------


@dataclass
class TransplantConfig:
    """Knobs of KVTransplant (Molt core #1)."""

    #: how many of the *destination* model's final layers are recomputed
    #: natively instead of being projected (requirement iii).  0 = pure
    #: projection (cheapest); ``n_layers`` would equal a full re-prefill.
    #: Every tier taps its hidden states at ``L - k`` so that exactly one trace
    #: per tier serves every outgoing route, whatever the depths involved.
    recompute_top_k: int = 6
    #: alternative to ``recompute_top_k``: recompute this fraction of depth.
    #: When set (non-None) it wins, which keeps a single knob meaningful across
    #: ladders whose rungs have very different depths.
    recompute_frac: Optional[float] = None
    #: use the learned linear projector when head_dim / depth differ (req. i)
    use_projection: bool = True
    #: rescale K/V when crossing a quantisation boundary (req. ii)
    use_scale_realign: bool = True
    #: un-rotate / re-rotate RoPE around the projection.  Ablation arm: turning
    #: this off is what shows the position-dependence problem is real.
    use_rope_realign: bool = True
    #: fall back to full re-prefill if a projector is missing for the route
    fallback_to_reprefill: bool = False
    #: cap on how many past tokens are carried across (None = all)
    max_carry_tokens: Optional[int] = None
    #: directory holding trained projector state-dicts
    projector_dir: str = "artifacts/projectors"

    def top_k_for(self, n_layers: int) -> int:
        """How many of a model's final layers this config wants recomputed."""
        if self.recompute_frac is not None:
            k = int(round(self.recompute_frac * n_layers))
        else:
            k = int(self.recompute_top_k)
        return max(0, min(k, n_layers))

    def boundary_layer(self, n_layers: int) -> int:
        """Index of the layer whose *input* hidden state is tapped.

        Layers ``[boundary, n_layers)`` are the ones a destination recomputes,
        and a source taps at the same index so a single trace per tier serves
        every outgoing route.  ``boundary == n_layers`` means "no recompute".
        """
        return n_layers - self.top_k_for(n_layers)


@dataclass
class CalibrationConfig:
    """Knobs of MigrationCalibration (Molt core #3)."""

    mode: str = "dual"  # "dual" | "frozen_ref" | "none"
    blend_steps: int = 6
    schedule: str = "cosine"  # "linear" | "cosine" | "exp"
    #: under critical pressure, dual-model blending is downgraded to frozen_ref
    #: (keeping two models resident is exactly what we cannot afford).
    critical_mode: str = "frozen_ref"
    #: cap the blend window when pressure is high
    critical_blend_steps: int = 3


@dataclass
class MigrationConfig:
    """Knobs of MidStreamMigration (Molt core #2)."""

    enabled: bool = True
    allow_up_shift: bool = True
    #: consecutive low-pressure decode steps required before climbing back up
    up_shift_patience: int = 24
    #: minimum decode steps between two migrations (anti-thrash)
    cooldown_steps: int = 8
    #: pressure must exceed (down) / fall below (up) these to trigger
    down_threshold: float = 0.85
    up_threshold: float = 0.55


@dataclass
class MoltConfig:
    ladder: TierLadder = field(default_factory=lambda: QWEN_LADDER)
    device: str = "auto"
    dtype: str = "auto"
    seed: int = 0
    transplant: TransplantConfig = field(default_factory=TransplantConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    migration: MigrationConfig = field(default_factory=MigrationConfig)
    max_new_tokens: int = 96
    temperature: float = 0.0  # greedy by default -> deterministic benchmarks
    top_p: float = 1.0
    log_dir: str = "benchmarks/results"

    @property
    def torch_device(self) -> torch.device:
        return resolve_device(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        return resolve_dtype(self.dtype, self.torch_device)

    def with_(self, **kw) -> "MoltConfig":
        return replace(self, **kw)


def artifacts_root() -> str:
    return os.environ.get("MOLT_ARTIFACTS", "artifacts")
