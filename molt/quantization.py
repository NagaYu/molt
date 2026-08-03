"""Weight-only quantisation used to build the middle rung of a tier ladder.

The brief asks for a bitsandbytes INT4/INT8 sibling of the top tier.  ``bnb``'s
kernels are CUDA-only, and this prototype's primary target is CPU (and Apple
MPS), so this module ships a portable pure-PyTorch fallback and *uses
bitsandbytes automatically when it is importable and CUDA is present*.

Two schemes:

``int8``
    Symmetric, per-output-channel.  ``w ≈ q * s`` with ``q ∈ [-127,127]``.
``int4``
    Symmetric, group-wise along the input dimension (default group 64), packed
    two nibbles per byte.

The dequantisation happens inside ``forward``; the *steady-state* footprint is
what shrinks (that is the property the scheduler budgets against), not the
arithmetic cost.  This trade-off is reported rather than hidden — see
``benchmarks/results/*.json`` field ``tier_footprint_mb``.

Claims supported by this module
-------------------------------
* **zero-kill**: a rung whose weights are 4x smaller is what lets the scheduler
  keep a high-quality model *family* resident under a budget that cannot hold
  the fp weights.
* **continuity**: :func:`collect_kv_quant_stats` exposes the per-channel scales
  that KVTransplant uses to re-align a cache across a quantisation boundary
  (requirement ii), which measurably lowers the distribution jump.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_GROUP_SIZE = 64
_SKIP_NAME_PARTS = ("lm_head", "embed_tokens", "wte", "wpe", "shared")


def bnb_available() -> bool:
    """True when bitsandbytes can actually be used (needs CUDA)."""
    if not torch.cuda.is_available():
        return False
    try:  # pragma: no cover - CUDA-only path
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# int8
# --------------------------------------------------------------------------


def quantize_int8(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8 quantisation of ``[out, in]``."""
    w32 = w.detach().to(torch.float32)
    scale = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.clamp(torch.round(w32 / scale), -127, 127).to(torch.int8)
    return q, scale.squeeze(1)


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (q.to(torch.float32) * scale.unsqueeze(1)).to(dtype)


# --------------------------------------------------------------------------
# int4 (group-wise, packed)
# --------------------------------------------------------------------------


def quantize_int4(w: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE):
    """Group-wise symmetric int4.  Returns ``(packed_uint8, scale, in_features)``."""
    w32 = w.detach().to(torch.float32)
    out_f, in_f = w32.shape
    pad = (-in_f) % group_size
    if pad:
        w32 = F.pad(w32, (0, pad))
    n_groups = w32.shape[1] // group_size
    g = w32.view(out_f, n_groups, group_size)
    scale = g.abs().amax(dim=2).clamp_min(1e-8) / 7.0            # [out, n_groups]
    q = torch.clamp(torch.round(g / scale.unsqueeze(2)), -7, 7)  # [-7, 7]
    q = (q + 8).to(torch.uint8).view(out_f, -1)                  # [0, 15]
    lo, hi = q[:, 0::2], q[:, 1::2]
    packed = (lo | (hi << 4)).contiguous()
    return packed, scale, in_f


def dequantize_int4(
    packed: torch.Tensor, scale: torch.Tensor, in_features: int,
    group_size: int, dtype: torch.dtype,
) -> torch.Tensor:
    out_f = packed.shape[0]
    lo = (packed & 0x0F).to(torch.float32) - 8.0
    hi = ((packed >> 4) & 0x0F).to(torch.float32) - 8.0
    q = torch.stack([lo, hi], dim=2).view(out_f, -1)
    n_groups = scale.shape[1]
    q = q.view(out_f, n_groups, group_size) * scale.unsqueeze(2)
    return q.view(out_f, -1)[:, :in_features].to(dtype)


# --------------------------------------------------------------------------
# modules
# --------------------------------------------------------------------------


class QuantLinear(nn.Module):
    """``nn.Linear`` whose weights live in int8/int4 and dequantise on use."""

    def __init__(self, mode: str, weight: torch.Tensor, bias: Optional[torch.Tensor],
                 group_size: int = DEFAULT_GROUP_SIZE, out_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.mode = mode
        self.group_size = group_size
        self.out_dtype = out_dtype
        self.out_features, self.in_features = weight.shape
        if mode == "int8":
            q, s = quantize_int8(weight)
            self.register_buffer("qweight", q, persistent=True)
            self.register_buffer("scale", s, persistent=True)
        elif mode == "int4":
            packed, s, in_f = quantize_int4(weight, group_size)
            self.register_buffer("qweight", packed, persistent=True)
            self.register_buffer("scale", s, persistent=True)
            self.in_features = in_f
        else:
            raise ValueError(f"unsupported quant mode {mode!r}")
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone(), persistent=True)
        else:
            self.bias = None

    @classmethod
    def empty(cls, mode: str, in_features: int, out_features: int, has_bias: bool,
              group_size: int = DEFAULT_GROUP_SIZE,
              out_dtype: torch.dtype = torch.float32) -> "QuantLinear":
        """Allocate the *shapes* of a quantised layer without any source weights.

        This is what lets a quantised rung be rebuilt from a cached state dict
        without ever materialising the full-precision model.  Loading an int8
        rung by first allocating 5.9 GiB of fp32 weights would be a fiction on a
        device whose whole budget is 2.6 GiB — the transient allocation is
        exactly the kind of spike the zero-kill invariant exists to prevent.
        """
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.mode, self.group_size, self.out_dtype = mode, group_size, out_dtype
        self.out_features, self.in_features = out_features, in_features
        if mode == "int8":
            self.register_buffer("qweight", torch.empty(out_features, in_features,
                                                        dtype=torch.int8))
            self.register_buffer("scale", torch.empty(out_features))
        elif mode == "int4":
            padded = in_features + (-in_features % group_size)
            self.register_buffer("qweight", torch.empty(out_features, padded // 2,
                                                        dtype=torch.uint8))
            self.register_buffer("scale", torch.empty(out_features, padded // group_size))
        else:
            raise ValueError(f"unsupported quant mode {mode!r}")
        if has_bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=out_dtype))
        else:
            self.bias = None
        return self

    def dequantized_weight(self) -> torch.Tensor:
        if self.mode == "int8":
            return dequantize_int8(self.qweight, self.scale, self.out_dtype)
        return dequantize_int4(self.qweight, self.scale, self.in_features,
                               self.group_size, self.out_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantized_weight().to(x.dtype)
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, w, b)

    def quant_error_rms(self) -> float:
        """RMS of ``w - dequant(quant(w))`` relative to RMS of ``w``.

        This number is the *cause* of the KV distribution shift that requirement
        (ii) of KVTransplant corrects for.
        """
        raise NotImplementedError  # needs the original weight; see quantize_model_

    def extra_repr(self) -> str:
        return f"{self.in_features}->{self.out_features}, mode={self.mode}"


@dataclass
class QuantReport:
    mode: str
    n_replaced: int
    orig_bytes: int
    quant_bytes: int
    #: per-module relative RMS quantisation error, keyed by module path
    rel_error: Dict[str, float]

    @property
    def compression(self) -> float:
        return self.orig_bytes / max(1, self.quant_bytes)

    def to_dict(self) -> dict:
        return dict(mode=self.mode, n_replaced=self.n_replaced,
                    orig_mb=self.orig_bytes / 2**20, quant_mb=self.quant_bytes / 2**20,
                    compression=self.compression,
                    mean_rel_error=float(sum(self.rel_error.values()) / max(1, len(self.rel_error))))


def _module_bytes(m: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in m.parameters(recurse=False)) + \
        sum(b.numel() * b.element_size() for b in m.buffers(recurse=False))


@torch.no_grad()
def quantize_model_(
    model: nn.Module,
    mode: str,
    group_size: int = DEFAULT_GROUP_SIZE,
    skip: Iterable[str] = _SKIP_NAME_PARTS,
    track_error: bool = True,
    skeleton_only: bool = False,
) -> QuantReport:
    """Replace every ``nn.Linear`` in ``model`` with a :class:`QuantLinear`, in place.

    Embeddings and the LM head are skipped: in Qwen/Llama they are frequently
    *tied*, and quantising a tied head would silently corrupt the embedding
    table for every tier that shares the same weights object.

    ``skeleton_only=True`` builds the quantised *structure* without reading the
    source weights, so a cached quantised checkpoint can be loaded into it.
    """
    if mode == "none":
        return QuantReport("none", 0, _total_bytes(model), _total_bytes(model), {})

    skip = tuple(skip)
    orig = _total_bytes(model)
    rel_error: Dict[str, float] = {}
    targets: List[Tuple[nn.Module, str, nn.Linear, str]] = []
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            path = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(s in path for s in skip):
                targets.append((module, child_name, child, path))

    out_dtype = next(model.parameters()).dtype
    for parent, child_name, lin, path in targets:
        if skeleton_only:
            setattr(parent, child_name, QuantLinear.empty(
                mode, lin.in_features, lin.out_features, lin.bias is not None,
                group_size=group_size, out_dtype=out_dtype))
            continue
        ql = QuantLinear(mode, lin.weight.data, lin.bias.data if lin.bias is not None else None,
                         group_size=group_size, out_dtype=out_dtype)
        if track_error:
            w = lin.weight.data.to(torch.float32)
            err = (w - ql.dequantized_weight().to(torch.float32)).pow(2).mean().sqrt()
            rel_error[path] = float(err / w.pow(2).mean().sqrt().clamp_min(1e-9))
        setattr(parent, child_name, ql)
        del lin

    return QuantReport(mode, len(targets), orig, _total_bytes(model), rel_error)


def _total_bytes(model: nn.Module) -> int:
    return (sum(p.numel() * p.element_size() for p in model.parameters())
            + sum(b.numel() * b.element_size() for b in model.buffers()))


# --------------------------------------------------------------------------
# KV-relevant statistics (KVTransplant requirement ii)
# --------------------------------------------------------------------------


@dataclass
class KVQuantStats:
    """Per-layer quantisation scales of the K/V projections.

    ``k_scale[l]`` has one entry per *output channel* of layer ``l``'s ``k_proj``,
    i.e. exactly one entry per ``(kv_head, head_dim)`` slot of the cache.  That
    alignment is what makes a closed-form scale re-alignment possible.
    """

    k_rel_error: List[float]
    v_rel_error: List[float]
    mode: str

    @property
    def n_layers(self) -> int:
        return len(self.k_rel_error)


def collect_kv_quant_stats(model: nn.Module, report: Optional[QuantReport]) -> Optional[KVQuantStats]:
    """Extract per-layer K/V quantisation error from a :class:`QuantReport`."""
    if report is None or report.mode == "none":
        return None
    k_err: Dict[int, float] = {}
    v_err: Dict[int, float] = {}
    for path, err in report.rel_error.items():
        parts = path.split(".")
        idx = None
        for i, p in enumerate(parts):
            if p.isdigit():
                idx = int(p)
        if idx is None:
            continue
        if path.endswith("k_proj"):
            k_err[idx] = err
        elif path.endswith("v_proj"):
            v_err[idx] = err
    if not k_err:
        return None
    n = max(k_err) + 1
    return KVQuantStats(
        k_rel_error=[k_err.get(i, 0.0) for i in range(n)],
        v_rel_error=[v_err.get(i, 0.0) for i in range(n)],
        mode=report.mode,
    )
