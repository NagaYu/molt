"""Loading, sharing and *evicting* the models of a tier ladder.

The zoo is refcounted because the QoS scheduler runs several pseudo-apps that
may sit on the same rung: loading Qwen-0.5B twice would double-count against a
memory budget that is the whole point of the experiment.

Claims supported by this module
-------------------------------
* **zero-kill**: :meth:`ModelZoo.footprint_mb` is the accounting the scheduler
  uses to guarantee it never admits work it cannot hold, and :meth:`release`
  makes eviction actually return memory instead of leaking references.
* **low migration cost**: :meth:`ModelZoo.acquire` is where a migration's
  *model-load* component is timed; keeping the incoming rung warm is what turns
  a multi-second load into a sub-millisecond handle bump.
"""

from __future__ import annotations

import gc
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .adapters import LMAdapter, ModelGeometry
from .config import TierSpec
from .metrics import MB, Stopwatch
from .quantization import (KVQuantStats, QuantReport, collect_kv_quant_stats,
                           quantize_model_)

_SYNTHETIC_SEED = 1234


@dataclass
class LoadedTier:
    """A resident model plus everything Molt needs to reason about it."""

    spec: TierSpec
    model: nn.Module
    adapter: LMAdapter
    quant_report: Optional[QuantReport] = None
    kv_quant_stats: Optional[KVQuantStats] = None
    load_ms: float = 0.0
    refcount: int = 0

    @property
    def geometry(self):
        return self.adapter.geometry

    @property
    def bytes(self) -> int:
        return self.adapter.weight_bytes()

    @property
    def mb(self) -> float:
        return self.bytes / MB

    @property
    def intermediate_size(self) -> int:
        return int(getattr(self.model.config, "intermediate_size",
                           4 * self.geometry.hidden_size))


# --------------------------------------------------------------------------
# synthetic models (hermetic tests, zero downloads)
# --------------------------------------------------------------------------


def build_synthetic_model(cfg_kwargs: dict, dtype: torch.dtype, device: torch.device,
                          seed: int = _SYNTHETIC_SEED) -> nn.Module:
    """A randomly-initialised Qwen2 with the requested geometry.

    Structurally identical to the real ladder (GQA, RoPE, tied embeddings) so the
    transplant code path under test is the same one that runs in the benchmark,
    but small enough that the whole pytest suite finishes in seconds.
    """
    from transformers.models.qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    cfg = Qwen2Config(
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        attn_implementation="eager",
        **cfg_kwargs,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Qwen2ForCausalLM(cfg)
    # Shrink the initialisation so a random model still produces a non-degenerate
    # (non-uniform, non-saturated) next-token distribution.
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.6)
    return model.to(device=device, dtype=dtype).eval()


class SyntheticTokenizer:
    """Minimal byte-ish tokenizer for the synthetic ladder."""

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1

    def encode(self, text: str) -> List[int]:
        return [(b % (self.vocab_size - 1)) for b in text.encode("utf-8")] or [1]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return " ".join(str(int(i)) for i in ids)

    def __call__(self, text, return_tensors=None, **kw):
        ids = self.encode(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": [ids]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(m["content"] for m in messages)


# --------------------------------------------------------------------------
# zoo
# --------------------------------------------------------------------------


class ModelZoo:
    """Refcounted cache of loaded tiers with explicit eviction."""

    def __init__(self, device: torch.device, dtype: torch.dtype,
                 local_files_only: Optional[bool] = None, verbose: bool = False,
                 allow_evict: bool = True,
                 quant_cache_dir: Optional[str] = "artifacts/quantized"):
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.quant_cache_dir = quant_cache_dir
        #: When False, rungs are never unloaded.  Used by the *warm-ladder*
        #: ablation, which isolates the KV-migration cost from the model-load
        #: cost so "transplant beats re-prefill" is measured on its own terms.
        self.allow_evict = allow_evict
        self.local_files_only = (
            local_files_only if local_files_only is not None
            else os.environ.get("MOLT_OFFLINE", "0") == "1"
        )
        self._loaded: Dict[str, LoadedTier] = {}
        self._lock = threading.RLock()
        self.load_events: List[Tuple[str, float]] = []
        #: measured footprints, remembered across evictions so the scheduler can
        #: budget with real numbers instead of the static estimates in TierSpec
        self._measured_mb: Dict[str, float] = {}
        self._measured_geom: Dict[str, ModelGeometry] = {}

    # -- loading -----------------------------------------------------------
    def _load(self, spec: TierSpec) -> LoadedTier:
        with Stopwatch(self.device) as sw:
            cached = (self._load_quantized_from_cache(spec)
                      if spec.quant != "none" and not spec.is_synthetic else None)
            if cached is not None:
                base, report = cached
                base.config.use_cache = True
                adapter = LMAdapter(base)
                lt = LoadedTier(spec=spec, model=base, adapter=adapter,
                                quant_report=report,
                                kv_quant_stats=collect_kv_quant_stats(base, report),
                                load_ms=0.0)
                lt.load_ms = sw.ms
                if self.verbose:
                    print(f"[zoo] loaded {spec.name} ({spec.label}) in {sw.ms:.0f} ms, "
                          f"{lt.mb:.0f} MiB (from quantised cache)")
                self.load_events.append((spec.name, sw.ms))
                return lt
            if spec.is_synthetic:
                base = build_synthetic_model(spec.synthetic_config or {}, self.dtype, self.device)
            else:
                from transformers import AutoModelForCausalLM

                base = AutoModelForCausalLM.from_pretrained(
                    spec.model_id,
                    dtype=self.dtype,
                    local_files_only=self.local_files_only,
                    attn_implementation="eager",
                )
                base = base.to(self.device).eval()
            base.config.use_cache = True
            report = None
            if spec.quant != "none":
                report = self._quantize(base, spec)
                gc.collect()
            adapter = LMAdapter(base)
        lt = LoadedTier(spec=spec, model=base, adapter=adapter, quant_report=report,
                        kv_quant_stats=collect_kv_quant_stats(base, report), load_ms=sw.ms)
        if self.verbose:
            print(f"[zoo] loaded {spec.name} ({spec.label}) in {sw.ms:.0f} ms, {lt.mb:.0f} MiB")
        self.load_events.append((spec.name, sw.ms))
        return lt

    def _quant_cache_path(self, spec: TierSpec) -> Optional[str]:
        if not self.quant_cache_dir or spec.is_synthetic:
            return None
        stem = spec.model_id.replace("/", "__")
        return os.path.join(self.quant_cache_dir, f"{stem}__{spec.quant}.pt")

    def _load_quantized_from_cache(self, spec: TierSpec):
        """Rebuild a quantised rung *without* materialising the fp model.

        The model is instantiated on the ``meta`` device (no allocation), its
        linear layers are swapped for empty :class:`QuantLinear` shells, and the
        cached quantised checkpoint is assigned straight in.  Peak allocation is
        therefore the quantised size, not the full-precision size.

        This matters beyond speed: loading an int8 rung by first allocating the
        fp32 weights would spike to 5.9 GiB under a 2.6 GiB budget.  The
        **zero-kill** invariant has to survive the *loading* of the rung it is
        migrating to, not only the steady state afterwards.
        """
        path = self._quant_cache_path(spec)
        if not path or not os.path.exists(path):
            return None
        try:
            from accelerate import init_empty_weights
            from transformers import AutoConfig, AutoModelForCausalLM

            cfg = AutoConfig.from_pretrained(spec.model_id,
                                             local_files_only=self.local_files_only)
            cfg.attn_implementation = "eager"
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
            report = quantize_model_(model, spec.quant, skeleton_only=True)
            blob = torch.load(path, map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(blob["state"], strict=False,
                                                        assign=True)
            # ``assign=True`` installs the checkpoint's tensors verbatim, which
            # breaks weight tying: lm_head and embed_tokens end up as two
            # separate 900 MiB tables.  Re-tie before measuring anything.
            model.tie_weights()
            still_meta = [n for n, p in list(model.named_parameters())
                          + list(model.named_buffers()) if p.device.type == "meta"]
            if still_meta:
                raise RuntimeError(f"{len(still_meta)} tensors left on meta "
                                   f"(e.g. {still_meta[:3]})")
            model = model.to(self.device).eval()
            report.rel_error = blob.get("rel_error", {})
            if self.verbose:
                print(f"[zoo] rebuilt {spec.name} from cached {spec.quant} checkpoint")
            return model, report
        except Exception as exc:  # pragma: no cover - stale/corrupt cache
            if self.verbose:
                print(f"[zoo] quant cache unusable ({exc!r}); re-quantising")
            return None

    def _quantize(self, model: nn.Module, spec: TierSpec) -> QuantReport:
        """Quantise in place and cache the full quantised checkpoint on disk."""
        report = quantize_model_(model, spec.quant)
        path = self._quant_cache_path(spec)
        if path and not os.path.exists(path):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save({"state": model.state_dict(), "rel_error": report.rel_error},
                           path + ".tmp")
                os.replace(path + ".tmp", path)
                if self.verbose:
                    print(f"[zoo] cached {spec.quant} checkpoint to {path}")
            except Exception:  # pragma: no cover - read-only fs
                pass
        return report

    def acquire(self, spec: TierSpec) -> LoadedTier:
        """Get (loading if necessary) a tier and bump its refcount."""
        with self._lock:
            lt = self._loaded.get(spec.cache_key)
            if lt is None:
                lt = self._load(spec)
                self._loaded[spec.cache_key] = lt
                self._measured_mb[spec.cache_key] = lt.mb
                self._measured_geom[spec.cache_key] = lt.geometry
            else:
                lt.load_ms = 0.0  # warm hit: no load cost this time
            lt.refcount += 1
            return lt

    def release(self, spec: TierSpec, evict_if_unused: bool = False) -> None:
        """Drop a reference; optionally free the weights when nobody holds it."""
        with self._lock:
            lt = self._loaded.get(spec.cache_key)
            if lt is None:
                return
            lt.refcount = max(0, lt.refcount - 1)
            if evict_if_unused and lt.refcount == 0:
                self.evict(spec)

    def evict(self, spec: TierSpec) -> bool:
        """Force-free a tier's weights.  Returns True if anything was freed."""
        with self._lock:
            if not self.allow_evict:
                return False
            lt = self._loaded.pop(spec.cache_key, None)
            if lt is None:
                return False
            if self.verbose:
                print(f"[zoo] evicting {spec.name} ({lt.mb:.0f} MiB)")
            del lt.model, lt.adapter, lt
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            return True

    def evict_all(self) -> None:
        for key in list(self._loaded):
            spec = self._loaded[key].spec
            self.evict(spec)

    # -- accounting --------------------------------------------------------
    def is_resident(self, spec: TierSpec) -> bool:
        return spec.cache_key in self._loaded

    def resident(self) -> List[LoadedTier]:
        return list(self._loaded.values())

    def footprint_mb(self) -> float:
        """Total weight bytes of everything resident (shared rungs counted once)."""
        return sum(lt.mb for lt in self._loaded.values())

    def known_mb(self, spec: TierSpec) -> float:
        """Best available footprint estimate: measured if we ever loaded it,
        otherwise the static estimate the ladder declares.

        The scheduler budgets against this, so an admission decision is never
        based on a number the runtime has already proven wrong.
        """
        if spec.cache_key in self._measured_mb:
            return self._measured_mb[spec.cache_key]
        return spec.est_weight_mb

    def incremental_mb(self, spec: TierSpec) -> float:
        """MiB that loading ``spec`` would *add* (0 if already resident)."""
        return 0.0 if self.is_resident(spec) else self.known_mb(spec)

    def known_geometry(self, spec: TierSpec) -> Optional[ModelGeometry]:
        """Cache geometry of a rung, remembered across evictions (or None)."""
        return self._measured_geom.get(spec.cache_key)

    def kv_mb_per_token(self, spec: TierSpec) -> Optional[float]:
        """Exact MiB of KV cache one token costs on this rung.

        The scheduler admits work against this rather than a hand-tuned "MiB per
        token" constant: on a long prompt the difference between a guess and the
        real number is the difference between a safe admission and an OOM at
        prefill time.
        """
        geom = self.known_geometry(spec)
        if geom is None:
            return None
        kv = geom.bytes_per_token(self.dtype)
        # hidden-state trace kept for the cheap top-k recompute path
        trace = geom.hidden_size * torch.finfo(self.dtype).bits // 8
        return (kv + trace) / MB

    def exclusive_mb(self, spec: TierSpec) -> float:
        """MiB that releasing one reference to ``spec`` would actually free.

        Zero when another tenant still holds it — which is why demoting a single
        app off a shared rung frees nothing and the scheduler demotes by group.
        """
        with self._lock:
            lt = self._loaded.get(spec.cache_key)
            if lt is None or lt.refcount > 1:
                return 0.0
            return lt.mb

    def measure_footprint(self, spec: TierSpec) -> float:
        """Actual MiB of a tier, loading it if needed (used to calibrate the
        scheduler's static estimates)."""
        lt = self.acquire(spec)
        mb = lt.mb
        self.release(spec)
        return mb

    def describe(self) -> str:
        parts = [f"{lt.spec.name}:{lt.mb:.0f}MiB(rc={lt.refcount})"
                 for lt in self._loaded.values()]
        return f"ModelZoo[{self.footprint_mb():.0f} MiB] " + " ".join(parts)


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------


_TOKENIZER_CACHE: Dict[str, object] = {}


def load_tokenizer(tokenizer_id: str, local_files_only: Optional[bool] = None):
    """Every rung of a ladder must share this tokenizer — a migration mid-stream
    is only meaningful when token ids mean the same thing on both sides."""
    if tokenizer_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[tokenizer_id]
    if tokenizer_id.startswith("synthetic:"):
        tok = SyntheticTokenizer()
    else:
        from transformers import AutoTokenizer

        lfo = (local_files_only if local_files_only is not None
               else os.environ.get("MOLT_OFFLINE", "0") == "1")
        tok = AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=lfo)
    _TOKENIZER_CACHE[tokenizer_id] = tok
    return tok


def assert_ladder_compatible(zoo: ModelZoo, ladder) -> None:
    """Fail loudly if two rungs disagree on vocabulary."""
    vocabs = {}
    for spec in ladder:
        lt = zoo.acquire(spec)
        vocabs[spec.name] = lt.geometry.vocab_size
        zoo.release(spec)
    uniq = set(vocabs.values())
    if len(uniq) != 1:
        raise ValueError(
            "tier ladder is not migration-compatible: vocab sizes differ "
            f"{vocabs}. Mid-stream migration requires a shared tokenizer."
        )
