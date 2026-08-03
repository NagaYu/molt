"""The elastic generation loop — where the four Molt cores meet.

A :class:`Generation` is a *steppable* object rather than a blocking loop, so a
scheduler can interleave several of them and demote one mid-answer.  One decode
step is::

    observe pressure -> ask the controller -> (maybe migrate) -> forward ->
    calibrate logits -> sample -> record telemetry

The four benchmark conditions are four :class:`Policy` values over the same
loop, which is what makes their comparison meaningful — they share the sampler,
the stopwatch, the memory accountant and the pressure source.

============  ==================================================================
``STATIC``    never change tier.  ``tier0`` = condition **A**, ``tier2`` = **B**.
``RESTART``   change tier under pressure but **discard the cache** and re-prefill
              the whole prefix = condition **C**.
``MOLT``      change tier and **transplant the cache** = condition **D**.
============  ==================================================================

Claims supported by this module
-------------------------------
* **no-stall**: :meth:`Generation.step` emits a token on every call, including
  the call during which a migration happened; the per-token latency series is
  the evidence.
* **zero-kill**: :class:`MemoryBroker` enforces the budget *before* an
  allocation, and only ``STATIC`` is allowed to die from it — which is the
  point of condition A.
* **low migration cost / continuity**: every migration records a
  :class:`~molt.kv_transplant.TransplantReport` and opens a calibration window.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .calibration import MigrationCalibrator
from .config import MoltConfig, TierLadder, TierSpec
from .kv_cache import CacheMeta, MoltCache
from .kv_transplant import (KVTransplant, ProjectorRegistry, TierRef,
                            TransplantReport)
from .metrics import (MB, DiscontinuityProbe, EventLog, LatencyRecorder,
                      MemoryProbe, Stopwatch, TokenRecord, js_divergence, now,
                      repetition_rate, sync)
from .migration import (MigrationController, MigrationDecision,
                        RequestBoundaryController)
from .model_zoo import LoadedTier, ModelZoo
from .projector import rebase_rope


class Policy(str, Enum):
    STATIC = "static"      # conditions A and B
    RESTART = "restart"    # condition C
    MOLT = "molt"          # condition D


class OOMKilled(RuntimeError):
    """The process footprint exceeded the budget and the OS reclaimed it.

    In a real device this is jetsam.  Here it is raised so the benchmark can
    count *forced terminations* — the metric condition A exists to produce and
    conditions B/C/D exist to avoid.
    """

    def __init__(self, usage_mb: float, budget_mb: float, tier: str, token_index: int):
        super().__init__(
            f"OOM: {usage_mb:.0f} MiB in use > {budget_mb:.0f} MiB budget "
            f"(tier {tier}, token {token_index})")
        self.usage_mb, self.budget_mb, self.tier, self.token_index = (
            usage_mb, budget_mb, tier, token_index)


# --------------------------------------------------------------------------
# memory accounting
# --------------------------------------------------------------------------


class MemoryBroker:
    """Single source of truth for "how much are we using, and may we?".

    Tracks resident model weights (via the zoo, shared rungs counted once) plus
    every live KV cache.  The **zero-kill** invariant is
    ``usage_mb() <= budget_mb`` at every decode step, and this class is where it
    is checked.
    """

    def __init__(self, zoo: ModelZoo, log: Optional[EventLog] = None):
        self.zoo = zoo
        self.log = log
        self._caches: Dict[str, float] = {}
        self.peak_mb = 0.0
        self.violations: List[Dict[str, Any]] = []

    def set_cache_mb(self, owner: str, mb: float) -> None:
        self._caches[owner] = mb
        self.peak_mb = max(self.peak_mb, self.usage_mb())

    def drop_cache(self, owner: str) -> None:
        self._caches.pop(owner, None)

    @property
    def cache_mb(self) -> float:
        return sum(self._caches.values())

    def usage_mb(self) -> float:
        return self.zoo.footprint_mb() + self.cache_mb

    def pressure(self, budget_mb: float) -> float:
        """0 = empty, 1 = exactly at the budget, >1 = would be killed."""
        if budget_mb <= 0:
            return float("inf")
        return self.usage_mb() / budget_mb

    def would_fit(self, extra_mb: float, budget_mb: float) -> bool:
        return (self.usage_mb() + extra_mb) <= budget_mb

    def check(self, budget_mb: float, tier: str, token_index: int,
              enforce: bool = True) -> None:
        use = self.usage_mb()
        self.peak_mb = max(self.peak_mb, use)
        if use > budget_mb:
            rec = dict(usage_mb=use, budget_mb=budget_mb, tier=tier, token=token_index)
            self.violations.append(rec)
            if self.log:
                self.log.log("budget_violation", **rec)
            if enforce:
                raise OOMKilled(use, budget_mb, tier, token_index)

    def summary(self) -> Dict[str, Any]:
        return dict(peak_mb=self.peak_mb, n_violations=len(self.violations),
                    violations=self.violations[:16])


# --------------------------------------------------------------------------
# a request and its result
# --------------------------------------------------------------------------


@dataclass
class GenerationRequest:
    app_id: str
    prompt: str
    max_new_tokens: int = 96
    priority: int = 0            # 0 = foreground/interactive, higher = batch
    kind: str = "chat"           # "chat" | "batch"
    reference: Optional[str] = None   # expected answer, for accuracy scoring
    stop_strings: Sequence[str] = ()


@dataclass
class GenerationResult:
    app_id: str
    prompt: str
    text: str
    token_ids: List[int]
    tier_per_token: List[str]
    latency: LatencyRecorder
    discontinuity: DiscontinuityProbe
    transplants: List[TransplantReport]
    migrations: List[dict]
    killed: bool = False
    kill_reason: str = ""
    finished: bool = False
    peak_mb: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self, stall_threshold_ms: float = 0.0) -> Dict[str, Any]:
        tp_ms = sum(t.wall_ms for t in self.transplants)
        return dict(
            app_id=self.app_id,
            killed=self.killed, kill_reason=self.kill_reason, finished=self.finished,
            migration_ms_total=tp_ms,
            migration_ms_mean=tp_ms / len(self.transplants) if self.transplants else 0.0,
            transplant_flops_saving=(
                sum(t.flops_saving for t in self.transplants) / len(self.transplants)
                if self.transplants else 0.0),
            repetition_rate_3=repetition_rate(self.token_ids, 3),
            peak_mb=self.peak_mb,
            tiers_used=sorted(set(self.tier_per_token)),
            frac_tokens_on_top_tier=(
                sum(1 for t in self.tier_per_token if t == "tier0") / len(self.tier_per_token)
                if self.tier_per_token else 0.0),
            **self.latency.summary(stall_threshold_ms),
            **self.discontinuity.summary(),
        )


# --------------------------------------------------------------------------
# one live generation
# --------------------------------------------------------------------------


class Generation:
    """A single steppable generation that can change tier between tokens."""

    def __init__(
        self,
        request: GenerationRequest,
        cfg: MoltConfig,
        zoo: ModelZoo,
        tokenizer,
        registry: ProjectorRegistry,
        broker: MemoryBroker,
        policy: Policy,
        start_tier: TierSpec,
        log: EventLog,
        controller: Optional[MigrationController] = None,
        enforce_oom: bool = True,
        pressure_fn: Optional[Callable[[], float]] = None,
        budget_fn: Optional[Callable[[], float]] = None,
        self_migrate: bool = True,
    ):
        self.req = request
        self.cfg = cfg
        self.zoo = zoo
        self.tok = tokenizer
        self.registry = registry
        self.broker = broker
        self.policy = policy
        self.log = log
        self.device = cfg.torch_device
        self.enforce_oom = enforce_oom
        self.pressure_fn = pressure_fn or (lambda: 0.0)
        self.budget_fn = budget_fn or (lambda: float("inf"))
        #: When False this generation only migrates on command.  Under the QoS
        #: scheduler there must be exactly one decision authority: three tenants
        #: each reacting to *global* pressure independently would each load a
        #: destination rung while the others still hold the source, and the
        #: transient sum is what kills the process.
        self.self_migrate = self_migrate

        self.ladder: TierLadder = cfg.ladder
        self.controller = controller or MigrationController(self.ladder, cfg.migration)
        # let the policy see the budget: without this it can only step one rung
        # per cooldown, which loses a race against a single large pressure event
        self.controller.affordable = lambda t: self.can_afford(t, evict_first=True)
        self.calibrator = MigrationCalibrator(cfg.calibration)
        self.transplanter = KVTransplant(cfg.transplant)

        # live state -------------------------------------------------------
        self.tier: TierSpec = start_tier
        self.loaded: Optional[LoadedTier] = None
        self.cache: Optional[MoltCache] = None
        self.token_ids: List[int] = []
        self.prompt_ids: List[int] = []
        self.tier_per_token: List[str] = []
        self._hooks: List[Any] = []
        self._trace_layer: Optional[int] = None
        self._pending_trace: Optional[torch.Tensor] = None
        #: logits already computed and not yet consumed (produced by prefill)
        self._pending_logits: Optional[torch.Tensor] = None
        #: the token that is *not yet* in the cache and must be fed next.
        #: ``None`` means the cache is fully up to date and ``_pending_logits``
        #: holds the next distribution.  Keeping this explicit is what makes a
        #: migration safe at any token index, including index 0.
        self._next_input_id: Optional[int] = None
        #: kept alive only while a ``dual`` calibration window is open
        self._shadow: Optional[Tuple[LoadedTier, MoltCache]] = None
        #: the outgoing model's distribution at the switch point, held for one
        #: step so the incoming model's first output can be compared with it
        self._handoff_ref: Optional[torch.Tensor] = None
        #: wall-ms of measurement-only work banked for the current token
        self._instrument_ms = 0.0
        #: measure the handoff (one extra decode step of the outgoing model per
        #: migration).  On by default for experiments; a deployment that does not
        #: need the metric can turn it off.
        self.measure_handoff = True

        # telemetry --------------------------------------------------------
        self.latency = LatencyRecorder()
        self.disc = DiscontinuityProbe(window=max(4, cfg.calibration.blend_steps))
        self.transplants: List[TransplantReport] = []
        self.done = False
        self.killed = False
        self.kill_reason = ""
        self.pending_migration_ms = 0.0
        #: set by :class:`~molt.scheduler.QoSScheduler` to command a tier change
        self.forced_target: Optional[TierSpec] = None
        self.forced_reason: str = ""
        self.paused = False
        self.n_pauses = 0
        self._paused_ref: Optional[TierRef] = None

    # -- model plumbing ----------------------------------------------------
    def _attach(self, spec: TierSpec) -> LoadedTier:
        lt = self.zoo.acquire(spec)
        self.loaded = lt
        self.tier = spec
        k = self.cfg.transplant.top_k_for(lt.geometry.n_layers)
        self._trace_layer = (lt.geometry.n_layers - k) if k > 0 else None
        self._install_hooks()
        return lt

    def _install_hooks(self) -> None:
        self._remove_hooks()
        if self._trace_layer is None or self.policy is not Policy.MOLT:
            return
        if not self.loaded.adapter.supports_partial_recompute:
            self._trace_layer = None
            return
        self._pending_trace = None

        def sink(h: torch.Tensor) -> None:
            self._pending_trace = h

        self._hooks.append(self.loaded.adapter.capture_hook(self._trace_layer, sink))

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _detach(self, evict: bool = True) -> None:
        self._remove_hooks()
        if self.loaded is not None:
            self.zoo.release(self.tier, evict_if_unused=evict)
            self.loaded = None

    # -- accounting --------------------------------------------------------
    def _sync_cache_accounting(self) -> None:
        mb = 0.0
        if self.cache is not None:
            mb += self.cache.nbytes() / MB
        if self._shadow is not None:
            mb += self._shadow[1].nbytes() / MB
        self.broker.set_cache_mb(self.req.app_id, mb)

    def _check_budget(self) -> None:
        self.broker.check(self.budget_fn(), self.tier.name, len(self.token_ids),
                          enforce=self.enforce_oom)

    def can_afford(self, target: TierSpec, evict_first: bool) -> bool:
        """Would moving to ``target`` keep us inside the budget?

        Climbing back up is only elastic if it is also *safe*.  A pressure
        reading below the up-threshold says "there is slack"; it does not say
        "there is 2.5x more slack", which is what a rung three times the size
        actually needs.  Without this veto the controller oscillates: it
        up-shifts on a low reading, immediately breaches the budget, gets
        demoted, and thrashes — the exact failure the hysteresis was meant to
        prevent but cannot, because hysteresis is about *time*, not *size*.
        """
        budget = self.budget_fn()
        if budget == float("inf"):
            return True
        inc = self.zoo.incremental_mb(target)
        freed = self.zoo.exclusive_mb(self.tier) if evict_first else 0.0
        return (self.broker.usage_mb() - freed + inc) <= budget

    # -- prefill -----------------------------------------------------------
    def prefill(self) -> None:
        """Read the prompt and produce the first logits.

        The cost measured here is TTFT.  Condition C pays it a second time at
        every pressure event; conditions A/B/D pay it once.
        """
        enc = self.tok(self.req.prompt, return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        self.prompt_ids = ids[0].tolist()

        self._attach(self.tier)
        self._check_budget()

        self._pending_trace = None
        with Stopwatch(self.device) as sw:
            with torch.no_grad():
                out = self.loaded.model(ids, use_cache=True)
        meta = CacheMeta(self.tier.name, self.loaded.geometry, list(self.prompt_ids),
                         [self._trace_layer] if self._trace_layer is not None else [],
                         len(self.prompt_ids))
        traces = {}
        if self._trace_layer is not None and self._pending_trace is not None:
            traces[self._trace_layer] = self._pending_trace.detach()
        self.cache = MoltCache.from_hf(out.past_key_values, meta, traces)
        self._pending_logits = out.logits[:, -1, :].detach()
        self._next_input_id = None      # cache already covers the whole prompt
        self.calibrator.observe(self._pending_logits)
        self._sync_cache_accounting()
        self._check_budget()
        self.log.log("prefill", app=self.req.app_id, tier=self.tier.name,
                     n_prompt_tokens=len(self.prompt_ids), ms=sw.ms)

    # -- migration ---------------------------------------------------------
    def _migrate(self, decision: MigrationDecision) -> float:
        """Switch tiers.  Returns wall-ms spent, which is charged to this token.

        Under ``critical`` pressure the outgoing weights are evicted *before* the
        incoming ones are loaded — possible only because
        :class:`~molt.kv_transplant.TierRef` keeps the few kilobytes of RoPE
        state the transplant actually needs.  That is what keeps peak memory at
        ``max(src, dst)`` instead of ``src + dst`` and preserves **zero-kill**.
        """
        target = decision.target
        src_ref = TierRef.from_loaded(self.loaded, snapshot_rope=True)
        old_tier_name = self.tier.name
        budget = self.budget_fn()

        # The logits held by ``_pending_logits`` belong to the *outgoing* model,
        # so they must not be emitted by the incoming one.  Drop the last cached
        # position and let the destination re-derive it in the next ordinary
        # decode step — one token of work, and it anchors the destination on a
        # natively-computed position.
        old_cache = self.cache
        if self._next_input_id is None and old_cache.seq_len > 1:
            self._next_input_id = old_cache.meta.token_ids[-1]
            old_cache = old_cache.truncate(old_cache.seq_len - 1)
        self._pending_logits = None

        # ---- instrumentation: what would the outgoing model have said? ------
        # Taken *before* the source is evicted, over exactly the position the
        # incoming model is about to answer.  This is the continuity metric; it
        # costs one decode step of the outgoing model and no extra residency,
        # and it works for every policy — including RESTART, whose value is the
        # "two different models disagree this much anyway" floor that a
        # transplant should be compared against rather than to zero.
        if self.measure_handoff and self._next_input_id is not None:
            try:
                with Stopwatch(self.device) as probe_sw:
                    inp = torch.tensor([[self._next_input_id]], dtype=torch.long,
                                       device=self.device)
                    with torch.no_grad():
                        self._handoff_ref = self.loaded.model(
                            inp, past_key_values=old_cache.to_hf(),
                            use_cache=False).logits[:, -1, :].detach().cpu()
                self._instrument_ms += probe_sw.ms
            except Exception:  # pragma: no cover - instrumentation must not break a run
                self._handoff_ref = None

        # Can we afford to keep the outgoing model alive for a dual blend?
        # If not, the source is evicted *before* the destination is loaded and
        # the calibrator falls back to its memory-free mode.
        keep_old = (
            self.policy is Policy.MOLT
            and self.cfg.calibration.mode == "dual"
            and not decision.critical
            and self.broker.would_fit(self.zoo.incremental_mb(target), budget)
        )

        with Stopwatch(self.device) as sw:
            if keep_old:
                shadow_loaded = self.loaded
                shadow_cache = old_cache
                self._remove_hooks()
                self.loaded = None
            else:
                shadow_loaded = shadow_cache = None
                self._detach(evict=True)

            new_lt = self._attach(target)

            if self.policy is Policy.MOLT:
                proj = self.registry.get(old_tier_name, target.name)
                try:
                    new_cache, rep = self.transplanter.transplant(
                        old_cache, src_ref, new_lt, proj, self.device)
                except Exception as exc:  # pragma: no cover - safety net
                    self.log.log("transplant_failed", app=self.req.app_id,
                                 error=repr(exc), src=old_tier_name, dst=target.name)
                    new_cache, rep = self.transplanter.reprefill(
                        list(old_cache.meta.token_ids), new_lt, self.device,
                        old_cache.meta.n_prompt_tokens, self._trace_layer, old_tier_name)
                    rep.degraded_reason = f"transplant raised {type(exc).__name__}"
            else:
                # condition C: the cache is thrown away and re-read from scratch
                new_cache, rep = self.transplanter.reprefill(
                    list(old_cache.meta.token_ids), new_lt, self.device,
                    old_cache.meta.n_prompt_tokens, self._trace_layer, old_tier_name)

            if keep_old:
                # the shadow must cover exactly the same positions as the new
                # cache, or the two models would disagree about what has been
                # attended to when they decode the same token side by side
                shadow_cache = old_cache
            else:
                old_cache.free()
            self.cache = new_cache
            self._shadow = (shadow_loaded, shadow_cache) if keep_old else None

        rep.model_load_ms = new_lt.load_ms
        self.transplants.append(rep)
        self.controller.note_migration(decision, old_tier_name, len(self.token_ids), sw.ms)
        self._sync_cache_accounting()
        self.log.log("migration", app=self.req.app_id, src=old_tier_name, dst=target.name,
                     direction=decision.direction, reason=decision.reason,
                     token=len(self.token_ids), wall_ms=sw.ms, method=rep.method,
                     transplant_ms=rep.wall_ms, load_ms=rep.model_load_ms,
                     flops_saving=rep.flops_saving, degraded=rep.degraded_reason,
                     critical=decision.critical)

        # open the calibration window (core #3)
        self.calibrator.begin(pressure=self.pressure_fn(), critical=decision.critical or not keep_old)
        self.disc.mark_migration(len(self.token_ids))
        return sw.ms

    def apply_forced_migration(self) -> float:
        """Execute a scheduler-commanded tier change **immediately**.

        The QoS layer cannot wait for this generation's turn in the round-robin:
        the budget has to be met *before anyone steps*, or a different app pays
        for this one's footprint.  The wall time is banked and charged to this
        generation's next token, so the latency series stays honest.
        """
        if self.forced_target is None or self.paused or self.cache is None:
            self.forced_target, self.forced_reason = None, ""
            return 0.0
        decision = self.controller.decide(
            self.tier, self.pressure_fn(), self.forced_target, self.forced_reason)
        self.forced_target, self.forced_reason = None, ""
        if not decision:
            return 0.0
        ms = self._migrate(decision)
        self.pending_migration_ms += ms
        return ms

    def _release_shadow(self) -> None:
        if self._shadow is None:
            return
        shadow_loaded, shadow_cache = self._shadow
        self._shadow = None
        if shadow_cache is not None:
            shadow_cache.free()
        if shadow_loaded is not None:
            self.zoo.release(shadow_loaded.spec, evict_if_unused=True)
        self._sync_cache_accounting()

    MIN_CONTEXT_TOKENS = 16

    def shed_context(self, target_free_mb: float) -> int:
        """Crop the oldest cached positions to free ``target_free_mb``.

        Public counterpart of :meth:`_shed_context_if_needed`, used by the QoS
        scheduler to unblock a device that cannot even hold the cheapest rung
        plus everyone's context.  Losing the head of a conversation is a real
        quality loss — but it is a *recoverable* one, unlike termination.
        """
        if self.cache is None or target_free_mb <= 0:
            return 0
        if self.cache.seq_len <= self.MIN_CONTEXT_TOKENS:
            return 0
        mb_per_token = (self.cache.nbytes() / MB) / max(1, self.cache.seq_len)
        if mb_per_token <= 0:
            return 0
        drop = int(target_free_mb / mb_per_token) + 1
        keep = max(self.MIN_CONTEXT_TOKENS, self.cache.seq_len - drop)
        if keep >= self.cache.seq_len:
            return 0
        dropped = self.cache.seq_len - keep
        self.cache = self.cache.crop(keep)
        if self.loaded is not None and self.loaded.adapter.supports_partial_recompute:
            rebase_rope(self.cache, self.loaded.adapter, 0)
        self._sync_cache_accounting()
        self.log.log("context_shed", app=self.req.app_id, dropped_tokens=dropped,
                     kept_tokens=keep, target_free_mb=target_free_mb, tier=self.tier.name)
        return dropped

    def _shed_context_if_needed(self) -> int:
        """Drop the oldest cached positions when even the cheapest rung is too big.

        The ladder bottoms out, but the KV cache does not: on a long context the
        cache alone can exceed a tight budget.  Rather than be reclaimed, the
        generation gives up its oldest history — a graceful degradation that
        keeps the token stream alive.  This is the final rung of the "never
        die" argument, and it is logged loudly because it *is* a quality loss.

        Returns the number of tokens dropped.
        """
        budget = self.budget_fn()
        if budget == float("inf") or self.cache is None:
            return 0
        over = self.broker.usage_mb() - budget
        if over <= 0 or self.cache.seq_len <= self.MIN_CONTEXT_TOKENS:
            return 0
        mb_per_token = (self.cache.nbytes() / MB) / max(1, self.cache.seq_len)
        if mb_per_token <= 0:
            return 0
        drop = int(over / mb_per_token) + 1
        keep = max(self.MIN_CONTEXT_TOKENS, self.cache.seq_len - drop)
        if keep >= self.cache.seq_len:
            return 0
        dropped = self.cache.seq_len - keep
        self.cache = self.cache.crop(keep)
        # the surviving keys still carry their original RoPE angles; the model
        # will index the shortened cache from zero, so re-base them
        if self.loaded is not None and self.loaded.adapter.supports_partial_recompute:
            rebase_rope(self.cache, self.loaded.adapter, 0)
        self._sync_cache_accounting()
        self.log.log("context_shed", app=self.req.app_id, dropped_tokens=dropped,
                     kept_tokens=keep, over_mb=over, tier=self.tier.name)
        return dropped

    # -- one decode step ---------------------------------------------------
    def step(self) -> Optional[int]:
        """Emit exactly one token (or None when finished).

        A migration that happens during this call is *inside* the same step —
        the caller still gets a token.  That is the operational meaning of
        **no-stall**, and the reason latency is recorded per token rather than
        per request.
        """
        if self.done:
            return None
        t_start = now()
        pressure = self.pressure_fn()
        self.controller.observe(pressure)

        # --- decide & migrate --------------------------------------------
        mig_ms = self.pending_migration_ms   # banked by a scheduler-forced move
        self.pending_migration_ms = 0.0
        if self.policy is not Policy.STATIC and self.cache is not None \
                and (self.self_migrate or self.forced_target is not None):
            decision = self.controller.decide(
                self.tier, pressure, self.forced_target, self.forced_reason)
            self.forced_target, self.forced_reason = None, ""
            if not self.self_migrate and not decision.forced:
                decision = MigrationDecision(None, "scheduler-owned policy")
            if decision and decision.direction == "up" and not decision.forced \
                    and not self.can_afford(decision.target, evict_first=True):
                self.log.log("up_shift_vetoed", app=self.req.app_id,
                             target=decision.target.name, tier=self.tier.name,
                             usage_mb=self.broker.usage_mb(), budget_mb=self.budget_fn())
                decision = MigrationDecision(None, "up-shift would not fit")
            if decision:
                try:
                    mig_ms = self._migrate(decision)
                except Exception as exc:  # pragma: no cover
                    self.log.log("migration_failed", app=self.req.app_id, error=repr(exc))
                    raise
        self.controller.tick()

        # --- last resort: shed context rather than die --------------------
        if self.policy is not Policy.STATIC:
            self._shed_context_if_needed()

        # --- forward ------------------------------------------------------
        try:
            self._check_budget()
        except OOMKilled as exc:
            self.killed = True
            self.done = True
            self.kill_reason = str(exc)
            self.log.log("oom_kill", app=self.req.app_id, tier=self.tier.name,
                         token=len(self.token_ids), usage_mb=exc.usage_mb,
                         budget_mb=exc.budget_mb)
            return None

        inp = None
        if self._next_input_id is None and self._pending_logits is not None:
            logits = self._pending_logits       # prefill already produced these
            self._pending_logits = None
        else:
            last_id = self._next_input_id
            inp = torch.tensor([[last_id]], dtype=torch.long, device=self.device)
            self._pending_trace = None
            with torch.no_grad():
                out = self.loaded.model(inp, past_key_values=self.cache.to_hf(), use_cache=True)
            logits = out.logits[:, -1, :].detach()
            meta = self.cache.meta
            meta.token_ids.append(last_id)
            self.cache = MoltCache.from_hf(out.past_key_values, meta, self.cache.hidden_traces)
            if self._trace_layer is not None and self._pending_trace is not None:
                self.cache.append_trace(self._trace_layer, self._pending_trace)
            del out

        # --- calibration (core #3) ---------------------------------------
        old_logits = None
        if self.calibrator.needs_old_model and self._shadow is not None and inp is not None:
            shadow_loaded, shadow_cache = self._shadow
            with torch.no_grad():
                s_out = shadow_loaded.model(inp, past_key_values=shadow_cache.to_hf(),
                                            use_cache=True)
            old_logits = s_out.logits[:, -1, :].detach()
            s_meta = shadow_cache.meta
            s_meta.token_ids.append(int(inp[0, 0]))
            self._shadow = (shadow_loaded,
                            MoltCache.from_hf(s_out.past_key_values, s_meta,
                                              shadow_cache.hidden_traces))
            del s_out

        emitted = self.calibrator.apply(logits, old_logits)

        # The switch's cost in distribution space, against the outgoing model's
        # own answer for this exact position.  Two numbers, because they answer
        # different questions:
        #   raw      — the *transplant's* own error, before any smoothing.
        #   handoff  — what the caller actually receives, after calibration.
        # Recording only the raw one would make the calibration ablation
        # unfalsifiable: blending changes the emitted distribution, not the
        # incoming model's logits.
        if self._handoff_ref is not None:
            ref = self._handoff_ref[0]
            self.disc.handoff_jsd_raw.append(
                js_divergence(ref, logits[0].detach().cpu()))
            self.disc.handoff_jsd.append(
                js_divergence(ref, emitted.reshape(1, -1)[0].detach().cpu()))
            self._handoff_ref = None
        self.calibrator.advance()
        if not self.calibrator.needs_old_model:
            self._release_shadow()

        # --- sample -------------------------------------------------------
        token_id = self._sample(emitted)
        self.token_ids.append(token_id)
        self.tier_per_token.append(self.tier.name)
        self.calibrator.observe(logits)
        self.disc.observe(emitted)
        self._next_input_id = token_id   # not in the cache yet

        self._sync_cache_accounting()
        sync(self.device)
        rec = TokenRecord(
            index=len(self.token_ids) - 1, t_start=t_start, t_end=now(),
            tier=self.tier.name, token_id=token_id, migration_ms=mig_ms,
            pressure=pressure, blending=self.calibrator.state.active,
            resident_mb=self.broker.usage_mb(),
            instrument_ms=self._instrument_ms,
        )
        self._instrument_ms = 0.0
        self.latency.add(rec)

        if self._is_finished(token_id):
            self.done = True
        return token_id

    def _sample(self, logits: torch.Tensor) -> int:
        z = logits.reshape(-1).float()
        if self.cfg.temperature <= 0:
            return int(z.argmax())
        z = z / self.cfg.temperature
        probs = torch.softmax(z, dim=-1)
        if self.cfg.top_p < 1.0:
            srt, idx = torch.sort(probs, descending=True)
            keep = (torch.cumsum(srt, 0) - srt) < self.cfg.top_p
            srt = srt * keep
            srt = srt / srt.sum().clamp_min(1e-12)
            return int(idx[torch.multinomial(srt, 1)])
        return int(torch.multinomial(probs, 1))

    def _is_finished(self, token_id: int) -> bool:
        if len(self.token_ids) >= self.req.max_new_tokens:
            return True
        eos = getattr(self.tok, "eos_token_id", None)
        if eos is not None and token_id == eos:
            return True
        if self.req.stop_strings:
            tail = self.decode_text()[-64:]
            return any(s in tail for s in self.req.stop_strings)
        return False

    # -- finishing ---------------------------------------------------------
    def decode_text(self) -> str:
        try:
            return self.tok.decode(self.token_ids, skip_special_tokens=True)
        except Exception:
            return " ".join(str(t) for t in self.token_ids)

    def result(self) -> GenerationResult:
        return GenerationResult(
            app_id=self.req.app_id, prompt=self.req.prompt, text=self.decode_text(),
            token_ids=list(self.token_ids), tier_per_token=list(self.tier_per_token),
            latency=self.latency, discontinuity=self.disc,
            transplants=list(self.transplants),
            migrations=list(self.controller.state.history),
            killed=self.killed, kill_reason=self.kill_reason,
            finished=self.done and not self.killed, peak_mb=self.broker.peak_mb,
            extra=dict(policy=self.policy.value,
                       calibration=self.calibrator.summary(),
                       start_tier=self.tier_per_token[0] if self.tier_per_token else None,
                       end_tier=self.tier.name),
        )

    # -- suspend / resume (used by the QoS scheduler) ----------------------
    def pause(self) -> float:
        """Release this generation's *weights* while keeping its KV cache.

        A background batch job can be parked for the duration of a pressure
        spike and resumed afterwards with its context intact.  This is the
        scheduler's second-cheapest lever after demotion, and it is why the
        no-kill invariant can be met without ever discarding work.

        Returns the MiB released.
        """
        if self.paused:
            return 0.0
        before = self.zoo.footprint_mb()
        self._release_shadow()
        # snapshot the RoPE parameters so this generation can be resumed onto a
        # *different* rung later without ever reloading the one it left
        self._paused_ref = TierRef.from_loaded(self.loaded, snapshot_rope=True)
        self._detach(evict=True)
        self.paused = True
        self.n_pauses += 1
        freed = before - self.zoo.footprint_mb()
        self.log.log("pause", app=self.req.app_id, tier=self.tier.name, freed_mb=freed)
        return freed

    def resume(self, target: Optional[TierSpec] = None) -> None:
        """Bring the weights back, optionally on a different (cheaper) rung.

        Resuming onto a cheaper rung transplants the parked cache on the way in,
        so a job that was paused under pressure comes back with its context and
        without a re-prefill.
        """
        if not self.paused:
            return
        target = target or self.tier
        src_ref = self._paused_ref
        old_name = self.tier.name
        with Stopwatch(self.device) as sw:
            new_lt = self._attach(target)
            if target.name != old_name and self.cache is not None and src_ref is not None:
                old_cache = self.cache
                if self._next_input_id is None and old_cache.seq_len > 1:
                    self._next_input_id = old_cache.meta.token_ids[-1]
                    old_cache = old_cache.truncate(old_cache.seq_len - 1)
                self._pending_logits = None
                proj = self.registry.get(old_name, target.name)
                if self.policy is Policy.MOLT:
                    new_cache, rep = self.transplanter.transplant(
                        old_cache, src_ref, new_lt, proj, self.device)
                else:
                    new_cache, rep = self.transplanter.reprefill(
                        list(old_cache.meta.token_ids), new_lt, self.device,
                        old_cache.meta.n_prompt_tokens, self._trace_layer, old_name)
                old_cache.free()
                self.cache = new_cache
                self.transplants.append(rep)
                self.disc.mark_migration(len(self.token_ids))
                self.calibrator.begin(pressure=self.pressure_fn(), critical=True)
        self.paused = False
        self._paused_ref = None
        self._sync_cache_accounting()
        self.log.log("resume", app=self.req.app_id, tier=self.tier.name,
                     from_tier=old_name, ms=sw.ms)

    def close(self, evict: bool = True) -> None:
        self._release_shadow()
        self._detach(evict=evict)
        if self.cache is not None:
            self.cache.free()
            self.cache = None
        self.broker.drop_cache(self.req.app_id)
        gc.collect()


# --------------------------------------------------------------------------
# convenience driver for single-app runs
# --------------------------------------------------------------------------


class MoltRuntime:
    """Builds and drives generations.  The benchmark's single-app entry point."""

    def __init__(self, cfg: MoltConfig, zoo: ModelZoo, tokenizer,
                 registry: Optional[ProjectorRegistry] = None,
                 log: Optional[EventLog] = None):
        self.cfg = cfg
        self.zoo = zoo
        self.tok = tokenizer
        self.log = log or EventLog()
        self.registry = registry or ProjectorRegistry(
            cfg.transplant.projector_dir, cfg.ladder.name, cfg.torch_device)
        self.broker = MemoryBroker(zoo, self.log)
        self.memory = MemoryProbe(cfg.torch_device)

    def make_generation(
        self, request: GenerationRequest, policy: Policy, start_tier: TierSpec,
        pressure_fn: Callable[[], float], budget_fn: Callable[[], float],
        enforce_oom: bool = True, request_boundary: bool = False,
        self_migrate: bool = True,
    ) -> Generation:
        controller_cls = RequestBoundaryController if request_boundary else MigrationController
        return Generation(
            request=request, cfg=self.cfg, zoo=self.zoo, tokenizer=self.tok,
            registry=self.registry, broker=self.broker, policy=policy,
            start_tier=start_tier, log=self.log,
            controller=controller_cls(self.cfg.ladder, self.cfg.migration),
            enforce_oom=enforce_oom, pressure_fn=pressure_fn, budget_fn=budget_fn,
            self_migrate=self_migrate,
        )

    def run(self, generation: Generation, on_token: Optional[Callable[[int], None]] = None,
            tick: Optional[Callable[[], None]] = None) -> GenerationResult:
        """Drive one generation to completion (or to its forced termination)."""
        try:
            generation.prefill()
        except OOMKilled as exc:
            generation.killed = True
            generation.kill_reason = str(exc)
            generation.done = True
            self.log.log("oom_kill", app=generation.req.app_id, phase="prefill",
                         usage_mb=exc.usage_mb, budget_mb=exc.budget_mb)
            res = generation.result()
            generation.close()
            return res
        while not generation.done:
            tid = generation.step()
            if tid is not None and on_token is not None:
                on_token(tid)
            if tick is not None:
                tick()
            self.memory.sample(self.broker.usage_mb())
        res = generation.result()
        res.peak_mb = self.broker.peak_mb
        generation.close()
        return res
