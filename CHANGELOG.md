# Changelog

## 0.2.0 — research rebuild on MLX

Scope narrowed to what prior work does not cover: switching models **mid-decode**,
**under a per-process memory limit**, with the two models **never co-resident**.

### Withdrawn from 0.1
- RoPE un-rotate/re-rotate around a linear KV map, and closed-form ridge maps
  between sizes of one family, are prior work (Heo et al., arXiv:2608.03893).
  0.1 presented the former as its own key detail; that claim is withdrawn.
- Throughput is no longer a headline metric.

### Added
- `mlx/molt_mlx`: MLX implementation, including a faithful reproduction of the
  Heo et al. mapper (baseline C). The rest:
  - phys_footprint probe and kernel lifetime max;
  - external C watchdog that SIGKILLs over a cap (jetsam emulation);
  - memory-ordered migration (synchronize + clear + wait before loading);
  - native sink/tail anchors;
  - rootless energy (per-process CPU, system GPU via IOReport).
- Test guaranteeing the ordered schedule's kernel peak footprint stays under the
  cap while colocated and del-only are killed.
- `bench/`: experiments I/Q/M with process isolation, judge scoring, bootstrap
  CIs, human-eval kit, `powermetrics` cross-check.
- `ios/`: MoltCore Swift package (self-tested on macOS); device app and protocol
  (not yet run on a device).
- `paper/`: arXiv draft whose numbers are all generated macros.

## 0.1.0 — first complete prototype

Elastic on-device inference: a running generation migrates between models of
different sizes *between two tokens*, carrying its KV cache across.

### Core
- **KVTransplant** — learned per-layer affine maps for K and V inside a RoPE
  un-rotate/re-rotate sandwich (the map is only position-independent if the keys
  are de-rotated first), a diagonal scale re-alignment for same-weights /
  different-precision routes, and selective top-k native recompute from a
  projected boundary hidden state. Partial layer-range recompute verified
  bit-exact against a full forward.
- **MidStreamMigration** — bidirectional, hysteresis + cooldown + up-shift
  patience, plus an *affordability veto*: an up-shift is checked for fit, not
  only for pressure.
- **MigrationCalibration** — `dual` (probability-space mixture with a cosine
  anneal), `frozen_ref` (entropy interpolation, zero extra memory), `none`.
- **QoSScheduler** — admission deferral, grouped demotion, parking, context
  shedding; applied synchronously before any tenant steps. Nothing is ever
  terminated.
- **TierRef** — a transplant needs the source model's RoPE parameters, not its
  weights, so the outgoing rung is evicted *before* the incoming one loads and
  peak memory is `max(src, dst)` rather than `src + dst`.

### Service
- `molt/service.py`: single-worker streaming HTTP server (SSE), per-token tier
  reporting, explicit in-band migration events, `/stats` exposing `kills` as an
  SLO, admission backpressure instead of rejection, and an operator endpoint for
  driving memory pressure. Browser demo at `/`. Stdlib only.
- `examples/client.py`, `Dockerfile`, `Makefile`.

### Measurement corrections made during the work
These changed reported numbers and are listed because the discarded metrics are
as informative as the kept ones.
- `use_projection=False` was defined but never read — the "no learned
  projection" ablation was silently identical to the full system. Now honoured
  at the point of use.
- The same ablation then fell through an exception path into a full re-prefill,
  because `can_recompute` was decided against the registry's projector before
  the identity fallback replaced it. Now re-derived after resolution.
- Step-to-step "excess JSD" was replaced by **handoff JSD**. On real text the
  step-to-step measure saturates at `ln 2 ≈ 0.693` (measured steady state
  0.65–0.69), so a migration's contribution is below the floor and the "excess"
  came out negative.
- Handoff JSD is recorded twice — against the incoming model's raw logits (the
  transplant's own error) and against the emitted distribution (what the caller
  receives). Recording only the raw one made the calibration ablation
  unfalsifiable.
- Instrumentation (the handoff probe forward) is excluded from measured token
  latency.
- Judge perplexity is no longer reported alone: degenerate repetition scores
  well on it. Added judge top-1 agreement.
- The needle task was rewritten with competing distractors; the original scored
  1.00 for every condition including the reclaimed ones, because the answer sat
  in the prompt.
- Projector residuals are reported **held out**, not on the fitting data. A
  dense 257×128 map fitted on a few hundred tokens drives training residual to
  ~0.01 while generalising not at all.
- The QoS scheduler now *waits* when the budget falls below the cheapest rung
  instead of giving up; traces recover, and parked work keeps its context.
