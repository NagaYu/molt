# Superseded measurements

Kept for provenance and never read by `bench/analyze.py`.

## `2026-10-18-before-per-layer-eval/`

Memory sweeps and resume-cost runs made before two fixes:

1. **Mapper evaluation order.** `KVMapper.apply` used to evaluate every layer
   lazily at once. For the 252 MiB k=all 3B→1.5B mapper, that materialised
   hundreds of MiB of float32 weight copies together. Condition C was killed by
   this transient even with twice the headroom it needs.
2. **Post-switch cleanup.** The careful schedules (ordered and colocated) did
   not clear MLX's buffer cache after the switch. Re-prefill (A′) left about
   80 MiB more cache resident, enough to be killed 5 MiB over the cap in the two
   largest-headroom cells for 3B→1.5B. The colocated schedule also released the
   source without synchronising.

Neither fix changes any generated token. Quality and intrinsic results did not
need re-running.
