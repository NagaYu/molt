# RESULTS

Every number below is generated from `bench/results/` by `bench/make_results_md.py`.
Intervals are 95% bootstrap CIs. A killed run scores 0 and is never dropped.
Measured on: Apple M2, 16 GB, macOS 27.0.1, MLX 0.31.2, mlx-lm 0.31.3. Other users' jobs were running and swap was
nearly full during these runs. Timings carry that noise; footprint and survival do not.

## Qwen2.5-3B → 1.5B (4-bit)

### Mapper fit and selection (held-out documents only)

PENDING

### Experiment I: ΔNLL of the transferred cache vs. the receiver reading natively

PENDING

### Experiment Q: quality when the switch lands mid-decode (no cap)

PENDING

### Experiment M: survival under a per-process cap (every run its own process)

PENDING

## Qwen2.5-1.5B → 0.5B (4-bit)

### Mapper fit and selection (held-out documents only)

PENDING

### Experiment I: ΔNLL of the transferred cache vs. the receiver reading natively

PENDING

### Experiment Q: quality when the switch lands mid-decode (no cap)

PENDING

### Experiment M: survival under a per-process cap (every run its own process)

PENDING

## Device, energy cross-check, human ratings

- iPhone measurements: **PENDING** (protocol: `ios/README.md`). Device folders present: none.
- `powermetrics` cross-check: **PENDING** (needs sudo; `bench/powermetrics.sh`).
- Human seam ratings: **PENDING** (`bench/human_eval/`).
