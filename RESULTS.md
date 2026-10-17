# RESULTS

Every number below is generated from `bench/results/` by `bench/make_results_md.py`.
Intervals are 95% bootstrap CIs. A killed run scores 0 and is never dropped.
Measured on: Apple M2, 16 GB, macOS 27.0.1, MLX 0.31.2, mlx-lm 0.31.3. Other users' jobs were running and swap was
nearly full during these runs. Timings carry that noise; footprint and survival do not.

## Summary (Qwen2.5-1.5B → 0.5B, 4-bit; 3B → 1.5B still running)

| | C ridge mid-decode (Heo et al., hand-off shape) | D molt | A′ re-prefill | B always small |
|---|---:|---:|---:|---:|
| needle exact | 0.208 | 0.125 | **0.958** | 0.958 |
| GSM8K exact (big alone: 0.450) | 0.000 | 0.050 | **0.250** | 0.000 |
| judge agreement after the switch | 0.686 | 0.751 | **0.858** | 0.860 |
| seam jump (nats) | 1.779 | 1.441 | **0.352** | — |
| survives h = 345 MiB (co-resident needs 409) | 0/3 | 3/3 | 3/3 | 3/3 |
| time to resume at h = 345 (s) | killed | 0.79 | **0.48** | — |

## Where the hypothesis fails, stated plainly

**The transferred cache does not carry the content.** The design hypothesis was
that carrying the KV cache across the switch would let the small model continue
the same sentence better than re-reading the text, and that molt's additions
would make the prior closed-form map sufficient. It does not. With a mapped
cache, the receiver keeps the topic and the voice but loses the specifics:

- needle accuracy is 0.208 (C) and 0.125 (D), against
  0.958 for re-prefill;
- GSM8K is 0.000 and 0.050, against 0.250.

molt's native anchors make the seam smaller than C's: paired agreement
+0.065 [0.035, 0.096], seam
-0.338 [-0.561, -0.118] nats. Task score does not move
(-0.023 [-0.114, 0.045]). "Better than C" and "good
enough" are different claims, and only the first holds here.

**Re-prefill wins in exactly the setting we claim.** A′ switches mid-decode, is
triggered by headroom, and releases the sender before loading the receiver, so
it satisfies all three of molt's conditions. It survives everywhere C dies.
It resumes faster (0.48 s vs 0.79 s)
with less CPU energy (0.83 J vs 1.30 J).
The mapping is cheaper only in compute, and only at long context
(1.419 s vs 2.669 s at 4 096 tokens). Our switches
happened at 89–298 tokens, and the
intrinsic quality gap grows with context (+0.117 nats at 768).

**The careful release is a fix for one allocator setting.** Under MLX's default
buffer cache:

- drop-then-load is killed in every cell (h = 818: 0/3);
- clear-without-sync is erratic (0/3,
  3/3, 0/3,
  2/3, 0/3).

Under the 20 MiB cache limit that mlx-swift's examples use on iOS, both survive
(h = 345: 3/3, 3/3).
Only the order is robust: never load the receiver before releasing the sender.

**molt v0.1's mapper does not survive.** Its depth-linear layer pairing scores
held-out NLL 4.285, against 2.923 for Heo et
al.'s top-k=16.

**What holds.** Below the headroom a co-resident switch needs
(409 MiB), the prefill-hand-off shape is killed and a release-first
switch survives with no retracted token. At the context lengths we measured,
the receiver should be given the transcript, not the transferred KV.


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

Calibration rows: 128,000 (HuggingFaceFW/fineweb-edu sample-10BT 000_00000.parquet). Selected: **heo_k16**. molt anchors: sink 4, tail 32.

| mapper | k | params (M) | MiB fp16 | R² in-sample | R² K held-out | R² V held-out | held-out NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| depth_linear | 1 | 1.6 | 3 | 0.592 | 0.702 | 0.468 | 4.285 [4.115, 4.452] |
| heo_k1 | 1 | 1.6 | 3 | 0.616 | 0.720 | 0.497 | 3.326 [3.200, 3.450] |
| heo_k2 | 2 | 3.2 | 6 | 0.682 | 0.769 | 0.578 | 3.096 [2.977, 3.215] |
| heo_k4 | 4 | 6.3 | 12 | 0.723 | 0.797 | 0.629 | 2.996 [2.877, 3.112] |
| heo_k8 | 8 | 12.6 | 24 | 0.760 | 0.820 | 0.671 | 2.948 [2.829, 3.064] |
| heo_k16 | 16 | 25.2 | 48 | 0.787 | 0.834 | 0.697 | 2.923 [2.803, 3.040] |
| heo_kall | 28 | 44.0 | 84 | 0.806 | 0.842 | 0.708 | 2.937 [2.816, 3.056] |
| *native_dst (reference)* | | | | | | | 2.800 [2.676, 2.923] |
| *native_src (reference)* | | | | | | | 2.448 [2.328, 2.564] |

λ sweep at k*: λ=0.0: 2.864, λ=0.001: 2.865, λ=0.01: 2.869, λ=0.1: 2.897, λ=1.0: 3.144

Anchor sweep (sink/tail → held-out NLL): 0/0: 2.869, 4/0: 2.860, 0/8: 2.867, 0/32: 2.854, 4/8: 2.857, 4/32: 2.844, 4/128: 2.813, 0/all: 2.761

### Experiment I: ΔNLL of the transferred cache vs. the receiver reading natively

| arm | P=32 | P=128 | P=256 | P=512 | P=768 |
|---|---:|---:|---:|---:|---:|
| C (ridge) | +0.023 [+0.012, +0.035] | +0.054 [+0.036, +0.074] | +0.090 [+0.070, +0.112] | +0.114 [+0.090, +0.139] | +0.146 [+0.111, +0.196] |
| D (molt anchors) | +0.000 [+0.000, +0.000] | +0.029 [+0.013, +0.048] | +0.061 [+0.045, +0.077] | +0.078 [+0.057, +0.101] | +0.117 [+0.080, +0.167] |
| D_depth | +0.000 [+0.000, +0.000] | +0.105 [+0.072, +0.144] | +0.159 [+0.133, +0.184] | +0.196 [+0.164, +0.231] | +0.243 [+0.187, +0.322] |
| native_dst | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| native_src | -0.408 [-0.443, -0.375] | -0.360 [-0.391, -0.330] | -0.375 [-0.417, -0.334] | -0.351 [-0.385, -0.318] | -0.362 [-0.398, -0.327] |

n = 64 evaluation documents, 128 scored tokens after the switch.

### Experiment Q: quality when the switch lands mid-decode (no cap)

| condition | GSM8K | needle | judge agreement after switch | seam jump (nats) | lost tokens | time to new content (s) |
|---|---:|---:|---:|---:|---:|---:|
| big, no pressure | 0.45 [0.25, 0.65] | 1.00 [1.00, 1.00] | 0.998 [0.997, 0.999] | — | 0.0 [0.0, 0.0] | — |
| (A) restart | 0.00 [0.00, 0.00] | 0.96 [0.88, 1.00] | 0.860 [0.824, 0.893] | — | 29.2 [25.2, 33.3] | 1.76 [1.58, 1.94] |
| (A′) re-prefill | 0.25 [0.05, 0.45] | 0.96 [0.88, 1.00] | 0.858 [0.824, 0.890] | +0.352 [+0.239, +0.481] | 0.0 [0.0, 0.0] | 1.33 [1.13, 1.54] |
| (B) always small | 0.00 [0.00, 0.00] | 0.96 [0.88, 1.00] | 0.860 [0.824, 0.893] | — | 0.0 [0.0, 0.0] | — |
| (C) ridge mid-decode, colocated | 0.00 [0.00, 0.00] | 0.21 [0.04, 0.38] | 0.686 [0.651, 0.721] | +1.779 [+1.525, +2.039] | 0.0 [0.0, 0.0] | 1.64 [1.37, 1.92] |
| (D) molt | 0.05 [0.00, 0.15] | 0.12 [0.00, 0.29] | 0.751 [0.720, 0.781] | +1.441 [+1.152, +1.725] | 0.0 [0.0, 0.0] | 2.15 [1.95, 2.35] |
| C + ordered | 0.00 [0.00, 0.00] | 0.21 [0.04, 0.38] | 0.686 [0.648, 0.722] | +1.779 [+1.527, +2.041] | 0.0 [0.0, 0.0] | 1.43 [1.27, 1.60] |
| D, depth-linear mapper | 0.05 [0.00, 0.15] | 0.00 [0.00, 0.00] | 0.720 [0.686, 0.752] | +3.107 [+2.385, +3.875] | 0.0 [0.0, 0.0] | 1.52 [1.35, 1.71] |

Paired differences (same items):

| comparison | seam jump | agreement after switch | task score |
|---|---:|---:|---:|
| A-A2 | — | +0.002 [-0.015, +0.019] | -0.114 [-0.205, -0.023] |
| A-B | — | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| A-C | — | +0.173 [+0.121, +0.228] | +0.409 [+0.250, +0.568] |
| A2-B | — | -0.002 [-0.019, +0.015] | +0.114 [+0.023, +0.205] |
| A2-C | -1.427 [-1.739, -1.112] | +0.172 [+0.121, +0.227] | +0.523 [+0.364, +0.659] |
| B-A2 | — | +0.002 [-0.015, +0.019] | -0.114 [-0.205, -0.023] |
| B-C | — | +0.173 [+0.121, +0.228] | +0.409 [+0.250, +0.568] |
| C-A2 | +1.427 [+1.112, +1.739] | -0.172 [-0.227, -0.121] | -0.523 [-0.659, -0.364] |
| C-B | — | -0.173 [-0.228, -0.121] | -0.409 [-0.568, -0.250] |
| C_ordered-A2 | +1.427 [+1.112, +1.739] | -0.172 [-0.227, -0.121] | -0.523 [-0.659, -0.364] |
| C_ordered-B | — | -0.173 [-0.228, -0.121] | -0.409 [-0.568, -0.250] |
| C_ordered-C | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| D-A2 | +1.089 [+0.777, +1.422] | -0.107 [-0.151, -0.066] | -0.545 [-0.705, -0.364] |
| D-B | — | -0.108 [-0.153, -0.066] | -0.432 [-0.591, -0.273] |
| D-C | -0.338 [-0.561, -0.118] | +0.065 [+0.035, +0.096] | -0.023 [-0.114, +0.045] |
| D_depth-A2 | +2.755 [+1.981, +3.580] | -0.138 [-0.192, -0.087] | -0.614 [-0.773, -0.455] |
| D_depth-B | — | -0.140 [-0.194, -0.087] | -0.500 [-0.659, -0.341] |
| D_depth-C | +1.328 [+0.761, +1.941] | +0.034 [+0.013, +0.056] | -0.091 [-0.205, +0.000] |
| big-A2 | — | +0.140 [+0.108, +0.173] | +0.114 [+0.000, +0.227] |
| big-B | — | +0.138 [+0.105, +0.174] | +0.227 [+0.114, +0.364] |
| big-C | — | +0.312 [+0.276, +0.350] | +0.636 [+0.500, +0.773] |

### Experiment M: survival under a per-process cap (every run its own process)

S (big model decoding) = 1380 MiB, small = 763 MiB, receiver weights 265 MiB, mapper 48 MiB → a co-resident switch needs ≈ 409 MiB of headroom.

| h (MiB) | cap | condition | survived | peak MiB (max) | killed in | time to resume (s) | migration CPU J | migration GPU J |
|---:|---:|---|---:|---:|---|---:|---:|---:|
| 128 | 1508 | (A) restart | 3/3 | 1445 | — | 0.46 [0.39, 0.51] | 0.78 [0.70, 0.83] | 2.94 [2.44, 3.84] |
| 128 | 1508 | (A′) re-prefill | 3/3 | 1465 | — | 0.49 [0.47, 0.53] | 0.84 [0.82, 0.87] | -0.20 [-0.26, -0.13] |
| 128 | 1508 | (B) always small | 3/3 | 1419 | — | — | — | — |
| 128 | 1508 | (C) ridge mid-decode, colocated | 0/3 | 1510 | load_dst | — | — | — |
| 128 | 1508 | C + ordered | 3/3 | 1449 | — | 0.88 [0.80, 1.00] | 1.39 [1.36, 1.43] | 0.01 [-0.94, 1.51] |
| 128 | 1508 | (D) molt | 3/3 | 1450 | — | 0.87 [0.84, 0.90] | 1.41 [1.32, 1.47] | 1.16 [-3.24, 7.48] |
| 128 | 1508 | D, clear-no-wait | 0/3 | 1514 | decode_dst | — | — | — |
| 128 | 1508 | D, colocated | 0/3 | 1512 | load_dst | — | — | — |
| 128 | 1508 | D, del-only | 0/3 | 1513 | load_dst | — | — | — |
| 128 | 1508 | big, ignores pressure | 0/3 | 1510 | decode_src | — | — | — |
| 205 | 1585 | (A) restart | 3/3 | 1494 | — | 0.53 [0.42, 0.63] | 0.76 [0.72, 0.85] | 4.49 [3.23, 5.97] |
| 205 | 1585 | (A′) re-prefill | 3/3 | 1529 | — | 0.56 [0.53, 0.63] | 0.86 [0.80, 0.94] | -0.17 [-0.45, -0.01] |
| 205 | 1585 | (B) always small | 3/3 | 1487 | — | — | — | — |
| 205 | 1585 | (C) ridge mid-decode, colocated | 0/3 | 1588 | load_dst | — | — | — |
| 205 | 1585 | C + ordered | 3/3 | 1450 | — | 0.83 [0.75, 0.93] | 1.35 [1.25, 1.41] | 0.04 [-0.59, 1.18] |
| 205 | 1585 | (D) molt | 3/3 | 1484 | — | 0.84 [0.76, 0.90] | 1.37 [1.29, 1.42] | 2.63 [-1.06, 5.13] |
| 205 | 1585 | D, clear-no-wait | 3/3 | 1573 | — | 0.40 [0.39, 0.43] | 0.71 [0.68, 0.75] | -0.91 [-3.38, 1.26] |
| 205 | 1585 | D, colocated | 0/3 | 1589 | load_dst | — | — | — |
| 205 | 1585 | D, del-only | 0/3 | 1588 | load_dst | — | — | — |
| 205 | 1585 | big, ignores pressure | 0/3 | 1590 | decode_src | — | — | — |
| 345 | 1725 | (A) restart | 3/3 | 1681 | — | 0.51 [0.47, 0.56] | 0.86 [0.81, 0.93] | -0.20 [-1.76, 1.37] |
| 345 | 1725 | (A′) re-prefill | 3/3 | 1721 | — | 0.48 [0.43, 0.56] | 0.83 [0.79, 0.90] | 1.99 [-0.21, 5.17] |
| 345 | 1725 | (B) always small | 3/3 | 1676 | — | — | — | — |
| 345 | 1725 | (C) ridge mid-decode, colocated | 0/3 | 1728 | transplant | — | — | — |
| 345 | 1725 | C + ordered | 3/3 | 1637 | — | 0.66 [0.57, 0.79] | 1.26 [1.15, 1.41] | -0.62 [-1.26, -0.09] |
| 345 | 1725 | (D) molt | 3/3 | 1671 | — | 0.79 [0.71, 0.85] | 1.30 [1.22, 1.40] | -0.41 [-0.64, -0.05] |
| 345 | 1725 | D, clear-no-wait | 0/3 | 1731 | decode_dst | — | — | — |
| 345 | 1725 | D, colocated | 0/3 | 1735 | transplant | — | — | — |
| 345 | 1725 | D, del-only | 0/3 | 1727 | load_dst | — | — | — |
| 345 | 1725 | big, ignores pressure | 0/3 | 1734 | decode_src | — | — | — |
| 569 | 1949 | (A) restart | 3/3 | 1871 | — | 0.79 [0.54, 1.25] | 0.96 [0.74, 1.29] | 2.97 [1.07, 4.85] |
| 569 | 1949 | (A′) re-prefill | 3/3 | 1916 | — | 0.58 [0.55, 0.60] | 0.88 [0.82, 0.93] | -0.35 [-0.70, 0.09] |
| 569 | 1949 | (B) always small | 3/3 | 1868 | — | — | — | — |
| 569 | 1949 | (C) ridge mid-decode, colocated | 3/3 | 1909 | — | 1.23 [0.40, 2.90] | 0.96 [0.75, 1.39] | -1.41 [-5.12, 1.16] |
| 569 | 1949 | C + ordered | 3/3 | 1838 | — | 0.81 [0.74, 0.88] | 1.40 [1.33, 1.45] | 0.05 [-2.89, 3.40] |
| 569 | 1949 | (D) molt | 3/3 | 1864 | — | 1.34 [0.83, 2.35] | 1.66 [1.28, 2.31] | -2.29 [-4.82, -0.93] |
| 569 | 1949 | D, clear-no-wait | 2/3 | 1949 | decode_dst | 3.12 [0.45, 5.78] | 1.38 [0.76, 1.99] | -18.95 [-36.46, -1.43] |
| 569 | 1949 | D, colocated | 3/3 | 1909 | — | 1.63 [0.48, 2.74] | 1.19 [0.81, 1.43] | -2.45 [-9.70, 3.92] |
| 569 | 1949 | D, del-only | 0/3 | 1951 | load_dst | — | — | — |
| 569 | 1949 | big, ignores pressure | 0/3 | 1959 | decode_src | — | — | — |
| 818 | 2198 | (A) restart | 3/3 | 2127 | — | 1.87 [0.71, 3.02] | 1.45 [1.00, 1.85] | -4.37 [-14.58, 1.71] |
| 818 | 2198 | (A′) re-prefill | 3/3 | 2176 | — | 1.49 [1.03, 1.90] | 1.22 [1.02, 1.33] | 8.08 [3.56, 11.94] |
| 818 | 2198 | (B) always small | 3/3 | 2126 | — | — | — | — |
| 818 | 2198 | (C) ridge mid-decode, colocated | 3/3 | 2180 | — | 2.93 [1.69, 4.55] | 1.53 [1.23, 1.79] | -14.01 [-23.37, -4.55] |
| 818 | 2198 | C + ordered | 3/3 | 2093 | — | 2.43 [0.92, 3.26] | 1.81 [1.49, 2.01] | -9.61 [-15.39, -0.66] |
| 818 | 2198 | (D) molt | 3/3 | 2121 | — | 1.49 [1.08, 2.26] | 1.62 [1.58, 1.69] | 0.34 [-5.86, 9.97] |
| 818 | 2198 | D, clear-no-wait | 0/3 | 2211 | decode_dst | — | — | — |
| 818 | 2198 | D, colocated | 3/3 | 2166 | — | 3.08 [2.82, 3.33] | 1.60 [1.53, 1.67] | -13.40 [-19.68, -3.29] |
| 818 | 2198 | D, del-only | 0/3 | 2199 | load_dst | — | — | — |
| 818 | 2198 | big, ignores pressure | 0/3 | 2206 | decode_src | — | — | — |

## Device, energy cross-check, human ratings

- iPhone measurements: **PENDING** (protocol: `ios/README.md`). Device folders present: none.
- `powermetrics` cross-check: **PENDING** (needs sudo; `bench/powermetrics.sh`).
- Human seam ratings: **PENDING** (`bench/human_eval/`).
