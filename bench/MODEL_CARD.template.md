---
license: mit
library_name: mlx
tags:
  - mlx
  - kv-cache
  - model-migration
  - on-device
  - apple-silicon
base_model:
  - mlx-community/Qwen2.5-3B-Instruct-4bit
  - mlx-community/Qwen2.5-1.5B-Instruct-4bit
  - mlx-community/Qwen2.5-0.5B-Instruct-4bit
datasets:
  - HuggingFaceFW/fineweb-edu
---

# molt KV mappers for Qwen2.5 (MLX)

Closed-form ridge maps from a larger Qwen2.5 Instruct model's KV cache to a
smaller one's: **3B → 1.5B** and **1.5B → 0.5B**, fitted on the 4-bit MLX builds
an on-device app would ship. They are used by
[molt](https://github.com/NagaYu/molt) to move a generation to the smaller model
**mid-decode, under a per-app memory limit, without holding both models in
memory**.

## Prior work: the method is not ours

The mapping is the method of Heo et al., *Cross-Model KV Cache Transfer in LLM
Families: A Closed-Form Linear Mapping for Prefill Reuse*
([arXiv:2608.03893](https://arxiv.org/abs/2608.03893)), reproduced as written:

- per target layer and KV head, separate K and V maps;
- input = all KV heads of the top-k source layers (keys with RoPE removed);
- centred ridge with λ = 0.01;
- top-k chosen by single-source R²;
- 500 × 1 024 FineWeb-Edu tokens at stride 4.

They apply it at the prefill hand-off point. molt claims only the setting it
applies the map in: mid-decode, under memory exhaustion, with sender and
receiver never resident together.

Two interpretations differ from or extend the paper, and are documented in
`config.json` and the molt paper draft:

- λ is applied to the centred covariance.
- k is chosen by held-out prefix-KV NLL, not downstream accuracy.

## Files

| pair | file | k | params | role |
|---|---|---:|---:|---|
| 3B → 1.5B | `3b-1.5b/heo_k{{fit.3b-15b.kstar}}.safetensors` | {{fit.3b-15b.kstar}} | {{fit.3b-15b.kstar_mib}} MiB | condition C and molt |
| 3B → 1.5B | `3b-1.5b/depth_linear.safetensors` | 1 | | ablation (molt v0.1 layer pairing) |
| 1.5B → 0.5B | `1.5b-0.5b/heo_k{{fit.15b-05b.kstar}}.safetensors` | {{fit.15b-05b.kstar}} | {{fit.15b-05b.kstar_mib}} MiB | condition C and molt |
| 1.5B → 0.5B | `1.5b-0.5b/depth_linear.safetensors` | 1 | | ablation |

The safetensors header metadata carries the geometry, both RoPE specs, the
per-layer source lists, λ and the number of calibration rows. `index.json`
lists SHA-256 checksums.

## Use (MLX)

```python
from molt_mlx.mappers import KVMapper            # pip: see the molt repository
m = KVMapper.load("1.5b-0.5b/heo_kK.safetensors")
dst_pairs = m.apply(src_pairs, offset=0)          # src_pairs: [(K post-RoPE, V)] per source layer
```

## Measured quality: held-out FineWeb-Edu, receiver decoding from a transferred 768-token prefix

| pair | mapped (k*) NLL | receiver reading natively | gap | held-out R² K / V |
|---|---:|---:|---:|---:|
| 3B → 1.5B | {{fit.3b-15b.kstar_nll}} | {{fit.3b-15b.native_dst_nll}} | {{fit.3b-15b.gap_nll}} | {{fit.3b-15b.kstar_r2k}} / {{fit.3b-15b.kstar_r2v}} |
| 1.5B → 0.5B | {{fit.15b-05b.kstar_nll}} | {{fit.15b-05b.native_dst_nll}} | {{fit.15b-05b.gap_nll}} | {{fit.15b-05b.kstar_r2k}} / {{fit.15b-05b.kstar_r2v}} |

A positive gap means the transferred cache is worse than letting the small
model re-read the text. Results across all conditions, including where these
mappers lose to re-prefill, are in
[RESULTS.md](https://github.com/NagaYu/molt/blob/main/RESULTS.md).

## Limits

- Fitted on English web text only.
- Large → small only.
- 4-bit Qwen2.5 Instruct weights only. A different quantisation of the same
  model is a different source.
- Not measured on an iPhone yet.
