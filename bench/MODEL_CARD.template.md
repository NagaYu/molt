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

## Read this first

In molt's own benchmark (1.5B → 0.5B, switch mid-decode), the receiver
decoding from a cache mapped with these weights keeps the topic and voice of
the text but loses its specifics. Needle accuracy is {{q.15b-05b.C.needle}} with the map
alone and {{q.15b-05b.D.needle}} with molt's native anchors, against
{{q.15b-05b.A2.needle}} when the small model simply re-reads the transcript.
Re-reading was also faster to resume at the context lengths tested. Use these
mappers to study KV transfer. If you are switching models in an app, the data
in [RESULTS.md](https://github.com/NagaYu/molt/blob/main/RESULTS.md) recommend
releasing the big model first and then re-prefilling.

## Files

{{FILES_TABLE}}

The safetensors header metadata carries the geometry, both RoPE specs, the
per-layer source lists, λ and the number of calibration rows. `mlx/index.json`
lists SHA-256 checksums.

## Use (MLX)

```python
from molt_mlx.mappers import KVMapper            # from the molt repository (mlx/)
m = KVMapper.load("mlx/1.5b-0.5b/heo_k16.safetensors")
dst_pairs = m.apply(src_pairs, offset=0)          # src_pairs: [(K post-RoPE, V)] per source layer
```

## Measured transfer quality: held-out FineWeb-Edu, receiver decoding from a transferred 768-token prefix

{{QUALITY_TABLE}}

A positive gap means the transferred cache is worse than letting the small
model re-read the text.

## Limits

- Fitted on English web text only.
- Large → small only.
- 4-bit Qwen2.5 Instruct weights only. A different quantisation of the same
  model is a different source.
- Not measured on an iPhone yet.
