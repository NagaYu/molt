"""molt_mlx — mid-decode, memory-bounded KV migration between model sizes, on MLX.

Scope of the claim this package exists to test (and nothing beyond it):
a generation running on a large model is moved, **between two decode steps**,
onto a smaller model of the same family, **under a memory limit that cannot hold
both models**, without the two ever being resident at the same time, and the
smaller model continues the same sentence from a transferred KV cache.

The KV mapping itself is prior work (Heo et al., arXiv:2608.03893: closed-form,
per-head ridge on RoPE-stripped keys) and is implemented here faithfully as a
baseline, not claimed.  See ``paper/`` for the boundary with prior work.
"""

__version__ = "0.2.0"
