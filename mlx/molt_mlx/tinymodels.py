"""Tiny, randomly-initialised mlx-lm Qwen2 models for hermetic tests.

Structurally the real thing — GQA, RoPE, tied embeddings, the same ``Attention``
code path that writes post-RoPE keys into ``KVCache`` — but small enough that
every self-test runs in seconds with no download.  The "big" and "small" models
differ in depth **and** head_dim, like Qwen2.5-1.5B → 0.5B.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models import qwen2

BIG = dict(hidden_size=128, num_hidden_layers=6, intermediate_size=256,
           num_attention_heads=4, num_key_value_heads=2)        # head_dim 32
SMALL = dict(hidden_size=64, num_hidden_layers=4, intermediate_size=128,
             num_attention_heads=4, num_key_value_heads=2)      # head_dim 16
VOCAB = 256


def make(spec: dict, seed: int) -> qwen2.Model:
    mx.random.seed(seed)
    args = qwen2.ModelArgs(model_type="qwen2", rms_norm_eps=1e-6, vocab_size=VOCAB,
                           rope_theta=10000.0, tie_word_embeddings=True, **spec)
    m = qwen2.Model(args)
    mx.eval(m.parameters())
    return m


def pair():
    return make(BIG, 1), make(SMALL, 2)
