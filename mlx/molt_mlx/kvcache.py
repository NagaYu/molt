"""Reading and writing mlx-lm ``KVCache`` objects as plain arrays.

A migration needs the source cache as data — ``[B, n_kv, T, head_dim]`` K and V
per layer — after the source model is gone, and needs to hand the destination a
cache it can keep decoding from.  These helpers do exactly that and nothing
else, so the rest of the code never touches mlx-lm internals directly.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import mlx.core as mx
from mlx_lm.models.cache import KVCache

KV = Tuple[mx.array, mx.array]


def extract(cache: Sequence[KVCache]) -> List[KV]:
    """Per-layer (K, V), trimmed to the filled length."""
    out = []
    for c in cache:
        k, v = c.state
        out.append((k, v))
    return out


def build(pairs: Sequence[KV]) -> List[KVCache]:
    """Fresh ``KVCache`` objects holding ``pairs``; offset = their length."""
    caches = []
    for k, v in pairs:
        c = KVCache()
        c.state = (k, v)
        caches.append(c)
    return caches


def length(pairs: Sequence[KV]) -> int:
    return 0 if not pairs else int(pairs[0][0].shape[2])


def truncate(pairs: Sequence[KV], n: int) -> List[KV]:
    """Keep the first ``n`` positions (used to hand the boundary token back to the
    destination, which must produce its own logits for it)."""
    return [(k[:, :, :n, :], v[:, :, :n, :]) for k, v in pairs]


def nbytes(pairs: Sequence[KV]) -> int:
    return sum(k.nbytes + v.nbytes for k, v in pairs)


def selftest() -> None:
    from mlx_lm.models.cache import make_prompt_cache

    from .tinymodels import make, BIG

    m = make(BIG, 4)
    ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    c1 = make_prompt_cache(m)
    ref = m(ids, cache=c1)
    pairs = extract(c1)
    assert length(pairs) == 9 and len(pairs) == 6
    # decoding one more token from a rebuilt cache == decoding from the original
    nxt = mx.array([[10]])
    a = m(nxt, cache=build(pairs))
    b = m(nxt, cache=c1)
    err = mx.abs(a - b).max().item()
    assert err < 1e-4, err
    # truncation + re-feeding the dropped token reproduces the full-prefix logits
    t = build(truncate(pairs, 8))
    c = m(ids[:, 8:9], cache=t)
    err2 = mx.abs(c[:, -1] - ref[:, -1]).max().item()
    assert err2 < 1e-4, err2
    print(f"  rebuild/truncate round-trips exactly (max err {max(err, err2):.1e}); "
          f"{nbytes(pairs)} bytes for 6 layers x 9 tokens")
    print("kvcache: ok")


if __name__ == "__main__":
    selftest()
