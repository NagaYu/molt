"""A minimal greedy decode loop with an explicit token boundary.

The loop keeps one invariant that a mid-decode migration relies on:

    ``cache`` holds KV for exactly ``tokens`` (every id fed so far), and
    ``pending`` is the id already *emitted to the user* but not yet fed.

A migration therefore happens between "emit" and "feed": the destination model
receives the source cache for ``tokens`` and feeds ``pending`` itself, so no
emitted token is ever retracted.  Decoding is greedy so that conditions differ
only by what the migration did, never by sampling noise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from . import kvcache

PREFILL_CHUNK = 512


def prefill(model, ids: List[int], cache=None, chunk: int = PREFILL_CHUNK):
    """Feed ``ids`` in chunks (bounded activation memory); return last logits."""
    cache = make_prompt_cache(model) if cache is None else cache
    logits = None
    for i in range(0, len(ids), chunk):
        logits = model(mx.array([ids[i:i + chunk]]), cache=cache)
        mx.eval(logits, [c.state for c in cache])
    return logits[:, -1], cache


@dataclass
class Session:
    """One live generation on one model."""

    model: object
    tier: str
    cache: list = None
    tokens: List[int] = field(default_factory=list)
    pending: Optional[int] = None
    emitted: List[int] = field(default_factory=list)

    def start(self, prompt_ids: List[int]) -> None:
        logits, self.cache = prefill(self.model, prompt_ids)
        self.tokens = list(prompt_ids)
        self.pending = int(mx.argmax(logits, axis=-1).item())
        self.emitted = [self.pending]

    def resume_from(self, logits: mx.array, cache, tokens: List[int]) -> None:
        """Adopt a cache built elsewhere (e.g. a transplant) whose last logits are
        ``logits``; ``tokens`` must be exactly what ``cache`` covers."""
        assert kvcache.length(kvcache.extract(cache)) == len(tokens), "cache/token mismatch"
        self.cache, self.tokens = cache, list(tokens)
        self.pending = int(mx.argmax(logits, axis=-1).item())
        self.emitted.append(self.pending)

    def step(self) -> int:
        """Feed ``pending``; emit and return the next token."""
        logits = self.model(mx.array([[self.pending]]), cache=self.cache)
        self.tokens.append(self.pending)
        self.pending = int(mx.argmax(logits[:, -1], axis=-1).item())
        self.emitted.append(self.pending)
        return self.pending

    def kv_pairs(self):
        return kvcache.extract(self.cache)


def selftest() -> None:
    from .tinymodels import make, BIG

    m = make(BIG, 3)
    prompt = [1, 2, 3, 4, 5, 6, 7]
    s = Session(m, "big")
    s.start(prompt)
    for _ in range(9):
        s.step()
    # invariant: cache covers tokens; pending is emitted but not fed
    assert kvcache.length(s.kv_pairs()) == len(s.tokens) == len(prompt) + 9
    assert s.emitted[-1] == s.pending and len(s.emitted) == 10
    # chunked prefill of the same ids == the incremental decode's last logits
    ref, _ = prefill(m, s.tokens, chunk=3)
    nxt = int(mx.argmax(ref, axis=-1).item())
    assert nxt == s.pending, (nxt, s.pending)
    print(f"  cache covers {len(s.tokens)} fed tokens; pending={s.pending} matches a "
          f"3-token-chunk re-prefill")
    print("generate: ok")


if __name__ == "__main__":
    selftest()
