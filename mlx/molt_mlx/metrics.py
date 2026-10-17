"""Measurements used by the benchmark and the mapper selection.

Every quantity here is computed from tokens and logits, never from a model's own
self-report, and every aggregate comes with a bootstrap interval:

* ``prefix_kv_nll``     teacher-forced NLL of a held-out continuation when the
                        destination decodes from a (mapped) source cache — the
                        intrinsic quality of a transfer at a mid-sequence boundary.
* ``judge``             per-token NLL and top-1 agreement of a fixed reference
                        model over generated text (agreement is the primary judge
                        score: repetitive text earns low NLL but not agreement).
* ``boundary_jump``     mean judge NLL just after the migration boundary minus
                        just before it — the "visible seam" in surprisal units.
* ``gsm8k_answer``      final-number extraction for task accuracy.
* ``bootstrap_ci`` / ``paired_bootstrap_ci``.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

from .generate import prefill
from .migrate import transplant


def _nll(logits: mx.array, targets: Sequence[int]) -> np.ndarray:
    """``logits [T, V]`` (float32 log-softmax) against ``targets [T]``."""
    lp = logits.astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    t = mx.array(list(targets))
    return -np.array(mx.take_along_axis(lp, t[:, None], axis=-1)[:, 0])


def continue_teacher_forced(model, cache, first_logits: mx.array,
                            continuation: Sequence[int]) -> np.ndarray:
    """NLL of ``continuation`` given a cache whose next-token logits are
    ``first_logits``; the continuation (minus its last token) is fed in one pass."""
    cont = list(continuation)
    first = _nll(first_logits.reshape(1, -1), cont[:1])
    if len(cont) == 1:
        return first
    rest = model(mx.array([cont[:-1]]), cache=cache)[0]
    return np.concatenate([first, _nll(rest, cont[1:])])


def prefix_kv_nll(seq: Sequence[int], P: int, dst_model, src_pairs=None, mapper=None,
                  sink: int = 0, tail: Union[int, str] = 0) -> np.ndarray:
    """Cache covers ``seq[:P]``, ``seq[P]`` is the pending token; score ``seq[P+1:]``.

    ``mapper=None`` (or ``tail='all'``) gives the destination-native reference.
    """
    seq = list(seq)
    logits, cache, _ = transplant(src_pairs, seq[:P], seq[P], dst_model, mapper, sink, tail)
    return continue_teacher_forced(dst_model, cache, logits, seq[P + 1:])


def native_nll(model, seq: Sequence[int], P: int) -> np.ndarray:
    """The same scored span as :func:`prefix_kv_nll`, model reading its own cache."""
    seq = list(seq)
    logits, cache = prefill(model, seq[:P + 1])
    return continue_teacher_forced(model, cache, logits, seq[P + 1:])


def judge(model, context: Sequence[int], generated: Sequence[int]) -> Dict[str, np.ndarray]:
    """Teacher-forced NLL and top-1 agreement of ``model`` for each generated token."""
    ids = list(context) + list(generated)
    logits = []
    cache = make_prompt_cache(model)
    for i in range(0, len(ids) - 1, 512):
        out = model(mx.array([ids[i:min(i + 512, len(ids) - 1)]]), cache=cache)[0]
        logits.append(out)
    lg = mx.concatenate(logits, axis=0)[len(context) - 1:]
    nll = _nll(lg, generated)
    agree = (np.array(mx.argmax(lg, axis=-1)) == np.array(generated)).astype(np.float64)
    return dict(nll=nll, agree=agree)


def boundary_jump(values: np.ndarray, b: int, w: int = 16) -> float:
    """``mean(values[b:b+w]) − mean(values[b-w:b])``; ``nan`` if a side is empty."""
    pre, post = values[max(0, b - w):b], values[b:b + w]
    if len(pre) == 0 or len(post) == 0:
        return float("nan")
    return float(np.mean(post) - np.mean(pre))


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def gsm8k_answer(text: str) -> Optional[str]:
    """Number after '####' if present, else the last number in the text."""
    if "####" in text:
        text = text.split("####")[-1]
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        text = m[-1]
    nums = _NUM.findall(text)
    if not nums:
        return None
    n = nums[-1].replace(",", "").rstrip(".")
    try:
        f = float(n)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def bootstrap_ci(x: Sequence[float], n: int = 4000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap; NaNs dropped."""
    a = np.asarray([v for v in x if not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(a) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(n, len(a)))].mean(1)
    return float(a.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_ci(x: Sequence[float], y: Sequence[float], **kw):
    """CI of mean(x − y) over paired items (same prompt under two conditions)."""
    d = [a - b for a, b in zip(x, y)]
    return bootstrap_ci(d, **kw)


def selftest() -> None:
    from .fit import collect, fit
    from .tinymodels import make, BIG, SMALL

    big, small = make(BIG, 1), make(SMALL, 2)
    rng = np.random.default_rng(0)
    seqs = [rng.integers(0, 256, 40).tolist() for _ in range(8)]
    seq = seqs[0]
    a = native_nll(small, seq, 20)
    b = prefix_kv_nll(seq, 20, small, None, None)
    assert a.shape == (19,) and np.allclose(a, b, atol=1e-4)
    # judge NLL on the same span equals native NLL (one implementation checks the other)
    j = judge(small, seq[:21], seq[21:])
    assert np.allclose(j["nll"], a, atol=1e-3), np.abs(j["nll"] - a).max()
    st = collect({"b": big, "s": small}, [("b", "s")], seqs, stride=1)[("b", "s")]
    m, _ = fit(st, "r2_topk", 6)
    _, cache = prefill(big, seq[:20])
    from . import kvcache
    pk = prefix_kv_nll(seq, 20, small, kvcache.extract(cache), m)
    assert pk.shape == a.shape and np.isfinite(pk).all()
    assert boundary_jump(np.r_[np.zeros(16), np.ones(16)], 16) == 1.0
    assert gsm8k_answer("so 3+4 = 7.\n#### 1,234") == "1234"
    assert gsm8k_answer("The answer is \\boxed{42}.") == "42"
    assert gsm8k_answer("costs $18.50 total") == "18.5"
    mu, lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0])
    assert lo <= mu <= hi
    print(f"  native == prefix_kv(no mapper) == judge NLL on the same span; "
          f"mapped prefix NLL {pk.mean():.3f} vs native {a.mean():.3f} (random tiny models)")
    print("metrics: ok")


if __name__ == "__main__":
    selftest()
