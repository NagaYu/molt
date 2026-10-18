"""Mid-decode migration between two models that are never resident together.

Two separable pieces:

``transplant``  — pure computation.  Map the source cache to the destination's
    geometry with a fitted :class:`~molt_mlx.mappers.KVMapper` (prior work: Heo
    et al.), optionally overwrite the first ``sink`` positions with the
    destination's *native* KV (exact: causal positions depend only on earlier
    tokens) and re-feed the last ``tail`` positions natively over the mapped
    history, then feed the already-emitted ``pending`` token.  ``tail='all'`` is a
    full re-prefill of the transcript and needs no mapper.

``migrate``     — the memory order in which that computation runs.  This is the
    part the paper claims for the memory-exhausted, non-co-resident setting.  Four
    schedules are implemented because the obvious ones are not what they look like
    on MLX (measured in ``footprint.selftest`` and ``tests/test_peak_footprint.py``):

    ``colocated``      load destination (+mapper) while the source is resident,
                       map, then release the source properly (synchronise,
                       clear, wait) — the prefill-handoff shape.
    ``del_only``       drop every reference to the source first.  MLX keeps the
                       freed buffers in its allocator cache and does not reuse them
                       for differently-sized arrays, so footprint does not fall:
                       still co-resident.
    ``clear_no_wait``  drop + ``gc.collect()`` + ``mx.clear_cache()`` — the call a
                       developer writes — then load at once.  Buffers return to
                       MLX's cache only when the GPU finishes with them, and the
                       OS reclaims asynchronously, so this frees late or not at all.
    ``ordered``        drop + synchronize + clear + **wait until phys_footprint
                       has fallen** by the source's weight bytes; map; drop the
                       mapper the same way; only then load the destination.

The ``ordered`` peak is ``max(src_w + kv, kv + mapper + kv', dst_w + kv')`` plus
whatever else the process holds; every other schedule pays ``src_w + dst_w``.
"""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache

from . import kvcache
from .footprint import PhaseMarker, footprint_bytes, release_and_wait
from .generate import Session, prefill
from .mappers import KVMapper

SCHEDULES = ("colocated", "del_only", "clear_no_wait", "ordered")
MiB = 2 ** 20


def weight_bytes(model) -> int:
    return sum(v.nbytes for _, v in tree_flatten(model.parameters()))


def cache_dtype(model) -> mx.Dtype:
    return model.model.embed_tokens.scales.dtype if hasattr(model.model.embed_tokens, "scales") \
        else model.model.embed_tokens.weight.dtype


# --------------------------------------------------------------------------
# transplant (pure computation)
# --------------------------------------------------------------------------
def transplant(src_pairs: Sequence[kvcache.KV], tokens: List[int], pending: int,
               dst_model, mapper: Optional[KVMapper], sink: int = 0,
               tail: Union[int, str] = 0):
    """Return ``(logits, cache, fed_tokens)`` for the destination.

    ``src_pairs`` covers exactly ``tokens``.  After the call the destination cache
    covers ``tokens + [pending]`` and ``logits`` predicts the next token.
    """
    n = len(tokens)
    if tail == "all" or mapper is None:
        logits, cache = prefill(dst_model, tokens + [pending])
        return logits, cache, tokens + [pending]
    assert kvcache.length(src_pairs) == n, (kvcache.length(src_pairs), n)
    keep = max(0, n - int(tail))
    mapped = mapper.apply(kvcache.truncate(src_pairs, keep), offset=0,
                          out_dtype=cache_dtype(dst_model)) if keep else []
    s = min(sink, keep)
    if s > 0:
        _, native = prefill(dst_model, tokens[:s])
        nat = kvcache.extract(native)
        mapped = [(mx.concatenate([nk, mk[:, :, s:]], axis=2),
                   mx.concatenate([nv, mv[:, :, s:]], axis=2))
                  for (nk, nv), (mk, mv) in zip(nat, mapped)]
    cache = kvcache.build(mapped) if keep else make_prompt_cache(dst_model)
    logits, cache = prefill(dst_model, tokens[keep:] + [pending], cache=cache)
    return logits, cache, tokens + [pending]


# --------------------------------------------------------------------------
# memory-ordered migration
# --------------------------------------------------------------------------
@dataclass
class Plan:
    schedule: str = "ordered"
    use_mapper: bool = True         # False → re-prefill the transcript (A′)
    sink: int = 0
    tail: Union[int, str] = 0
    restart: bool = False           # True → discard the generation, start over (A)
    release_timeout_s: float = 3.0

    def __post_init__(self):
        assert self.schedule in SCHEDULES, self.schedule


@dataclass
class MigrationReport:
    plan: dict
    phases: List[dict] = field(default_factory=list)
    src_weight_mib: float = 0.0
    dst_weight_mib: float = 0.0
    mapper_mib: float = 0.0
    kv_src_mib: float = 0.0
    kv_dst_mib: float = 0.0
    release_waits_ms: List[float] = field(default_factory=list)
    release_reached: List[bool] = field(default_factory=list)
    mlx_peak_mib: float = 0.0
    seconds: float = 0.0
    lost_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class _Clock:
    def __init__(self, report: MigrationReport, marker: Optional[PhaseMarker]):
        self.r, self.m, self.t0 = report, marker, time.perf_counter()

    def __call__(self, name: str):
        if self.m:
            self.m(name)
        self.r.phases.append(dict(name=name, t=time.perf_counter() - self.t0,
                                  footprint_mib=footprint_bytes() / MiB))


def _release(report: MigrationReport, target: int, timeout: float) -> None:
    rr = release_and_wait(target, timeout_s=timeout)
    report.release_waits_ms.append(rr.waited_ms)
    report.release_reached.append(rr.reached_target)


def migrate(session: Session, plan: Plan, load_dst: Callable[[], object],
            load_mapper: Optional[Callable[[], KVMapper]] = None,
            prompt_len: Optional[int] = None, dst_tier: str = "dst",
            marker: Optional[PhaseMarker] = None):
    """Move ``session`` to the destination; returns ``(new_session, report)``.

    ``session.model`` is released by this call: callers must hold no other
    reference to it, or the memory cannot come back (``tests`` check this).
    """
    rep = MigrationReport(plan=asdict(plan))
    clock = _Clock(rep, marker)
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    clock("snapshot")
    pairs = session.kv_pairs()
    tokens, pending, emitted = list(session.tokens), session.pending, list(session.emitted)
    rep.kv_src_mib = kvcache.nbytes(pairs) / MiB
    rep.src_weight_mib = weight_bytes(session.model) / MiB
    src_w = int(rep.src_weight_mib * MiB)

    if plan.restart:
        prompt = tokens[:prompt_len]
        rep.lost_tokens = len(emitted)
        pairs = None
    mapper = None
    need_mapper = plan.use_mapper and not plan.restart and plan.tail != "all"

    def drop_source(clear: bool, wait: bool):
        nonlocal session
        fp = footprint_bytes()
        session.model = None
        session.cache = None
        gc.collect()
        if clear:
            mx.clear_cache()
        if wait:
            _release(rep, fp - int(0.9 * src_w), plan.release_timeout_s)

    def do_map():
        nonlocal pairs, mapper
        mapper = load_mapper()
        rep.mapper_mib = mapper.nbytes() / MiB

    sched = plan.schedule
    if sched == "colocated":
        clock("load_dst")
        dst = load_dst(); mx.eval(dst.parameters())
        if need_mapper:
            clock("load_mapper"); do_map()
    else:
        clock("release_src")
        drop_source(clear=sched != "del_only", wait=sched == "ordered")
        if need_mapper:
            clock("load_mapper"); do_map()
            if sched == "ordered":
                # map first, then drop mapper + source KV before the weights arrive
                clock("map")
                mapped = mapper.apply(pairs, offset=0, out_dtype=pairs[0][0].dtype)
                mx.eval(mapped)
                fp = footprint_bytes()
                mbytes = mapper.nbytes() + kvcache.nbytes(pairs)
                pairs, mapper = mapped, _Premapped(mapper)
                gc.collect(); mx.clear_cache()
                _release(rep, fp - int(0.9 * mbytes), plan.release_timeout_s)
        clock("load_dst")
        dst = load_dst(); mx.eval(dst.parameters())
    rep.dst_weight_mib = weight_bytes(dst) / MiB

    clock("transplant")
    if plan.restart:
        new = Session(dst, dst_tier)
        new.start(prompt)
    else:
        logits, cache, fed = transplant(pairs, tokens, pending, dst,
                                        mapper if need_mapper else None,
                                        sink=plan.sink, tail=plan.tail)
        new = Session(dst, dst_tier)
        new.emitted = emitted
        new.resume_from(logits, cache, fed)
    mx.eval([c.state for c in new.cache])
    rep.kv_dst_mib = kvcache.nbytes(new.kv_pairs()) / MiB
    pairs = mapper = None
    if sched == "colocated":
        # the hand-off shape loads before it releases; once done, it releases
        # properly (synchronise, clear, wait) — sloppy freeing is not what C tests
        clock("release_src")
        drop_source(clear=True, wait=True)
    if sched in ("ordered", "colocated"):
        # careful schedules return switch-time temporaries (mapping inputs,
        # prefill activations) to the OS; the naive ablations deliberately do not
        mx.synchronize()
        mx.clear_cache()
    clock("resumed")
    rep.seconds = time.perf_counter() - t0
    rep.mlx_peak_mib = mx.get_peak_memory() / MiB
    return new, rep


class _Premapped:
    """Stands in for a mapper whose output was computed before the source left:
    ``apply`` returns the stored destination-geometry pairs (truncated as asked)."""

    def __init__(self, mapper: KVMapper):
        self.dst = mapper.dst

    def apply(self, pairs, offset=0, out_dtype=None):
        return [(k.astype(out_dtype) if out_dtype else k, v.astype(out_dtype) if out_dtype else v)
                for k, v in pairs]


def selftest() -> None:
    from . import rope as R
    from .mappers import Geometry, LayerMap
    from .tinymodels import make, BIG, SMALL

    big = make(BIG, 5)
    g = Geometry.from_model(big)
    spec = R.RopeSpec.from_model(big)
    eye = mx.eye(g.width)
    ident = KVMapper(g, g, spec, spec, "identity", 1,
                     [LayerMap([l], eye, mx.zeros((g.width,)), eye, mx.zeros((g.width,)))
                      for l in range(g.n_layers)])
    s = Session(big, "big"); s.start([3, 1, 4, 1, 5, 9, 2, 6])
    for _ in range(5):
        s.step()
    ref, _ = prefill(big, s.tokens + [s.pending])
    errs = {}
    for sink, tail in ((0, 0), (4, 0), (0, 3), (2, 2), (0, "all")):
        lg, cache, fed = transplant(s.kv_pairs(), s.tokens, s.pending, big, ident, sink, tail)
        errs[(sink, tail)] = mx.abs(lg - ref).max().item()
        assert kvcache.length(kvcache.extract(cache)) == len(fed) == len(s.tokens) + 1
    assert max(errs.values()) < 1e-3, errs
    print(f"  identity mapper, same model: transplant == native re-prefill for all "
          f"(sink, tail) settings (max err {max(errs.values()):.1e})")

    small = make(SMALL, 6)
    from .fit import collect, fit
    import numpy as np
    rng = np.random.default_rng(1)
    st = collect({"b": big, "s": small}, [("b", "s")],
                 [rng.integers(0, 256, 48).tolist() for _ in range(10)], stride=1)[("b", "s")]
    m, _ = fit(st, "r2_topk", 2)
    for sched in SCHEDULES:
        src = make(BIG, 5)
        s = Session(src, "big"); s.start([3, 1, 4, 1, 5, 9, 2, 6]); s.step()
        del src
        new, rep = migrate(s, Plan(schedule=sched, sink=2, tail=2),
                           load_dst=lambda: make(SMALL, 6), load_mapper=lambda: m)
        for _ in range(4):
            new.step()
        assert len(new.emitted) == 2 + 1 + 4 and rep.lost_tokens == 0
        assert [p["name"] for p in rep.phases][-1] == "resumed"
    s = Session(make(BIG, 5), "big"); s.start([3, 1, 4]); s.step(); s.step()
    new, rep = migrate(s, Plan(restart=True), load_dst=lambda: make(SMALL, 6), prompt_len=3)
    assert rep.lost_tokens == 3 and len(new.tokens) == 3
    print(f"  all four schedules and restart run end to end on the tiny pair; "
          f"ordered phases: {[p['name'] for p in rep.phases]}")
    print("migrate: ok")


if __name__ == "__main__":
    selftest()
