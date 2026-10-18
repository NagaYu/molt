"""Collect every result file into ``bench/results/summary.json``.

Nothing is filtered: killed runs stay in every denominator (task score 0),
missing experiments appear as ``null`` so downstream renders them PENDING.

    python -m bench.analyze
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional

from bench._env import RESULTS

import numpy as np

PAIRS = ("3b-1.5b", "1.5b-0.5b")


def ci(xs, seed=0):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None
    a = np.asarray(xs)
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, len(a), size=(4000, len(a)))].mean(1)
    return dict(mean=float(a.mean()), lo=float(np.quantile(m, 0.025)),
                hi=float(np.quantile(m, 0.975)), n=len(a))


def paired(xs: Dict[str, float], ys: Dict[str, float]):
    keys = sorted(set(xs) & set(ys))
    return ci([xs[k] - ys[k] for k in keys])


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def fit_summary(pair):
    d = load_json(RESULTS / "fit" / f"{pair}.json")
    if not d:
        return None
    rows = {n: dict(k=m["k"], policy=m["policy"], params_m=m["params"] / 1e6, mib=m["mib_fp16"],
                    insample_r2=(m["insample_r2_k"] + m["insample_r2_v"]) / 2,
                    heldout_r2_k=m["heldout_r2_k"], heldout_r2_v=m["heldout_r2_v"],
                    heldout_nll=m["heldout_nll"]) for n, m in d["mappers"].items()}
    return dict(rows=rows, natives=d["natives"], selected=d["selected"], lam_sweep=d["lam_sweep"],
                anchors=d["anchors"], molt_anchor=d["molt_anchor"], n_rows=d["n_rows"],
                corpus=d["corpus"], seconds=d["seconds"], machine=d["machine_start"],
                single_source_r2=d["single_source_r2"])


def intrinsic_summary(pair):
    d = load_json(RESULTS / "intrinsic" / f"{pair}.json")
    if not d:
        return None
    return dict(n=d["n"], span=d["span"], mapper=d["mapper"], sink=d["sink"], tail=d["tail"],
                summary=d["summary"])


def _max_new():
    pj = RESULTS.parent / "prompts.json"
    return {p["id"]: p["max_new"] for p in json.loads(pj.read_text())} if pj.exists() else {}


def quality_summary(pair):
    p = RESULTS / "scored" / "quality" / f"{pair}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    mx_new = _max_new()
    by = defaultdict(list)
    for r in rows:
        by[r["cond"]].append(r)
    out = {}
    for cond, rs in by.items():
        acc = {}
        for kind in ("gsm8k", "needle", "longneedle"):
            v = [r["task_correct"] for r in rs if r["kind"] == kind]
            acc[kind] = ci(v)
        trunc = {kind: ci([float(r.get("n_gen") == mx_new.get(r["id"])) for r in rs
                           if r["kind"] == kind and r.get("n_gen") is not None])
                 for kind in ("gsm8k", "open")} if mx_new else {}
        out[cond] = dict(
            n=len(rs), acc=acc, truncated=trunc,
            agree_post=ci([r.get("agree_post") for r in rs]),
            agree_post_open=ci([r.get("agree_post") for r in rs if r["kind"] == "open"]),
            nll_post=ci([r.get("nll_post") for r in rs]),
            jump=ci([r.get("jump") for r in rs]),
            lost_tokens=ci([r.get("lost_tokens") or 0 for r in rs]),
            time_to_resume_s=ci([r.get("time_to_resume_s") for r in rs]),
            time_to_new_content_s=ci([r.get("time_to_new_content_s") for r in rs]),
            migration_s=ci([r.get("migration_s") for r in rs]))
    # paired comparisons against A2 (re-prefill) and C (ridge), per item
    def per_item(cond, key):
        return {r["id"]: r[key] for r in by.get(cond, []) if r.get(key) is not None}
    comps = {}
    def per_kind(cond, key, kind):
        return {r["id"]: r[key] for r in by.get(cond, []) if r.get(key) is not None and r["kind"] == kind}
    for a in by:
        for b in ("A2", "C", "B"):
            if a == b or b not in by:
                continue
            comps[f"{a}-{b}-longneedle"] = dict(
                task=paired(per_kind(a, "task_correct", "longneedle"), per_kind(b, "task_correct", "longneedle")),
                resume=paired(per_kind(a, "time_to_new_content_s", "longneedle"),
                              per_kind(b, "time_to_new_content_s", "longneedle")))
            comps[f"{a}-{b}"] = dict(
                jump=paired(per_item(a, "jump"), per_item(b, "jump")),
                agree_post=paired(per_item(a, "agree_post"), per_item(b, "agree_post")),
                task=paired({k: v for k, v in per_item(a, "task_correct").items()},
                            per_item(b, "task_correct")))
    # context length at the switch (prompt incl. chat template + tokens already emitted)
    ctx = defaultdict(list)
    for f in (RESULTS / "raw" / "quality" / pair).glob("*/shard_*.json"):
        for r in json.loads(f.read_text())["runs"]:
            if r.get("boundary") is not None:
                ctx[r["kind"]].append(r["prompt_len"] + r["boundary"])
    context = {k: dict(median=median(v), min=min(v), max=max(v)) for k, v in ctx.items()}
    return dict(conds=out, paired=comps, n_items=len({r["id"] for r in rows}), context=context)


def _phase_durations(migs):
    out = defaultdict(list)
    for m in migs:
        ph = m.get("phases") or []
        ts = [p["t"] for p in ph] + [m.get("seconds", 0)]
        for i, p in enumerate(ph):
            out[p["name"]].append(ts[i + 1] - ts[i])
    return out


def long_quality(pair):
    """Judge scores of the long-context memory runs (killed runs excluded here;
    their survival is reported in the memory table)."""
    p = RESULTS / "scored" / "memory" / f"{pair}_long.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    by = defaultdict(list)
    for r in rows:
        if r.get("completed"):
            by[r["cond"]].append(r)
    out = {c: dict(n=len(rs), agree_post=ci([r.get("agree_post") for r in rs]),
                   jump=ci([r.get("jump") for r in rs]), nll_post=ci([r.get("nll_post") for r in rs]))
           for c, rs in by.items()}
    item = lambda c, k: {r["id"]: r[k] for r in by.get(c, []) if r.get(k) is not None}
    paired = {f"{a}-{b}": dict(agree_post=paired_(item(a, "agree_post"), item(b, "agree_post")),
                               jump=paired_(item(a, "jump"), item(b, "jump")))
              for a in ("D", "C_ordered") for b in ("A2",) if a in by and b in by}
    return dict(conds=out, paired=paired)


def paired_(x, y):
    return paired(x, y)


def memory_summary(pair, sub=None):
    root = RESULTS / "raw" / "memory" / (sub or pair)
    plan = load_json(root / "plan.json")
    if not plan:
        return None
    cells = defaultdict(list)
    for f in sorted(root.glob("h*/*.json")):
        if f.name.endswith(".worker.json"):
            continue
        r = json.loads(f.read_text())
        h = int(f.parent.name[1:])
        cells[(h, r["cond"])].append(r)
    out = defaultdict(dict)
    for (h, cond), rs in sorted(cells.items()):
        w = [r["worker"] for r in rs if r.get("worker")]
        mig = [x.get("migration") for x in w if x.get("migration")]
        en = [x.get("energy_migration") for x in w if x.get("energy_migration")]
        ok = [r for r in rs if not r.get("error")]          # a crash is not a survival
        out[str(h)][cond] = dict(
            n=len(ok), errors=len(rs) - len(ok), survived=sum(not r["killed"] for r in ok),
            survival=ci([0.0 if r["killed"] else 1.0 for r in ok]),
            peak_mib=dict(median=median([r["peak_mib"] for r in rs]),
                          max=max(r["peak_mib"] for r in rs), min=min(r["peak_mib"] for r in rs)),
            cap_mib=rs[0]["cap_mib"],
            killed_in=sorted({r["killed_in"] for r in rs if r["killed"]}),
            triggered=sum(bool(r.get("triggered")) for r in rs),
            time_to_resume_s=ci([x.get("time_to_resume_s") for x in w]),
            time_to_new_content_s=ci([x.get("time_to_new_content_s") for x in w]),
            migration_s=ci([m.get("seconds") for m in mig]),
            release_wait_ms=ci([sum(m.get("release_waits_ms") or [0]) for m in mig]),
            energy_mig_cpu_j=ci([e.get("cpu_j") for e in en]),
            energy_mig_gpu_j=ci([e.get("gpu_j") for e in en]),
            gpu_baseline_w=ci([x.get("gpu_baseline_w") for x in w]),
            phase_s={name: median(vals) for name, vals in _phase_durations(mig).items()},
            resumed_mib=ci([r["phase_peaks_mib"].get("resumed") for r in ok if not r["killed"] or "resumed" in r["phase_peaks_mib"]]),
            phase_peaks=[r["phase_peaks_mib"] for r in rs][:1])
    return dict(plan=plan, cells=out)


def main():
    s = dict(pairs={})
    for pair in PAIRS:
        s["pairs"][pair] = dict(fit=fit_summary(pair), intrinsic=intrinsic_summary(pair),
                                quality=quality_summary(pair), memory=memory_summary(pair),
                                memory_cache20=memory_summary(pair, f"{pair}_cache20"),
                                memory_long=memory_summary(pair, f"{pair}_long"),
                                long_quality=long_quality(pair))
    s["resume_cost"] = {p: load_json(RESULTS / "resume_cost" / f"{p}.json") for p in PAIRS}
    s["facts"] = dict(platform=load_json(RESULTS / "facts" / "platform.json"),
                      peak_test=load_json(RESULTS / "facts" / "peak_test.json"))
    dev = RESULTS / "device"
    s["device"] = sorted(p.name for p in dev.iterdir()) if dev.exists() else []
    (RESULTS / "summary.json").write_text(json.dumps(s, indent=1))
    print("wrote", RESULTS / "summary.json")


if __name__ == "__main__":
    main()
