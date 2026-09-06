#!/usr/bin/env python3
"""Assemble the Hugging Face dataset payload from a benchmark run.

    python scripts/build_hf_dataset.py --out artifacts/hf-dataset

The raw benchmark output is deeply nested JSON, which the dataset viewer cannot
show. This flattens the parts a reader actually wants to sort and filter into
CSVs, and ships the raw JSON alongside so nothing is lost.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil


def _g(x):
    return "" if x is None or x != x else round(x, 4)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="benchmarks/results")
    p.add_argument("--traces", default="benchmarks/pressure_traces")
    p.add_argument("--figures", default="figures")
    p.add_argument("--projectors", default="artifacts/projectors")
    p.add_argument("--session", default="artifacts/demo_session.json")
    p.add_argument("--out", default="artifacts/hf-dataset")
    args = p.parse_args(argv)

    with open(os.path.join(args.results, "summary.json")) as fh:
        d = json.load(fh)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for k, r in d["conditions"].items():
        a, q = r["aggregate"], r["quality"]
        rows.append(dict(
            condition=k, label=r["label"], policy=r["policy"], kills=r["kills"],
            n_prompts=a["n_prompts"], worst_itl_ms=_g(a["worst_itl_ms"]),
            mean_itl_ms=_g(a["mean_itl_ms"]), mean_ttft_ms=_g(a["mean_ttft_ms"]),
            migrations=_g(a["mean_migrations"]),
            switch_cost_ms=_g(a["mean_migration_ms"]),
            handoff_jsd=_g(a.get("handoff_jsd")),
            handoff_jsd_raw=_g(a.get("handoff_jsd_raw")),
            judge_agreement=_g(q.get("judge_agreement")),
            judge_ppl=_g(q.get("judge_ppl")), accuracy=_g(q["accuracy"]),
            needle_recall=_g(q["needle_recall"]),
            repetition_rate=_g(a["repetition_rate"]), peak_mb=_g(a["peak_mb"]),
            frac_tokens_top_tier=_g(a["frac_top_tier"])))
    _write(os.path.join(args.out, "conditions.csv"), rows)

    keys = ["route", "tokens", "transplant_ms", "reprefill_ms", "speedup",
            "flops_saving", "project_ms", "recompute_ms", "rope_ms", "mb_in",
            "mb_out", "recomputed_layers", "projected_layers"]
    _write(os.path.join(args.out, "cost_sweep.csv"),
           [{k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in r.items() if k in keys}
            for r in d.get("cost_sweep", {}).get("rows", [])], keys)

    with open(os.path.join(args.out, "token_series.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "prompt_id", "token_index", "itl_ms", "tier",
                    "pressure", "resident_mb", "killed"])
        for k, r in d["conditions"].items():
            for s in r.get("series", []):
                itl, tiers = s["itl_ms"], s["tiers"]
                pr, rm = s.get("pressure") or [], s.get("resident_mb") or []
                for i in range(len(itl)):
                    w.writerow([k, s["prompt_id"], i, round(itl[i], 3), tiers[i],
                                round(pr[i], 4) if i < len(pr) else "",
                                round(rm[i], 1) if i < len(rm) else "",
                                int(bool(s.get("killed")))])

    idx = os.path.join(args.projectors, f"{d['meta']['ladder']}__index.json")
    if os.path.exists(idx):
        with open(idx) as fh:
            pi = json.load(fh)
        _write(os.path.join(args.out, "projector_residuals.csv"),
               [dict(route=r, kind=v["kind"], rope_realign=v["rope_realign"],
                     fit_tokens=v["n_calib_tokens"], val_tokens=v["n_val_tokens"],
                     val_k=_g(v["val_k"]), val_v=_g(v["val_v"]),
                     val_hidden=_g(v["val_hidden"]), train_k=_g(v["train_k"]),
                     train_v=_g(v["train_v"])) for r, v in pi.items()])

    for src, dst in ((args.results, "results"), (args.traces, "pressure_traces"),
                     (args.figures, "figures")):
        if os.path.isdir(src):
            out = os.path.join(args.out, dst)
            shutil.rmtree(out, ignore_errors=True)
            shutil.copytree(src, out,
                            ignore=shutil.ignore_patterns("*.py", "*.md", "__pycache__"))
    if os.path.exists(args.session):
        shutil.copy(args.session, os.path.join(args.out, "demo_session.json"))

    print(f"wrote {args.out}")
    return 0


def _write(path, rows, keys=None):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys or list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
