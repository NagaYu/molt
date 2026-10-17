"""Turn files pulled from iPhones into result JSON (see ios/README.md).

Expected layout, one folder per device (model identifier, e.g. iPhone15,2):

    bench/results/device/<model>/budget-summary*.json     MoltCore.BudgetTest.Summary
    bench/results/device/<model>/JetsamEvent-*.ips        exported from Settings → Analytics
    bench/results/device/<model>/bench-<pair>-<cond>-<prompt>*.jsonl   BenchRun telemetry

    python -m bench.ingest_device      # -> bench/results/device_summary.json

Jetsam reports are read for the entry of our app (``dev.molt.MoltBench``): its
``reason`` and ``rpages × pageSize`` (Apple: "multiply the number of memory pages
reported in the rpages field by the page size").  A bench run is *killed* if its
telemetry has a trigger or tokens but no ``done`` event and a matching jetsam
record exists; *crashed* if neither.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from bench._env import RESULTS

BUNDLE = "MoltBench"


def parse_jetsam(path: Path):
    """.ips files are a JSON header line followed by a JSON body."""
    text = path.read_text(errors="replace")
    parts = text.split("\n", 1)
    try:
        body = json.loads(parts[1]) if len(parts) > 1 else json.loads(parts[0])
    except json.JSONDecodeError:
        return None
    page = body.get("pageSize") or body.get("header", {}).get("pageSize") or 16384
    for p in body.get("processes", []):
        if BUNDLE.lower() in str(p.get("name", "")).lower():
            return dict(file=path.name, reason=p.get("reason"), rpages=p.get("rpages"),
                        mib=(p.get("rpages") or 0) * page / 2 ** 20, page_size=page,
                        largest=body.get("largestProcess"))
    return dict(file=path.name, reason=None, note="app not in report")


def parse_bench(path: Path):
    ev = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    first = next((e for e in ev if e.get("ev") == "start"), {})
    trig = next((e for e in ev if e.get("ev") == "trigger"), None)
    mig = next((e for e in ev if e.get("ev") == "migrated"), None)
    done = next((e for e in ev if e.get("ev") == "done"), None)
    toks = [e for e in ev if e.get("ev") == "token"]
    after = [e["t"] for e in toks if trig and e["t"] > trig["t"]]
    return dict(file=path.name, cond=first.get("cond"), prompt=first.get("prompt"), pair=first.get("pair"),
                device=first.get("device"), start_available_mib=first.get("available_mib"),
                completed=done is not None, n_tokens=len(toks),
                triggered=trig is not None, trigger_available_mib=(trig or {}).get("available_mib"),
                migration_s=(mig or {}).get("seconds"), migration_peak_mib=(mig or {}).get("peak_mib"),
                time_to_resume_s=(after[0] - trig["t"]) if after else None,
                peak_mib=(done or {}).get("peak_mib"),
                signals=[e.get("signal") for e in ev if e.get("ev") == "signal"])


def main():
    root = RESULTS / "device"
    out = {}
    if root.exists():
        for dev in sorted(p for p in root.iterdir() if p.is_dir()):
            budgets = [json.loads(p.read_text()) for p in sorted(dev.glob("budget-summary*.json"))]
            jets = [j for j in (parse_jetsam(p) for p in sorted(dev.glob("JetsamEvent-*.ips"))) if j]
            runs = [parse_bench(p) for p in sorted(dev.glob("bench-*.jsonl"))]
            cells = defaultdict(list)
            for r in runs:
                cells[r["cond"]].append(r)
            out[dev.name] = dict(
                budget=budgets, jetsam=jets,
                per_process_limit_kills=sum(1 for j in jets if j.get("reason") == "per-process-limit"),
                runs=runs,
                by_cond={c: dict(n=len(v), completed=sum(r["completed"] for r in v),
                                 resume_s=[r["time_to_resume_s"] for r in v if r["time_to_resume_s"]],
                                 peak_mib=[r["peak_mib"] for r in v if r["peak_mib"]])
                         for c, v in cells.items()})
    (RESULTS / "device_summary.json").write_text(json.dumps(out, indent=1))
    print(f"{len(out)} device folder(s) ->", RESULTS / "device_summary.json")


if __name__ == "__main__":
    main()
