"""Run every experiment for every pair, in priority order, resumably.

    nohup python -m bench.run_all > bench/results/run_all.log 2>&1 &

Order per pair: fit → memory (the claim) → intrinsic → quality (needle, open,
gsm8k; primary conditions before ablations) → judge scoring → summary/figures.
Every step skips work whose output already exists, so the driver can be
stopped and restarted at any point.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from bench._env import RESULTS

PY = sys.executable


def step(*args) -> int:
    t = time.time()
    print(time.strftime("%H:%M:%S"), "→", " ".join(args), flush=True)
    rc = subprocess.call([PY, "-m", *args])
    print(time.strftime("%H:%M:%S"), f"  rc={rc} ({time.time() - t:.0f}s)", flush=True)
    return rc


def refresh():
    step("bench.analyze")
    subprocess.call([PY, "figures/make_paper_figures.py"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="1.5b-0.5b,3b-1.5b")
    ap.add_argument("--skip-fit", action="store_true")
    a = ap.parse_args()
    if not (RESULTS / "facts" / "platform.json").exists():
        step("bench.facts")
    for pair in a.pairs.split(","):
        if not a.skip_fit and not (RESULTS / "fit" / f"{pair}.json").exists():
            if step("bench.fit_mappers", "--pair", pair) != 0:
                print("fit failed; skipping pair", pair, flush=True)
                continue
        refresh()
        smoke = RESULTS / "smoke" / pair
        bad = []
        for cond in ("C", "D", "A2", "D_depth"):
            out = smoke / f"{cond}.json"
            if not out.exists() and step("bench.worker", "--pair", pair, "--cond", cond, "--ids",
                                         "needle-00", "--max-new", "12", "--pressure-at", "6",
                                         "--out", str(out)) != 0:
                bad.append(cond)
        if bad:
            print("smoke failed for", bad, "- stopping before the long runs", flush=True)
            return
        step("bench.run_memory", "--pair", pair)
        refresh()
        if not (RESULTS / "resume_cost" / f"{pair}.json").exists():
            step("bench.resume_cost", "--pair", pair)
        if not (RESULTS / "intrinsic" / f"{pair}.json").exists():
            step("bench.intrinsic", "--pair", pair)
        refresh()
        for kinds, conds in (("needle", "C,D,A2,B,A,big"), ("open", "C,D,A2,B,A,big"),
                             ("gsm8k", "C,D,A2,B,A,big"),
                             ("needle,open,gsm8k", "C_ordered,D_depth")):
            step("bench.run_quality", "--pair", pair, "--kinds", kinds, "--conds", conds,
                 "--limit-gsm8k", "20")
            step("bench.score", "--pair", pair, "--exp", "quality")
            refresh()
    print("all done", flush=True)


if __name__ == "__main__":
    main()
