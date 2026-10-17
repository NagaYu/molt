"""Experiment Q — quality of the continuation when the switch happens mid-decode.

No memory cap: every migrating condition switches at the same token
(``pressure_at`` of the item), so the conditions differ only in *how* they
switch.  Each (condition, shard of prompts) is a fresh process.

    python -m bench.run_quality --pair 1.5b-0.5b [--conds A,A2,B,C,D,...] [--shard 8]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from bench._env import RESULTS
from bench import prompts as P
from bench.conditions import ABLATIONS, PRIMARY

QUALITY_CONDS = ("big",) + PRIMARY + ("C_ordered", "D_depth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--conds", default=",".join(QUALITY_CONDS))
    ap.add_argument("--kinds", default="gsm8k,needle,open")
    ap.add_argument("--shard", type=int, default=8)
    ap.add_argument("--limit-gsm8k", type=int, default=None, help="first N GSM8K items")
    a = ap.parse_args()
    items = []
    for k in a.kinds.split(","):
        ks = P.load([k])
        items += ks[: a.limit_gsm8k] if (k == "gsm8k" and a.limit_gsm8k) else ks
    ids = [it["id"] for it in items]
    shards = [ids[i:i + a.shard] for i in range(0, len(ids), a.shard)]
    root = RESULTS / "raw" / "quality" / a.pair
    t0 = time.time()
    for cond in a.conds.split(","):
        for si, sh in enumerate(shards):
            out = root / cond / f"shard_{sh[0]}_{len(sh)}.json"   # named by content
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            t1 = time.time()
            rc = subprocess.call([sys.executable, "-m", "bench.worker", "--pair", a.pair,
                                  "--cond", cond, "--ids", ",".join(sh), "--out", str(out)])
            print(f"[{time.time() - t0:7.0f}s] {cond} shard {si + 1}/{len(shards)} rc={rc} "
                  f"({time.time() - t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
