"""Score collected human ratings (see README.md).  Writes results/human/<pair>.json."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from bench._env import RESULTS

SEAM = {"no": 0, "slight": 1, "clear": 2}


def kripp_alpha_ordinal(units):
    """Krippendorff's α (interval metric on the 0/1/2 seam scale)."""
    vals = [v for u in units for v in u]
    if len(vals) < 2:
        return float("nan")
    allv = np.array(vals, dtype=float)
    De = ((allv[:, None] - allv[None, :]) ** 2).sum() / (len(allv) * (len(allv) - 1))
    num, den = 0.0, 0
    for u in units:
        u = np.array(u, dtype=float)
        if len(u) < 2:
            continue
        num += ((u[:, None] - u[None, :]) ** 2).sum() / (len(u) - 1)
        den += len(u)
    Do = num / den if den else float("nan")
    return float(1 - Do / De) if De else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    a = ap.parse_args()
    d = RESULTS / "human" / a.pair
    key = json.loads((d / "key.json").read_text())
    rat = list(csv.DictReader(open(d / "ratings.csv")))
    per = defaultdict(lambda: defaultdict(list))
    units = defaultdict(list)
    for r in rat:
        k = key[r["pair_id"]]
        cs = k["condition_side"]
        rs = "right" if cs == "left" else "left"
        x, y = SEAM[r[f"seam_{cs}"]], SEAM[r[f"seam_{rs}"]]
        per[k["cond"]]["clear_diff"].append(float(x == 2) - float(y == 2))
        per[k["cond"]]["pref"].append(1.0 if r["pref"] == cs else (0.5 if r["pref"] == "tie" else 0.0))
        units[(r["pair_id"], "c")].append(x)
        units[(r["pair_id"], "r")].append(y)
    rng = np.random.default_rng(0)
    out = {}
    for cond, m in per.items():
        out[cond] = {}
        for name, v in m.items():
            v = np.array(v)
            bs = v[rng.integers(0, len(v), size=(4000, len(v)))].mean(1)
            out[cond][name] = dict(mean=float(v.mean()), lo=float(np.quantile(bs, .025)),
                                   hi=float(np.quantile(bs, .975)), n=len(v))
    out["alpha_seam"] = kripp_alpha_ordinal(list(units.values()))
    (d / "scores.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
