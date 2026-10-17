"""Export blinded continuation pairs for human raters (see README.md)."""

from __future__ import annotations

import argparse
import csv
import json
import random

from bench._env import RESULTS

COMPARE = ("C", "D", "B")
REF = "A2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--seed", type=int, default=20261017)
    a = ap.parse_args()
    from molt_mlx import tiers
    tok = tiers.load_tokenizer()
    runs = {}
    for f in (RESULTS / "raw" / "quality" / a.pair).glob("*/shard_*.json"):
        for r in json.loads(f.read_text())["runs"]:
            if r["kind"] == "open":
                runs[(r["cond"], r["id"])] = r
    rng = random.Random(a.seed)
    out_dir = RESULTS / "human" / a.pair
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, key = [], {}
    for (cond, pid), r in sorted(runs.items()):
        if cond not in COMPARE or (REF, pid) not in runs:
            continue
        ref = runs[(REF, pid)]
        b = r.get("boundary") or ref.get("boundary") or 48
        prefix = tok.decode(ref["generated"][:b])
        cont_x = tok.decode(r["generated"][b:]) if cond != "B" else tok.decode(r["generated"])
        cont_r = tok.decode(ref["generated"][b:])
        flip = rng.random() < 0.5
        pid_ = f"{a.pair}:{pid}:{cond}"
        left, right = (cont_r, cont_x) if flip else (cont_x, cont_r)
        rows.append(dict(pair_id=pid_, prompt=r["id"], opening=prefix if cond != "B" else "",
                         left=left, right=right))
        key[pid_] = dict(cond=cond, condition_side="right" if flip else "left", switch_token=b)
    rng.shuffle(rows)
    with open(out_dir / "pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["pair_id"])
        w.writeheader(); w.writerows(rows)
    (out_dir / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(rows)} pairs -> {out_dir / 'pairs.csv'}")


if __name__ == "__main__":
    main()
