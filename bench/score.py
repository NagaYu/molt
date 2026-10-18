"""Score generated text with a fixed judge, after all generation has finished.

The judge is the pair's *source* (big) model — the quality the user would have
had without pressure.  Run in its own process, after generation, so judging
never shares memory with a benchmarked condition.

Per run:
  task_correct     gsm8k / needle exact match (killed or missing → 0, never dropped)
  agree_post       judge top-1 agreement over tokens after the boundary
  nll_post         judge NLL over tokens after the boundary
  jump             boundary_jump of judge NLL (w = 16) at the boundary
  lost_tokens      tokens shown to the user and then retracted

    python -m bench.score --pair 1.5b-0.5b --exp quality
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench._env import RESULTS
from bench import prompts as P

import numpy as np

W = 16


def task_score(item, text):
    from molt_mlx.metrics import gsm8k_answer
    if item["kind"] == "gsm8k":
        return float(gsm8k_answer(text) == item["answer"])
    if item["kind"] in ("needle", "longneedle"):
        nums = [t for t in __import__("re").findall(r"\d{4}", text)]
        return float(bool(nums) and nums[0] == item["answer"])
    return None


def iter_runs(exp: str, pair: str, sub: str = ""):
    root = RESULTS / "raw" / exp / (pair + sub)
    if exp == "quality":
        for f in sorted(root.glob("*/shard_*.json")):
            for r in json.loads(f.read_text())["runs"]:
                yield f, r
    else:
        for f in sorted(root.glob("h*/*.json")):
            if f.name.endswith(".worker.json"):
                continue
            rec = json.loads(f.read_text())
            yield f, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--exp", default="quality", choices=["quality", "memory"])
    ap.add_argument("--sub", default="", help="memory sub-experiment folder suffix, e.g. _long")
    a = ap.parse_args()
    import mlx.core as mx
    mx.set_cache_limit(1 << 30)   # offline script: bounded MLX buffer cache
    from molt_mlx import tiers
    from molt_mlx.metrics import boundary_jump, judge
    src, _ = a.pair.split("-")
    tok = tiers.load_tokenizer()
    model = tiers.load_model(src)
    items = {p["id"]: p for p in P.load()}
    out_path = RESULTS / "scored" / a.exp / f"{a.pair}{a.sub}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {(d["cond"], d["id"], d.get("tag"), d.get("cap_mib")) for d in map(json.loads, out_path.read_text().splitlines())}
    with open(out_path, "a") as fh:
        for f, r in iter_runs(a.exp, a.pair, a.sub):
            if a.exp == "memory":
                rec, w = r, r.get("worker")
                key = (r["cond"], r["id"], r["tag"], r["cap_mib"])
            else:
                rec, w = dict(cond=r["cond"], id=r["id"], tag=None, cap_mib=None), r
                key = (r["cond"], r["id"], None, None)
            if key in done:
                continue
            item = items[rec["id"]]
            row = dict(cond=rec["cond"], id=rec["id"], kind=item["kind"], tag=rec.get("tag"),
                       cap_mib=rec.get("cap_mib"), killed=bool(rec.get("killed")))
            if w is None or row["killed"]:
                row.update(task_correct=0.0 if item["kind"] != "open" else None, completed=False,
                           lost_tokens=rec.get("emitted_before_kill"))
            else:
                ids = tok.apply_chat_template([{"role": "user", "content": item["text"]}],
                                              add_generation_prompt=True)
                gen = w["generated"]
                j = judge(model, ids, gen)
                b = w.get("boundary")
                post = slice(b, None) if b is not None else slice(item["pressure_at"], None)
                row.update(
                    completed=True, task_correct=task_score(item, w["text"]),
                    boundary=b, n_gen=len(gen), lost_tokens=w.get("lost_tokens", 0),
                    agree_post=float(j["agree"][post].mean()) if len(j["agree"][post]) else None,
                    nll_post=float(j["nll"][post].mean()) if len(j["nll"][post]) else None,
                    agree_all=float(j["agree"].mean()), nll_all=float(j["nll"].mean()),
                    jump=boundary_jump(j["nll"], b, W) if b is not None and not w["cond"] == "A" else None,
                    agree_jump=boundary_jump(j["agree"], b, W) if b is not None and not w["cond"] == "A" else None,
                    time_to_resume_s=w.get("time_to_resume_s"),
                    time_to_new_content_s=w.get("time_to_new_content_s"),
                    migration_s=(w.get("migration") or {}).get("seconds"),
                    text=w["text"])
                if a.exp == "memory":
                    row.update(peak_mib=rec["peak_mib"], phase_peaks_mib=rec["phase_peaks_mib"])
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(row["cond"], row["id"], row.get("task_correct"), row.get("agree_post"), flush=True)


if __name__ == "__main__":
    main()
