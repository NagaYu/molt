"""Experiment M — survival and peak footprint under a per-process memory cap.

Every run is its own process under the external watchdog, which SIGKILLs it the
instant phys_footprint exceeds the cap (jetsam emulation).  The sweep variable is
the *headroom* ``h``: how much memory the app has spare while the big model is
decoding.

    cap     = S + h          S = measured footprint of the big model decoding
    ballast = h + (S − S_small) − end_margin
            (the foreign demand: more than the big model can absorb, small
             enough that the small model plus the demand fits)

so every condition must give up the big model, and the small model always fits
afterwards.  What differs is whether the *transition* fits.

    python -m bench.run_memory --pair 1.5b-0.5b [--repeats 2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from bench._env import ARTIFACTS, MLX_DIR, RESULTS
from bench.conditions import CONDITIONS

from molt_mlx import watchdog

MEM_CONDS = ("static_big", "A", "A2", "B", "C", "D", "C_ordered", "D_colocated",
             "D_del_only", "D_clear_no_wait")
END_MARGIN_MIB = 128
CHUNK_MIB = 64
PROBE_ID = "open-00"
PROMPT_IDS = ("open-01", "open-02", "open-03")


def run(pair, cond, pid, cap, ballast, out_dir: Path, tag: str):
    out = out_dir / f"{cond}__{pid}__{tag}.json"
    if out.exists():
        return json.loads(out.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = str(out.with_suffix(".csv"))
    cmd = [sys.executable, "-m", "bench.worker", "--pair", pair, "--cond", cond, "--ids", pid,
           "--out", str(out.with_suffix(".worker.json")), "--cap-mib", str(cap),
           "--ballast-mib", str(ballast), "--chunk-mib", str(CHUNK_MIB)]
    t0 = time.time()
    rc, res = watchdog.run_under_watchdog(cmd, cap_mib=cap, csv_path=csv, timeout_s=900)
    wj = out.with_suffix(".worker.json")
    worker = json.loads(wj.read_text())["runs"][0] if wj.exists() else None
    prog = Path(str(wj) + ".progress.jsonl")
    events = [json.loads(l) for l in prog.read_text().splitlines()] if prog.exists() else []
    rec = dict(pair=pair, cond=cond, id=pid, cap_mib=cap, ballast_mib=ballast, tag=tag, rc=rc,
               error=(not res.killed) and (rc != 0 or worker is None),
               killed=res.killed, killed_in=res.killed_in_phase(), peak_mib=res.peak_mib,
               phase_peaks_mib=res.phase_peaks_mib(), max_gap_ms=res.max_gap_us() / 1000,
               samples=res.samples, wall_s=time.time() - t0, worker=worker,
               emitted_before_kill=max([e.get("i", 0) for e in events if e.get("ev") == "token"] or [0]) + 1
               if res.killed else None,
               triggered=any(e.get("ev") == "trigger" for e in events))
    out.write_text(json.dumps(rec))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--conds", default=",".join(MEM_CONDS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--h", default=None, help="comma-separated headroom values (MiB)")
    a = ap.parse_args()
    root = RESULTS / "raw" / "memory" / a.pair
    # 1) probe S and S_small: uncapped, no ballast, watchdog in monitor-only mode
    big = run(a.pair, "static_big", PROBE_ID, 0, 0, root / "probe", "probe")
    small = run(a.pair, "B", PROBE_ID, 0, 0, root / "probe", "probe")
    S = big["phase_peaks_mib"]["decode_src"]
    S_small = small["phase_peaks_mib"]["decode_src"]
    src, dst = a.pair.split("-")
    from molt_mlx import tiers
    from bench.worker import dir_bytes
    man = json.loads((ARTIFACTS / "mappers" / a.pair / "manifest.json").read_text())
    dst_w = dir_bytes(tiers.local_path(dst)) / 2**20
    map_w = (ARTIFACTS / "mappers" / a.pair / f"{man['selected']}.safetensors").stat().st_size / 2**20
    need_c = dst_w + map_w + 96
    hs = [float(x) for x in a.h.split(",")] if a.h else sorted({
        128.0, round(need_c * 0.5), round(need_c - 64), round(need_c + 160), round(need_c * 2)})
    plan = dict(S=S, S_small=S_small, dst_w=dst_w, mapper_w=map_w, need_colocated=need_c, h=hs,
                end_margin=END_MARGIN_MIB, chunk=CHUNK_MIB)
    (root / "plan.json").write_text(json.dumps(plan, indent=1))
    print("plan", json.dumps(plan), flush=True)
    t0 = time.time()
    for h in hs:
        cap = round(S + h)
        ballast = max(CHUNK_MIB, round(h + (S - S_small) - END_MARGIN_MIB))
        for rep in range(a.repeats):
            for pid in PROMPT_IDS:
                for cond in a.conds.split(","):
                    r = run(a.pair, cond, pid, cap, ballast, root / f"h{int(h)}", f"r{rep}")
                    print(f"[{time.time() - t0:6.0f}s] h={h:.0f} cap={cap} {cond:16s} {pid} "
                          f"{'KILLED in ' + str(r['killed_in']) if r['killed'] else ('ERROR rc=' + str(r['rc']) if r.get('error') else 'ok')} "
                          f"peak={r['peak_mib']:.0f}", flush=True)


if __name__ == "__main__":
    main()
