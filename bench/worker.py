"""One benchmark process: one condition, one or more prompts.

    python -m bench.worker --pair 3b-1.5b --cond D --ids gsm8k-000,needle-03 \
        --out bench/results/raw/... [--cap-mib 3000 --ballast-mib 900 --chunk-mib 64]

The orchestrators start every worker in a fresh process (under the external
watchdog when a cap is set), so no condition inherits another's allocator state,
page cache warmth or compiled kernels.

Pressure model (memory runs).  At token ``pressure_at`` a foreign demand of
``ballast_mib`` starts arriving, ``chunk_mib`` per token — the app's other
features needing memory.  It arrives whether or not the model reacts.  Before
every token the worker reads the emulated ``os_proc_available_memory`` (cap −
phys_footprint) and migrates once headroom drops below what *that condition's*
migration needs plus the next chunk: each condition gets its own best trigger,
so a co-resident schedule is not killed merely for reacting late.

Quality runs set no cap: the migration fires exactly at ``pressure_at`` so every
condition switches at the same token.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from bench._env import ARTIFACTS, machine
from bench.conditions import CONDITIONS
from bench import prompts as P

from molt_mlx.watchdog import wait_until_watched  # noqa: E402  (stdlib only)

MiB = 2 ** 20


def dir_bytes(path: Path, pattern: str = "*.safetensors") -> int:
    return sum(p.stat().st_size for p in path.glob(pattern))


def prewarm(paths) -> float:
    """Read model/mapper files once so every condition loads from a warm page
    cache.  Without this, whichever condition runs first pays the cold disk read
    and time-to-resume depends on run order (observed: 5.1 s vs 1.4 s).  Page
    cache is not part of phys_footprint, so this does not affect any memory
    number.  Returns seconds spent (outside every timed window)."""
    t = time.perf_counter()
    for p in paths:
        p = Path(p)
        for f in ([p] if p.is_file() else sorted(p.glob("*.safetensors"))):
            with open(f, "rb") as fh:
                while fh.read(8 << 20):
                    pass
    return time.perf_counter() - t


class Ballast:
    """Foreign memory demand: plain process memory, pages touched on arrival."""

    def __init__(self, total_mib: float, chunk_mib: float):
        self.total, self.chunk, self.blocks = total_mib, chunk_mib, []

    @property
    def held_mib(self) -> float:
        return len(self.blocks) * self.chunk

    def next_chunk_mib(self) -> float:
        return self.chunk if self.held_mib + 1e-9 < self.total else 0.0

    def grow(self) -> None:
        import numpy as np
        if self.next_chunk_mib():
            self.blocks.append(np.ones(int(self.chunk * MiB // 8), dtype=np.float64))


def run_prompt(item, cond, pair, tok, args, mark, log):
    import mlx.core as mx
    from molt_mlx import tiers
    from molt_mlx.footprint import emulated_available_memory, footprint_bytes
    from molt_mlx.generate import Session
    from molt_mlx.mappers import KVMapper
    from molt_mlx.migrate import Plan, migrate
    from molt_mlx.energy import EnergyMeter

    src, dst = pair.split("-")
    mdir = ARTIFACTS / "mappers" / pair
    man = json.loads((mdir / "manifest.json").read_text()) if (cond.mapper or cond.anchors) else None
    mapper_file = (mdir / (f"{man['selected']}.safetensors" if cond.mapper == "heo"
                           else "depth_linear.safetensors")) if cond.mapper else None
    sink, tail = (man["molt_anchor"]["sink"], man["molt_anchor"]["tail"]) if cond.anchors else (0, 0)
    eos = set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}

    ids = tok.apply_chat_template([{"role": "user", "content": item["text"]}],
                                  add_generation_prompt=True)
    cap = int(args.cap_mib * MiB) if args.cap_mib else 0
    ballast = Ballast(args.ballast_mib, args.chunk_mib)
    dst_w = dir_bytes(tiers.local_path(dst)) / MiB
    map_w = mapper_file.stat().st_size / MiB if cond.mapper else 0.0
    margin = args.margin_mib
    need = margin + ((dst_w + map_w) if cond.schedule == "colocated" else 0.0)

    rec = dict(id=item["id"], kind=item["kind"], cond=cond.name, pair=pair, prompt_len=len(ids),
               cap_mib=args.cap_mib, ballast_mib=args.ballast_mib, chunk_mib=args.chunk_mib,
               trigger_need_mib=need, sink=sink, tail=tail, mapper=mapper_file.name if mapper_file else None)
    rec["prewarm_s"] = prewarm([tiers.local_path(src), tiers.local_path(dst)] +
                               ([mapper_file] if mapper_file else []))
    meter = EnergyMeter()
    rec["gpu_baseline_w"] = meter.baseline(0.4)
    meter.start()
    t0 = time.perf_counter()
    mark("load_src")
    model = tiers.load_model(dst if cond.start_tier == "dst" else src)
    rec["t_loaded"] = time.perf_counter() - t0
    mark("prefill")
    s = Session(model, dst if cond.start_tier == "dst" else src)
    del model
    s.start(ids)
    times = [time.perf_counter() - t0]
    tiers_of = [s.tier]
    log(dict(ev="token", i=0, t=times[0]))
    mark("decode_src")
    migrated, rep, trig = False, None, None
    fp_trace = []
    while len(s.emitted) < item["max_new"] and s.pending not in eos:
        n = len(s.emitted)
        pressure = n >= item["pressure_at"]
        if pressure and cap:
            ballast.grow()
        fp = footprint_bytes()
        fp_trace.append((n, round(fp / MiB, 1)))
        if pressure and cond.migrates and not migrated:
            head = emulated_available_memory(cap) / MiB if cap else 0.0
            fire = (not cap) or head < need + ballast.next_chunk_mib()
            if fire:
                trig = dict(t=time.perf_counter() - t0, n_emitted=n, headroom_mib=head,
                            footprint_mib=fp / MiB, ballast_held_mib=ballast.held_mib,
                            shown=list(s.emitted))
                log(dict(ev="trigger", **trig))
                plan = Plan(schedule=cond.schedule, use_mapper=cond.mapper is not None,
                            sink=sink, tail="all" if cond.reprefill else tail,
                            restart=cond.restart)
                mm = EnergyMeter(); mm.base_w = meter.base_w; mm.start()
                s, rep = migrate(s, plan, load_dst=lambda: tiers.load_model(dst),
                                 load_mapper=(lambda: KVMapper.load(str(mapper_file))) if cond.mapper else None,
                                 prompt_len=len(ids), dst_tier=dst, marker=mark)
                rec["energy_migration"] = mm.stop().to_dict()
                migrated = True
                mark("decode_dst")
                times.append(time.perf_counter() - t0)
                tiers_of.append(s.tier)
                log(dict(ev="token", i=len(s.emitted) - 1, t=times[-1]))
                continue
        s.step()
        times.append(time.perf_counter() - t0)
        tiers_of.append(s.tier)
        if len(times) % 16 == 0:
            log(dict(ev="token", i=len(s.emitted) - 1, t=times[-1]))
    mark("done")
    rec["energy_run"] = meter.stop().to_dict()
    gen = s.emitted
    rec.update(
        generated=gen, text=tok.decode(gen), tiers=tiers_of, token_times=times,
        boundary=(trig["n_emitted"] if trig else None), trigger=trig,
        migration=rep.to_dict() if rep else None, migrated=migrated,
        footprint_trace=fp_trace[::4], ballast_held_mib=ballast.held_mib,
        lost_tokens=(rep.lost_tokens if rep else 0),
        total_s=time.perf_counter() - t0)
    if trig:
        # first token emitted after the trigger, and (for restart) the time until
        # the regenerated text first exceeds what had already been shown
        after = [t for t in times if t > trig["t"]]
        rec["time_to_resume_s"] = (after[0] - trig["t"]) if after else None
        if cond.restart:
            k = trig["n_emitted"]
            rec["time_to_new_content_s"] = (after[k] - trig["t"]) if len(after) > k else None
        else:
            rec["time_to_new_content_s"] = rec["time_to_resume_s"]
    del s
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--cond", required=True, choices=sorted(CONDITIONS))
    ap.add_argument("--ids", required=True, help="comma-separated prompt ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-mib", type=float, default=0.0)
    ap.add_argument("--ballast-mib", type=float, default=0.0)
    ap.add_argument("--chunk-mib", type=float, default=64.0)
    ap.add_argument("--margin-mib", type=float, default=96.0)
    ap.add_argument("--pressure-at", type=int, default=None, help="override per-item value")
    ap.add_argument("--max-new", type=int, default=None, help="override per-item value")
    args = ap.parse_args()

    wait_until_watched()
    from molt_mlx.footprint import PhaseMarker
    from molt_mlx import tiers
    mark = PhaseMarker(os.environ.get("MOLT_WATCHDOG_CSV"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    progress = open(str(out) + ".progress.jsonl", "a", buffering=1)
    log = lambda d: progress.write(json.dumps(d) + "\n")
    tok = tiers.load_tokenizer()
    want = args.ids.split(",")
    items = {p["id"]: p for p in P.load()}
    cond = CONDITIONS[args.cond]
    env = machine()
    recs = []
    for pid in want:
        it = dict(items[pid])
        if args.pressure_at is not None:
            it["pressure_at"] = args.pressure_at
        if args.max_new is not None:
            it["max_new"] = args.max_new
        log(dict(ev="start", id=pid))
        recs.append(run_prompt(it, cond, args.pair, tok, args, mark, log))
        log(dict(ev="end", id=pid))
    out.write_text(json.dumps(dict(machine=env, runs=recs)))


if __name__ == "__main__":
    main()
