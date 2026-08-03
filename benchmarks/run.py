#!/usr/bin/env python3
"""Molt benchmark driver.

Runs every condition over **the same pressure trace and the same prompts**, then
writes one JSON per run plus a combined ``summary.json`` that ``figures/`` reads.

Typical use::

    # hermetic, no downloads, ~1 minute
    python benchmarks/run.py --ladder synthetic --trace spike_mid_answer

    # the real thing (Qwen2.5 1.5B / int8 / 0.5B), CPU
    python scripts/train_projectors.py --ladder qwen
    python benchmarks/run.py --ladder qwen --trace spike_mid_answer --device cpu

    # multi-tenant QoS experiment (foreground chat + 2 background batches)
    python benchmarks/run.py --ladder qwen --qos

Every reported number is defined in :mod:`molt.metrics` and every claim it backs
is named in the corresponding docstring.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.conditions import (ABLATIONS, CONDITIONS, MAIN_CONDITIONS,
                                   Condition, get_conditions)
from benchmarks.quality import (QualityReport, TaskPrompt, default_suite,
                                judge_nll, score_generation)
from molt.config import (CalibrationConfig, MigrationConfig, MoltConfig,
                         TransplantConfig, get_ladder, resolve_device,
                         resolve_dtype)
from molt.kv_transplant import ProjectorRegistry
from molt.metrics import (MB, EventLog, distinct_n, dump_json, repetition_rate,
                          system_available_mb)
from molt.model_zoo import ModelZoo, assert_ladder_compatible, load_tokenizer
from molt.pressure import (ConstantSource, ProcessHogSource, PressureTrace,
                           TraceReplaySource, builtin_traces, load_trace)
from molt.runtime import GenerationRequest, MoltRuntime, Policy
from molt.scheduler import AppSpec, QoSScheduler, SchedulerConfig

TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pressure_traces")


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------


def build_config(args, ladder) -> MoltConfig:
    return MoltConfig(
        ladder=ladder,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        transplant=TransplantConfig(
            recompute_top_k=args.recompute_top_k,
            projector_dir=args.projector_dir,
            max_carry_tokens=args.max_carry_tokens,
        ),
        calibration=CalibrationConfig(
            mode=args.calibration, blend_steps=args.blend_steps,
            schedule=args.blend_schedule),
        migration=MigrationConfig(
            down_threshold=args.down_threshold, up_threshold=args.up_threshold,
            cooldown_steps=args.cooldown, up_shift_patience=args.up_patience,
            allow_up_shift=not args.no_upshift),
        log_dir=args.out,
    )


def measure_ladder_footprints(zoo: ModelZoo, ladder, verbose: bool = True) -> Dict[str, float]:
    """Load each rung once to learn its true size.

    The pressure traces are expressed as multiples of tier0's *measured*
    footprint, so a "0.55x" event is genuinely below what the top rung needs
    rather than a threshold chosen to make the result come out right.
    """
    out: Dict[str, float] = {}
    for spec in ladder:
        mb = zoo.measure_footprint(spec)
        out[spec.name] = mb
        if verbose:
            print(f"  {spec.name:6s} {spec.label:26s} {mb:8.1f} MiB")
    zoo.evict_all()
    return out


def make_pressure_source(args, trace: PressureTrace):
    if args.no_pressure:
        return ConstantSource(trace.max_budget_mb, trace.duration_s)
    if args.real_hog:
        return ProcessHogSource(trace)
    return TraceReplaySource(trace, time_scale=args.time_scale,
                             virtual_step_s=args.virtual_step)


# --------------------------------------------------------------------------
# single-app benchmark
# --------------------------------------------------------------------------


def run_condition(
    cond: Condition, args, ladder, footprints: Dict[str, float],
    trace: PressureTrace, prompts: List[TaskPrompt],
) -> Dict[str, Any]:
    cfg = cond.apply(build_config(args, ladder))
    device = cfg.torch_device
    dtype = cfg.torch_dtype
    zoo = ModelZoo(device, dtype, verbose=args.verbose, allow_evict=not args.warm_ladder)
    tok = load_tokenizer(ladder.tokenizer_id)
    registry = ProjectorRegistry(cfg.transplant.projector_dir, ladder.name, device)
    log = EventLog()
    rt = MoltRuntime(cfg, zoo, tok, registry, log)

    if args.warm_ladder:
        for spec in ladder:
            zoo.acquire(spec)          # pin every rung; nothing is ever unloaded

    results: List[Dict[str, Any]] = []
    quality = QualityReport()
    per_prompt_series: List[Dict[str, Any]] = []
    transcripts: List[Dict[str, Any]] = []
    kills = 0

    for task in prompts:
        src = make_pressure_source(args, trace)
        src.start()
        budget_fn = src.budget_mb
        pressure_fn = lambda: rt.broker.pressure(src.budget_mb())

        req = GenerationRequest(
            app_id=f"{cond.key}:{task.id}", prompt=task.prompt,
            max_new_tokens=min(task.max_new_tokens, cfg.max_new_tokens),
            reference=task.answer)
        gen = rt.make_generation(
            req, cond.policy, ladder.by_name(cond.start_tier),
            pressure_fn=pressure_fn, budget_fn=budget_fn,
            # --warm-ladder pins every rung, so the budget is meaningless there;
            # that mode measures migration *cost* in isolation, not survival.
            enforce_oom=not args.warm_ladder,
            request_boundary=cond.request_boundary)

        tick = getattr(src, "tick", None)
        res = rt.run(gen, tick=tick if callable(tick) else None)
        src.stop()

        if res.killed:
            kills += 1
        # A "stall" is a token that took far longer than this run's own typical
        # token.  Deriving the threshold from the median (rather than fixing it)
        # keeps the definition fair across rungs of very different speeds.
        thr = args.stall_threshold_ms
        if thr <= 0:
            thr = max(1.0, 4.0 * res.latency.pct_itl_ms(50))
        summ = res.summary(stall_threshold_ms=thr)
        summ["stall_threshold_ms"] = thr
        summ["prompt_id"] = task.id
        summ["text_head"] = res.text[:200]
        results.append(summ)

        qrec: Dict[str, Any] = dict(id=task.id, kind=task.kind,
                                    repetition_rate=repetition_rate(res.token_ids, 3),
                                    distinct_2=distinct_n(res.token_ids, 2))
        qrec.update(score_generation(task, res.text, killed=res.killed))
        quality.add(qrec)
        # judged later, once every rung has been evicted — loading the judge now
        # would double the resident footprint the whole experiment is about
        transcripts.append(dict(prompt_id=task.id, token_ids=list(res.token_ids),
                                killed=res.killed))

        per_prompt_series.append(dict(
            prompt_id=task.id,
            itl_ms=res.latency.itl_ms,
            tiers=res.tier_per_token,
            pressure=[t.pressure for t in res.latency.tokens],
            resident_mb=[t.resident_mb for t in res.latency.tokens],
            t_rel=[t.t_end - res.latency.t_request for t in res.latency.tokens],
            migrations=res.migrations,
            transplants=[t.to_dict() for t in res.transplants],
            killed=res.killed,
        ))
        # release everything this prompt held before the next one starts
        if not args.warm_ladder:
            zoo.evict_all()

    if args.warm_ladder:
        for spec in ladder:
            zoo.release(spec)
    zoo.evict_all()

    agg = aggregate(results)
    return dict(
        condition=cond.key, label=cond.label, description=cond.description,
        policy=cond.policy.value, start_tier=cond.start_tier,
        kills=kills, expect_kills=cond.expect_kills,
        aggregate=agg, quality=quality.summary(), quality_raw=quality.per_prompt,
        transcripts=transcripts,
        per_prompt=results, series=per_prompt_series,
        events=log.to_list(), footprints=footprints,
        config=dict(calibration=asdict(cfg.calibration),
                    transplant=asdict(cfg.transplant),
                    migration=asdict(cfg.migration),
                    warm_ladder=args.warm_ladder),
    )


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def m(key, only_alive=True):
        vals = [r[key] for r in rows
                if key in r and isinstance(r[key], (int, float)) and r[key] == r[key]
                and (not only_alive or not r.get("killed"))]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def mx(key):
        vals = [r[key] for r in rows
                if key in r and isinstance(r[key], (int, float)) and r[key] == r[key]]
        return max(vals) if vals else float("nan")

    return dict(
        n_prompts=len(rows), n_killed=sum(1 for r in rows if r.get("killed")),
        n_finished=sum(1 for r in rows if r.get("finished")),
        mean_ttft_ms=m("ttft_ms"), max_ttft_ms=mx("ttft_ms"),
        mean_itl_ms=m("mean_itl_ms"),
        worst_itl_ms=mx("max_itl_ms"), mean_p99_itl_ms=m("p99_itl_ms"),
        mean_stall_time_ms=m("stall_time_ms"), total_stalls=sum(
            r.get("n_stalls", 0) for r in rows),
        mean_migrations=m("n_migrations", only_alive=False),
        mean_migration_ms=m("migration_ms_total", only_alive=False),
        mean_flops_saving=m("transplant_flops_saving", only_alive=False),
        handoff_jsd=m("mean_handoff_jsd", only_alive=False),
        handoff_jsd_raw=m("mean_handoff_jsd_raw", only_alive=False),
        max_handoff_jsd=mx("max_handoff_jsd"),
        excess_jsd=m("excess_jsd", only_alive=False),
        peak_jsd_at_migration=m("peak_jsd_at_migration", only_alive=False),
        repetition_rate=m("repetition_rate_3", only_alive=False),
        peak_mb=mx("peak_mb"),
        mean_tokens=m("n_tokens", only_alive=False),
        frac_top_tier=m("frac_tokens_on_top_tier", only_alive=False),
    )


def _text_ids(tok, n: int, device) -> torch.Tensor:
    """``n`` tokens of ordinary prose, tiled as needed."""
    from molt.fit_projector import DEFAULT_CALIB_TEXTS

    seed = " ".join(DEFAULT_CALIB_TEXTS[:8])
    ids = tok(seed, return_tensors="pt")["input_ids"][0].tolist()
    while len(ids) < n:
        ids = ids + ids
    return torch.tensor([ids[:n]], dtype=torch.long, device=device)


def run_cost_sweep(args, ladder, device, dtype, lengths=(64, 128, 256, 512, 1024)
                   ) -> Dict[str, Any]:
    """Transplant vs re-prefill, as a function of carried context length.

    Both rungs are kept warm so the comparison isolates the KV work: model
    loading is a cost *both* strategies pay and it would otherwise swamp the
    difference this experiment is about.  This is the direct evidence for the
    **low migration cost** claim, independent of any policy or trace.
    """
    from molt.kv_cache import CacheMeta, MoltCache
    from molt.kv_transplant import KVTransplant, TierRef

    cfg = build_config(args, ladder)
    tp_cfg = cfg.transplant
    zoo = ModelZoo(device, dtype, verbose=False)
    tok = load_tokenizer(ladder.tokenizer_id)
    reg = ProjectorRegistry(tp_cfg.projector_dir, ladder.name, device)
    rows: List[Dict[str, Any]] = []

    for src_name, dst_name in [("tier0", "tier2"), ("tier0", "tier1")]:
        s_spec, d_spec = ladder.by_name(src_name), ladder.by_name(dst_name)
        src, dst = zoo.acquire(s_spec), zoo.acquire(d_spec)
        proj = reg.get(src_name, dst_name)
        tp = KVTransplant(tp_cfg)
        k = tp_cfg.top_k_for(src.geometry.n_layers)
        boundary = src.geometry.n_layers - k
        src_ref = TierRef.from_loaded(src, snapshot_rope=True)

        # Real text, not random ids: the projector is fitted on natural token
        # statistics, and a uniformly random prefix is out of distribution for
        # both the map and the models.  Timing barely cares, but any error the
        # sweep also happens to expose should be the one users would hit.
        for T in lengths:
            ids = _text_ids(tok, T, device)
            trace: Dict[int, torch.Tensor] = {}
            h = (src.adapter.capture_hook(boundary,
                                          lambda x, _b=boundary: trace.__setitem__(_b, x))
                 if k > 0 else None)
            with torch.no_grad():
                out = src.model(ids, use_cache=True)
            if h is not None:
                h.remove()
            meta = CacheMeta(src_name, src.geometry, ids[0].tolist(), sorted(trace), T)
            cache = MoltCache.from_hf(out.past_key_values, meta, trace)
            del out

            tp.transplant(cache, src_ref, dst, proj, device)      # warm
            tp.reprefill(cache.meta.token_ids, dst, device)
            t_ms = min(tp.transplant(cache, src_ref, dst, proj, device)[1].wall_ms
                       for _ in range(args.sweep_repeats))
            _, rt_rep = tp.transplant(cache, src_ref, dst, proj, device)
            r_ms = min(tp.reprefill(cache.meta.token_ids, dst, device)[1].wall_ms
                       for _ in range(args.sweep_repeats))
            rows.append(dict(
                route=f"{src_name}->{dst_name}", tokens=T,
                transplant_ms=t_ms, reprefill_ms=r_ms,
                speedup=r_ms / max(1e-6, t_ms),
                transplant_flops=rt_rep.flops_total,
                reprefill_flops=rt_rep.flops_reprefill_equiv,
                flops_saving=rt_rep.flops_saving,
                project_ms=rt_rep.project_ms, recompute_ms=rt_rep.recompute_ms,
                rope_ms=rt_rep.rope_ms,
                mb_in=rt_rep.bytes_in / MB, mb_out=rt_rep.bytes_out / MB,
                recomputed_layers=rt_rep.n_recomputed_layers,
                projected_layers=rt_rep.n_projected_layers))
            print(f"  {src_name}->{dst_name} T={T:5d}: transplant {t_ms:8.1f} ms  "
                  f"reprefill {r_ms:8.1f} ms  speedup {r_ms/max(1e-6,t_ms):5.2f}x  "
                  f"FLOPs saved {rt_rep.flops_saving*100:5.1f}%")
            cache.free()
        zoo.release(s_spec); zoo.release(d_spec)
    zoo.evict_all()
    return dict(rows=rows, recompute_top_k=tp_cfg.recompute_top_k)


def run_judge_pass(out: Dict[str, Any], args, ladder, device, dtype,
                   prompts: List[TaskPrompt]) -> None:
    """Score every condition's transcripts with the top rung, *after* all runs.

    Deferring the judge matters: loading a second copy of the largest model
    while the experiment is running would inflate exactly the footprint the
    experiment measures.  Every condition is judged by the same model on the
    same prompts, so the comparison is apples-to-apples.
    """
    print("\n--- judging (top rung, all rungs already evicted) ---")
    by_id = {p.id: p for p in prompts}
    jz = ModelZoo(device, dtype)
    judge = jz.acquire(ladder.top).model
    judge_tok = load_tokenizer(ladder.tokenizer_id)
    try:
        for key, rec in out["conditions"].items():
            q = QualityReport()
            raw = {r["id"]: r for r in rec.get("quality_raw", [])}
            for tr in rec.get("transcripts", []):
                r = dict(raw.get(tr["prompt_id"], dict(id=tr["prompt_id"])))
                task = by_id.get(tr["prompt_id"])
                if task is not None and tr["token_ids"] and not tr["killed"]:
                    try:
                        r.update(judge_nll(judge, judge_tok, task.prompt,
                                           tr["token_ids"], device))
                    except Exception as exc:  # pragma: no cover
                        r["judge_error"] = repr(exc)
                q.add(r)
            rec["quality"] = q.summary()
            rec["quality_raw"] = q.per_prompt
            q = rec["quality"]
            print(f"  {rec['label'][:28]:28s} ppl={q['judge_ppl']:7.2f} "
                  f"agree={q['judge_agreement']:.3f} acc={q['accuracy']:.2f} "
                  f"needle={q['needle_recall']:.2f} rep={q['repetition_rate']:.3f}")
    finally:
        jz.evict_all()


# --------------------------------------------------------------------------
# multi-tenant QoS benchmark
# --------------------------------------------------------------------------


def run_qos(args, ladder, footprints: Dict[str, float], trace: PressureTrace,
            prompts: List[TaskPrompt]) -> Dict[str, Any]:
    """One foreground conversation plus two background batch jobs.

    The result that matters is ``kills == 0`` together with
    ``max_overshoot_mb <= 0``: nobody died and nobody exceeded the budget.
    """
    cfg = build_config(args, ladder)
    device = cfg.torch_device
    zoo = ModelZoo(device, cfg.torch_dtype, verbose=args.verbose)
    # teach the zoo every rung's real size *and* cache geometry before the
    # scheduler has to make admission decisions against them
    measure_ladder_footprints(zoo, ladder, verbose=False)
    tok = load_tokenizer(ladder.tokenizer_id)
    registry = ProjectorRegistry(cfg.transplant.projector_dir, ladder.name, device)
    rt = MoltRuntime(cfg, zoo, tok, registry, EventLog())
    src = make_pressure_source(args, trace)
    sched = QoSScheduler(cfg, rt, src, SchedulerConfig(
        safety_margin=args.safety_margin,
        kv_mb_per_token_hint=args.kv_mb_per_token,
        background_step_divisor=2))

    fg, bg1, bg2 = prompts[0], prompts[min(1, len(prompts) - 1)], prompts[-1]
    sched.submit(AppSpec(GenerationRequest("fg-chat", fg.prompt,
                                           max_new_tokens=args.max_new_tokens,
                                           priority=0, kind="chat"),
                         Policy.MOLT, "tier0", 0))
    sched.submit(AppSpec(GenerationRequest("bg-batch-1", bg1.prompt,
                                           max_new_tokens=args.max_new_tokens,
                                           priority=1, kind="batch"),
                         Policy.MOLT, "tier0", 1))
    sched.submit(AppSpec(GenerationRequest("bg-batch-2", bg2.prompt,
                                           max_new_tokens=args.max_new_tokens,
                                           priority=2, kind="batch"),
                         Policy.MOLT, "tier0", 2))
    report = sched.run()
    zoo.evict_all()
    return dict(summary=report.summary(),
                budget_series=report.budget_series,
                actions=report.actions,
                admissions=report.admissions,
                events=rt.log.to_list(),
                footprints=footprints)


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Molt benchmark: elastic inference under memory pressure")
    p.add_argument("--ladder", default="synthetic", choices=["synthetic", "qwen", "qwen-3b"])
    p.add_argument("--trace", default="spike_mid_answer",
                   help="builtin trace name or path to a JSON trace")
    p.add_argument("--conditions", default=",".join(MAIN_CONDITIONS),
                   help="comma-separated condition keys, or 'all'")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--out", default="benchmarks/results")
    p.add_argument("--projector-dir", default="artifacts/projectors")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--prompt-chars", type=int, default=1400,
                   help="filler length; longer prompts make condition C's re-prefill "
                        "cost realistic instead of negligible")
    p.add_argument("--n-prompts", type=int, default=5)

    p.add_argument("--recompute-top-k", type=int, default=6)
    p.add_argument("--max-carry-tokens", type=int, default=None)
    p.add_argument("--calibration", default="dual", choices=["dual", "frozen_ref", "none"])
    p.add_argument("--blend-steps", type=int, default=6)
    p.add_argument("--blend-schedule", default="cosine", choices=["cosine", "linear", "exp"])

    p.add_argument("--down-threshold", type=float, default=0.85)
    p.add_argument("--up-threshold", type=float, default=0.55)
    p.add_argument("--cooldown", type=int, default=8)
    p.add_argument("--up-patience", type=int, default=16)
    p.add_argument("--no-upshift", action="store_true")

    p.add_argument("--time-scale", type=float, default=1.0,
                   help=">1 stretches the trace; use on slow CPUs so the spike lands "
                        "mid-answer rather than after it")
    p.add_argument("--virtual-step", type=float, default=None,
                   help="seconds of virtual time per decode step (deterministic mode)")
    p.add_argument("--real-hog", action="store_true",
                   help="use a real child process that allocates memory")
    p.add_argument("--no-pressure", action="store_true", help="control run, flat budget")
    p.add_argument("--warm-ladder", action="store_true",
                   help="pin every rung in memory: isolates KV-migration cost from "
                        "model-load cost")
    p.add_argument("--stall-threshold-ms", type=float, default=0.0,
                   help="0 = derive from the median inter-token latency")

    p.add_argument("--cost-sweep", action="store_true",
                   help="measure transplant vs re-prefill across context lengths")
    p.add_argument("--sweep-lengths", default="64,128,256,512",
                   help="comma-separated context lengths for --cost-sweep")
    p.add_argument("--sweep-repeats", type=int, default=3)
    p.add_argument("--qos", action="store_true", help="also run the multi-tenant experiment")
    p.add_argument("--qos-only", action="store_true")
    p.add_argument("--safety-margin", type=float, default=0.10)
    p.add_argument("--kv-mb-per-token", type=float, default=0.02)

    p.add_argument("--judge", action="store_true", default=True,
                   help="score generations with the top rung as judge (default on)")
    p.add_argument("--no-judge", dest="judge", action="store_false")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    ladder = get_ladder(args.ladder)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    os.makedirs(args.out, exist_ok=True)

    print("== Molt benchmark ==")
    print(f"ladder     : {ladder.name}  ({len(ladder)} rungs)")
    print(f"device     : {device}  dtype={dtype}")
    print(f"system     : {system_available_mb():.0f} MiB available")

    print("measuring rung footprints…")
    probe_zoo = ModelZoo(device, dtype, verbose=False)
    footprints = measure_ladder_footprints(probe_zoo, ladder)
    assert_ladder_compatible(probe_zoo, ladder)
    probe_zoo.evict_all()
    top_mb = footprints[ladder.top.name]

    trace = load_trace(args.trace, top_tier_mb=top_mb)
    if args.time_scale != 1.0:
        trace = trace.time_scaled(args.time_scale)
    print(f"trace      : {trace.name} — {trace.description}")
    print(f"             budget {trace.min_budget_mb:.0f}..{trace.max_budget_mb:.0f} MiB "
          f"over {trace.duration_s:.0f}s (tier0 needs {top_mb:.0f} MiB)")

    prompts = default_suite(args.prompt_chars)[:args.n_prompts]

    out: Dict[str, Any] = dict(
        meta=dict(ladder=ladder.name, device=str(device), dtype=str(dtype),
                  trace=trace.to_dict(), footprints=footprints,
                  args=vars(args), timestamp=time.time()),
        conditions={},
    )

    if not args.qos_only:
        keys = (MAIN_CONDITIONS + ABLATIONS) if args.conditions == "all" \
            else [k.strip() for k in args.conditions.split(",") if k.strip()]
        for cond in get_conditions(keys):
            print(f"\n--- {cond.label} ---")
            t0 = time.perf_counter()
            rec = run_condition(cond, args, ladder, footprints, trace, prompts)
            rec["wall_s"] = time.perf_counter() - t0
            out["conditions"][cond.key] = rec
            a = rec["aggregate"]
            print(f"  kills={rec['kills']}  worst ITL={a['worst_itl_ms']:.0f}ms  "
                  f"mean ITL={a['mean_itl_ms']:.0f}ms  migrations={a['mean_migrations']:.1f}  "
                  f"mig cost={a['mean_migration_ms']:.0f}ms  "
                  f"handoff JSD={a['handoff_jsd']:.4f}  peak={a['peak_mb']:.0f}MiB  "
                  f"({rec['wall_s']:.0f}s)")

        if args.judge:
            run_judge_pass(out, args, ladder, device, dtype, prompts)
        for key, rec in out["conditions"].items():
            dump_json(rec, os.path.join(args.out, f"condition_{key}.json"))

    if args.cost_sweep:
        print("\n--- migration cost sweep (warm ladder, KV work only) ---")
        lens = tuple(int(x) for x in args.sweep_lengths.split(",") if x.strip())
        out["cost_sweep"] = run_cost_sweep(args, ladder, device, dtype, lens)
        dump_json(out["cost_sweep"], os.path.join(args.out, "cost_sweep.json"))

    if args.qos or args.qos_only:
        print("\n--- QoS: foreground chat + 2 background batches ---")
        qos = run_qos(args, ladder, footprints, trace, prompts)
        out["qos"] = qos
        s = qos["summary"]
        print(f"  kills={s['kills']}  max overshoot={s['max_overshoot_mb']:.1f} MiB  "
              f"demotions={s['n_demotions']}  pauses={s['n_pauses']}  "
              f"deferred admissions={s['n_deferred_admissions']}")
        dump_json(qos, os.path.join(args.out, "qos.json"))

    dump_json(out, os.path.join(args.out, "summary.json"))
    print(f"\nwrote {os.path.join(args.out, 'summary.json')}")
    print_table(out)
    return 0


def print_table(out: Dict[str, Any]) -> None:
    conds = out.get("conditions") or {}
    if not conds:
        return
    cols = [("cond", 24, "s"), ("kills", 6, "d"), ("worstITL", 10, ".0f"),
            ("meanITL", 9, ".1f"), ("migr", 6, ".1f"), ("migMs", 9, ".0f"),
            ("agree", 7, ".3f"), ("acc", 6, ".2f"), ("hoJSD", 9, ".4f"),
            ("peakMB", 9, ".0f")]
    hdr = " ".join(f"{n:>{w}s}" if i else f"{n:<{w}s}" for i, (n, w, _) in enumerate(cols))
    print("\n" + hdr)
    print("-" * len(hdr))
    for rec in conds.values():
        a, q = rec["aggregate"], rec["quality"]
        vals = [rec["label"][:24], rec["kills"], a["worst_itl_ms"], a["mean_itl_ms"],
                a["mean_migrations"], a["mean_migration_ms"], q.get("judge_agreement", float("nan")),
                q["accuracy"], a.get("handoff_jsd", float("nan")), a["peak_mb"]]
        cells = []
        for i, ((_, w, f), v) in enumerate(zip(cols, vals)):
            if i == 0:
                cells.append(f"{v:<{w}s}")
            elif v != v:                       # NaN
                cells.append(f"{'--':>{w}s}")
            else:
                cells.append(f"{v:>{w}{f}}")
        print(" ".join(cells))


if __name__ == "__main__":
    raise SystemExit(main())
