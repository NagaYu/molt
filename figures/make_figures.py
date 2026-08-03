#!/usr/bin/env python3
"""Render the benchmark's figures from ``benchmarks/results/summary.json``.

    python figures/make_figures.py --results benchmarks/results --out figures

Every figure states the claim it is evidence for in its own title/caption, and
every one of them is a rendering of measured data — nothing here is drawn by
hand or idealised.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# A restrained, colour-blind-safe palette; tiers are ordered light -> dark so the
# "which rung am I on" band reads as a gradient rather than a categorical mess.
COND_COLORS = {
    "A": "#c1443c",   # static-large: the one that dies
    "B": "#8a8f98",   # static-small: the flat floor
    "C": "#d98c2b",   # restart: the one that stalls
    "D": "#2f6f9f",   # molt
    "D-nocal": "#7fb3d5",
    "D-noproj": "#a9cce3",
    "D-norecompute": "#5499c7",
    "D-reqbound": "#6c3483",
}
TIER_COLORS = {"tier0": "#dce8f2", "tier1": "#f4e2c4", "tier2": "#e8dcea"}
GRID = dict(alpha=0.25, linewidth=0.6)


def _style(ax, title=None, xlabel=None, ylabel=None, legend=False):
    if title:
        ax.set_title(title, fontsize=11, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    if legend:
        ax.legend(fontsize=8, frameon=False)


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def _series(rec, prompt_id=None):
    ser = rec.get("series") or []
    if not ser:
        return None
    if prompt_id is not None:
        for s in ser:
            if s["prompt_id"] == prompt_id:
                return s
    # the longest surviving series makes the clearest picture
    alive = [s for s in ser if not s.get("killed")] or ser
    return max(alive, key=lambda s: len(s["itl_ms"]))


# --------------------------------------------------------------------------
# 1. the hero figure
# --------------------------------------------------------------------------


def fig_hero(data: Dict[str, Any], out_dir: str) -> Optional[str]:
    """Per-token latency through a pressure event, all four conditions.

    Claim: **no-stall.**  Condition C shows a tall spike where it re-reads the
    prompt; condition D crosses the same event with a bump a fraction of the
    size, and unlike B it spends the uncontended part of the answer on the
    better model.
    """
    conds = data.get("conditions", {})
    picks = [k for k in ("A", "B", "C", "D") if k in conds]
    if not picks:
        return None

    # use a prompt that every surviving condition ran
    common = None
    for k in picks:
        ids = {s["prompt_id"] for s in conds[k].get("series", [])}
        common = ids if common is None else (common & ids)
    prompt_id = sorted(common)[0] if common else None

    fig, (ax, axt) = plt.subplots(
        2, 1, figsize=(10, 5.6), sharex=True,
        gridspec_kw=dict(height_ratios=[3.2, 1.0], hspace=0.12))

    tier_order = ["tier0", "tier1", "tier2"]
    for key in picks:
        rec, s = conds[key], _series(conds[key], prompt_id)
        if not s:
            continue
        y = s["itl_ms"]
        x = list(range(len(y)))
        ax.plot(x, y, lw=1.6, color=COND_COLORS.get(key, "#444"),
                label=f"{rec['label']}", zorder=3)
        if s.get("killed"):
            ax.scatter([len(y) - 1], [y[-1]], marker="X", s=110, zorder=5,
                       color=COND_COLORS.get(key), edgecolor="white", linewidth=1.2)
            ax.annotate("reclaimed by the OS", (len(y) - 1, y[-1]),
                        textcoords="offset points", xytext=(8, 10), fontsize=8,
                        color=COND_COLORS.get(key))
        for m in s.get("migrations", []):
            ax.axvline(m["token_index"], color=COND_COLORS.get(key), alpha=0.28,
                       lw=1.0, ls="--", zorder=1)

    ax.set_yscale("log")
    _style(ax, ylabel="inter-token latency (ms, log)", legend=True,
           title="A pressure event mid-answer — Molt crosses it without stopping")

    # tier band for condition D
    sD = _series(conds.get("D", {}), prompt_id) if "D" in conds else None
    if sD:
        tiers = sD["tiers"]
        for i, t in enumerate(tiers):
            axt.axvspan(i - 0.5, i + 0.5, color=TIER_COLORS.get(t, "#eee"), lw=0)
        press = sD.get("pressure") or []
        if press:
            axt.plot(range(len(press)), np.clip(press, 0, 2), color="#333", lw=1.2,
                     label="memory pressure (usage / budget)")
            axt.axhline(1.0, color="#c1443c", lw=0.9, ls=":", label="budget")
        handles = [mpatches.Patch(color=TIER_COLORS[t], label=f"Molt on {t}")
                   for t in tier_order if t in set(tiers)]
        # Keep this legend *inside* its own panel: anchoring it above spills it
        # into the latency plot and covers the very spike the figure exists for.
        axt.legend(handles=handles + axt.get_legend_handles_labels()[0],
                   fontsize=6.5, frameon=True, framealpha=0.92, edgecolor="none",
                   ncol=2, loc="upper right", borderpad=0.4)
        axt.set_ylim(0, 2.6)
    _style(axt, xlabel="generated token index", ylabel="pressure")
    return _save(fig, out_dir, "hero_latency_timeline.png")


# --------------------------------------------------------------------------
# 2. survival vs quality
# --------------------------------------------------------------------------


def fig_survival_quality(data: Dict[str, Any], out_dir: str) -> Optional[str]:
    """The trade-off the four conditions span.

    Claim: **zero-kill + quality.**  A dies; B survives at its floor; C and D
    both survive, but D pays a fraction of C's stall to get there.
    """
    conds = data.get("conditions", {})
    keys = [k for k in ("A", "B", "C", "D") if k in conds]
    if not keys:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3))
    labels = [conds[k]["label"].split("·")[-1].strip() for k in keys]
    colors = [COND_COLORS.get(k, "#444") for k in keys]

    def bar(ax, vals, title, ylabel, fmt="{:.0f}", log=False):
        vv = [0 if (v is None or v != v) else v for v in vals]
        ax.bar(range(len(vv)), vv, color=colors, width=0.62)
        for i, (v, raw) in enumerate(zip(vv, vals)):
            txt = "n/a" if (raw is None or raw != raw) else fmt.format(raw)
            ax.text(i, v, txt, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(vv)))
        ax.set_xticklabels(labels, fontsize=7.5, rotation=18, ha="right")
        if log:
            ax.set_yscale("log")
        _style(ax, title=title, ylabel=ylabel)

    bar(axes[0], [conds[k]["kills"] for k in keys],
        "Forced terminations", "count", "{:.0f}")
    bar(axes[1], [conds[k]["aggregate"]["worst_itl_ms"] for k in keys],
        "Worst inter-token latency", "ms", "{:.0f}", log=True)
    bar(axes[2], [conds[k]["quality"]["judge_agreement"] for k in keys],
        "Judge agreement (higher better)", "fraction", "{:.3f}")
    bar(axes[3], [conds[k]["quality"]["accuracy"] for k in keys],
        "Reference-task accuracy\n(a killed run scores 0)", "fraction", "{:.2f}")
    fig.suptitle("Survive, and stay good: what each strategy costs",
                 fontsize=11, x=0.09, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out_dir, "survival_vs_quality.png")


# --------------------------------------------------------------------------
# 3. why request-boundary switching is not enough
# --------------------------------------------------------------------------


def fig_request_boundary(data: Dict[str, Any], out_dir: str) -> Optional[str]:
    """The single figure the README leads with for this argument.

    Claim: elasticity that can only act *between* requests cannot serve a spike
    that arrives *inside* one.  Left: the timeline, with the window in which a
    request-boundary policy is powerless shaded.  Right: what that costs.
    """
    conds = data.get("conditions", {})
    if "D" not in conds:
        return None
    d, rb = conds["D"], conds.get("D-reqbound")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 3.9),
                                  gridspec_kw=dict(width_ratios=[2.1, 1.0]))

    sD = _series(d)
    sR = _series(rb) if rb else None
    n = len(sD["itl_ms"]) if sD else 0

    press = (sD or {}).get("pressure") or []
    over = [i for i, p in enumerate(press) if p >= 1.0]
    if over:
        ax.axvspan(min(over), max(over), color="#f3d9d7", lw=0,
                   label="pressure above budget")
    if sD:
        ax.plot(sD["itl_ms"], color=COND_COLORS["D"], lw=1.7,
                label="Molt — may switch between any two tokens")
        for m in sD.get("migrations", []):
            ax.annotate("", xy=(m["token_index"], max(sD["itl_ms"]) * 0.55),
                        xytext=(m["token_index"], max(sD["itl_ms"]) * 0.95),
                        arrowprops=dict(arrowstyle="->", color=COND_COLORS["D"], lw=1.2))
            ax.text(m["token_index"], max(sD["itl_ms"]) * 1.0,
                    f"{m['from_tier']}→{m['to_tier']}", fontsize=7, ha="center",
                    color=COND_COLORS["D"])
    if sR:
        ax.plot(sR["itl_ms"], color=COND_COLORS["D-reqbound"], lw=1.7, ls="--",
                label="request-boundary policy — cannot act until the answer ends")
        if sR.get("killed"):
            ax.scatter([len(sR["itl_ms"]) - 1], [sR["itl_ms"][-1]], marker="X",
                       s=110, color=COND_COLORS["D-reqbound"], zorder=5,
                       edgecolor="white")
    ax.set_yscale("log")
    _style(ax, title="The spike arrives mid-answer; the request boundary is far away",
           xlabel="generated token index", ylabel="inter-token latency (ms, log)",
           legend=True)
    if n:
        ax.annotate("request ends here →", (n - 1, min(sD["itl_ms"])),
                    textcoords="offset points", xytext=(-105, -2), fontsize=7.5,
                    color="#555")

    rows = [("Molt\n(mid-stream)", d)] + ([("request-boundary", rb)] if rb else [])
    metrics = [("forced terminations", lambda r: r["kills"]),
               ("tokens on the top rung", lambda r: r["aggregate"]["frac_top_tier"]),
               ("worst ITL (s)", lambda r: r["aggregate"]["worst_itl_ms"] / 1000.0)]
    w = 0.36
    for j, (mname, fn) in enumerate(metrics):
        for i, (label, rec) in enumerate(rows):
            v = fn(rec)
            v = 0 if (v is None or v != v) else v
            ax2.bar(j + (i - 0.5) * w, v, width=w,
                    color=COND_COLORS["D"] if i == 0 else COND_COLORS["D-reqbound"],
                    label=label if j == 0 else None)
            ax2.text(j + (i - 0.5) * w, v, f"{v:.2f}", ha="center", va="bottom",
                     fontsize=7)
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels([m[0] for m in metrics], fontsize=7.5, rotation=14, ha="right")
    _style(ax2, title="What the boundary costs", legend=True)
    fig.tight_layout()
    return _save(fig, out_dir, "request_boundary_gap.png")


# --------------------------------------------------------------------------
# 4. migration cost
# --------------------------------------------------------------------------


def fig_migration_cost(data: Dict[str, Any], out_dir: str) -> Optional[str]:
    """Transplant vs re-prefill as context grows.

    Claim: **low migration cost**, and an advantage that widens with the length
    of the conversation being carried.
    """
    sweep = data.get("cost_sweep")
    if not sweep or not sweep.get("rows"):
        return None
    rows = sweep["rows"]
    routes = sorted({r["route"] for r in rows})
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.6))
    marks = ["o", "s", "^", "D"]
    for i, route in enumerate(routes):
        rs = sorted([r for r in rows if r["route"] == route], key=lambda r: r["tokens"])
        x = [r["tokens"] for r in rs]
        ax.plot(x, [r["reprefill_ms"] for r in rs], marker=marks[i % 4], lw=1.6,
                color=COND_COLORS["C"], ls="--", label=f"re-prefill  {route}")
        ax.plot(x, [r["transplant_ms"] for r in rs], marker=marks[i % 4], lw=1.6,
                color=COND_COLORS["D"], label=f"transplant  {route}")
        ax2.plot(x, [r["speedup"] for r in rs], marker=marks[i % 4], lw=1.6,
                 color=COND_COLORS["D"], label=route)
        ax3.plot(x, [r["flops_saving"] * 100 for r in rs], marker=marks[i % 4], lw=1.6,
                 color="#2e7d5b", label=route)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _style(ax, title="Cost of one migration", xlabel="tokens carried across",
           ylabel="wall time (ms, log)", legend=True)
    ax2.axhline(1.0, color="#999", lw=0.9, ls=":")
    ax2.set_xscale("log", base=2)
    _style(ax2, title="Speed-up over re-prefill", xlabel="tokens carried across",
           ylabel="x faster", legend=True)
    ax3.set_xscale("log", base=2)
    _style(ax3, title="FLOPs avoided", xlabel="tokens carried across",
           ylabel="% of a re-prefill", legend=True)
    fig.suptitle(f"Carrying the cache beats re-reading the prompt "
                 f"(top-k recompute = {sweep.get('recompute_top_k')} layers)",
                 fontsize=11, x=0.07, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, out_dir, "migration_cost.png")


# --------------------------------------------------------------------------
# 5. calibration
# --------------------------------------------------------------------------


def fig_calibration(data: Dict[str, Any], out_dir: str) -> Optional[str]:
    """Distribution continuity at the seam, with and without blending.

    Claim: **continuity.**  ``excess JSD`` is the step-to-step divergence around
    the switch minus the generation's own steady-state drift.
    """
    conds = data.get("conditions", {})
    if "D" not in conds or "D-nocal" not in conds:
        return None
    keys = ["D", "D-nocal"]
    names = ["Molt (calibrated)", "Molt (no calibration)"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.6),
                                  gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    w = 0.34
    metrics = [("handoff JSD (mean)", "handoff_jsd"),
               ("handoff JSD (worst)", "max_handoff_jsd"),
               ("3-gram repetition", "repetition_rate")]
    for j, (mname, key) in enumerate(metrics):
        for i, k in enumerate(keys):
            v = conds[k]["aggregate"].get(key, float("nan"))
            v = 0 if (v is None or v != v) else v
            ax.bar(j + (i - 0.5) * w, v, width=w,
                   color=COND_COLORS["D"] if i == 0 else "#bbbbbb",
                   label=names[i] if j == 0 else None)
            ax.text(j + (i - 0.5) * w, v, f"{v:.4f}", ha="center", va="bottom",
                    fontsize=7)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[0] for m in metrics], fontsize=8, rotation=12, ha="right")
    _style(ax, title="Continuity across the seam — JSD between what the outgoing model\n"
                     "would have said and what the incoming model did say (lower is smoother)",
           legend=True)

    for i, k in enumerate(keys):
        s = _series(conds[k])
        if not s:
            continue
        ax2.plot(s["itl_ms"], lw=1.4,
                 color=COND_COLORS["D"] if i == 0 else "#bbbbbb", label=names[i])
        for m in s.get("migrations", []):
            ax2.axvline(m["token_index"], color="#c1443c", alpha=0.35, lw=1.0, ls="--")
    ax2.set_yscale("log")
    _style(ax2, title="Both arms migrate at the same point",
           xlabel="token index", ylabel="ITL (ms, log)", legend=True)
    fig.tight_layout()
    return _save(fig, out_dir, "calibration_ablation.png")


# --------------------------------------------------------------------------
# 6. QoS
# --------------------------------------------------------------------------


def fig_qos(data: Dict[str, Any], out_dir: str) -> Optional[str]:
    """Three tenants, one moving budget, nobody killed.

    Claim: **zero-kill.**  The usage line never crosses the budget line, and
    every action taken to keep it there is annotated.
    """
    qos = data.get("qos")
    if not qos or not qos.get("budget_series"):
        return None
    ser = qos["budget_series"]
    t = [r[0] for r in ser]
    budget = [r[1] for r in ser]
    usage = [r[3] for r in ser]
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.plot(t, budget, color="#c1443c", lw=1.8, label="memory budget (from the trace)")
    ax.plot(t, usage, color=COND_COLORS["D"], lw=1.6, label="Molt footprint (tracked)")
    ax.fill_between(t, usage, budget, where=[u <= b for u, b in zip(usage, budget)],
                    color="#dbe9f4", alpha=0.7, lw=0, label="headroom")

    style = {"demote": ("v", "#d98c2b"), "promote": ("^", "#2e7d5b"),
             "pause": ("s", "#8a8f98"), "resume": ("o", "#2f6f9f"),
             "shed_context": ("x", "#7d3c98")}
    seen = set()
    for a in qos.get("actions", []):
        m, c = style.get(a["action"], ("o", "#555"))
        i = min(len(t) - 1, max(0, a["round"] - 1))
        ax.scatter([t[i]], [usage[i]], marker=m, s=42, color=c, zorder=5,
                   label=a["action"] if a["action"] not in seen else None)
        seen.add(a["action"])
    s = qos["summary"]
    ax.text(0.99, 0.04,
            f"forced terminations: {s['kills']}    "
            f"max overshoot: {s['max_overshoot_mb']:.1f} MiB",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#eef6ee", ec="#2e7d5b", lw=0.8))
    _style(ax, title="Foreground chat + two background batches under a moving budget",
           xlabel="time (s, trace clock)", ylabel="MiB", legend=True)
    fig.tight_layout()
    return _save(fig, out_dir, "qos_timeline.png")


# --------------------------------------------------------------------------
# 7. ablations
# --------------------------------------------------------------------------


def fig_ablations(data: Dict[str, Any], out_dir: str) -> Optional[str]:
    """Each mechanism, removed one at a time.

    Claim: every core is load-bearing, or the table says plainly that it is not.
    """
    conds = data.get("conditions", {})
    keys = [k for k in ("D", "D-nocal", "D-noproj", "D-norecompute", "D-reqbound")
            if k in conds]
    if len(keys) < 2:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
    labels = [conds[k]["label"].split("·")[-1].strip() for k in keys]
    colors = [COND_COLORS.get(k, "#666") for k in keys]

    def bar(ax, vals, title, ylabel, fmt="{:.3f}"):
        vv = [0 if (v is None or v != v) else v for v in vals]
        ax.bar(range(len(vv)), vv, color=colors, width=0.62)
        for i, (v, raw) in enumerate(zip(vv, vals)):
            ax.text(i, v, "n/a" if (raw is None or raw != raw) else fmt.format(raw),
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(vv)))
        ax.set_xticklabels(labels, fontsize=7, rotation=22, ha="right")
        _style(ax, title=title, ylabel=ylabel)

    bar(axes[0], [conds[k]["aggregate"]["handoff_jsd"] for k in keys],
        "Handoff JSD (lower = smoother)", "nats", "{:.4f}")
    bar(axes[1], [conds[k]["quality"]["judge_agreement"] for k in keys],
        "Judge agreement (higher better)", "fraction", "{:.3f}")
    bar(axes[2], [conds[k]["aggregate"]["mean_migration_ms"] for k in keys],
        "Mean migration cost", "ms", "{:.0f}")
    fig.suptitle("Ablations: what each mechanism contributes", fontsize=11,
                 x=0.07, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out_dir, "ablations.png")


# --------------------------------------------------------------------------
# markdown table for the README
# --------------------------------------------------------------------------


def write_table(data: Dict[str, Any], out_dir: str) -> str:
    conds = data.get("conditions", {})
    lines = ["| condition | kills | worst ITL (ms) | mean ITL (ms) | migrations | "
             "migration cost (ms) | handoff JSD | judge agree | accuracy | needle | peak MiB |",
             "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]

    def f(v, spec=".2f"):
        if v is None or v != v:
            return "—"
        return format(v, spec)

    for key, rec in conds.items():
        a, q = rec["aggregate"], rec["quality"]
        lines.append(
            f"| {rec['label']} | {rec['kills']} | {f(a['worst_itl_ms'], '.0f')} | "
            f"{f(a['mean_itl_ms'], '.0f')} | {f(a['mean_migrations'], '.1f')} | "
            f"{f(a['mean_migration_ms'], '.0f')} | {f(a.get('handoff_jsd'), '.4f')} | "
            f"{f(q.get('judge_agreement'), '.3f')} | "
            f"{f(q['accuracy'], '.2f')} | {f(q['needle_recall'], '.2f')} | "
            f"{f(a['peak_mb'], '.0f')} |")
    sweep = data.get("cost_sweep")
    if sweep:
        lines += ["", "| route | tokens | transplant (ms) | re-prefill (ms) | speed-up | "
                  "FLOPs saved |", "|---|--:|--:|--:|--:|--:|"]
        for r in sweep["rows"]:
            lines.append(f"| {r['route']} | {r['tokens']} | {r['transplant_ms']:.1f} | "
                         f"{r['reprefill_ms']:.1f} | {r['speedup']:.2f}x | "
                         f"{r['flops_saving']*100:.0f}% |")
    qos = data.get("qos")
    if qos:
        s = qos["summary"]
        lines += ["", "| QoS (3 tenants) | value |", "|---|--:|",
                  f"| forced terminations | {s['kills']} |",
                  f"| max budget overshoot | {s['max_overshoot_mb']:.1f} MiB |",
                  f"| demotions / promotions | {s['n_demotions']} / {s['n_promotions']} |",
                  f"| pauses / resumes | {s['n_pauses']} / {s['n_resumes']} |",
                  f"| context sheds | {s.get('n_context_sheds', 0)} |",
                  f"| deferred admissions | {s['n_deferred_admissions']} |"]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results_table.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {path}")
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="benchmarks/results")
    p.add_argument("--out", default="figures")
    args = p.parse_args(argv)

    path = os.path.join(args.results, "summary.json")
    if not os.path.exists(path):
        print(f"no results at {path}; run benchmarks/run.py first", file=sys.stderr)
        return 1
    with open(path) as fh:
        data = json.load(fh)

    print("rendering figures…")
    for fn in (fig_hero, fig_survival_quality, fig_request_boundary,
               fig_migration_cost, fig_calibration, fig_qos, fig_ablations):
        try:
            if fn(data, args.out) is None:
                print(f"  skipped {fn.__name__} (missing data)")
        except Exception as exc:
            print(f"  FAILED {fn.__name__}: {exc!r}", file=sys.stderr)
    write_table(data, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
