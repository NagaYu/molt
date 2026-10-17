"""Every figure, table and in-text number of the paper, from bench/results.

    python -m bench.analyze && python figures/make_paper_figures.py

Writes figures/paper/*.pdf|png, paper/generated/{results.tex,tables.tex} and
bench/results/macros.json (the same key → value map RESULTS.md is filled from).
A value that was not measured is simply absent: LaTeX renders it as PENDING.
Figure numbers come from LaTeX (\\cref), never from this script.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("MOLT_RESULTS", ROOT / "bench" / "results"))
FIG = Path(os.environ.get("MOLT_FIG_DIR", ROOT / "figures" / "paper"))
GEN = Path(os.environ.get("MOLT_GEN_DIR", ROOT / "paper" / "generated"))
FIG.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)

S = json.loads((RES / "summary.json").read_text())
M: dict = {}                                        # macro key -> string

PAIR_LABEL = {"3b-1.5b": "3B→1.5B", "1.5b-0.5b": "1.5B→0.5B"}
COND_LABEL = {"big": "big (no pressure)", "static_big": "big, ignores pressure", "A": "A restart",
              "A2": "A′ re-prefill", "B": "B always small", "C": "C ridge mid-decode",
              "D": "D molt", "C_ordered": "C + ordered", "D_colocated": "D, colocated",
              "D_del_only": "D, del-only", "D_clear_no_wait": "D, clear-no-wait",
              "D_depth": "D, depth-linear"}
def tex(label: str) -> str:
    """Figure labels use Unicode; LaTeX tables need math-mode arrows and primes."""
    return label.replace("→", r"$\to$").replace("A′", "A$'$").replace("_", r"\_")


COLORS = {"static_big": "#888888", "A": "#d62728", "A2": "#9467bd", "B": "#8c564b",
          "C": "#ff7f0e", "D": "#1f77b4", "C_ordered": "#e6ab02", "D_colocated": "#17becf",
          "D_del_only": "#bcbd22", "D_clear_no_wait": "#7f7f7f", "D_depth": "#2ca02c",
          "big": "#000000"}


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return f"{x:.{nd}f}"


def put(key, val, nd=2):
    if isinstance(val, (int, float)):
        val = fmt(float(val), nd) if not isinstance(val, int) else str(val)
    if val is not None:
        M[key] = val


def put_ci(key, c, nd=2, scale=1.0):
    if not c:
        return
    put(key, c["mean"] * scale, nd)
    M[key + ".ci"] = f"[{c['lo'] * scale:.{nd}f}, {c['hi'] * scale:.{nd}f}]"
    M[key + ".n"] = str(c["n"])


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


pairs = [p for p, d in S["pairs"].items() if any(d.values())]

# ------------------------------------------------------------------ facts
pf = (S["facts"] or {}).get("platform")
if pf:
    for k in ("alloc_fp", "alloc_rss", "after_del", "after_clear"):
        put(f"mlx.{k}", pf["mlx"][k], 0)
    for k in ("release_ms", "release_ms_min", "release_ms_max", "after_clear_max"):
        if k in pf["mlx"]:
            put(f"mlx.{k}", pf["mlx"][k], 0)
    for k in ("clear_alone_freed", "release_runs"):
        if k in pf["mlx"]:
            M[f"mlx.{k}"] = str(pf["mlx"][k])
    for k, v in pf["energy"].items():
        put(f"energy.{k}", v, 2)
    if pf.get("identity"):
        idt = pf["identity"]
        M["id.same"] = str(idt["identical_continuations"]); M["id.items"] = str(idt["items"])
        put("id.maxdiff", idt["max_logit_diff"], 3); M["id.tokens"] = str(idt["continuation_tokens"])
    mc = pf["machine"]
    M["env.machine"] = f"{mc['chip']}, {mc['ram_gib']:.0f} GB, {mc['os']}, MLX {mc['mlx']}, mlx-lm {mc['mlx_lm']}"
pt = (S["facts"] or {}).get("peak_test")
if pt and "ordered" in pt:
    o = pt["ordered"]
    put("peak.src_mib", o["src_mib"], 0); put("peak.dst_mib", o["dst_mib"], 0)
    put("peak.ordered_mib", o["peak_mib"], 0); put("peak.cap_mib", o["cap_mib"], 0)
    if "clear_no_wait" in pt:
        M["peak.cnw_survived"] = str(sum(not r["killed"] for r in pt["clear_no_wait"]["runs"]))

# ------------------------------------------------------------------ fit
tab_fit, tab_lam = [], []
fig, axes = plt.subplots(1, max(1, len(pairs)), figsize=(5.2 * max(1, len(pairs)), 3.4), squeeze=False)
for ax, pair in zip(axes[0], pairs or ["-"]):
    f = S["pairs"].get(pair, {}).get("fit")
    if not f:
        ax.set_axis_off(); continue
    pl = pair.replace(".", "")
    ks = [(r["k"], r) for n, r in f["rows"].items() if n.startswith("heo_")]
    ks.sort()
    x = [k for k, _ in ks]
    y = [r["heldout_nll"][0] for _, r in ks]
    lo = [r["heldout_nll"][1] for _, r in ks]
    hi = [r["heldout_nll"][2] for _, r in ks]
    ax.fill_between(x, lo, hi, alpha=0.2, color=COLORS["C"])
    ax.plot(x, y, "o-", color=COLORS["C"], label="Heo top-k (C's mapper)")
    dl = f["rows"].get("depth_linear")
    if dl:
        ax.axhline(dl["heldout_nll"][0], color=COLORS["D_depth"], ls=":", label="depth-linear (k=1)")
    for nat, ls, lab in (("native_dst", "--", "receiver reads natively (A′)"), ("native_src", "-.", "sender reads natively")):
        ax.axhline(f["natives"][nat][0], color="k", ls=ls, lw=1, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("source layers per target layer, k"); ax.set_ylabel("held-out NLL (nats/token)")
    ax.set_title(f"{PAIR_LABEL[pair]}: mapper selection (held-out)")
    ax.legend(fontsize=7)
    sel = f["rows"][f["selected"]]
    M[f"fit.{pl}.kstar"] = str(sel["k"])
    put(f"fit.{pl}.kstar_nll", sel["heldout_nll"][0], 3)
    put(f"fit.{pl}.kstar_r2k", sel["heldout_r2_k"], 3); put(f"fit.{pl}.kstar_r2v", sel["heldout_r2_v"], 3)
    put(f"fit.{pl}.kstar_mib", sel["mib"], 0)
    put(f"fit.{pl}.native_dst_nll", f["natives"]["native_dst"][0], 3)
    put(f"fit.{pl}.native_src_nll", f["natives"]["native_src"][0], 3)
    put(f"fit.{pl}.gap_nll", sel["heldout_nll"][0] - f["natives"]["native_dst"][0], 3)
    M[f"fit.{pl}.n_rows"] = f"{f['n_rows']:,}".replace(",", "{,}")
    M[f"fit.{pl}.anchor"] = f"$s={f['molt_anchor']['sink']}$, $t={f['molt_anchor']['tail']}$"
    put(f"fit.{pl}.anchor_nll", f["molt_anchor"]["nll"][0], 3)
    for key, r in f["anchors"].items():
        put(f"fit.{pl}.anc.{key.replace('/', '_')}", r["nll"][0], 3)
    for lam, c in f["lam_sweep"].items():
        put(f"fit.{pl}.lam.{lam}", c[0], 3)
    put(f"fit.{pl}.depth_nll", f["rows"]["depth_linear"]["heldout_nll"][0], 3)
    put(f"fit.{pl}.depth_r2k", f["rows"]["depth_linear"]["heldout_r2_k"], 3)
    put(f"fit.{pl}.depth_r2v", f["rows"]["depth_linear"]["heldout_r2_v"], 3)
    for n, r in f["rows"].items():
        if n.startswith("heo_"):
            put(f"fit.{pl}.{n.replace('_', '')}.nll", r["heldout_nll"][0], 3)
            put(f"fit.{pl}.{n.replace('_', '')}.r2k", r["heldout_r2_k"], 3)
            put(f"fit.{pl}.{n.replace('_', '')}.r2v", r["heldout_r2_v"], 3)
    M[f"fit.{pl}.nsrc"] = str(len(f["single_source_r2"][0]))
    for n, r in sorted(f["rows"].items(), key=lambda kv: (kv[1]["policy"], kv[1]["k"])):
        tab_fit.append((tex(PAIR_LABEL[pair]), n.replace("_", r"\_"), r["k"], f"{r['params_m']:.1f}", f"{r['mib']:.0f}",
                        f"{r['insample_r2']:.3f}", f"{r['heldout_r2_k']:.3f}", f"{r['heldout_r2_v']:.3f}",
                        f"{r['heldout_nll'][0]:.3f}"))
    for lam, c in f["lam_sweep"].items():
        tab_lam.append((tex(PAIR_LABEL[pair]), lam, f"{c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]"))
save(fig, "fit_ksweep")

# ------------------------------------------------------------------ intrinsic
fig, axes = plt.subplots(1, max(1, len(pairs)), figsize=(5.2 * max(1, len(pairs)), 3.4), squeeze=False)
for ax, pair in zip(axes[0], pairs or ["-"]):
    d = S["pairs"].get(pair, {}).get("intrinsic")
    if not d:
        ax.set_axis_off(); continue
    pl = pair.replace(".", "")
    Ps = sorted(int(p) for p in d["summary"])
    for arm, col in (("C (ridge)", COLORS["C"]), ("D (molt anchors)", COLORS["D"]),
                     ("D_depth", COLORS["D_depth"]), ("native_src", "#000000")):
        if arm not in d["summary"][str(Ps[0])]:
            continue
        m = [d["summary"][str(p)][arm]["delta_vs_native_dst"][0] for p in Ps]
        lo = [d["summary"][str(p)][arm]["delta_vs_native_dst"][1] for p in Ps]
        hi = [d["summary"][str(p)][arm]["delta_vs_native_dst"][2] for p in Ps]
        ax.fill_between(Ps, lo, hi, alpha=0.15, color=col)
        ax.plot(Ps, m, "o-", color=col, label={"native_src": "sender, native"}.get(arm, arm))
        key = {"C (ridge)": "C", "D (molt anchors)": "D", "D_depth": "Ddepth", "native_src": "src"}[arm]
        for p in Ps:
            c = d["summary"][str(p)][arm]["delta_vs_native_dst"]
            M[f"int.{pl}.{key}.P{p}"] = f"{c[0]:+.3f}"
            M[f"int.{pl}.{key}.P{p}.ci"] = f"[{c[1]:+.3f}, {c[2]:+.3f}]"
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("switch position P (tokens of context)"); ax.set_ylabel("ΔNLL vs receiver reading natively")
    ax.set_title(f"{PAIR_LABEL[pair]}: cost of the transferred cache")
    ax.legend(fontsize=7)
    M[f"int.{pl}.n"] = str(d["n"])
save(fig, "intrinsic_context")

# ------------------------------------------------------------------ quality
QC = ["big", "A", "A2", "B", "C", "D", "C_ordered", "D_depth"]
tab_q = []
fig, axes = plt.subplots(len(pairs) or 1, 3, figsize=(13, 3.2 * max(1, len(pairs))), squeeze=False)
for row, pair in enumerate(pairs or ["-"]):
    q = S["pairs"].get(pair, {}).get("quality")
    if not q:
        for ax in axes[row]:
            ax.set_axis_off()
        continue
    pl = pair.replace(".", "")
    conds = [c for c in QC if c in q["conds"]]
    for ax, (metric, title) in zip(axes[row], (("gsm8k", "GSM8K exact"), ("needle", "needle exact"),
                                              ("agree_post", "judge agreement after switch"))):
        vals, los, his = [], [], []
        for c in conds:
            cc = q["conds"][c]["acc"].get(metric) if metric != "agree_post" else q["conds"][c]["agree_post"]
            vals.append(cc["mean"] if cc else float("nan"))
            los.append(cc["mean"] - cc["lo"] if cc else 0); his.append(cc["hi"] - cc["mean"] if cc else 0)
        ax.bar(range(len(conds)), vals, yerr=[los, his], color=[COLORS[c] for c in conds], capsize=2)
        ax.set_xticks(range(len(conds))); ax.set_xticklabels([c.replace("A2", "A′").replace("_", "\n") for c in conds], fontsize=7)
        ax.set_title(f"{PAIR_LABEL[pair]}: {title}", fontsize=9); ax.set_ylim(0, 1)
    for c in conds:
        cq = q["conds"][c]
        key = c.replace("_", "")
        put_ci(f"q.{pl}.{key}.gsm8k", cq["acc"]["gsm8k"], 3)
        put_ci(f"q.{pl}.{key}.needle", cq["acc"]["needle"], 3)
        put_ci(f"q.{pl}.{key}.agree", cq["agree_post"], 3)
        put_ci(f"q.{pl}.{key}.jump", cq["jump"], 3)
        put_ci(f"q.{pl}.{key}.lost", cq["lost_tokens"], 1)
        put_ci(f"q.{pl}.{key}.resume", cq["time_to_resume_s"], 2)
        put_ci(f"q.{pl}.{key}.newcontent", cq["time_to_new_content_s"], 2)
        for kind, t in (cq.get("truncated") or {}).items():
            if t:
                M[f"q.{pl}.{key}.trunc.{kind}"] = f"{round(t['mean'] * t['n'])}/{t['n']}"
        g = cq["acc"]["gsm8k"]; nd = cq["acc"]["needle"]; ag = cq["agree_post"]; jp = cq["jump"]
        rs = cq["time_to_new_content_s"]
        tab_q.append((tex(PAIR_LABEL[pair]), tex(COND_LABEL[c]),
                      f"{g['mean']:.2f}" if g else "--", f"{nd['mean']:.2f}" if nd else "--",
                      f"{ag['mean']:.3f}" if ag else "--", f"{jp['mean']:+.3f}" if jp else "--",
                      f"{cq['lost_tokens']['mean']:.0f}" if cq["lost_tokens"] else "--",
                      f"{rs['mean']:.2f}" if rs else "--"))
    for k, v in q["paired"].items():
        a, b = k.split("-")
        for metric in ("jump", "agree_post", "task"):
            if v.get(metric):
                put_ci(f"qp.{pl}.{a.replace('_', '')}.{b}.{metric}", v[metric], 3)
    M[f"q.{pl}.n_items"] = str(q["n_items"])
    for kind, c in (q.get("context") or {}).items():
        M[f"q.{pl}.ctx.{kind}"] = f"{c['median']:.0f}"
        M[f"q.{pl}.ctx.{kind}.range"] = f"{c['min']:.0f}--{c['max']:.0f}"
save(fig, "quality")

# ------------------------------------------------------------------ memory
MC = ["static_big", "A", "A2", "B", "C", "D", "C_ordered", "D_colocated", "D_del_only", "D_clear_no_wait"]
tab_m = []
fig, axes = plt.subplots(1, max(1, len(pairs)), figsize=(5.8 * max(1, len(pairs)), 4.0), squeeze=False)
for ax, pair in zip(axes[0], pairs or ["-"]):
    mm = S["pairs"].get(pair, {}).get("memory")
    if not mm or not mm["cells"]:
        ax.set_axis_off(); continue
    pl = pair.replace(".", "")
    hs = sorted(int(h) for h in mm["cells"])
    rows = [c for c in MC if any(c in mm["cells"][str(h)] for h in hs)]
    grid = [[(mm["cells"][str(h)].get(c) or {}).get("survival", {}) or {} for h in hs] for c in rows]
    vals = [[g.get("mean", float("nan")) for g in r] for r in grid]
    ax.imshow(vals, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i, c in enumerate(rows):
        for j, h in enumerate(hs):
            cell = mm["cells"][str(h)].get(c)
            if cell:
                v = vals[i][j]
                ax.text(j, i, f"{cell['survived']}/{cell['n']}", ha="center", va="center", fontsize=8,
                        color="white" if (v < 0.25 or v > 0.75) else "black")
    ax.set_xticks(range(len(hs))); ax.set_xticklabels([str(h) for h in hs])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([COND_LABEL[c] for c in rows], fontsize=8)
    need = mm["plan"]["need_colocated"]
    xpos = sum(1 for h in hs if h < need) - 0.5
    ax.axvline(xpos, color="k", ls=":", lw=1)
    ax.text(xpos + 0.05, len(rows) - 0.55, "← below: a co-resident switch cannot fit", fontsize=7,
            va="top", color="k", bbox=dict(facecolor="white", alpha=0.8, lw=0, pad=1))
    ax.set_xlabel("headroom h over the big model's footprint (MiB)")
    ax.set_title(f"{PAIR_LABEL[pair]}: survived / runs under a per-process cap", fontsize=10)
    p = mm["plan"]
    put(f"m.{pl}.S", p["S"], 0); put(f"m.{pl}.Ssmall", p["S_small"], 0); put(f"m.{pl}.dstw", p["dst_w"], 0)
    put(f"m.{pl}.mapw", p["mapper_w"], 0); put(f"m.{pl}.need", p["need_colocated"], 0)
    M[f"m.{pl}.hgrid"] = ", ".join(str(int(h)) for h in hs)
    for h in hs:
        for c, cell in mm["cells"][str(h)].items():
            key = c.replace("_", "")
            M[f"m.{pl}.h{h}.{key}.surv"] = f"{cell['survived']}/{cell['n']}"
            put(f"m.{pl}.h{h}.{key}.peak", cell["peak_mib"]["max"], 0)
            put(f"m.{pl}.h{h}.cap", cell["cap_mib"], 0)
            if cell["killed_in"]:
                M[f"m.{pl}.h{h}.{key}.killedin"] = ", ".join(x.replace("_", r"\_") for x in cell["killed_in"])
            put_ci(f"m.{pl}.h{h}.{key}.resume", cell["time_to_resume_s"], 2)
            put_ci(f"m.{pl}.h{h}.{key}.mig", cell["migration_s"], 2)
            put_ci(f"m.{pl}.h{h}.{key}.ecpu", cell["energy_mig_cpu_j"], 2)
            put_ci(f"m.{pl}.h{h}.{key}.egpu", cell["energy_mig_gpu_j"], 2)
            tab_m.append((tex(PAIR_LABEL[pair]), h, f"{cell['cap_mib']:.0f}", tex(COND_LABEL[c]), f"{cell['survived']}/{cell['n']}",
                          f"{cell['peak_mib']['max']:.0f}",
                          ", ".join(cell["killed_in"]).replace("_", r"\_") or "--",
                          f"{cell['time_to_resume_s']['mean']:.2f}" if cell["time_to_resume_s"] else "--"))
save(fig, "memory_survival")

# footprint timelines at the band point: one run per schedule
for pair in pairs:
    mm = S["pairs"].get(pair, {}).get("memory")
    if not mm or not mm["cells"]:
        continue
    need = mm["plan"]["need_colocated"]
    band = [int(h) for h in mm["cells"] if int(h) < need]
    if not band:
        continue
    h = max(band)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    cap = None
    SWITCH = ("snapshot", "release_src", "load_dst", "load_mapper")
    for c in ("C", "D", "D_del_only", "D_clear_no_wait", "A2"):
        files = sorted((RES / "raw" / "memory" / pair / f"h{h}").glob(f"{c}__*__r0.csv"))
        if not files:
            continue
        rows = [r for r in csv.reader(open(files[0])) if r and not r[0].startswith(("#", "t_us"))]
        t0 = next((int(r[0]) for r in rows if r[2] in SWITCH), None)
        if t0 is None:
            continue
        pts = [((int(r[0]) - t0) / 1e6, int(r[1]) / 2**20, r[2]) for r in rows
               if -1.0 <= (int(r[0]) - t0) / 1e6 <= 3.0 and r[2] not in ("done",)]
        ax.plot([t for t, _, _ in pts], [m for _, m, _ in pts], color=COLORS[c], lw=1.2, label=COND_LABEL[c])
        kill = [(t, m) for t, m, ph in pts if ph == "KILLED"]
        if kill:
            ax.plot(*kill[0], "x", color=COLORS[c], ms=9, mew=2)
        cap = mm["cells"][str(h)][c]["cap_mib"]
    if cap:
        ax.axhline(cap, color="r", ls="--", lw=1, label="cap (jetsam emulation)")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("time from the start of the switch (s);  × = killed by the watchdog")
    ax.set_ylabel("phys_footprint (MiB)")
    ax.set_title(f"{PAIR_LABEL[pair]}: footprint through the switch, h = {h} MiB")
    ax.legend(fontsize=7, loc="lower left")
    save(fig, f"footprint_timeline_{pair.replace('.', '')}")
    M[f"m.{pair.replace('.', '')}.bandh"] = str(h)

# ------------------------------------------------------------------ resume cost
rc_pairs = [p for p, d in (S.get("resume_cost") or {}).items() if d]
if rc_pairs:
    fig, axes = plt.subplots(1, len(rc_pairs), figsize=(5.2 * len(rc_pairs), 3.4), squeeze=False)
    for ax, pair in zip(axes[0], rc_pairs):
        d = S["resume_cost"][pair]
        pl = pair.replace(".", "")
        Ps = sorted(int(p) for p in d["by_context"])
        for arm, col in (("A2 re-prefill", COLORS["A2"]), ("C ridge", COLORS["C"]), ("D molt anchors", COLORS["D"])):
            ys = [d["by_context"][str(p)][arm]["median_s"] for p in Ps]
            ax.plot(Ps, ys, "o-", color=col, label=arm)
            key = {"A2 re-prefill": "A2", "C ridge": "C", "D molt anchors": "D"}[arm]
            for p, y in zip(Ps, ys):
                put(f"r.{pl}.{key}.P{p}", y, 3)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("context at the switch (tokens)"); ax.set_ylabel("resume compute (s, median)")
        ax.set_title(f"{PAIR_LABEL[pair]}: resume compute, models resident")
        ax.legend(fontsize=7)
        put(f"r.{pl}.load_dst", d["load_s"]["dst_model"], 3)
        put(f"r.{pl}.load_mapper", d["load_s"]["mapper"], 3)
        M[f"r.{pl}.repeats"] = str(d["repeats"])
    save(fig, "resume_cost")

# phase durations (median over runs) at every h, for the conditions that switch
for pair in pairs:
    for sub, tag in (("memory", ""), ("memory_cache20", "c20.")):
        mm = S["pairs"].get(pair, {}).get(sub)
        if not mm or not mm["cells"]:
            continue
        pl = pair.replace(".", "")
        for h, cells in mm["cells"].items():
            for c, cell in cells.items():
                key = c.replace("_", "")
                for ph, v in (cell.get("phase_s") or {}).items():
                    put(f"m.{tag}{pl}.h{h}.{key}.ph.{ph.replace('_', '')}", v, 3)
                if cell.get("resumed_mib"):
                    put(f"m.{tag}{pl}.h{h}.{key}.resumed", cell["resumed_mib"]["mean"], 0)
                if tag:
                    M[f"m.{tag}{pl}.h{h}.{key}.surv"] = f"{cell['survived']}/{cell['n']}"
                    put(f"m.{tag}{pl}.h{h}.{key}.peak", cell["peak_mib"]["max"], 0)
                put(f"m.{tag}{pl}.h{h}.{key}.gpubase", (cell.get("gpu_baseline_w") or {}).get("mean"), 1)

# ------------------------------------------------------------------ write
def tex_table(rows, header, caption, label, colspec):
    if not rows:
        return (f"\\begin{{table}}[t]\\centering\\caption{{{caption}}}\\label{{{label}}}"
                f"\\textcolor{{red}}{{\\textbf{{PENDING}}}}\\end{{table}}\n")
    body = "\n".join(" & ".join(str(c) for c in r) + r" \\" for r in rows)
    return (f"\\begin{{table}}[t]\\centering\\small\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{colspec}}}\\toprule\n{' & '.join(header)} \\\\\\midrule\n{body}\n"
            f"\\bottomrule\\end{{tabular}}\\end{{table}}\n")


tables = [
    tex_table(tab_fit, ["pair", "mapper", "$k$", "params (M)", "MiB", "$R^2$ in", "$R^2_K$ held", "$R^2_V$ held", "NLL held"],
              "Every fitted mapper. Held-out = documents disjoint from calibration and evaluation.",
              "tab:fit", "llrrrrrrr"),
    tex_table(tab_lam, ["pair", "$\\lambda$", "held-out NLL [95\\% CI]"],
              "Ridge $\\lambda$ sweep at the selected $k^*$ (centred-covariance reading).", "tab:lambda", "llr"),
    tex_table(tab_q, ["pair", "condition", "GSM8K", "needle", "agree$_{post}$", "jump", "lost", "resume (s)"],
              "Experiment Q. Accuracy counts every item; jump = judge-NLL jump at the switch (16 tokens each side).",
              "tab:quality", "llrrrrrr"),
    tex_table(tab_m, ["pair", "$h$", "cap", "condition", "survived", "peak", "killed in", "resume (s)"],
              "Experiment M: every cell. Peak = kernel lifetime-max phys\\_footprint (MiB).",
              "tab:memory", "lrrlrrlr"),
]
(GEN / "tables.tex").write_text("\n".join(tables))
lines = ["% generated by figures/make_paper_figures.py — do not edit"]
for k, v in sorted(M.items()):
    lines.append(f"\\expandafter\\def\\csname res:{k}\\endcsname{{{v}}}")
(GEN / "results.tex").write_text("\n".join(lines) + "\n")
(RES / "macros.json").write_text(json.dumps(M, indent=1, ensure_ascii=False))
print(f"{len(M)} macros, {len(list(FIG.glob('*.pdf')))} figures")
