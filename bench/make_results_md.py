"""Write RESULTS.md and the README results block from measured data only.

Tables come straight from bench/results/summary.json (every cell, including the
ones molt loses).  The prose sections come from bench/RESULTS.findings.md, whose
``{{key}}`` placeholders are filled from bench/results/macros.json — the same
values the paper's \\res{key} macros print.  An unmeasured key renders as
**PENDING**, never as a guess.

    python -m bench.analyze && python figures/make_paper_figures.py && python -m bench.make_results_md
"""

from __future__ import annotations

import json
import re

from bench._env import RESULTS, ROOT

COND = {"big": "big, no pressure", "static_big": "big, ignores pressure", "A": "(A) restart",
        "A2": "(A′) re-prefill", "B": "(B) always small", "C": "(C) ridge mid-decode, colocated",
        "D": "(D) molt", "C_ordered": "C + ordered", "D_colocated": "D, colocated",
        "D_del_only": "D, del-only", "D_clear_no_wait": "D, clear-no-wait",
        "D_depth": "D, depth-linear mapper"}
PAIR = {"3b-1.5b": "Qwen2.5-3B → 1.5B (4-bit)", "1.5b-0.5b": "Qwen2.5-1.5B → 0.5B (4-bit)"}


def c(x, nd=3, sign=False):
    if not x:
        return "—"
    f = f"{{:{'+' if sign else ''}.{nd}f}}"
    return f"{f.format(x['mean'])} [{f.format(x['lo'])}, {f.format(x['hi'])}]"


def fill(text, M):
    return re.sub(r"\{\{([^}]+)\}\}", lambda m: M.get(m.group(1).strip(), "**PENDING**"), text)


def main():
    S = json.loads((RESULTS / "summary.json").read_text())
    M = json.loads((RESULTS / "macros.json").read_text()) if (RESULTS / "macros.json").exists() else {}
    out = ["# RESULTS", "",
           "Every number below is generated from `bench/results/` by `bench/make_results_md.py`.",
           "Intervals are 95% bootstrap CIs. A killed run scores 0 and is never dropped.",
           "Measured on: " + M.get("env.machine", "PENDING") + ". Other users' jobs were running and swap was",
           "nearly full during these runs. Timings carry that noise; footprint and survival do not.", ""]
    findings = ROOT / "bench" / "RESULTS.findings.md"
    if findings.exists():
        out += [fill(findings.read_text(), M), ""]
    for pair, d in S["pairs"].items():
        out += [f"## {PAIR[pair]}", ""]
        f = d.get("fit")
        out += ["### Mapper fit and selection (held-out documents only)", ""]
        if f:
            out += [f"Calibration rows: {f['n_rows']:,} ({f['corpus']['source']}). Selected: **{f['selected']}**. "
                    f"molt anchors: sink {f['molt_anchor']['sink']}, tail {f['molt_anchor']['tail']}.", "",
                    "| mapper | k | params (M) | MiB fp16 | R² in-sample | R² K held-out | R² V held-out | held-out NLL |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for n, r in sorted(f["rows"].items(), key=lambda kv: (kv[1]["policy"], kv[1]["k"])):
                h = r["heldout_nll"]
                out.append(f"| {n} | {r['k']} | {r['params_m']:.1f} | {r['mib']:.0f} | {r['insample_r2']:.3f} | "
                           f"{r['heldout_r2_k']:.3f} | {r['heldout_r2_v']:.3f} | {h[0]:.3f} [{h[1]:.3f}, {h[2]:.3f}] |")
            for k, v in f["natives"].items():
                out.append(f"| *{k} (reference)* | | | | | | | {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}] |")
            out += ["", "λ sweep at k*: " + ", ".join(f"λ={l}: {v[0]:.3f}" for l, v in f["lam_sweep"].items()), "",
                    "Anchor sweep (sink/tail → held-out NLL): " +
                    ", ".join(f"{k}: {v['nll'][0]:.3f}" for k, v in f["anchors"].items()), ""]
        else:
            out += ["PENDING", ""]
        it = d.get("intrinsic")
        out += ["### Experiment I: ΔNLL of the transferred cache vs. the receiver reading natively", ""]
        if it:
            Ps = sorted(it["summary"], key=int)
            arms = list(it["summary"][Ps[0]])
            out += ["| arm | " + " | ".join(f"P={p}" for p in Ps) + " |", "|---|" + "---:|" * len(Ps)]
            for a in arms:
                cells = []
                for p in Ps:
                    x = it["summary"][p][a]["delta_vs_native_dst"]
                    cells.append(f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]")
                out.append(f"| {a} | " + " | ".join(cells) + " |")
            out += ["", f"n = {it['n']} evaluation documents, {it['span']} scored tokens after the switch.", ""]
        else:
            out += ["PENDING", ""]
        q = d.get("quality")
        out += ["### Experiment Q: quality when the switch lands mid-decode (no cap)", ""]
        if q:
            out += ["| condition | GSM8K | needle | judge agreement after switch | seam jump (nats) | lost tokens | time to new content (s) |",
                    "|---|---:|---:|---:|---:|---:|---:|"]
            for cond in ("big", "A", "A2", "B", "C", "D", "C_ordered", "D_depth"):
                x = q["conds"].get(cond)
                if not x:
                    continue
                out.append(f"| {COND[cond]} | {c(x['acc']['gsm8k'], 2)} | {c(x['acc']['needle'], 2)} | "
                           f"{c(x['agree_post'])} | {c(x['jump'], 3, True)} | {c(x['lost_tokens'], 1)} | "
                           f"{c(x['time_to_new_content_s'], 2)} |")
            out += ["", "Paired differences (same items):", "",
                    "| comparison | seam jump | agreement after switch | task score |", "|---|---:|---:|---:|"]
            for k, v in sorted(q["paired"].items()):
                out.append(f"| {k} | {c(v['jump'], 3, True)} | {c(v['agree_post'], 3, True)} | {c(v['task'], 3, True)} |")
            out.append("")
        else:
            out += ["PENDING", ""]
        mm = d.get("memory")
        out += ["### Experiment M: survival under a per-process cap (every run its own process)", ""]
        if mm:
            p = mm["plan"]
            out += [f"S (big model decoding) = {p['S']:.0f} MiB, small = {p['S_small']:.0f} MiB, "
                    f"receiver weights {p['dst_w']:.0f} MiB, mapper {p['mapper_w']:.0f} MiB → a co-resident switch "
                    f"needs ≈ {p['need_colocated']:.0f} MiB of headroom.", "",
                    "| h (MiB) | cap | condition | survived | peak MiB (max) | killed in | time to resume (s) | migration CPU J | migration GPU J |",
                    "|---:|---:|---|---:|---:|---|---:|---:|---:|"]
            for h in sorted(mm["cells"], key=int):
                for cond, x in mm["cells"][h].items():
                    out.append(f"| {h} | {x['cap_mib']:.0f} | {COND.get(cond, cond)} | {x['survived']}/{x['n']} | "
                               f"{x['peak_mib']['max']:.0f} | {', '.join(x['killed_in']) or '—'} | "
                               f"{c(x['time_to_resume_s'], 2)} | {c(x['energy_mig_cpu_j'], 2)} | {c(x['energy_mig_gpu_j'], 2)} |")
            out.append("")
        else:
            out += ["PENDING", ""]
    out += ["## Device, energy cross-check, human ratings", "",
            "- iPhone measurements: **PENDING** (protocol: `ios/README.md`). Device folders present: "
            + (", ".join(S.get("device") or []) or "none") + ".",
            "- `powermetrics` cross-check: **PENDING** (needs sudo; `bench/powermetrics.sh`).",
            "- Human seam ratings: **PENDING** (`bench/human_eval/`).", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(out))
    # README block: the headline numbers only, from macros
    readme = (ROOT / "README.md").read_text()
    glance = ROOT / "bench" / "README.glance.md"
    block = fill(glance.read_text(), M) if glance.exists() else "*See [RESULTS.md](RESULTS.md).*"
    readme = re.sub(r"<!-- results:begin -->.*?<!-- results:end -->",
                    "<!-- results:begin -->\n" + block.strip() + "\n<!-- results:end -->", readme, flags=re.S)
    (ROOT / "README.md").write_text(readme)
    print("wrote RESULTS.md and README block;", sum(1 for _ in re.finditer("PENDING", "\n".join(out))), "PENDING marks")


if __name__ == "__main__":
    main()
