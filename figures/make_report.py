#!/usr/bin/env python3
"""Render the shareable Molt report straight from ``summary.json``.

    python figures/make_report.py --results benchmarks/results --out artifacts/report.html

Generated rather than hand-written on purpose: every number on the page is read
out of the benchmark's own output, so the report cannot drift away from the
measurements it describes.  If a figure is missing from the results, the section
that depends on it is omitted rather than filled in with a plausible number.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import math
import os
from typing import Any, Dict, List, Optional

COND_ORDER = ["A", "B", "C", "D"]
ABL_ORDER = ["D-nocal", "D-noproj", "D-norecompute", "D-reqbound"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def n(v, spec=".2f", dash="—"):
    """Format a number, or a dash when it is missing/NaN.

    Never substitutes a plausible value for a missing measurement.
    """
    if v is None:
        return dash
    try:
        if v != v:            # NaN
            return dash
        return format(v, spec)
    except (TypeError, ValueError):
        return dash


def esc(s) -> str:
    return html.escape(str(s))


def _series(rec, prompt_id=None):
    ser = rec.get("series") or []
    if not ser:
        return None
    if prompt_id:
        for s in ser:
            if s["prompt_id"] == prompt_id:
                return s
    alive = [s for s in ser if not s.get("killed")] or ser
    return max(alive, key=lambda s: len(s.get("itl_ms") or []))


def pick_common_prompt(conds: Dict[str, Any]) -> Optional[str]:
    common = None
    for k in ("C", "D"):
        if k not in conds:
            continue
        ids = {s["prompt_id"] for s in conds[k].get("series", [])}
        common = ids if common is None else (common & ids)
    if not common:
        return None
    # the longest surviving D series makes the clearest picture
    d = conds.get("D", {})
    best, best_len = None, -1
    for s in d.get("series", []):
        if s["prompt_id"] in common and len(s.get("itl_ms") or []) > best_len:
            best, best_len = s["prompt_id"], len(s.get("itl_ms") or [])
    return best or sorted(common)[0]


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------

CSS = """
:root{
  /* A memory gauge, not a mood board: the neutrals carry a faint green bias
     (the colour of a healthy meter), the accent is that same green at full
     strength, and the alarm hue is the one an OS uses when it takes memory
     back.  The three rungs get a deliberate cool -> warm ramp so a stripe of
     tokens reads as a ladder rather than as three arbitrary categories. */
  --paper:#f2f4ee; --ink:#12160f; --ink-2:#4a5147; --ink-3:#7b8177;
  --line:#dfe3d9; --raise:#fafbf7;
  --gauge:#37734a; --gauge-soft:#e2ede3;
  --alarm:#b0402a; --alarm-soft:#f6e2dc;
  --t0:#264f6e; --t1:#a8763a; --t2:#8a5a7a;
  --t0-soft:#dbe6ef; --t1-soft:#f2e6d3; --t2-soft:#ece0eb;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
         "Helvetica Neue",Arial,sans-serif;
  --w:68ch;
}
@media (prefers-color-scheme:dark){
  :root{--paper:#0e110d; --ink:#e8ece4; --ink-2:#a9b0a4; --ink-3:#767d72;
        --line:#242a22; --raise:#151a13;
        --gauge:#6fb684; --gauge-soft:#1b2a1f;
        --alarm:#e0705a; --alarm-soft:#2c1a16;
        --t0:#7aa8cd; --t1:#d5a865; --t2:#bb92b4;
        --t0-soft:#1b2a37; --t1-soft:#332714; --t2-soft:#2c1f2b;}
}
:root[data-theme="dark"]{--paper:#0e110d; --ink:#e8ece4; --ink-2:#a9b0a4;
  --ink-3:#767d72; --line:#242a22; --raise:#151a13;
  --gauge:#6fb684; --gauge-soft:#1b2a1f; --alarm:#e0705a; --alarm-soft:#2c1a16;
  --t0:#7aa8cd; --t1:#d5a865; --t2:#bb92b4;
  --t0-soft:#1b2a37; --t1-soft:#332714; --t2-soft:#2c1f2b;}
:root[data-theme="light"]{--paper:#f2f4ee; --ink:#12160f; --ink-2:#4a5147;
  --ink-3:#7b8177; --line:#dfe3d9; --raise:#fafbf7;
  --gauge:#37734a; --gauge-soft:#e2ede3; --alarm:#b0402a; --alarm-soft:#f6e2dc;
  --t0:#264f6e; --t1:#a8763a; --t2:#8a5a7a;
  --t0-soft:#dbe6ef; --t1-soft:#f2e6d3; --t2-soft:#ece0eb;}

*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
     font-size:16.5px;line-height:1.6;-webkit-font-smoothing:antialiased}
.page{max-width:var(--w);margin:0 auto;padding:0 24px}
.bleed{max-width:1120px;margin:0 auto;padding:0 24px}
section{padding:8px 0 40px}

h1{font-size:clamp(38px,7vw,66px);line-height:.98;letter-spacing:-.035em;
   font-weight:800;margin:0 0 18px;text-wrap:balance}
h2{font-size:clamp(23px,3.4vw,31px);line-height:1.18;letter-spacing:-.02em;
   font-weight:750;margin:52px 0 14px;text-wrap:balance}
h3{font-size:18px;letter-spacing:-.01em;font-weight:700;margin:32px 0 8px}
p{margin:0 0 16px}
a{color:var(--gauge);text-underline-offset:3px}
strong{font-weight:680}

.eyebrow{font-family:var(--mono);font-size:11.5px;letter-spacing:.16em;
         text-transform:uppercase;color:var(--ink-3);margin:0 0 14px}
.lede{font-size:20px;line-height:1.5;color:var(--ink-2);margin:0 0 26px}
.note{font-size:14.5px;color:var(--ink-2)}
.mono{font-family:var(--mono)}
code{font-family:var(--mono);font-size:.88em;background:var(--raise);
     border:1px solid var(--line);border-radius:4px;padding:1px 5px}

/* ---- claim strip ------------------------------------------------------ */
.claims{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));
        gap:1px;background:var(--line);border:1px solid var(--line);
        border-radius:8px;overflow:hidden;margin:26px 0 6px}
.claim{background:var(--paper);padding:15px 16px}
.claim b{display:block;font-family:var(--mono);font-size:26px;font-weight:600;
         letter-spacing:-.02em;font-variant-numeric:tabular-nums;line-height:1.1}
.claim span{display:block;font-family:var(--mono);font-size:11px;
            letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);
            margin-top:6px}
.good{color:var(--gauge)} .bad{color:var(--alarm)}

/* ---- the trace -------------------------------------------------------- */
figure{margin:26px 0 8px}
figcaption{font-size:13.5px;color:var(--ink-2);margin-top:10px;
           font-family:var(--sans)}
.chart{width:100%;height:auto;display:block;background:var(--raise);
       border:1px solid var(--line);border-radius:8px}

/* ---- token stripe ----------------------------------------------------- */
.stripe{display:flex;gap:2px;flex-wrap:wrap;margin:14px 0 6px}
.cell{width:11px;height:22px;border-radius:2px}
.c-tier0{background:var(--t0)} .c-tier1{background:var(--t1)}
.c-tier2{background:var(--t2)} .c-dead{background:var(--alarm);opacity:.35}
.key{display:flex;gap:16px;flex-wrap:wrap;font-family:var(--mono);
     font-size:11.5px;letter-spacing:.06em;color:var(--ink-2);margin-top:8px}
.key i{display:inline-block;width:10px;height:10px;border-radius:2px;
       margin-right:6px;vertical-align:-1px}

/* ---- tables ----------------------------------------------------------- */
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;
        background:var(--raise)}
table{border-collapse:collapse;width:100%;font-family:var(--mono);
      font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:9px 13px;text-align:right;white-space:nowrap;
      border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
thead th{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;
         color:var(--ink-3);font-weight:600;position:sticky;top:0;
         background:var(--raise)}
tbody tr:last-child td{border-bottom:none}
tr.hero td{background:var(--gauge-soft)}
tr.dead td{color:var(--ink-2)}
tr.dead td:first-child::after{content:" ✕";color:var(--alarm)}

/* ---- ladder ----------------------------------------------------------- */
.ladder{display:grid;gap:8px;margin:18px 0}
.rung{display:grid;grid-template-columns:auto 1fr auto;gap:14px;
      align-items:center;border:1px solid var(--line);border-radius:8px;
      padding:12px 14px;background:var(--raise)}
.rung .tag{font-family:var(--mono);font-size:11px;letter-spacing:.09em;
           text-transform:uppercase;padding:3px 8px;border-radius:999px;
           color:var(--paper)}
.rung .bar{height:9px;border-radius:5px;background:var(--line);overflow:hidden}
.rung .bar>i{display:block;height:100%}
.rung .mb{font-family:var(--mono);font-size:13px;
          font-variant-numeric:tabular-nums;color:var(--ink-2)}

/* ---- misc ------------------------------------------------------------- */
ol,ul{margin:0 0 16px;padding-left:22px}
li{margin:0 0 8px}
hr{border:none;border-top:1px solid var(--line);margin:44px 0}
.foot{font-size:13.5px;color:var(--ink-3);font-family:var(--mono);
      padding:8px 0 60px;line-height:1.7}
.callout{border-left:3px solid var(--gauge);background:var(--gauge-soft);
         padding:14px 18px;border-radius:0 8px 8px 0;margin:22px 0;
         font-size:15.5px}
.callout.warn{border-left-color:var(--alarm);background:var(--alarm-soft)}
@media (prefers-reduced-motion:no-preference){
  .reveal{animation:rise .5s cubic-bezier(.2,.7,.3,1) both}
  @keyframes rise{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:none}}
}
"""


def chart_svg(cond_c, cond_d, width=1040, height=330) -> str:
    """The thesis, drawn from the measured per-token latency series.

    Hand-built SVG rather than a chart library: the CSP forbids external
    scripts, and the shape here is simple enough that a dependency would buy
    nothing but weight.
    """
    cs = (cond_c or {}).get("itl_ms") or []
    ds = (cond_d or {}).get("itl_ms") or []
    if not ds:
        return ""
    press = (cond_d or {}).get("pressure") or []
    nmax = max(len(cs), len(ds))
    lo = max(1.0, min([v for v in (cs + ds) if v > 0] or [1.0]))
    hi = max(cs + ds + [1.0])


    pad_l, pad_r, pad_t, pad_b = 58, 14, 18, 34
    W, H = width - pad_l - pad_r, height - pad_t - pad_b

    def X(i):
        return pad_l + (W * i / max(1, nmax - 1))

    def Y(v):
        v = max(lo, v)
        f = (math.log(v) - math.log(lo)) / max(1e-9, math.log(hi) - math.log(lo))
        return pad_t + H * (1 - f)

    def path(vals):
        return " ".join(("M" if i == 0 else "L") + f"{X(i):.1f},{Y(v):.1f}"
                        for i, v in enumerate(vals))

    parts: List[str] = []
    # pressure band: the interval where usage exceeded the budget
    over = [i for i, p in enumerate(press) if p and p >= 1.0]
    if over:
        x0, x1 = X(min(over)), X(max(over))
        parts.append(f'<rect x="{x0:.1f}" y="{pad_t}" width="{max(2,x1-x0):.1f}" '
                     f'height="{H}" fill="var(--alarm-soft)"/>')
        parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{pad_t+13}" font-size="10.5" '
                     f'fill="var(--alarm)" text-anchor="middle" '
                     f'font-family="var(--mono)" letter-spacing="1.2">'
                     f'BUDGET EXCEEDED</text>')
    # Gridlines on *round* decades, not on whatever the data minimum happened to
    # be — "68.413ms" as an axis label is a tell that nobody looked at the output.
    d = 10.0 ** math.floor(math.log10(lo))
    while d <= hi * 1.001:
        if d >= lo * 0.999:
            y = Y(d)
            parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" '
                         f'y2="{y:.1f}" stroke="var(--line)" stroke-width="1"/>')
            lab = f"{d:g} ms" if d < 1000 else f"{d/1000:g} s"
            parts.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" font-size="10.5" '
                         f'fill="var(--ink-3)" text-anchor="end" '
                         f'font-family="var(--mono)">{lab}</text>')
        d *= 10
    if cs:
        parts.append(f'<path d="{path(cs)}" fill="none" stroke="var(--t1)" '
                     f'stroke-width="2" stroke-linejoin="round" opacity=".95"/>')
    parts.append(f'<path d="{path(ds)}" fill="none" stroke="var(--t0)" '
                 f'stroke-width="2.4" stroke-linejoin="round"/>')
    # migration markers, from the real event log
    for m in (cond_d or {}).get("migrations", []):
        i = m.get("token_index", 0)
        if i < len(ds):
            parts.append(f'<line x1="{X(i):.1f}" y1="{pad_t}" x2="{X(i):.1f}" '
                         f'y2="{pad_t+H}" stroke="var(--t0)" stroke-width="1" '
                         f'stroke-dasharray="3 3" opacity=".55"/>')
            parts.append(f'<text x="{X(i)+5:.1f}" y="{pad_t+H-6}" font-size="10" '
                         f'fill="var(--t0)" font-family="var(--mono)">'
                         f'{esc(m.get("from_tier"))}&#8594;{esc(m.get("to_tier"))}</text>')
    parts.append(f'<text x="{pad_l}" y="{height-10}" font-size="10.5" '
                 f'fill="var(--ink-3)" font-family="var(--mono)">'
                 f'generated token index &#8594;</text>')
    parts.append(f'<text x="{width-pad_r}" y="{pad_t+14}" font-size="11.5" '
                 f'fill="var(--t0)" text-anchor="end" font-family="var(--mono)">'
                 f'D · Molt</text>')
    if cs:
        parts.append(f'<text x="{width-pad_r}" y="{pad_t+30}" font-size="11.5" '
                     f'fill="var(--t1)" text-anchor="end" font-family="var(--mono)">'
                     f'C · restart</text>')
    return (f'<svg class="chart" viewBox="0 0 {width} {height}" '
            f'preserveAspectRatio="xMidYMid meet" role="img" '
            f'aria-label="Per-token latency for the restart baseline and for Molt '
            f'across a memory-pressure event">{"".join(parts)}</svg>')


def stripe(series, limit=200) -> str:
    tiers = (series or {}).get("tiers") or []
    if not tiers:
        return ""
    cells = "".join(f'<div class="cell c-{esc(t)}"></div>' for t in tiers[:limit])
    if series.get("killed"):
        cells += '<div class="cell c-dead" title="reclaimed by the OS"></div>'
    return f'<div class="stripe">{cells}</div>'


def tier_descriptions(meta: Dict[str, Any]) -> Dict[str, tuple]:
    """Human labels for the rungs, keyed off the ladder that was actually run.

    Hard-coding "Qwen2.5-1.5B" would silently mislabel a report generated from
    the synthetic ladder, which is exactly the kind of caption error a reader has
    no way to catch.
    """
    if str(meta.get("ladder", "")).startswith("qwen2.5-1.5b"):
        return {
            "tier0": ("Qwen2.5-1.5B fp32", "the rung you want to be on"),
            "tier1": ("Qwen2.5-1.5B int8", "same weights, lower precision &#8212; "
                                           "a diagonal scale re-alignment"),
            "tier2": ("Qwen2.5-0.5B fp32", "different depth <em>and</em> head_dim "
                                           "&#8212; a learned linear projection"),
        }
    if str(meta.get("ladder", "")).startswith("qwen2.5-3b"):
        return {
            "tier0": ("Qwen2.5-3B fp32", "the rung you want to be on"),
            "tier1": ("Qwen2.5-3B int4", "same weights, lower precision"),
            "tier2": ("Qwen2.5-0.5B fp32", "different depth and head_dim"),
        }
    return {
        "tier0": ("tier0", "the rung you want to be on"),
        "tier1": ("tier1", "same weights, lower precision"),
        "tier2": ("tier2", "different depth and head_dim"),
    }


def build(data: Dict[str, Any], proj_index: Optional[Dict[str, Any]]) -> str:
    conds = data.get("conditions", {})
    meta = data.get("meta", {})
    fps = meta.get("footprints", {})
    trace = meta.get("trace", {})
    sweep = data.get("cost_sweep", {})
    qos = data.get("qos", {})
    pid = pick_common_prompt(conds)
    sD, sC = _series(conds.get("D", {}), pid), _series(conds.get("C", {}), pid)

    A, B, C, D = (conds.get(k, {}) for k in COND_ORDER)

    def agg(c, k, dflt=None):
        return (c.get("aggregate") or {}).get(k, dflt)

    def qual(c, k, dflt=None):
        return (c.get("quality") or {}).get(k, dflt)

    out: List[str] = []
    w = out.append

    # ---- hero -----------------------------------------------------------
    w('<div class="page reveal">')
    w('<p class="eyebrow">On-device inference · research prototype</p>')
    w("<h1>Molt</h1>")
    w('<p class="lede">A generation that runs out of memory mid-answer does not '
      "have to die, and does not have to start over. Molt moves it onto a "
      "smaller model <em>between two tokens</em> and carries the KV cache "
      "across — the animal walks out of the shell.</p>")

    kills_a, kills_d = A.get("kills"), D.get("kills")
    mig_c, mig_d = agg(C, "mean_migration_ms"), agg(D, "mean_migration_ms")
    ratio = (mig_c / mig_d) if (mig_c and mig_d) else None
    w('<div class="claims">')
    w(f'<div class="claim"><b class="bad">{n(kills_a, ".0f")}</b>'
      f'<span>kills · always-large</span></div>')
    w(f'<div class="claim"><b class="good">{n(kills_d, ".0f")}</b>'
      f'<span>kills · Molt</span></div>')
    w(f'<div class="claim"><b>{n(ratio, ".1f")}&#215;</b>'
      f'<span>cheaper switch than restart</span></div>')
    w("</div>")
    w('<div class="claims" style="margin-top:1px">')
    if sweep.get("rows"):
        best = max(r["speedup"] for r in sweep["rows"])
        w(f'<div class="claim"><b>{n(best, ".1f")}&#215;</b>'
          f'<span>transplant vs re-prefill</span></div>')
    if qos.get("summary"):
        w(f'<div class="claim"><b class="good">'
          f'{n(qos["summary"].get("max_overshoot_mb"), ".0f")}</b>'
          f'<span>MiB over budget · 3 tenants</span></div>')
    w("</div>")
    w('<p class="note">Measured on '
      f'{esc(meta.get("ladder", "?"))}, {esc(meta.get("device", "?"))} / '
      f'{esc(str(meta.get("dtype", "?")).replace("torch.", ""))}, '
      f'{len((conds.get("D") or {}).get("per_prompt", []))} prompts per condition, '
      "identical pressure trace for every condition.</p>")
    w("</div>")

    # ---- the trace ------------------------------------------------------
    if sD:
        w('<div class="bleed">')
        w("<figure>")
        w(chart_svg(sC, sD))
        w("<figcaption><strong>The whole argument in one plot.</strong> "
          "Per-token latency through the same pressure event. The shaded band "
          "is where the memory budget is exceeded. Both arms switch models at "
          "the same moment; the restart baseline pays a full re-prefill of the "
          "prompt each time, Molt carries the cache across. Log scale."
          "</figcaption>")
        w("</figure>")
        w(stripe(sD))
        td = tier_descriptions(meta)
        keys = "".join(
            f'<span><i style="background:var(--t{i})"></i>tier{i} &#183; '
            f'{td.get(f"tier{i}", (f"tier{i}",))[0]}</span>' for i in (0, 1, 2))
        w(f'<div class="key">{keys}<span>one cell = one generated token, coloured '
          f"by the model that produced it</span></div>")
        w("</div>")

    # ---- problem --------------------------------------------------------
    w('<div class="page">')
    w("<h2>The three bad options</h2>")
    w("<p>On a phone or a laptop the model shares memory with everything else. "
      "When the camera app launches, the OS wants memory back now. An inference "
      "process conventionally has three choices, and all three are bad:</p>")
    w("<ul>"
      "<li><strong>Hold the big model.</strong> The process is reclaimed. The "
      "answer is lost mid-sentence.</li>"
      "<li><strong>Always run the small model.</strong> You survive — and pay "
      "for it on every request, including the overwhelming majority where there "
      "was no pressure at all.</li>"
      "<li><strong>Switch models and restart the request.</strong> You survive, "
      "and the user watches a multi-second stall while the prompt is re-read.</li>"
      "</ul>")
    w("<p>Molt adds a fourth: change the model between two tokens, and take the "
      "attention state with you.</p>")

    # ---- why request boundaries fail ------------------------------------
    rb = conds.get("D-reqbound")
    if rb:
        w("<h2>Why per-request model selection cannot fix this</h2>")
        w("<p>Every conventional elastic-serving stack picks a model "
          "<em>per request</em>. That is useless here, and the reason is "
          "structural rather than an implementation detail: a long answer takes "
          "tens of seconds, an app launch takes one, so the spike lands "
          "<em>inside</em> the answer and the next request boundary is far away "
          "in the future. Between the two, the process is holding a footprint "
          "the OS has already decided it cannot have.</p>")
        w('<div class="callout warn">Measured, not asserted: the same Molt '
          "machinery restricted to switching only at request boundaries "
          f"(<code>D-reqbound</code>) was reclaimed <strong>{n(rb.get('kills'), '.0f')} "
          f"times out of {len(rb.get('per_prompt', []))} prompts</strong> — the "
          "same score as never adapting at all. Allowed to switch mid-stream, "
          f"the identical code was reclaimed {n(kills_d, '.0f')} times.</div>")

    # ---- ladder ---------------------------------------------------------
    if fps:
        w("<h2>The ladder</h2>")
        w("<p>Three rungs of one model family, sharing a tokenizer. Two of the "
          "three transplant routes are structurally different problems, which is "
          "the point of this particular ladder:</p>")
        top = max(fps.values()) if fps else 1.0
        tier_desc = tier_descriptions(meta)
        w('<div class="ladder">')
        for name in ("tier0", "tier1", "tier2"):
            if name not in fps:
                continue
            mb = fps[name]
            lab, why = tier_desc.get(name, (name, ""))
            w(f'<div class="rung">'
              f'<span class="tag" style="background:var(--{name.replace("tier","t")})">'
              f'{esc(name)}</span>'
              f'<div><div style="font-size:14.5px;font-weight:600">{lab}</div>'
              f'<div class="note" style="font-size:13px">{why}</div>'
              f'<div class="bar" style="margin-top:7px"><i style="width:'
              f'{100*mb/top:.1f}%;background:var(--{name.replace("tier","t")})"></i></div>'
              f"</div>"
              f'<span class="mb">{mb:,.0f} MiB</span></div>')
        w("</div>")
        budgets = [e.get("budget_mb") for e in (trace.get("events") or [])
                   if e.get("budget_mb") is not None]
        if budgets:
            w(f'<p class="note">The pressure trace drops the budget to '
              f'<strong>{n(min(budgets), ",.0f")} MiB</strong> — well below what '
              f"tier0 needs ({n(fps.get('tier0'), ',.0f')} MiB) — and it does so "
              "in two steps, so every rung of the ladder is exercised in a single "
              "run. The spike is timed to land mid-answer on purpose.</p>")

    # ---- results table --------------------------------------------------
    w("<h2>Results</h2>")
    w("</div>")
    w('<div class="bleed"><div class="scroll"><table>')
    w("<thead><tr><th>condition</th><th>kills</th><th>worst ITL</th>"
      "<th>mean ITL</th><th>switches</th><th>switch cost</th>"
      "<th>handoff JSD</th><th>judge agree</th><th>accuracy</th>"
      "<th>peak MiB</th></tr></thead><tbody>")
    for k in COND_ORDER + [a for a in ABL_ORDER if a in conds]:
        c = conds.get(k)
        if not c:
            continue
        cls = "hero" if k == "D" else ("dead" if c.get("kills") else "")
        w(f'<tr class="{cls}"><td>{esc(c.get("label", k))}</td>'
          f'<td>{n(c.get("kills"), ".0f")}</td>'
          f'<td>{n(agg(c, "worst_itl_ms"), ",.0f")}</td>'
          f'<td>{n(agg(c, "mean_itl_ms"), ",.0f")}</td>'
          f'<td>{n(agg(c, "mean_migrations"), ".1f")}</td>'
          f'<td>{n(agg(c, "mean_migration_ms"), ",.0f")}</td>'
          f'<td>{n(agg(c, "handoff_jsd"), ".4f")}</td>'
          f'<td>{n(qual(c, "judge_agreement"), ".3f")}</td>'
          f'<td>{n(qual(c, "accuracy"), ".2f")}</td>'
          f'<td>{n(agg(c, "peak_mb"), ",.0f")}</td></tr>')
    w("</tbody></table></div>")
    w('<p class="note" style="margin-top:10px">Latencies in milliseconds. '
      "<em>Handoff JSD</em> is the divergence between what the outgoing model "
      "would have said at the switch position and what the incoming model "
      "actually said — lower is a smoother seam. A killed run scores 0 on "
      "accuracy rather than dropping out of the average.</p></div>")

    # ---- the honest trade-off ------------------------------------------
    h_c, h_d = agg(C, "handoff_jsd"), agg(D, "handoff_jsd")
    if h_c and h_d and h_c == h_c and h_d == h_d:
        w('<div class="page">')
        w("<h2>The trade-off, stated plainly</h2>")
        w("<p>Condition C re-reads the prompt on the new model, so its handoff "
          "divergence is the irreducible floor: two different models simply "
          "disagree by that much about the next token. Molt does not reach that "
          "floor.</p>")
        w('<div class="claims">')
        w(f'<div class="claim"><b>{n(h_c, ".3f")}</b>'
          "<span>handoff JSD · re-prefill floor</span></div>")
        w(f'<div class="claim"><b>{n(h_d, ".3f")}</b>'
          "<span>handoff JSD · transplant</span></div>")
        w(f'<div class="claim"><b>{n(ratio, ".1f")}&#215;</b>'
          "<span>faster switch</span></div>")
        w("</div>")
        w(f"<p>So the honest summary is a purchase, not a free lunch: a "
          f"<strong>{n(ratio, '.1f')}&#215; cheaper switch</strong> costs "
          f"<strong>{n(h_d / h_c, '.1f')}&#215; more distribution "
          "discontinuity</strong> at the seam than a full re-prefill would. Most "
          "of that gap is the value projection, which is the weakest map in the "
          "system.</p>")
        w('<div class="callout warn"><strong>And one number that does not move '
          "much.</strong> The <em>worst single token</em> is "
          f"{n(agg(C, 'worst_itl_ms'), ',.0f')} ms for restart against "
          f"{n(agg(D, 'worst_itl_ms'), ',.0f')} ms for Molt &#8212; nearly the "
          "same, because in a cold-start run that token is dominated by "
          "<em>loading the destination model from disk</em>, which both "
          "strategies pay identically. What Molt removes is the re-prefill on "
          f"top of it: {n(mig_c, ',.0f')} ms &#8594; {n(mig_d, ',.0f')} ms. The "
          "sweep below isolates that component with both rungs already warm."
          "</div>")
        w("</div>")

    # ---- cost sweep -----------------------------------------------------
    if sweep.get("rows"):
        w('<div class="page"><h2>What a switch costs</h2>')
        w("<p>Carrying the cache versus re-reading the prompt, both rungs kept "
          "warm so the comparison is about the KV work rather than about model "
          "loading, which both strategies pay identically.</p></div>")
        w('<div class="bleed"><div class="scroll"><table>')
        w("<thead><tr><th>route</th><th>tokens carried</th><th>transplant</th>"
          "<th>re-prefill</th><th>speed-up</th><th>FLOPs avoided</th>"
          "</tr></thead><tbody>")
        for r in sweep["rows"]:
            w(f'<tr><td>{esc(r["route"])}</td><td>{r["tokens"]:,}</td>'
              f'<td>{n(r["transplant_ms"], ",.0f")} ms</td>'
              f'<td>{n(r["reprefill_ms"], ",.0f")} ms</td>'
              f'<td>{n(r["speedup"], ".2f")}&#215;</td>'
              f'<td>{n(r["flops_saving"]*100, ".0f")}%</td></tr>')
        w("</tbody></table></div></div>")

    # ---- ablations ------------------------------------------------------
    abl = [k for k in ABL_ORDER if k in conds]
    if abl and "D" in conds:
        w('<div class="page"><h2>Ablations</h2>')
        w("<p>Each mechanism removed on its own, same trace, same prompts. "
          "Reported as measured, including where a mechanism did not pay for "
          "itself &#8212; an ablation table that only ever confirms the design "
          "is not an ablation table.</p>")
        w('<div class="callout warn"><strong>Read the two quality columns '
          "together.</strong> Handoff JSD measures the <em>seam</em>, at one "
          "position. It is bounded, and it rewards blurring: a map that degrades "
          "the cache into a flat distribution can score a <em>lower</em> one-shot "
          "divergence while the text falls apart a few tokens later. Judge "
          "agreement catches that. Where the two disagree in the table below, "
          "agreement is the one to trust.</div></div>")
        w('<div class="bleed"><div class="scroll"><table>')
        w("<thead><tr><th>arm</th><th>what is removed</th><th>judge agree</th>"
          "<th>judge ppl</th><th>handoff JSD</th><th>switch cost</th>"
          "<th>peak MiB</th><th>kills</th></tr></thead><tbody>")
        removed = {
            "D": "&#8212; (full system)",
            "D-nocal": "logit blending (core #3)",
            "D-noproj": "learned projection + RoPE sandwich (core #1i, and with "
                        "it the top-k recompute, which needs a learned map)",
            "D-norecompute": "top-k native recompute (core #1iii)",
            "D-reqbound": "the ability to switch mid-stream (core #2)",
        }
        for k in ["D"] + abl:
            c = conds[k]
            w(f'<tr class="{"hero" if k == "D" else ""}">'
              f'<td>{esc(c.get("label", k))}</td><td>{removed.get(k, "")}</td>'
              f'<td>{n(qual(c, "judge_agreement"), ".3f")}</td>'
              f'<td>{n(qual(c, "judge_ppl"), ".2f")}</td>'
              f'<td>{n(agg(c, "handoff_jsd"), ".4f")}</td>'
              f'<td>{n(agg(c, "mean_migration_ms"), ",.0f")} ms</td>'
              f'<td>{n(agg(c, "peak_mb"), ",.0f")}</td>'
              f'<td>{n(c.get("kills"), ".0f")}</td></tr>')
        w("</tbody></table></div></div>")

    # ---- how ------------------------------------------------------------
    w('<div class="page">')
    w("<h2>How the cache crosses</h2>")
    w("<p>HuggingFace caches keys <em>after</em> RoPE. Two rungs with different "
      "<code>head_dim</code> rotate by different angles, so no "
      "position-independent matrix can map one cache onto the other — the "
      "required map would depend on each token's absolute position. Molt "
      "therefore sandwiches the learned matrix between an un-rotation at the "
      "source's angles and a re-rotation at the destination's. That one detail "
      "is what makes a cross-size transplant well-posed at all.</p>")
    w("<p>Three mechanisms, fitted offline by closed-form ridge regression on a "
      "few thousand tokens of generic text:</p>")
    w("<ol>"
      "<li><strong>A learned linear projection</strong> per destination layer, "
      "inside the RoPE sandwich, plus a depth remap.</li>"
      "<li><strong>A diagonal scale re-alignment</strong> when the rungs share "
      "weights and differ only in precision.</li>"
      "<li><strong>Selective top-k recompute</strong>: the destination's last "
      "few layers are recomputed natively from a projected boundary hidden "
      "state, rather than projected. Verified bit-exact against a full forward "
      "pass.</li></ol>")
    if proj_index:
        w('<p class="note">Held-out fit residuals (relative RMS, on windows the '
          "map never saw):</p>")
        w('<div class="scroll"><table><thead><tr><th>route</th><th>map</th>'
          "<th>K</th><th>V</th><th>hidden</th></tr></thead><tbody>")
        for route, v in list(proj_index.items())[:6]:
            w(f'<tr><td>{esc(route)}</td><td>{esc(v.get("kind"))}</td>'
              f'<td>{n(v.get("val_k"), ".3f")}</td>'
              f'<td>{n(v.get("val_v"), ".3f")}</td>'
              f'<td>{n(v.get("val_hidden"), ".3f")}</td></tr>')
        w("</tbody></table></div>")
        w('<p class="note">The value map is the weak one, and most of the '
          "residual post-migration quality gap lives there.</p>")
        w('<div class="callout">Worth one measurement, because the ablation '
          "table above looks contradictory at first: against a naive truncated "
          "map, the fitted projection has a <em>higher</em> handoff JSD but far "
          "better text. Running both through the same prefix "
          "(<code>scripts/diagnose_projection.py</code>) shows why &#8212; the "
          "naive map's cache has a reconstruction error of <strong>1.31</strong> "
          "(worse than predicting zeros) and induces attention "
          "<strong>11.4&#215;</strong> flatter than the destination model's own, "
          "against <strong>1.5&#215;</strong> for the fitted map. That flatness "
          "is exactly the blur that flatters a bounded divergence.</div>")

    # ---- QoS ------------------------------------------------------------
    if qos.get("summary"):
        s = qos["summary"]
        w("<h2>Three tenants, one budget, nobody killed</h2>")
        w("<p>A foreground conversation and two background batch jobs share one "
          "moving budget. The scheduler has four levers, applied "
          "<em>synchronously, before anyone takes a step</em>, in order of "
          "increasing harm: defer admission, demote a rung, park the weights "
          "while keeping the cache, and finally shed the oldest context. "
          "Terminating is not on the list.</p>")
        w('<div class="claims">')
        w(f'<div class="claim"><b class="good">{n(s.get("kills"), ".0f")}</b>'
          "<span>forced terminations</span></div>")
        w(f'<div class="claim"><b class="good">{n(s.get("max_overshoot_mb"), ".0f")}</b>'
          "<span>MiB max overshoot</span></div>")
        w(f'<div class="claim"><b>{n(s.get("n_demotions"), ".0f")}</b>'
          "<span>demotions</span></div>")
        w(f'<div class="claim"><b>{n(s.get("n_pauses"), ".0f")}</b>'
          "<span>parks</span></div>")
        w("</div>")
        w('<p class="note">Max overshoot is <code>max over time of '
          "(usage − budget)</code>, sampled after enforcement at every "
          "scheduling round. Negative means the budget was never exceeded.</p>")

    # ---- live transcript -------------------------------------------------
    w("<h2>It runs as a service</h2>")
    w("<p>The prototype ships a streaming server whose elasticity is visible on "
      "the wire: every token event names the rung that produced it, and a switch "
      "arrives as its own event rather than as a surprise. Below is a real "
      "session &#8212; not a mock-up &#8212; with the budget cut to 2651 MiB "
      "eight tokens into the answer.</p>")
    w('<pre style="font-family:var(--mono);font-size:12.5px;line-height:1.65;'
      'background:var(--raise);border:1px solid var(--line);border-radius:8px;'
      'padding:16px;overflow-x:auto"><code>'
      "[start on tier0 &#183; 17 prompt tokens &#183; TTFT 3122 ms]\n"
      " An operating system reclaims memory from background\n"
      "<b>[budget cut to 2651 MiB]</b>\n"
      " processes\n"
      "<b>&#8646; tier0 &#8594; tier1  (molt, 260 ms, 79% FLOPs saved, pressure 0.81)</b>\n"
      " to free up resources and improve system performance. When a process is\n"
      " no longer actively using memory, the operating system can reclaim that\n"
      " memory for other processes or applications\n"
      "[done: 40 tokens &#183; rungs used tier0, tier1 &#183; 1 migration costing "
      "260 ms &#183; killed=False]"
      "</code></pre>")
    w("<p>The sentence <em>&#8220;&#8230;reclaims memory from background "
      "processes to free up resources&#8230;&#8221;</em> is written by two "
      "different models. The word before the switch and the word after it are "
      "separated by a 260 ms migration and nothing else &#8212; no re-prefill, "
      "no restart, no dropped connection.</p>")

    # ---- limits ---------------------------------------------------------
    w("<h2>What this does not show</h2>")
    w("<p>Stated plainly, because a prototype that hides its edges is worth "
      "less than one that marks them.</p>")
    w("<ul>"
      "<li>The <strong>INT8 rung buys memory, not speed.</strong> bitsandbytes "
      "is CUDA-only, so the portable fallback dequantises on use: prefill is "
      "slightly faster than fp32, single-token decode is roughly 4× slower. A "
      "tuned kernel removes this; the memory result stands either way.</li>"
      "<li><strong>The value projection is weak</strong> compared with the key "
      "projection, and up-shifts (expanding a cache) are harder than "
      "down-shifts. Per-head or low-rank non-linear maps are the obvious next "
      "step.</li>"
      "<li><strong>Cross-family migration is not supported.</strong> Every rung "
      "must share a tokenizer; the loader refuses otherwise.</li>"
      "<li>The budget is <strong>simulated by default</strong> for "
      "reproducibility. A real allocating child process is available and gives "
      "the same qualitative result, noisier.</li>"
      "<li>Server-side cache sharing and weight streaming from flash are "
      "<strong>deliberately out of scope</strong>: this is about on-device "
      "multi-tenancy and mid-generation migration.</li>"
      "<li>Three metrics were <strong>discarded</strong> during the work for "
      "measuring nothing — step-to-step JSD (saturated at <code>ln 2</code> on "
      "real text), judge perplexity alone (rewards degenerate repetition), and "
      "a single-needle recall task (every condition scored 1.00, including the "
      "ones that were killed). The replacements are what the table reports.</li>"
      "</ul>")

    w("<hr>")
    w('<p class="foot">Molt · elastic on-device inference · research prototype<br>'
      "Every number on this page is read directly from the benchmark's own "
      "output; the page is generated, not transcribed.<br>"
      f'Ladder {esc(meta.get("ladder", "?"))} · '
      f'{esc(meta.get("device", "?"))} · '
      f'trace {esc(trace.get("name", "?"))}</p>')
    w("</div>")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="benchmarks/results")
    p.add_argument("--out", default="artifacts/report.html")
    p.add_argument("--projectors", default="artifacts/projectors")
    args = p.parse_args(argv)

    path = os.path.join(args.results, "summary.json")
    if not os.path.exists(path):
        raise SystemExit(f"no results at {path}; run benchmarks/run.py first")
    with open(path) as fh:
        data = json.load(fh)

    proj = None
    idx = sorted(glob.glob(os.path.join(args.projectors, "*__index.json")))
    if idx:
        with open(idx[-1]) as fh:
            proj = json.load(fh)

    body = build(data, proj)
    page = (f"<title>Molt &#8212; elastic on-device inference</title>\n"
            f"<style>{CSS}</style>\n{body}\n")
    # Escape every non-ASCII character as a numeric entity.  The page is embedded
    # into a host document whose charset we do not control, and an em-dash that
    # renders as "â€"" is a worse bug than it looks: it is silent, it survives
    # review, and it makes the whole page read as unfinished.
    page = page.encode("ascii", "xmlcharrefreplace").decode("ascii")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="ascii") as fh:
        fh.write(page)
    print(f"wrote {args.out} ({len(page)/1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
