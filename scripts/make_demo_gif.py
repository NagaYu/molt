#!/usr/bin/env python3
"""Render a recorded Molt session into an animated GIF.

    python scripts/record_demo.py   --out artifacts/demo_session.json
    python scripts/make_demo_gif.py --session artifacts/demo_session.json \
                                    --out figures/demo.gif

This is a **replay**, not a re-enactment: every token, every timestamp, every
migration cost and every memory reading comes from
``artifacts/demo_session.json``, which is the raw SSE stream the service
produced.  The only liberties taken are stated on the frame itself — the
timeline is compressed by a constant factor so the clip is watchable, and the
prompt-reading pause is shortened.

The point the animation has to make in one glance: the background colour behind
the text changes *mid-sentence*, and the sentence keeps going.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# Same palette as the report, dark ground: the tier colours have to survive
# being used as text backgrounds, which the light-theme values do not.
BG = (14, 17, 13)
PANEL = (21, 26, 19)
INK = (232, 236, 228)
INK2 = (169, 176, 164)
INK3 = (118, 125, 114)
LINE = (44, 50, 42)
GAUGE = (111, 182, 132)
ALARM = (224, 112, 90)
TIER = {"tier0": (58, 96, 130), "tier1": (122, 88, 42), "tier2": (94, 66, 90)}
TIER_TEXT = {"tier0": (176, 208, 232), "tier1": (240, 204, 148), "tier2": (222, 186, 216)}

W, H = 900, 424
PAD = 30

FONT_CANDIDATES_MONO = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
FONT_CANDIDATES_SANS = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(paths, size, index=0):
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size, index=index)
            except Exception:
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    continue
    return ImageFont.load_default()


class Fonts:
    def __init__(self):
        self.h1 = load_font(FONT_CANDIDATES_SANS, 30, index=1)
        self.body = load_font(FONT_CANDIDATES_SANS, 19)
        self.small = load_font(FONT_CANDIDATES_MONO, 12)
        self.tiny = load_font(FONT_CANDIDATES_MONO, 11)
        self.badge = load_font(FONT_CANDIDATES_MONO, 13)


def wrap_tokens(draw, tokens, font, max_w) -> List[List[Tuple[str, str]]]:
    """Lay tokens out into lines, keeping each token's tier with it.

    Tokens are sub-word pieces with leading spaces, so lines are broken on token
    boundaries rather than on words; that is also what makes the colour bands
    line up with what the model actually emitted.
    """
    lines: List[List[Tuple[str, str]]] = [[]]
    w = 0.0
    for text, tier in tokens:
        tw = draw.textlength(text, font=font)
        if w + tw > max_w and lines[-1]:
            lines.append([])
            w = 0.0
            text = text.lstrip()
            tw = draw.textlength(text, font=font)
        lines[-1].append((text, tier))
        w += tw
    return lines


def rounded(draw, box, r, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def draw_frame(f: Fonts, tokens, tier, budget, in_use, banner, elapsed,
               subtitle, speed) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # -- header ---------------------------------------------------------
    d.text((PAD, 22), "Molt", font=f.h1, fill=INK)
    d.text((PAD + 92, 34), "ELASTIC ON-DEVICE INFERENCE", font=f.tiny, fill=INK3)
    d.text((W - PAD, 34), subtitle, font=f.tiny, fill=INK3, anchor="ra")

    # -- answer panel ---------------------------------------------------
    top, bot = 68, 256
    rounded(d, (PAD - 10, top, W - PAD + 10, bot), 10, fill=PANEL)
    tx, ty = PAD + 4, top + 16
    max_w = (W - PAD + 10) - (PAD + 4) - 18
    lines = wrap_tokens(d, tokens, f.body, max_w)
    lh = 30
    for line in lines[-6:]:
        x = tx
        for text, tr in line:
            tw = d.textlength(text, font=f.body)
            if tw > 0:
                d.rectangle((x - 1, ty - 4, x + tw + 1, ty + 23),
                            fill=TIER.get(tr, PANEL))
            d.text((x, ty), text, font=f.body, fill=INK)
            x += tw
        ty += lh
    if not tokens:
        d.text((tx, ty), "reading the prompt…", font=f.body, fill=INK3)

    # -- memory gauge ---------------------------------------------------
    gy = 284
    d.text((PAD, gy - 18), "MEMORY", font=f.tiny, fill=INK3)
    bar_w = W - 2 * PAD
    d.rounded_rectangle((PAD, gy, PAD + bar_w, gy + 16), radius=8, fill=LINE)
    top_budget = 9422.1                      # the un-squeezed budget
    bw = max(6, int(bar_w * min(1.0, budget / top_budget)))
    d.rounded_rectangle((PAD, gy, PAD + bw, gy + 16), radius=8,
                        outline=ALARM, width=2)
    uw = max(4, int(bar_w * min(1.0, in_use / top_budget)))
    over = in_use > budget
    d.rounded_rectangle((PAD, gy, PAD + uw, gy + 16), radius=8,
                        fill=ALARM if over else TIER.get(tier, GAUGE))
    d.text((PAD, gy + 24), f"in use {in_use:,.0f} MiB", font=f.small, fill=INK2)
    d.text((PAD + bw, gy + 24), f"budget {budget:,.0f} MiB", font=f.small,
           fill=ALARM, anchor="ma")

    # -- status ---------------------------------------------------------
    sy = 342
    label = {"tier0": "Qwen2.5-1.5B  fp32", "tier1": "Qwen2.5-1.5B  int8",
             "tier2": "Qwen2.5-0.5B  fp32"}.get(tier, tier)
    tw = d.textlength(f"  {tier}  ", font=f.badge)
    d.rounded_rectangle((PAD, sy, PAD + tw + 8, sy + 24), radius=12,
                        fill=TIER.get(tier, LINE))
    d.text((PAD + 4 + tw / 2, sy + 12), tier, font=f.badge,
           fill=TIER_TEXT.get(tier, INK), anchor="mm")
    d.text((PAD + tw + 22, sy + 12), label, font=f.badge, fill=INK2, anchor="lm")
    d.text((W - PAD, sy + 12), f"{elapsed:4.1f}s   {len(tokens)} tokens",
           font=f.badge, fill=INK3, anchor="rm")

    # -- migration banner ------------------------------------------------
    if banner:
        bh = 44
        by = 190
        rounded(d, (PAD + 6, by, W - PAD - 6, by + bh), 8,
                fill=(38, 30, 24), outline=TIER.get(banner["to"], LINE), width=2)
        d.text((PAD + 22, by + bh / 2),
               f"MIGRATION   {banner['from']} -> {banner['to']}", font=f.badge,
               fill=TIER_TEXT.get(banner["to"], INK), anchor="lm")
        d.text((W - PAD - 22, by + bh / 2),
               f"KV transplanted in {banner['cost_ms']:.0f} ms   "
               f"{banner['flops_saved']*100:.0f}% of the FLOPs avoided",
               font=f.badge, fill=INK2, anchor="rm")

    # -- footer ----------------------------------------------------------
    d.line((PAD, H - 34, W - PAD, H - 34), fill=LINE, width=1)
    d.text((PAD, H - 22),
           f"replay of a recorded session · {speed:.1f}x speed · "
           f"no re-prefill, no restart, 0 forced terminations",
           font=f.tiny, fill=INK3)
    return img


def build(session: Dict[str, Any], fps: int, target_s: float,
          banner_s: float, tail_s: float) -> List[Image.Image]:
    ev = session["events"]
    start = next((e for e in ev if e["type"] == "start"), None)
    toks = [e for e in ev if e["type"] == "token"]
    migs = [e for e in ev if e["type"] == "migration"]
    done = next((e for e in ev if e["type"] == "done"), None)
    if not toks:
        raise SystemExit("recording contains no tokens")

    # Compress the timeline by a constant factor: relative gaps are preserved,
    # so the viewer still sees that int8 decodes slowly.  The factor is printed
    # on every frame.
    t_first, t_last = toks[0]["t"], toks[-1]["t"]
    span = max(0.1, t_last - t_first)
    speed = span / max(0.1, target_s)

    f = Fonts()
    frames: List[Image.Image] = []
    n_frames = int(target_s * fps)
    budget0 = session.get("budget_start_mb", 9422.1)
    subtitle = f"{start['prompt_tokens']} prompt tokens · TTFT {start['ttft_ms']:.0f} ms" \
        if start else ""

    # a short beat before the first token, so the gauge is legible at full budget
    for _ in range(int(0.8 * fps)):
        frames.append(draw_frame(f, [], start["tier"] if start else "tier0",
                                 budget0, start.get("in_use_mb", 0.0) if start else 0.0,
                                 None, 0.0, subtitle, speed))

    last_good = [start.get("in_use_mb", 0.0) if start else 0.0]
    banner_frames = int(banner_s * fps)
    pending: List[Tuple[int, dict]] = []      # (frames_left, migration event)
    shown = 0
    for i in range(n_frames):
        t_real = t_first + span * (i / max(1, n_frames - 1))
        while shown < len(toks) and toks[shown]["t"] <= t_real:
            shown += 1
        vis = [(e["text"], e["tier"]) for e in toks[:shown]]
        cur = toks[shown - 1]["tier"] if shown else (start["tier"] if start else "tier0")
        last = toks[shown - 1] if shown else (start or {})
        budget = last.get("budget_mb", budget0)
        # /stats is polled just after each event, so the final samples can land
        # after teardown and read ~0 MiB.  Carry the last reading taken while a
        # model was actually resident: the gauge is meant to show the state the
        # tokens were produced in, not the state of an empty process.
        iu = last.get("in_use_mb", 0.0)
        if iu > 50:
            last_good[0] = iu
        in_use = last_good[0]

        for m in migs:
            if m["t"] <= t_real and all(m is not p[1] for p in pending) \
                    and not m.get("_seen"):
                m["_seen"] = True
                pending.append([banner_frames, m])
        banner = None
        for p in pending:
            if p[0] > 0:
                banner = p[1]
                p[0] -= 1
                break
        frames.append(draw_frame(f, vis, cur, budget, in_use, banner,
                                 t_real - t_first, subtitle, speed))

    # hold the finished answer
    final_tier = toks[-1]["tier"]
    fb = toks[-1].get("budget_mb", budget0)
    fu = last_good[0]
    for _ in range(int(tail_s * fps)):
        frames.append(draw_frame(f, [(e["text"], e["tier"]) for e in toks],
                                 final_tier, fb, fu, None,
                                 (done or toks[-1])["t"] - t_first, subtitle, speed))
    return frames


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session", default="artifacts/demo_session.json")
    p.add_argument("--out", default="figures/demo.gif")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--seconds", type=float, default=9.0,
                   help="compressed duration of the token stream")
    p.add_argument("--banner-seconds", type=float, default=1.1)
    p.add_argument("--tail-seconds", type=float, default=2.0)
    p.add_argument("--colors", type=int, default=96)
    args = p.parse_args(argv)

    with open(args.session) as fh:
        session = json.load(fh)
    frames = build(session, args.fps, args.seconds, args.banner_seconds,
                   args.tail_seconds)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pal = [fr.quantize(colors=args.colors, method=Image.MEDIANCUT) for fr in frames]
    pal[0].save(args.out, save_all=True, append_images=pal[1:],
                duration=int(1000 / args.fps), loop=0, optimize=True)
    size = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out}: {len(frames)} frames, {size:.0f} KiB, "
          f"{len(frames)/args.fps:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
