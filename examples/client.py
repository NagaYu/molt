#!/usr/bin/env python3
"""Streaming client for the Molt service — stdlib only, no dependencies.

    # terminal 1
    python -m molt.service --ladder qwen --port 8000

    # terminal 2: stream an answer and colour each token by the rung that made it
    python examples/client.py "Explain why an OS reclaims memory."

    # squeeze the device to 45% of what is currently in use, mid-answer
    python examples/client.py "Write a long paragraph about a workshop." --squeeze-at 20

    # replay a whole pressure trace underneath the request
    python examples/client.py "..." --trace spike_mid_answer

What this demonstrates is narrow and specific: the token stream does not stop,
does not restart, and tells you — in band — which model produced each token and
what the switch cost.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request

TIER_COLOR = {"tier0": "\033[48;5;189m", "tier1": "\033[48;5;223m",
              "tier2": "\033[48;5;225m"}
DIM, RESET, BOLD = "\033[2m", "\033[0m", "\033[1m"


def post(base: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.loads(r.read())


def stream(base: str, payload: dict, colour: bool):
    req = urllib.request.Request(base + "/v1/generate/stream",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    n_tokens = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            k = ev.get("type")
            if k == "start":
                print(f"{DIM}[start on {ev['tier']} · {ev['prompt_tokens']} prompt tokens "
                      f"· TTFT {ev['ttft_ms']:.0f} ms]{RESET}")
            elif k == "queued":
                print(f"{DIM}[queued: {ev['reason']} — {ev['in_use_mb']:.0f} MiB in use, "
                      f"{ev['budget_mb']:.0f} MiB budget]{RESET}")
            elif k == "token":
                c = TIER_COLOR.get(ev["tier"], "") if colour else ""
                sys.stdout.write(f"{c}{ev['text']}{RESET if c else ''}")
                sys.stdout.flush()
                n_tokens += 1
            elif k == "migration":
                sys.stdout.write(
                    f"\n{BOLD}⇄ {ev['from']} → {ev['to']}{RESET}{DIM} "
                    f"({ev['method']}, {ev['cost_ms']:.0f} ms, "
                    f"{(ev.get('flops_saved') or 0)*100:.0f}% FLOPs saved, "
                    f"pressure {ev['pressure']:.2f}){RESET}\n")
                sys.stdout.flush()
            elif k == "done":
                print(f"\n{DIM}[done: {ev['tokens']} tokens in {ev['total_ms']:.0f} ms · "
                      f"rungs used {', '.join(ev['tiers'])} · "
                      f"{ev['migrations']} migration(s) costing {ev['migration_ms']:.0f} ms · "
                      f"killed={ev['killed']}]{RESET}")
            elif k == "error":
                print(f"\n\033[31m[error] {ev['error']}{RESET}", file=sys.stderr)
            elif k == "end":
                break
    return n_tokens


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Molt streaming client")
    p.add_argument("prompt", nargs="?", default="Explain why an operating system "
                   "might reclaim memory from a background process.")
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--squeeze-at", type=int, default=0,
                   help="after N tokens, cut the budget to --squeeze-to of current usage")
    p.add_argument("--squeeze-to", type=float, default=0.45)
    p.add_argument("--trace", default=None,
                   help="start a named pressure trace before the request")
    p.add_argument("--no-colour", action="store_true")
    args = p.parse_args(argv)

    try:
        h = get(args.base, "/health")
    except Exception as exc:
        print(f"cannot reach {args.base}: {exc}\n"
              f"start the server with:  python -m molt.service --ladder qwen",
              file=sys.stderr)
        return 2
    print(f"{DIM}ladder {h['ladder']} on {h['device']} — "
          f"{' → '.join(t['label'] for t in h['tiers'])}{RESET}")

    if args.trace:
        r = post(args.base, "/admin/pressure", {"trace": args.trace})
        print(f"{DIM}trace {args.trace}: {r.get('description', '')}{RESET}")

    if args.squeeze_at > 0:
        def squeezer():
            # poll the server's own token counter so the squeeze lands *inside*
            # the answer regardless of how fast this machine decodes
            base_n = get(args.base, "/stats")["counters"]["tokens"]
            while True:
                s = get(args.base, "/stats")
                if s["counters"]["tokens"] - base_n >= args.squeeze_at:
                    target = max(600.0, s["in_use_mb"] * args.squeeze_to)
                    post(args.base, "/admin/pressure", {"budget_mb": target})
                    print(f"\n{DIM}[budget cut to {target:.0f} MiB]{RESET}")
                    return
                time.sleep(0.05)
        threading.Thread(target=squeezer, daemon=True).start()

    t0 = time.time()
    stream(args.base, dict(prompt=args.prompt, max_new_tokens=args.max_new_tokens),
           colour=not args.no_colour)
    s = get(args.base, "/stats")
    print(f"{DIM}[server: {s['counters']['migrations']} migrations, "
          f"{s['counters']['kills']} forced terminations, "
          f"peak {s['peak_mb']:.0f} MiB, wall {time.time()-t0:.1f}s]{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
