#!/usr/bin/env python3
"""Record a real streaming session (with a mid-answer squeeze) to JSON.

    python -m molt.service --ladder qwen --port 8765 &
    python scripts/record_demo.py --out artifacts/demo_session.json

The recording is the *actual* SSE event stream, timestamps included, so the
animation built from it in ``scripts/make_demo_gif.py`` is a replay rather than
a re-enactment.  That distinction matters for a project whose whole argument is
"here is what we measured".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.loads(r.read())


def post(base, path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8765")
    p.add_argument("--out", default="artifacts/demo_session.json")
    p.add_argument("--prompt", default="Explain in a few sentences why an operating "
                   "system reclaims memory from background processes.")
    p.add_argument("--max-new-tokens", type=int, default=44)
    p.add_argument("--squeeze-at", type=int, default=9)
    p.add_argument("--squeeze-to", type=float, default=0.42)
    args = p.parse_args(argv)

    health = get(args.base, "/health")
    stats0 = get(args.base, "/stats")
    budget0 = stats0["budget_mb"]

    def squeezer():
        base_n = get(args.base, "/stats")["counters"]["tokens"]
        while True:
            s = get(args.base, "/stats")
            if s["counters"]["tokens"] - base_n >= args.squeeze_at:
                post(args.base, "/admin/pressure",
                     {"budget_mb": max(600.0, s["in_use_mb"] * args.squeeze_to)})
                return
            time.sleep(0.03)

    threading.Thread(target=squeezer, daemon=True).start()

    req = urllib.request.Request(
        args.base + "/v1/generate/stream",
        data=json.dumps(dict(prompt=args.prompt,
                             max_new_tokens=args.max_new_tokens)).encode(),
        headers={"Content-Type": "application/json"})

    events, t0 = [], time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            ev["t"] = round(time.perf_counter() - t0, 4)
            try:
                s = get(args.base, "/stats")
                ev["budget_mb"] = round(s["budget_mb"], 1)
                ev["in_use_mb"] = round(s["in_use_mb"], 1)
            except Exception:
                pass
            events.append(ev)
            if ev.get("type") == "end":
                break

    post(args.base, "/admin/pressure", {"budget_mb": budget0})
    blob = dict(health=health, prompt=args.prompt, budget_start_mb=budget0,
                events=events)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(blob, fh, indent=1)
    kinds = {}
    for e in events:
        kinds[e["type"]] = kinds.get(e["type"], 0) + 1
    print(f"wrote {args.out}: {kinds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
