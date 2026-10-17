"""Record the measured platform facts the paper cites (appendix A/B) as JSON.

    python -m bench.facts            # MLX freeing facts + energy instrument facts
    python -m pytest mlx/tests/test_peak_footprint.py --import-mode=importlib
                                     # writes bench/results/facts/peak_test.json

Nothing in the paper's appendix is typed by hand; it is read from these files.
"""

from __future__ import annotations

import json
import time

from bench._env import RESULTS, machine

MiB = 2 ** 20


def mlx_facts() -> dict:
    import mlx.core as mx
    from molt_mlx.footprint import footprint_bytes, release_and_wait, rss_bytes

    base = footprint_bytes()
    a = mx.ones((64, 1024, 1024), dtype=mx.float32)      # 256 MiB
    mx.eval(a)
    alloc_fp, alloc_rss = (footprint_bytes() - base) / MiB, rss_bytes() / MiB
    del a
    after_del = (footprint_bytes() - base) / MiB
    mx.clear_cache()
    time.sleep(0.3)
    after_clear = (footprint_bytes() - base) / MiB
    rep = release_and_wait(base + 64 * MiB, timeout_s=3.0)
    return dict(alloc_fp=alloc_fp, alloc_rss=alloc_rss, after_del=after_del,
                after_clear=after_clear, release_ms=rep.waited_ms,
                released=rep.reached_target)


def energy_facts() -> dict:
    import mlx.core as mx
    from molt_mlx.energy import EnergyMeter

    m = EnergyMeter()
    base = m.baseline(0.5)
    a = mx.random.normal((2048, 2048)); mx.eval(a)
    m.start()
    t = time.time()
    while time.time() - t < 2.0:
        b = a
        for _ in range(10):
            b = b @ a
        mx.eval(b)
    r = m.stop()
    m.start()
    t = time.time()
    x = 0
    while time.time() - t < 1.5:
        x += sum(i * i for i in range(20000))
    c = m.stop()
    return dict(gpu_baseline_w=base, gpu_loop_gpu_w=(r.gpu_j_raw or 0) / r.seconds,
                gpu_loop_proc_w=r.cpu_j / r.seconds, cpu_loop_proc_w=c.cpu_j / c.seconds)


def identity_check(tier: str = "0.5b", n_cont: int = 8) -> dict:
    """The transplant path itself is lossless: map a real model's cache onto
    *itself* with an identity mapper (RoPE stripped and re-applied, fp16 →
    fp32 → fp16) at the same switch point as experiment Q's needle items, and
    compare the greedy continuation with native decoding."""
    import json as _json
    import mlx.core as mx
    from bench._env import ROOT
    from molt_mlx import rope as R, tiers
    from molt_mlx.generate import Session, prefill
    from molt_mlx.mappers import Geometry, KVMapper, LayerMap
    from molt_mlx.migrate import transplant

    m, tok = tiers.load_model(tier), tiers.load_tokenizer()
    g, spec = Geometry.from_model(m), R.RopeSpec.from_model(m)
    eye = mx.eye(g.width)
    ident = KVMapper(g, g, spec, spec, "identity", 1,
                     [LayerMap([l], eye, mx.zeros((g.width,)), eye, mx.zeros((g.width,)))
                      for l in range(g.n_layers)])
    items = [p for p in _json.loads((ROOT / "bench" / "prompts.json").read_text()) if p["kind"] == "needle"]
    same, worst = 0, 0.0
    for it in items:
        ids = tok.apply_chat_template([{"role": "user", "content": it["text"]}], add_generation_prompt=True)
        s = Session(m, tier); s.start(ids)
        while len(s.emitted) < it["pressure_at"]:
            s.step()
        ref, c_ref = prefill(m, s.tokens + [s.pending])
        lg, c_id, _ = transplant(s.kv_pairs(), s.tokens, s.pending, m, ident, 0, 0)
        worst = max(worst, mx.abs(lg.astype(mx.float32) - ref.astype(mx.float32)).max().item())

        def cont(logits, cache):
            out, nxt = [], int(mx.argmax(logits, -1).item())
            for _ in range(n_cont):
                out.append(nxt)
                nxt = int(mx.argmax(m(mx.array([[nxt]]), cache=cache)[:, -1], -1).item())
            return out
        same += cont(ref, c_ref) == cont(lg, c_id)
    return dict(tier=tier, items=len(items), identical_continuations=same,
                max_logit_diff=worst, continuation_tokens=n_cont)


def main():
    out = RESULTS / "facts"
    out.mkdir(parents=True, exist_ok=True)
    runs = [mlx_facts() for _ in range(5)]          # release time varies run to run
    mlx = dict(runs[0])
    rel = sorted(r["release_ms"] for r in runs)
    mlx.update(release_ms=rel[len(rel) // 2], release_ms_min=rel[0], release_ms_max=rel[-1],
               release_runs=len(runs), released=all(r["released"] for r in runs),
               after_clear_max=max(r["after_clear"] for r in runs),
               after_clear_runs=[r["after_clear"] for r in runs],
               clear_alone_freed=sum(r["after_clear"] < 64 for r in runs))
    res = dict(mlx=mlx, energy=energy_facts(), identity=identity_check(), machine=machine())
    (out / "platform.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
