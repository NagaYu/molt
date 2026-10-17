"""Experiment R — compute cost of resuming vs. context length.

The memory experiment's prompts are short (~10^2 tokens), where re-reading the
transcript on the small model is cheap.  This measures where (if anywhere) the
transfer becomes cheaper than re-prefill as context grows.  Models and mapper
are resident; only the resume computation is timed (fixed load costs are timed
separately), conditions interleaved, median of repeats.

    python -m bench.resume_cost --pair 1.5b-0.5b
"""

from __future__ import annotations

import argparse
import json
import time
from statistics import median

from bench._env import ARTIFACTS, RESULTS, machine

PS = (128, 256, 512, 1024, 2048, 4096)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    import mlx.core as mx
    from molt_mlx import calib_data, kvcache, tiers
    from molt_mlx.generate import prefill
    from molt_mlx.mappers import KVMapper
    from molt_mlx.migrate import transplant

    mx.set_cache_limit(1 << 30)
    src, dst = a.pair.split("-")
    tok = tiers.load_tokenizer()
    corpus = calib_data.load(lambda s: tok.encode(s, add_special_tokens=False), "qwen2.5")
    stream = [t for s in corpus.splits["eval"] for t in s]
    man = json.loads((ARTIFACTS / "mappers" / a.pair / "manifest.json").read_text())
    mfile = ARTIFACTS / "mappers" / a.pair / f"{man['selected']}.safetensors"
    sink, tail = man["molt_anchor"]["sink"], man["molt_anchor"]["tail"]

    # fixed costs, warm page cache
    loads = {"dst_model": [], "mapper": []}
    for _ in range(3):
        t = time.perf_counter(); m = tiers.load_model(dst); loads["dst_model"].append(time.perf_counter() - t); del m
        t = time.perf_counter(); mp = KVMapper.load(str(mfile)); mx.eval([l.Wk for l in mp.layers])
        loads["mapper"].append(time.perf_counter() - t); del mp
        mx.clear_cache()
    msrc, mdst, mapper = tiers.load_model(src), tiers.load_model(dst), KVMapper.load(str(mfile))
    arms = {"A2 re-prefill": dict(mapper=None, sink=0, tail="all"),
            "C ridge": dict(mapper=mapper, sink=0, tail=0),
            "D molt anchors": dict(mapper=mapper, sink=sink, tail=tail)}
    out = {}
    for P in PS:
        toks = stream[: P + 1]
        _, cs = prefill(msrc, toks[:P])
        pairs = kvcache.extract(cs)
        mx.eval([k for k, _ in pairs])
        times = {k: [] for k in arms}
        for r in range(a.repeats + 1):                     # first round is warm-up
            for name, arm in arms.items():
                mx.synchronize()
                t = time.perf_counter()
                lg, cache, _ = transplant(pairs, toks[:P], toks[P], mdst, arm["mapper"], arm["sink"], arm["tail"])
                mx.eval(lg, [c.state for c in cache])
                mx.synchronize()
                if r:
                    times[name].append(time.perf_counter() - t)
                del lg, cache
        out[P] = {k: dict(median_s=median(v), min_s=min(v), max_s=max(v)) for k, v in times.items()}
        print(P, {k: round(v["median_s"], 3) for k, v in out[P].items()}, flush=True)
        del pairs, cs
    res = dict(pair=a.pair, mapper=man["selected"], sink=sink, tail=tail, repeats=a.repeats,
               load_s={k: median(v) for k, v in loads.items()}, by_context=out, machine=machine())
    (RESULTS / "resume_cost").mkdir(parents=True, exist_ok=True)
    (RESULTS / "resume_cost" / f"{a.pair}.json").write_text(json.dumps(res, indent=1))
    print("loads", res["load_s"])


if __name__ == "__main__":
    main()
