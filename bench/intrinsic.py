"""Experiment I — transfer quality vs. boundary position (context-length sweep).

Teacher-forced, on the FineWeb-Edu *eval* split (never used for fitting or
selection): the source reads ``seq[:P]``, its cache is transferred, and the
destination scores the next ``SPAN`` tokens.  Reported as ΔNLL against the
destination reading the same prefix natively (= what re-prefill A′ achieves),
with paired bootstrap CIs.  One process per pair; this is an offline measurement
of the mapping, not an online condition, so it needs no memory isolation.

    python -m bench.intrinsic --pair 1.5b-0.5b
"""

from __future__ import annotations

import argparse
import json
import time

from bench._env import ARTIFACTS, RESULTS, machine

import numpy as np

PS = (32, 128, 256, 512, 768)
SPAN = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    import mlx.core as mx
    mx.set_cache_limit(1 << 30)   # offline script: bounded MLX buffer cache
    from molt_mlx import calib_data, kvcache, metrics, tiers
    from molt_mlx.generate import prefill
    from molt_mlx.mappers import KVMapper

    src, dst = a.pair.split("-")
    tok = tiers.load_tokenizer()
    corpus = calib_data.load(lambda s: tok.encode(s, add_special_tokens=False), "qwen2.5")
    seqs = corpus.splits["eval"][: a.n]
    mdir = ARTIFACTS / "mappers" / a.pair
    man = json.loads((mdir / "manifest.json").read_text())
    heo = KVMapper.load(str(mdir / f"{man['selected']}.safetensors"))
    depth = KVMapper.load(str(mdir / "depth_linear.safetensors"))
    sink, tail = man["molt_anchor"]["sink"], man["molt_anchor"]["tail"]
    msrc, mdst = tiers.load_model(src), tiers.load_model(dst)
    arms = {"C (ridge)": (heo, 0, 0), "D (molt anchors)": (heo, sink, tail),
            "D_depth": (depth, sink, tail)}
    res = {P: {k: [] for k in list(arms) + ["native_dst", "native_src"]} for P in PS}
    t0 = time.time()
    for i, seq in enumerate(seqs):
        for P in PS:
            s = seq[: P + 1 + SPAN]
            _, cs = prefill(msrc, s[:P])
            sp = kvcache.extract(cs)
            res[P]["native_dst"].append(float(metrics.native_nll(mdst, s, P).mean()))
            res[P]["native_src"].append(float(metrics.native_nll(msrc, s, P).mean()))
            for name, (m, sk, tl) in arms.items():
                res[P][name].append(float(metrics.prefix_kv_nll(s, P, mdst, sp, m, min(sk, P), min(tl, P)).mean()))
        if (i + 1) % 8 == 0:
            print(f"{i + 1}/{len(seqs)} ({time.time() - t0:.0f}s)", flush=True)
    summary = {}
    for P in PS:
        nd = res[P]["native_dst"]
        summary[P] = {k: dict(nll=metrics.bootstrap_ci(v),
                              delta_vs_native_dst=metrics.paired_bootstrap_ci(v, nd))
                      for k, v in res[P].items()}
    out = RESULTS / "intrinsic" / f"{a.pair}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(pair=a.pair, span=SPAN, n=len(seqs), mapper=man["selected"],
                                   sink=sink, tail=tail, raw=res, summary=summary,
                                   machine=machine()), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
