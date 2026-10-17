"""Fit, sweep and select KV mappers for one tier pair.

    python -m bench.fit_mappers --pair 1.5b-0.5b

Stage 1 (collect)  500 × 1 024 FineWeb-Edu tokens through both models, stride 4,
                   float32 moments → ``~/.cache/molt/stats/<pair>.safetensors``.
Stage 2 (solve)    float64 ridge; single-source R² table; Heo top-k for
                   k ∈ {1, 2, 4, 8, 16, all} and the depth-linear ablation.
Stage 3 (select)   on the *held-out* split only (64 documents, never the eval
                   split): prefix-KV NLL at P = 768 and held-out R² for every
                   mapper; λ sweep at the chosen k; anchor (sink, tail) sweep for
                   molt.  ``k*`` = argmin held-out NLL (Heo et al. select k on
                   downstream accuracy; see the paper's Methods for why we use
                   held-out NLL instead).

Writes ``bench/results/fit/<pair>.json`` and the selected mappers (fp16) under
``artifacts/mlx/mappers/<pair>/``.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time

from bench._env import ARTIFACTS, CACHE, RESULTS, machine

import mlx.core as mx
import numpy as np

from molt_mlx import calib_data, fit, kvcache, metrics, tiers
from molt_mlx.footprint import footprint_bytes
from molt_mlx.generate import prefill
from molt_mlx.mappers import KVMapper
from molt_mlx import rope as R

mx.set_cache_limit(256 << 20)  # offline script: keep MLX's buffer cache small (shared, swapping machine)

KS = (1, 2, 4, 8, 16, "all")
LAMS = (0.0, 1e-3, 1e-2, 1e-1, 1.0)
ANCHORS = ((0, 0), (4, 0), (0, 8), (0, 32), (4, 8), (4, 32), (4, 128), (0, "all"))
ANCHOR_TAIL_BUDGET = 32   # molt's tail is chosen among tails ≤ this (cost bound)
P = 768


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def corpus():
    tok = tiers.load_tokenizer()
    return calib_data.load(lambda s: tok.encode(s, add_special_tokens=False), "qwen2.5")


def stage_collect(src, dst, n_calib, stride, path):
    """Collect moments; checkpoints every 25 sequences so an interrupted run
    resumes where it stopped (the 3B pair takes hours on a shared machine)."""
    if path.exists():
        log("stats cached:", path)
        return
    c = corpus()
    seqs = c.splits["calib"][:n_calib]
    ck, ck_meta = path.with_suffix(".partial.safetensors"), path.with_suffix(".partial.json")
    stats, start = None, 0
    if ck.exists() and ck_meta.exists():
        start = json.loads(ck_meta.read_text())["consumed"]
        stats = {(src, dst): fit.PairStats.load_raw(str(ck))}
        log(f"resuming from checkpoint: {start} sequences consumed")
    models = {src: tiers.load_model(src), dst: tiers.load_model(dst)}
    t0 = time.time()
    last = [start]

    def checkpoint(consumed, st):
        if consumed - last[0] >= 25:
            path.parent.mkdir(parents=True, exist_ok=True)
            st[(src, dst)].save_raw(str(ck))
            ck_meta.write_text(json.dumps(dict(consumed=consumed)))
            last[0] = consumed

    st = fit.collect(models, [(src, dst)], seqs, stride=stride, stats=stats, start=start,
                     checkpoint=checkpoint,
                     progress=lambda i: (i + 1) % 25 == 0 and log(
                         f"collect {i + 1}/{len(seqs)} ({time.time() - t0:.0f}s, "
                         f"footprint {footprint_bytes() / 2**30:.2f} GiB)"))[(src, dst)]
    path.parent.mkdir(parents=True, exist_ok=True)
    st.save_raw(str(path))
    for p in (ck, ck_meta):
        if p.exists():
            p.unlink()
    log(f"collected n={st.n} rows in {time.time() - t0:.0f}s ->", path)
    del models, st
    gc.collect(); mx.clear_cache()


def mapper_name(policy, k):
    return "depth_linear" if policy == "depth_linear" else f"heo_k{k}"


def stage_solve(src, dst, stats_path, out_dir, lam):
    st = fit.PairStats.load_raw(str(stats_path))
    t0 = time.time()
    r2 = fit.single_source_r2(st, lam)
    log("single-source R² table", r2.shape, f"best per layer mean {r2.max(1).mean():.3f}")
    rows = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    configs, policies = {}, {}
    for k in KS:
        kk = st.src.n_layers if k == "all" else k
        configs[mapper_name("r2_topk", k)] = fit.heo_sources(r2, kk)
        policies[mapper_name("r2_topk", k)] = "r2_topk"
    configs["depth_linear"] = fit.depth_linear_sources(st.src.n_layers, st.dst.n_layers)
    policies["depth_linear"] = "depth_linear"
    t1 = time.time()
    solved = fit.solve_many(st, configs, lam, policies, meta=dict(pair=f"{src}-{dst}"))
    log(f"solved {len(solved)} mappers in {time.time() - t1:.0f}s")
    for name, (m, rep) in solved.items():
        m.save(str(out_dir / f"{name}.safetensors"))
        rows[name] = dict(policy=policies[name], k=m.k, insample_r2_k=float(rep.r2_k.mean()),
                          insample_r2_v=float(rep.r2_v.mean()), params=m.n_params(),
                          mib_fp16=m.n_params() * 2 / 2**20,
                          sources=[l.sources for l in m.layers])
        log(f"{name}: in-sample R² K {rep.r2_k.mean():.3f} V {rep.r2_v.mean():.3f} "
            f"({m.n_params() / 1e6:.1f} M params)")
    del solved
    return st, r2, rows


def stage_select(src, dst, st, r2tab, mapper_dir, rows, n_heldout):
    c = corpus()
    seqs = c.splits["heldout"][:n_heldout]
    msrc, mdst = tiers.load_model(src), tiers.load_model(dst)
    mappers = {n: KVMapper.load(str(mapper_dir / f"{n}.safetensors")) for n in rows}
    nll = {n: [] for n in mappers}
    nll["native_dst"], nll["native_src"] = [], []
    # per (layer, head): residual SS; per (layer, head, dim): Σt and Σt² (global centring)
    r2acc = {n: {kv: [np.zeros((m.dst.n_layers, m.dst.n_kv_heads)),
                      np.zeros((m.dst.n_layers, m.dst.n_kv_heads, m.dst.head_dim)),
                      np.zeros((m.dst.n_layers, m.dst.n_kv_heads, m.dst.head_dim))]
                 for kv in "kv"} for n, m in mappers.items()}
    n_pos = 0
    t0 = time.time()
    for i, seq in enumerate(seqs):
        _, cs = prefill(msrc, seq[:P]); sp = kvcache.extract(cs)
        _, cd = prefill(mdst, seq[:P]); dp = kvcache.extract(cd)
        nll["native_dst"].append(float(metrics.native_nll(mdst, seq, P).mean()))
        nll["native_src"].append(float(metrics.native_nll(msrc, seq, P).mean()))
        for n, m in mappers.items():
            nll[n].append(float(metrics.prefix_kv_nll(seq, P, mdst, sp, m).mean()))
            mapped = m.apply(sp, out_dtype=mx.float32)
            for l, ((pk, pv), (tk, tv)) in enumerate(zip(mapped, dp)):
                for kv, p_, t_ in (("k", R.remove(pk, m.dst_rope), R.remove(tk.astype(mx.float32), m.dst_rope)),
                                   ("v", pv, tv.astype(mx.float32))):
                    p_ = np.array(p_[0], dtype=np.float64); t_ = np.array(t_[0], dtype=np.float64)
                    a = r2acc[n][kv]
                    a[0][l] += ((p_ - t_) ** 2).sum(axis=(1, 2))
                    a[1][l] += t_.sum(axis=1)
                    a[2][l] += (t_ * t_).sum(axis=1)
        n_pos += P
        if (i + 1) % 8 == 0:
            log(f"select {i + 1}/{len(seqs)} ({time.time() - t0:.0f}s)")
    out = {}
    for n in mappers:
        hr2 = {}
        for kv in "kv":
            res, s1, s2 = r2acc[n][kv]
            tot = (s2 - s1 ** 2 / n_pos).sum(-1)
            hr2[kv] = float((1 - res / tot).mean())
        out[n] = dict(rows[n], heldout_nll=metrics.bootstrap_ci(nll[n]),
                      heldout_r2_k=hr2["k"], heldout_r2_v=hr2["v"])
        log(f"{n}: held-out NLL {out[n]['heldout_nll'][0]:.3f}  R² K {hr2['k']:.3f} V {hr2['v']:.3f}")
    natives = {k: metrics.bootstrap_ci(nll[k]) for k in ("native_dst", "native_src")}
    heo = [n for n in out if n.startswith("heo_")]
    best = min(heo, key=lambda n: out[n]["heldout_nll"][0])
    log("selected", best, "| natives", {k: round(v[0], 3) for k, v in natives.items()})

    # λ sweep at k*, and molt's anchor sweep with the selected mapper (held-out only)
    kbest = out[best]["k"]
    lam_rows, anchor_rows = {}, {}
    sub = seqs[: max(16, n_heldout // 2)]
    caches = []
    for seq in sub:
        _, cs = prefill(msrc, seq[:P]); caches.append(kvcache.extract(cs))
    for lam in LAMS:
        m, _ = fit.fit(st, "r2_topk", kbest, lam=lam, r2=r2tab)
        v = [float(metrics.prefix_kv_nll(s, P, mdst, sp, m).mean()) for s, sp in zip(sub, caches)]
        lam_rows[str(lam)] = metrics.bootstrap_ci(v)
        log(f"λ={lam}: held-out NLL {lam_rows[str(lam)][0]:.3f}")
    m = mappers[best]
    for sink, tail in ANCHORS:
        v = [float(metrics.prefix_kv_nll(s, P, mdst, sp, m, sink, tail).mean()) for s, sp in zip(sub, caches)]
        anchor_rows[f"{sink}/{tail}"] = dict(sink=sink, tail=tail, nll=metrics.bootstrap_ci(v))
        log(f"anchors sink={sink} tail={tail}: NLL {anchor_rows[f'{sink}/{tail}']['nll'][0]:.3f}")
    ok = [r for r in anchor_rows.values() if r["tail"] != "all" and r["tail"] <= ANCHOR_TAIL_BUDGET]
    molt_anchor = min(ok, key=lambda r: r["nll"][0])
    return out, natives, best, lam_rows, anchor_rows, molt_anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=[tiers.pair_name(*p) for p in tiers.PAIRS])
    ap.add_argument("--n-calib", type=int, default=500)
    ap.add_argument("--n-heldout", type=int, default=64)
    ap.add_argument("--stride", type=int, default=fit.HEO_STRIDE)
    ap.add_argument("--lam", type=float, default=fit.HEO_LAMBDA)
    a = ap.parse_args()
    src, dst = a.pair.split("-")
    env0 = machine()
    stats_path = CACHE / "stats" / f"{a.pair}_n{a.n_calib}_s{a.stride}.safetensors"
    mapper_dir = CACHE / "mappers" / a.pair
    t0 = time.time()
    stage_collect(src, dst, a.n_calib, a.stride, stats_path)
    st, r2, rows = stage_solve(src, dst, stats_path, mapper_dir, a.lam)
    out, natives, best, lam_rows, anchor_rows, molt_anchor = stage_select(
        src, dst, st, r2, mapper_dir, rows, a.n_heldout)
    c = corpus()
    res = dict(pair=a.pair, src=tiers.TIERS[src], dst=tiers.TIERS[dst], P=P,
               n_calib=a.n_calib, n_rows=st.n, stride=a.stride, lam=a.lam,
               corpus=dict(c.manifest(), doc_ids=None), single_source_r2=r2.tolist(),
               mappers=out, natives=natives, selected=best, lam_sweep=lam_rows,
               anchors=anchor_rows, molt_anchor=molt_anchor,
               anchor_tail_budget=ANCHOR_TAIL_BUDGET, seconds=time.time() - t0,
               machine_start=env0, machine_end=machine())
    (RESULTS / "fit").mkdir(parents=True, exist_ok=True)
    (RESULTS / "fit" / f"{a.pair}.json").write_text(json.dumps(res, indent=1))
    dest = ARTIFACTS / "mappers" / a.pair
    dest.mkdir(parents=True, exist_ok=True)
    for n in (best, "depth_linear"):
        shutil.copy(mapper_dir / f"{n}.safetensors", dest / f"{n}.safetensors")
    (dest / "manifest.json").write_text(json.dumps(dict(
        selected=best, molt_anchor=molt_anchor, corpus=c.manifest(),
        heldout_nll=out[best]["heldout_nll"], natives=natives), indent=1))
    log("done in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
