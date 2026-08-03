#!/usr/bin/env python3
"""Why a *lower* handoff JSD does not mean a better transplant.

    python scripts/diagnose_projection.py --ladder qwen

The benchmark turned up a genuinely confusing pair of numbers on the 1.5B->0.5B
route.  Against the naive truncated-identity map, the fitted map is:

* **worse** on handoff JSD (0.220 vs 0.192) — the local divergence at the one
  position where the switch happens; and
* **much better** on everything downstream (judge perplexity 3.1 vs 24.1, judge
  agreement 0.817 vs 0.717).

Both are real.  The resolution is that Jensen–Shannon divergence is bounded and
*rewards blurring*: a flatter output distribution overlaps more with almost
anything, so a map that degrades the cache into mush can score a lower one-shot
JSD while producing text that falls apart a few tokens later.  Handoff JSD is a
sound measure of the *seam*; it is not a measure of transplant quality on its
own, and this repository reports it beside judge agreement for that reason.

This script makes the mechanism visible rather than leaving it as an argument.
For each layer it prints the L2 reconstruction error, the variance ratio
``std(projected) / std(native)`` (1.0 = variance preserved; below 1 means the map
shrank the cache toward its mean, which ridge regression does by construction),
and the resulting attention-entropy ratio (above 1 = flatter attention than the
destination model's own).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molt.config import TransplantConfig, get_ladder, resolve_device, resolve_dtype
from molt.kv_cache import CacheMeta, MoltCache
from molt.kv_transplant import KVTransplant, ProjectorRegistry, TierRef
from molt.model_zoo import ModelZoo, load_tokenizer
from molt.projector import make_identity_projector


def attention_entropy(q: torch.Tensor, k: torch.Tensor) -> float:
    """Mean entropy of the attention distribution of the last query over ``k``."""
    d = k.shape[-1]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    p = torch.softmax(scores.float(), dim=-1)
    return float(-(p * p.clamp_min(1e-12).log()).sum(-1).mean())


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ladder", default="qwen")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--projector-dir", default="artifacts/projectors")
    p.add_argument("--tokens", type=int, default=192)
    p.add_argument("--src", default="tier0")
    p.add_argument("--dst", default="tier2")
    args = p.parse_args(argv)

    ladder = get_ladder(args.ladder)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    zoo = ModelZoo(device, dtype)
    tok = load_tokenizer(ladder.tokenizer_id)
    reg = ProjectorRegistry(args.projector_dir, ladder.name, device)
    cfg = TransplantConfig(recompute_top_k=0, projector_dir=args.projector_dir)

    from molt.fit_projector import DEFAULT_CALIB_TEXTS
    text = " ".join(DEFAULT_CALIB_TEXTS[8:])          # held out from the fit's head
    ids = tok(text, return_tensors="pt")["input_ids"][:, :args.tokens].to(device)
    print(f"route {args.src} -> {args.dst}, {ids.shape[1]} held-out tokens\n")

    src = zoo.acquire(ladder.by_name(args.src))
    with torch.no_grad():
        out = src.model(ids, use_cache=True)
    cache = MoltCache.from_hf(out.past_key_values,
                              CacheMeta(args.src, src.geometry, ids[0].tolist(),
                                        [], ids.shape[1]), {})
    ref = TierRef.from_loaded(src, snapshot_rope=True)
    del out
    zoo.release(ladder.by_name(args.src), evict_if_unused=True)

    dst = zoo.acquire(ladder.by_name(args.dst))
    with torch.no_grad():
        native = dst.model(ids, use_cache=True)
    nat = MoltCache.from_hf(native.past_key_values,
                            CacheMeta(args.dst, dst.geometry, ids[0].tolist(),
                                      [], ids.shape[1]), {})
    del native

    maps = {
        "fitted (ridge)": reg.get(args.src, args.dst),
        "truncated identity": make_identity_projector(
            src.geometry if hasattr(src, "geometry") else ref.geometry,
            dst.geometry, args.src, args.dst, rope_realign=False),
    }
    tp = KVTransplant(cfg)
    print(f"{'map':<20} {'layer':>5} {'k L2err':>9} {'v L2err':>9} "
          f"{'k std/nat':>10} {'v std/nat':>10} {'attn H/nat':>11}")
    print("-" * 78)
    for name, proj in maps.items():
        got, _ = tp.transplant(cache, ref, dst, proj, device)
        k_rat, v_rat, k_err, v_err, h_rat = [], [], [], [], []
        for l in range(dst.geometry.n_layers):
            kg, vg = got.layers[l]
            kn, vn = nat.layers[l]
            k_err.append(float((kg - kn).pow(2).mean().sqrt()
                               / kn.pow(2).mean().sqrt()))
            v_err.append(float((vg - vn).pow(2).mean().sqrt()
                               / vn.pow(2).mean().sqrt()))
            k_rat.append(float(kg.std() / kn.std()))
            v_rat.append(float(vg.std() / vn.std()))
            q = kn[:, :, -1:, :]                    # a real query direction
            h_nat = attention_entropy(q, kn)
            # Early layers can attend almost deterministically, which drives the
            # native entropy to ~0 and makes a *ratio* meaningless (it explodes).
            # Skip those rather than print a number that looks like a result.
            h_rat.append(attention_entropy(q, kg) / h_nat if h_nat > 0.05 else None)
        n = dst.geometry.n_layers
        ok = [x for x in h_rat if x is not None]

        def hs(l):
            return f"{h_rat[l]:>11.3f}" if h_rat[l] is not None else f"{'n/a':>11}"

        for l in (0, n // 2, n - 1):
            print(f"{name if l == 0 else '':<20} {l:>5} {k_err[l]:>9.3f} "
                  f"{v_err[l]:>9.3f} {k_rat[l]:>10.3f} {v_rat[l]:>10.3f} {hs(l)}")
        mh = f"{sum(ok)/len(ok):>11.3f}" if ok else f"{'n/a':>11}"
        print(f"{'  mean':<20} {'':>5} {sum(k_err)/n:>9.3f} {sum(v_err)/n:>9.3f} "
              f"{sum(k_rat)/n:>10.3f} {sum(v_rat)/n:>10.3f} {mh}")
        print(f"{'':<20} {'':>5} (entropy ratio averaged over the {len(ok)}/{n} "
              f"layers whose native attention is not near-deterministic)")
        print()
    zoo.evict_all()
    print("Reading: a std ratio below 1 means the map shrank the cache's variance;\n"
          "an attention-entropy ratio above 1 means the resulting attention is\n"
          "flatter than the destination model's own. Ridge shrinkage predicts both.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
