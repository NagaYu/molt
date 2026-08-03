#!/usr/bin/env python3
"""Fit the KV projections for every migration route of a tier ladder.

    python scripts/train_projectors.py --ladder qwen
    python scripts/train_projectors.py --ladder synthetic --out artifacts/projectors

Closed-form ridge regression on a small calibration corpus (see
:mod:`molt.fit_projector`).  Fitting all six routes of a three-rung ladder takes
seconds on CPU for the synthetic ladder and a couple of minutes for Qwen2.5.

The printed residuals are the diagnostic that matters: a route whose K/V
residual is near 1.0 has learned nothing and its migrations will show a large
distribution jump.  Residuals are also stored inside each ``.pt`` so the
benchmark can report them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molt.config import TransplantConfig, get_ladder, resolve_device, resolve_dtype
from molt.fit_projector import DEFAULT_CALIB_TEXTS, fit_all_routes, mean_residual
from molt.model_zoo import ModelZoo, load_tokenizer
from molt.projector import KVProjector


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ladder", default="synthetic", choices=["synthetic", "qwen", "qwen-3b"])
    p.add_argument("--out", default="artifacts/projectors")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--recompute-top-k", type=int, default=6)
    p.add_argument("--calib-file", default=None,
                   help="text file, one calibration sample per line")
    p.add_argument("--max-total-tokens", type=int, default=3072)
    p.add_argument("--ridge", type=float, default=1e-2,
                   help="ridge strength; higher generalises better on small corpora")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="fraction of calibration windows held out to measure "
                        "generalisation (training residual alone is misleading)")
    p.add_argument("--keep-resident", action="store_true",
                   help="keep both rungs loaded while fitting (faster, needs ~2x memory)")
    p.add_argument("--no-rope-realign", action="store_true",
                   help="ablation: fit without the RoPE un-rotate/re-rotate sandwich")
    args = p.parse_args(argv)

    ladder = get_ladder(args.ladder)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    cfg = TransplantConfig(recompute_top_k=args.recompute_top_k, projector_dir=args.out,
                           use_rope_realign=not args.no_rope_realign)

    texts = DEFAULT_CALIB_TEXTS
    if args.calib_file:
        with open(args.calib_file) as f:
            texts = [ln.strip() for ln in f if ln.strip()]

    print(f"ladder {ladder.name} on {device}/{dtype}; {len(texts)} calibration samples")
    zoo = ModelZoo(device, dtype, verbose=True)
    tok = load_tokenizer(ladder.tokenizer_id)
    projs = fit_all_routes(zoo, ladder, tok, cfg, device, out_dir=args.out,
                           texts=texts, verbose=True, sequential=not args.keep_resident,
                           max_total_tokens=args.max_total_tokens, ridge=args.ridge,
                           val_frac=args.val_frac)
    zoo.evict_all()

    index = {}
    for (s, d), proj in projs.items():
        index[f"{s}->{d}"] = dict(
            kind=proj.meta.kind, rope_realign=proj.meta.rope_realign,
            layer_map=proj.meta.layer_map, n_calib_tokens=proj.meta.n_calib_tokens,
            n_val_tokens=proj.meta.n_val_tokens,
            val_k=mean_residual(proj, "k"), val_v=mean_residual(proj, "v"),
            val_hidden=mean_residual(proj, "hidden"),
            train_k=mean_residual(proj, "k", held_out=False),
            train_v=mean_residual(proj, "v", held_out=False),
            file=KVProjector.route_filename(s, d, ladder.name),
        )
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{ladder.name}__index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nwrote {len(projs)} projectors to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
