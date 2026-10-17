"""Export fitted mappers as a Hugging Face model repository (MLX safetensors).

Layout written by :func:`export`::

    <out>/<pair>/<name>.safetensors   mapper weights (fp16) + metadata header
    <out>/<pair>/config.json          geometry, RoPE, policy, k, λ, data manifest
    <out>/index.json                  every pair, every file, SHA-256

Every exported file is reloaded and applied to a random cache before the index
is written, so a broken export fails here rather than on someone's phone.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx

from .mappers import KVMapper


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check(path: Path) -> Dict:
    m = KVMapper.load(str(path))
    T = 5
    src = [(mx.random.normal((1, m.src.n_kv_heads, T, m.src.head_dim)).astype(mx.float16),
            mx.random.normal((1, m.src.n_kv_heads, T, m.src.head_dim)).astype(mx.float16))
           for _ in range(m.src.n_layers)]
    out = m.apply(src, offset=3)
    mx.eval(out)
    assert len(out) == m.dst.n_layers
    assert out[0][0].shape == (1, m.dst.n_kv_heads, T, m.dst.head_dim)
    assert all(bool(mx.isfinite(k).all().item()) for k, _ in out)
    return dict(policy=m.policy, k=m.k, params=m.n_params(), src=m.src.__dict__,
                dst=m.dst.__dict__, src_rope=m.src_rope.__dict__, dst_rope=m.dst_rope.__dict__,
                lam=m.meta.get("lam"), n_rows=m.meta.get("n_rows"))


def export(mapper_root: Path, out: Path, repos: Optional[Dict[str, Dict[str, str]]] = None) -> Dict:
    """Copy ``mapper_root/<pair>/*.safetensors`` (+ manifest.json) into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for pair_dir in sorted(p for p in Path(mapper_root).iterdir() if p.is_dir()):
        dest = out / pair_dir.name
        dest.mkdir(exist_ok=True)
        files = {}
        for f in sorted(pair_dir.glob("*.safetensors")):
            shutil.copy(f, dest / f.name)
            files[f.name] = dict(check(dest / f.name), sha256=_sha(dest / f.name),
                                 bytes=(dest / f.name).stat().st_size)
        man = json.loads((pair_dir / "manifest.json").read_text()) if (pair_dir / "manifest.json").exists() else {}
        cfg = dict(format="molt-kv-mapper/1", pair=pair_dir.name, files=files,
                   selected=man.get("selected"), molt_anchor=man.get("molt_anchor"),
                   calibration=man.get("corpus"), heldout_nll=man.get("heldout_nll"),
                   natives=man.get("natives"), models=(repos or {}).get(pair_dir.name))
        (dest / "config.json").write_text(json.dumps(cfg, indent=1))
        index[pair_dir.name] = {n: v["sha256"] for n, v in files.items()}
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return index


def selftest() -> None:
    import tempfile

    from . import rope as R
    from .mappers import Geometry, LayerMap, depth_linear_sources

    tmp = Path(tempfile.mkdtemp())
    gs, gd = Geometry(6, 2, 32), Geometry(4, 2, 16)
    spec = R.RopeSpec(32, 1e4, False)
    layers = [LayerMap([s], mx.zeros((64, 32)), mx.zeros((32,)), mx.zeros((64, 32)), mx.zeros((32,)))
              for [s] in depth_linear_sources(6, 4)]
    (tmp / "src" / "big-small").mkdir(parents=True)
    KVMapper(gs, gd, spec, R.RopeSpec(16, 1e4, False), "depth_linear", 1, layers,
             dict(lam=0.01, n_rows=10)).save(str(tmp / "src" / "big-small" / "depth_linear.safetensors"))
    idx = export(tmp / "src", tmp / "out")
    cfg = json.loads((tmp / "out" / "big-small" / "config.json").read_text())
    assert cfg["files"]["depth_linear.safetensors"]["k"] == 1
    assert len(idx["big-small"]["depth_linear.safetensors"]) == 64
    print("  exported, reloaded and applied; index carries SHA-256 per file")
    print("convert: ok")


if __name__ == "__main__":
    selftest()
