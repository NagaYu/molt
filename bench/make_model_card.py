"""Fill bench/MODEL_CARD.template.md from bench/results/macros.json.

    python -m bench.make_model_card --out build/hf/README.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench._env import RESULTS, ROOT
from bench.make_results_md import fill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "hf" / "README.md"))
    a = ap.parse_args()
    M = json.loads((RESULTS / "macros.json").read_text())
    label = {"3b-1.5b": "3B → 1.5B", "1.5b-0.5b": "1.5B → 0.5B"}
    files = ["| pair | file | k | size | role |", "|---|---|---:|---:|---|"]
    qual = ["| pair | mapped (k*) NLL | receiver reading natively | gap | held-out R² K / V |",
            "|---|---:|---:|---:|---:|"]
    for pair in ("3b-1.5b", "1.5b-0.5b"):
        pl = pair.replace(".", "")
        if f"fit.{pl}.kstar" not in M:
            files.append(f"| {label[pair]} | *fit in progress* | | | |")
            continue
        k = M[f"fit.{pl}.kstar"]
        files.append(f"| {label[pair]} | `mlx/{pair}/heo_k{k}.safetensors` | {k} | {M[f'fit.{pl}.kstar_mib']} MiB | "
                     "conditions C and D (Heo et al. top-k) |")
        files.append(f"| {label[pair]} | `mlx/{pair}/depth_linear.safetensors` | 1 | | ablation: molt v0.1 layer pairing |")
        qual.append(f"| {label[pair]} | {M[f'fit.{pl}.kstar_nll']} | {M[f'fit.{pl}.native_dst_nll']} | "
                    f"+{M[f'fit.{pl}.gap_nll']} | {M[f'fit.{pl}.kstar_r2k']} / {M[f'fit.{pl}.kstar_r2v']} |")
    M = dict(M, FILES_TABLE="\n".join(files), QUALITY_TABLE="\n".join(qual))
    text = fill((ROOT / "bench" / "MODEL_CARD.template.md").read_text(), M)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print("wrote", a.out, "PENDING marks:", text.count("PENDING"))


if __name__ == "__main__":
    main()
