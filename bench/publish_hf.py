"""Publish the MLX mappers, the model card and the v0.2 measurements to the Hub.

    python -m bench.publish_hf --dry-run      # stage under build/hf and show what would go up
    python -m bench.publish_hf                # upload

Model repo NagaYu/molt-kv-projectors-qwen2.5: ``mlx/<pair>/*`` (weights + config,
checked by reload in convert.export) and README.md (generated card).  The v0.1
PyTorch ``*.pt`` files are left in place and labelled legacy in the card.
Dataset repo NagaYu/molt-benchmark-results: ``v0.2/`` (every JSON under
bench/results except bulky raw timelines).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from bench._env import ARTIFACTS, RESULTS, ROOT

MODEL_REPO = "NagaYu/molt-kv-projectors-qwen2.5"
DATA_REPO = "NagaYu/molt-benchmark-results"


def stage() -> Path:
    from molt_mlx import convert, tiers
    out = ROOT / "build" / "hf"
    if (out / "mlx").exists():
        shutil.rmtree(out / "mlx")
    repos = {f"{s}-{d}": dict(src=tiers.TIERS[s], dst=tiers.TIERS[d]) for s, d in tiers.PAIRS}
    convert.export(ARTIFACTS / "mappers", out / "mlx", repos)
    from bench import make_model_card
    import sys
    sys.argv = ["make_model_card", "--out", str(out / "README.md")]
    make_model_card.main()
    card = (out / "README.md").read_text()
    card += ("\n\n## Legacy (v0.1)\n\nThe `qwen2.5-1.5b-ladder__*.pt` files at the repository root are the "
             "PyTorch projectors of molt v0.1, superseded by `mlx/`. Their README claims about RoPE handling "
             "being novel are withdrawn: that handling is prior work (Heo et al.).\n")
    (out / "README.md").write_text(card)
    data = out / "dataset_v0.2"
    if data.exists():
        shutil.rmtree(data)
    for p in RESULTS.rglob("*"):
        if p.is_file() and p.suffix in (".json", ".jsonl") and not p.name.endswith(".progress.jsonl"):
            dst = data / p.relative_to(RESULTS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dst)
    shutil.copy(ROOT / "RESULTS.md", data / "RESULTS.md")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = stage()
    files = sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file())
    print(f"{len(files)} files staged under {out}")
    for f in files[:40]:
        print("  ", f)
    if a.dry_run:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(repo_id=MODEL_REPO, folder_path=str(out / "mlx"), path_in_repo="mlx",
                      commit_message="Add MLX KV mappers (v0.2): Heo et al. recipe, Qwen2.5 4-bit")
    api.upload_file(repo_id=MODEL_REPO, path_or_fileobj=str(out / "README.md"), path_in_repo="README.md",
                    commit_message="Model card for v0.2 (MLX); credit prior work")
    api.upload_folder(repo_id=DATA_REPO, repo_type="dataset", folder_path=str(out / "dataset_v0.2"),
                      path_in_repo="v0.2", commit_message="v0.2 measurements (MLX, jetsam emulation)")
    print("uploaded")


if __name__ == "__main__":
    main()
