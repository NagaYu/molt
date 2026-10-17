"""Stage everything the iOS harness reads from Documents/.

    python -m bench.export_device --out build/device
    xcrun devicectl device copy to --device <udid> --domain-type appDataContainer \
        --domain-identifier dev.molt.MoltBench --source build/device --destination Documents

Copies the selected mappers and their manifests (adding the destination's
weight size, which the co-resident schedule's trigger needs), and the prompt
suite (also bundled as an app resource).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from bench._env import ARTIFACTS, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "device"))
    a = ap.parse_args()
    from molt_mlx import tiers
    from bench.worker import dir_bytes
    out = Path(a.out)
    for pair_dir in sorted((ARTIFACTS / "mappers").iterdir()):
        if not pair_dir.is_dir() or not (pair_dir / "manifest.json").exists():
            continue
        dest = out / "mappers" / pair_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        man = json.loads((pair_dir / "manifest.json").read_text())
        _, dst = pair_dir.name.split("-")
        man["dst_weight_bytes"] = dir_bytes(tiers.local_path(dst))
        (dest / "manifest.json").write_text(json.dumps(man, indent=1))
        for f in pair_dir.glob("*.safetensors"):
            shutil.copy(f, dest / f.name)
        print("staged", pair_dir.name)
    shutil.copy(ROOT / "bench" / "prompts.json", out / "prompts.json")
    shutil.copy(ROOT / "bench" / "prompts.json", ROOT / "ios" / "App" / "prompts.json")
    print("->", out)


if __name__ == "__main__":
    main()
