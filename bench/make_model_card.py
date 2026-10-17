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
    text = fill((ROOT / "bench" / "MODEL_CARD.template.md").read_text(), M)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print("wrote", a.out, "PENDING marks:", text.count("PENDING"))


if __name__ == "__main__":
    main()
