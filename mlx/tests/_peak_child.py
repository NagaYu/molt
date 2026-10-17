"""Child process for test_peak_footprint: decode on SRC, migrate with the given
schedule, decode on DST.  Loads both models from disk like the real tiers."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molt_mlx.watchdog import wait_until_watched

wait_until_watched()

import mlx.core as mx  # noqa: E402
from mlx_lm.models import qwen2  # noqa: E402

from molt_mlx.footprint import PhaseMarker, footprint_bytes  # noqa: E402
from molt_mlx.generate import Session  # noqa: E402
from molt_mlx.mappers import KVMapper  # noqa: E402
from molt_mlx.migrate import Plan, migrate  # noqa: E402

root, schedule, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
mark = PhaseMarker(os.environ.get("MOLT_WATCHDOG_CSV"))
base = footprint_bytes()


def load(name):
    cfg = json.loads((root / name / "config.json").read_text())
    m = qwen2.Model(qwen2.ModelArgs(**cfg))           # parameters stay lazy …
    m.load_weights(str(root / name / "model.safetensors"))  # … and are replaced here
    mx.eval(m.parameters())
    return m


mark("load_src")
s = Session(load("src"), "src")
mark("decode_src")
s.start(list(range(1, 65)))
for _ in range(8):
    s.step()
fp_src = footprint_bytes()
new, rep = migrate(s, Plan(schedule=schedule, sink=2, tail=4), load_dst=lambda: load("dst"),
                   load_mapper=lambda: KVMapper.load(str(root / "mapper.safetensors")),
                   marker=mark)
mark("decode_dst")
for _ in range(8):
    new.step()
mark("done")
Path(out).write_text(json.dumps(dict(base=base, fp_src=fp_src, report=rep.to_dict())))
