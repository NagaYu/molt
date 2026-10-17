"""The non-co-residence guarantee, tested against an external 1 kHz watchdog.

Two synthetic Qwen2 models are written to disk (source ≈ 300 MiB, destination
≈ 110 MiB, fp16).  Each schedule runs in its own process under a cap of

    cap = (footprint while the source is resident) + 40 MiB

which is below source + destination.  The watchdog SIGKILLs the process the
moment phys_footprint exceeds the cap, like jetsam.  Required outcomes:

* ``ordered``   survives, and its sampled peak never exceeds the cap  ← guarantee
* ``colocated`` is killed (it needs source + destination)
* ``del_only``  is killed (MLX keeps the freed source buffers cached)
* ``clear_no_wait`` is reported, not asserted: it races asynchronous reclaim.

The peak asserted is the kernel's ``ri_lifetime_max_phys_footprint`` — the
ledger maximum, which cannot miss a spike between two 1 ms samples — and the
watchdog kills on it too, so a sub-millisecond excursion over the cap is a kill.
"""

import json
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models import qwen2  # noqa: E402

from molt_mlx import watchdog  # noqa: E402
from molt_mlx.mappers import Geometry, KVMapper, LayerMap, depth_linear_sources  # noqa: E402
from molt_mlx import rope as R  # noqa: E402

CHILD = Path(__file__).with_name("_peak_child.py")
FACTS = Path(__file__).resolve().parents[2] / "bench" / "results" / "facts" / "peak_test.json"


def _record(key, value):
    """Append this test's measured outcome to the facts file the paper reads."""
    FACTS.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(FACTS.read_text()) if FACTS.exists() else {}
    d[key] = value
    FACTS.write_text(json.dumps(d, indent=1))
MiB = 2 ** 20
SRC = dict(hidden_size=1024, num_hidden_layers=12, intermediate_size=2816,
           num_attention_heads=16, num_key_value_heads=4, vocab_size=16384)
DST = dict(hidden_size=640, num_hidden_layers=10, intermediate_size=1792,
           num_attention_heads=8, num_key_value_heads=4, vocab_size=16384)
MARGIN_MIB = 40


def _write(root: Path, name: str, spec: dict, seed: int) -> int:
    from mlx.utils import tree_flatten
    mx.random.seed(seed)
    cfg = dict(model_type="qwen2", rms_norm_eps=1e-6, rope_theta=10000.0,
               tie_word_embeddings=True, **spec)
    m = qwen2.Model(qwen2.ModelArgs(**cfg))
    params = {k: v.astype(mx.float16) for k, v in tree_flatten(m.parameters())}
    (root / name).mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(root / name / "model.safetensors"), params)
    (root / name / "config.json").write_text(json.dumps(cfg))
    return sum(v.nbytes for v in params.values())


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("peak")
    sw = _write(root, "src", SRC, 1)
    dw = _write(root, "dst", DST, 2)
    gs, gd = Geometry(12, 4, 64), Geometry(10, 4, 80)
    spec, dspec = R.RopeSpec(64, 10000.0, False), R.RopeSpec(80, 10000.0, False)
    layers = [LayerMap([s], mx.random.normal((gs.width, gd.width)) * 0.02, mx.zeros((gd.width,)),
                       mx.random.normal((gs.width, gd.width)) * 0.02, mx.zeros((gd.width,)))
              for [s] in depth_linear_sources(12, 10)]
    KVMapper(gs, gd, spec, dspec, "depth_linear", 1, layers).save(str(root / "mapper.safetensors"))
    return root, sw, dw


def _run(root, schedule, cap_mib, tmp):
    csv = str(tmp / f"{schedule}_{int(cap_mib)}.csv")
    out = str(tmp / f"{schedule}_{int(cap_mib)}.json")
    rc, res = watchdog.run_under_watchdog([sys.executable, str(CHILD), str(root), schedule, out],
                                          cap_mib=cap_mib, csv_path=csv, timeout_s=300)
    rep = json.loads(Path(out).read_text()) if Path(out).exists() else None
    return rc, res, rep


@pytest.fixture(scope="module")
def cap(models, tmp_path_factory):
    """Cap derived from an uncapped ordered run: source-resident footprint + margin."""
    root, sw, dw = models
    tmp = tmp_path_factory.mktemp("cal")
    rc, res, rep = _run(root, "ordered", 0, tmp)
    assert rc == 0, "uncapped calibration run failed"
    src_peak = res.phase_peaks_mib()["decode_src"]
    assert dw / MiB > MARGIN_MIB + 32, "destination must not fit in the margin"
    return src_peak + MARGIN_MIB, res, rep


def test_ordered_never_exceeds_cap(models, cap, tmp_path):
    root, sw, dw = models
    cap_mib, _, _ = cap
    rc, res, rep = _run(root, "ordered", cap_mib, tmp_path)
    phases = res.phase_peaks_mib()
    assert not res.killed and rc == 0, f"ordered was killed in {res.killed_in_phase()} ({phases})"
    assert res.lifetime_max_bytes > 0, "kernel lifetime max was not recorded"
    assert res.peak_mib <= cap_mib, (res.peak_mib, cap_mib)
    _record("ordered", dict(peak_mib=res.peak_mib, sampled_peak_mib=res.sampled_peak_mib,
                            cap_mib=cap_mib, src_mib=sw / MiB, dst_mib=dw / MiB,
                            phases=phases, max_gap_ms=res.max_gap_us() / 1000))
    print(f"\nordered: exact peak {res.peak_mib:.0f} MiB (sampled {res.sampled_peak_mib:.0f}) "
          f"≤ cap {cap_mib:.0f} MiB; src {sw / MiB:.0f} MiB, dst {dw / MiB:.0f} MiB; "
          f"phases {({k: round(v) for k, v in phases.items()})}")


@pytest.mark.parametrize("schedule", ["colocated", "del_only"])
def test_coresident_schedules_are_killed(models, cap, tmp_path, schedule):
    root, _, _ = models
    cap_mib, _, _ = cap
    rc, res, _ = _run(root, schedule, cap_mib, tmp_path)
    assert res.killed and rc == -9, f"{schedule} survived a cap it should exceed (peak {res.peak_mib:.0f})"
    _record(schedule, dict(killed=res.killed, killed_in=res.killed_in_phase(), cap_mib=cap_mib,
                           kill_reason=res.kill_reason))
    print(f"\n{schedule}: killed in phase {res.killed_in_phase()!r} at cap {cap_mib:.0f} MiB")


def test_clear_no_wait_is_reported(models, cap, tmp_path):
    root, _, _ = models
    cap_mib, _, _ = cap
    outcomes = []
    for i in range(3):
        (tmp_path / str(i)).mkdir()
        rc, res, _ = _run(root, "clear_no_wait", cap_mib, tmp_path / str(i))
        outcomes.append((res.killed, round(res.peak_mib)))
    _record("clear_no_wait", dict(runs=[dict(killed=k, peak_mib=p) for k, p in outcomes],
                                  cap_mib=cap_mib))
    print(f"\nclear_no_wait over 3 runs (killed, peak MiB): {outcomes} at cap {cap_mib:.0f}")
