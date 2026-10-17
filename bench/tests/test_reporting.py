"""The reporting chain maps data to numbers correctly.

Synthetic results with known values go through analyze → make_paper_figures →
make_results_md; the test checks the numbers that come out the other end.  If
this chain were wrong, every number in the paper would be wrong with it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _w(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def _synthetic(res: Path):
    pair = "1.5b-0.5b"
    m = lambda k, nll: dict(policy="r2_topk", k=k, insample_r2_k=0.9, insample_r2_v=0.8, params=1e6,
                            mib_fp16=2.0, sources=[], heldout_nll=[nll, nll - 0.1, nll + 0.1],
                            heldout_r2_k=0.7, heldout_r2_v=0.6)
    _w(res / "fit" / f"{pair}.json", dict(
        pair=pair, n_rows=128000, corpus=dict(source="fake"), single_source_r2=[[0.5]],
        mappers={"heo_k1": m(1, 3.0), "heo_k8": m(8, 2.5), "depth_linear": dict(m(1, 3.2), policy="depth_linear")},
        natives=dict(native_dst=[2.25, 2.2, 2.3], native_src=[2.0, 1.9, 2.1]), selected="heo_k8",
        lam_sweep={"0.01": [2.5, 2.4, 2.6]}, anchors={"4/32": dict(sink=4, tail=32, nll=[2.4, 2.3, 2.5])},
        molt_anchor=dict(sink=4, tail=32, nll=[2.4, 2.3, 2.5]), seconds=1, machine_start={}))
    rows = []
    for i in range(4):
        for cond, acc in (("C", 1.0), ("D", 0.0), ("A2", 1.0)):
            rows.append(dict(cond=cond, id=f"needle-{i}", kind="needle", task_correct=acc, agree_post=0.5,
                             jump=0.25 if cond == "C" else 0.0, lost_tokens=0, time_to_resume_s=1.0,
                             time_to_new_content_s=1.0, migration_s=0.5))
    p = res / "scored" / "quality" / f"{pair}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows))
    _w(res / "raw" / "memory" / pair / "plan.json",
       dict(S=1000, S_small=400, dst_w=300, mapper_w=50, need_colocated=446, h=[128]))
    for i, (cond, killed) in enumerate((("C", True), ("C", True), ("D", False), ("D", False), ("D", True))):
        _w(res / "raw" / "memory" / pair / "h128" / f"{cond}__open-0{i}__r0.json",
           dict(pair=pair, cond=cond, id=f"open-0{i}", cap_mib=1128, killed=killed, rc=-9 if killed else 0,
                error=False, killed_in="load_dst" if killed else None, peak_mib=1100 if not killed else 1129,
                phase_peaks_mib={}, worker=None if killed else dict(time_to_resume_s=2.0, migration=dict(seconds=1.0))))


def test_reporting_chain(tmp_path):
    res = tmp_path / "results"
    _synthetic(res)
    env = dict(os.environ, MOLT_RESULTS=str(res), MOLT_FIG_DIR=str(tmp_path / "fig"),
               MOLT_GEN_DIR=str(tmp_path / "gen"), MOLT_RESULTS_MD=str(tmp_path / "RESULTS.md"))
    for cmd in (["-m", "bench.analyze"], ["figures/make_paper_figures.py"], ["-m", "bench.make_results_md"]):
        subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, check=True, capture_output=True)
    M = json.loads((res / "macros.json").read_text())
    assert M["fit.15b-05b.kstar"] == "8"
    assert M["fit.15b-05b.gap_nll"] == "0.250"             # 2.5 − 2.25
    assert M["q.15b-05b.C.needle"] == "1.000" and M["q.15b-05b.D.needle"] == "0.000"
    assert M["qp.15b-05b.C.A2.jump"] == "0.250"             # paired C − A′ seam
    assert M["m.15b-05b.h128.C.surv"] == "0/2" and M["m.15b-05b.h128.D.surv"] == "2/3"
    assert M["m.15b-05b.h128.C.killedin"] == r"load\_dst"
    md = (tmp_path / "RESULTS.md").read_text()
    assert "| 128 | 1128 | (D) molt | 2/3 |" in md
    assert "| (C) ridge mid-decode, colocated | — | 1.00 [1.00, 1.00] |" in md
    tex = (tmp_path / "gen" / "results.tex").read_text()
    assert r"\csname res:m.15b-05b.h128.D.surv\endcsname{2/3}" in tex
