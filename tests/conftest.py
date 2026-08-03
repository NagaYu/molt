"""Shared fixtures.  The whole suite is hermetic: no downloads, no network.

Every test runs on the ``synthetic`` ladder — randomly-initialised Qwen2 models
that are structurally identical to the real ones (GQA, RoPE, tied embeddings,
different depth *and* different head_dim between the top and bottom rungs) but
small enough that the suite finishes in seconds.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molt.config import (CalibrationConfig, MigrationConfig, MoltConfig,
                         TransplantConfig, get_ladder)
from molt.fit_projector import fit_all_routes
from molt.kv_transplant import ProjectorRegistry
from molt.metrics import EventLog
from molt.model_zoo import ModelZoo, load_tokenizer
from molt.runtime import MoltRuntime

CALIB_TEXTS = [
    "the quick brown fox jumps over the lazy dog and keeps running for a while",
    "memory pressure forces the runtime to move a generation onto a smaller model",
    "a projector is fitted by least squares on a small calibration corpus of text",
    "tokens continue to stream while the model underneath them is being replaced",
    "0123456789 punctuation ,.;:!? and some MiXeD case Words To Vary The Input",
    "second sample line with quite different characters $%&*()[]{}<>|\\/~`^",
]

# Held-out text used to *evaluate* a projector.  Evaluating on uniformly random
# token ids would be unfair in both directions: the maps are fitted on natural
# text, and a random-id prefix is out of distribution for the models too.
HELD_OUT_TEXT = ("a different sentence that was never part of the calibration set "
                 "and mentions unrelated things like harbours, ledgers and rainfall")

RECOMPUTE_K = 2


def held_out_ids(tokenizer, n: int = 48):
    """Tokenise the held-out text and pad/repeat to exactly ``n`` tokens."""
    import torch

    ids = tokenizer(HELD_OUT_TEXT, return_tensors="pt")["input_ids"][0].tolist()
    while len(ids) < n:
        ids = ids + ids
    return torch.tensor([ids[:n]], dtype=torch.long)


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def ladder():
    return get_ladder("synthetic")


@pytest.fixture(scope="session")
def tokenizer(ladder):
    return load_tokenizer(ladder.tokenizer_id)


@pytest.fixture(scope="session")
def projector_dir(tmp_path_factory, ladder, tokenizer, device):
    """Fit every route once for the whole session."""
    out = str(tmp_path_factory.mktemp("projectors"))
    zoo = ModelZoo(device, torch.float32)
    cfg = TransplantConfig(recompute_top_k=RECOMPUTE_K, projector_dir=out)
    fit_all_routes(zoo, ladder, tokenizer, cfg, device, out_dir=out,
                   texts=CALIB_TEXTS, verbose=False)
    zoo.evict_all()
    return out


def make_config(ladder, projector_dir, **kw) -> MoltConfig:
    cal = kw.pop("calibration", CalibrationConfig(mode="dual", blend_steps=4))
    mig = kw.pop("migration", MigrationConfig(down_threshold=0.85, up_threshold=0.55,
                                              cooldown_steps=3, up_shift_patience=6))
    tp = kw.pop("transplant", TransplantConfig(recompute_top_k=RECOMPUTE_K,
                                               projector_dir=projector_dir))
    return MoltConfig(ladder=ladder, device="cpu", dtype="float32",
                      transplant=tp, calibration=cal, migration=mig,
                      max_new_tokens=kw.pop("max_new_tokens", 24), **kw)


@pytest.fixture
def runtime_factory(ladder, tokenizer, projector_dir, device):
    """Build a fresh :class:`MoltRuntime` (and its zoo) per test."""
    made = []

    def _make(cfg=None, **kw):
        cfg = cfg or make_config(ladder, projector_dir, **kw)
        zoo = ModelZoo(device, torch.float32)
        reg = ProjectorRegistry(cfg.transplant.projector_dir, ladder.name, device)
        rt = MoltRuntime(cfg, zoo, tokenizer, reg, EventLog())
        made.append((rt, zoo))
        return rt

    yield _make
    for _rt, zoo in made:
        zoo.evict_all()


PROMPT = ("the runtime keeps generating while the model underneath it is exchanged "
          "for a smaller one and then for a larger one again as pressure changes "
          "over the course of a single long answer produced by the assistant")
