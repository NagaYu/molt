"""The service layer: does the deployable face keep the guarantees?

Claims under test
-----------------
* **no-stall / zero-kill as an SLO** — a request that spans a memory squeeze
  still returns a complete answer, on a smaller model, and ``kills`` stays 0.
* **backpressure, not rejection** — a request that cannot fit waits and says so,
  rather than being refused.
* the wire protocol actually reports the tier per token and an explicit
  migration event, which is what makes elasticity visible to a caller.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest
import torch

from molt.config import CalibrationConfig, MigrationConfig
from molt.service import ManualPressure, MoltService, _make_handler

from .conftest import PROMPT, make_config


@pytest.fixture
def service(ladder, projector_dir):
    cfg = make_config(ladder, projector_dir, max_new_tokens=24,
                      calibration=CalibrationConfig(mode="dual", blend_steps=3),
                      migration=MigrationConfig(down_threshold=0.85, up_threshold=0.5,
                                                cooldown_steps=2, up_shift_patience=4))
    svc = MoltService(cfg, ManualPressure(float("inf")))
    sizes = svc.warm()
    svc.pressure.set_budget(max(sizes.values()) * 2.0)
    svc.start()
    yield svc
    svc.stop()


def _drain(job, timeout=90.0):
    events, t0 = [], time.time()
    while time.time() - t0 < timeout:
        ev = job.events.get(timeout=timeout)
        if ev is None:
            break
        events.append(ev)
    return events


def test_service_streams_tokens_with_their_tier(service):
    job = service.submit(PROMPT, max_new_tokens=12)
    events = _drain(job)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "done", kinds[:3]
    toks = [e for e in events if e["type"] == "token"]
    assert len(toks) == 12
    assert all("tier" in t and t["tier"] for t in toks), "every token must name its rung"
    assert [t["index"] for t in toks] == list(range(12))
    assert service.stats()["counters"]["kills"] == 0


def test_a_squeeze_mid_answer_migrates_without_dropping_the_stream(service, ladder):
    """The headline behaviour, exercised through the public surface.

    The budget is cut while the answer is in flight; the stream must continue,
    emit a migration event, and finish with the full token count.
    """
    job = service.submit(PROMPT, max_new_tokens=48)

    def squeeze():
        # Trigger on *progress*, not on a sleep: the synthetic ladder decodes a
        # short answer in well under a second, and a time-based trigger would
        # race the generation and silently test nothing.
        base = service.counters["tokens"]
        deadline = time.time() + 30
        while time.time() < deadline:
            if service.counters["tokens"] - base >= 6:
                service.pressure.set_budget(service.zoo.known_mb(ladder.top) * 0.55)
                return
            time.sleep(0.01)

    threading.Thread(target=squeeze, daemon=True).start()
    events = _drain(job)

    done = [e for e in events if e["type"] == "done"]
    assert done, [e["type"] for e in events]
    assert done[0]["tokens"] == 48, "the stream was truncated by the squeeze"
    assert not done[0]["killed"]
    migs = [e for e in events if e["type"] == "migration"]
    assert migs, "the squeeze did not trigger a migration"
    m = migs[0]
    assert m["from"] != m["to"] and m["cost_ms"] is not None
    tiers = {t["tier"] for t in events if t["type"] == "token"}
    assert len(tiers) > 1, f"tokens all came from one rung: {tiers}"
    assert service.stats()["counters"]["kills"] == 0
    service.pressure.set_budget(float("inf"))


def test_request_that_does_not_fit_waits_instead_of_being_refused(service, ladder):
    """**Backpressure.**  Refusing under pressure would reintroduce the failure
    mode the whole project removes, so the contract is "queue and tell them"."""
    service.pressure.set_budget(service.zoo.known_mb(ladder.bottom) * 0.4)
    job = service.submit(PROMPT, max_new_tokens=4)
    try:
        first = job.events.get(timeout=30)
        assert first["type"] == "queued", first
        assert "budget_mb" in first and "in_use_mb" in first
    finally:
        job.cancelled = True
        service.pressure.set_budget(float("inf"))
        _drain(job, timeout=180)


def test_stats_and_health_expose_the_slo(service):
    h = service.health()
    assert h["status"] == "ok" and h["worker_alive"]
    assert len(h["tiers"]) == len(service.cfg.ladder)
    s = service.stats()
    for key in ("budget_mb", "in_use_mb", "headroom_mb", "utilisation",
                "counters", "slo_zero_kill"):
        assert key in s, key
    assert s["slo_zero_kill"] is True


def test_http_surface_round_trips(service):
    """The HTTP layer itself: JSON in, JSON out, on a real socket."""
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(service, None))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
            assert json.loads(r.read())["status"] == "ok"

        body = json.dumps(dict(prompt=PROMPT, max_new_tokens=6)).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            out = json.loads(r.read())
        assert out["tokens"] == 6 and isinstance(out["text"], str)
        assert isinstance(out["tiers"], list) and out["tiers"]

        bad = urllib.request.Request(f"http://127.0.0.1:{port}/v1/generate",
                                     data=b'{"max_new_tokens": 4}',
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(bad, timeout=10)
        assert exc.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_admin_can_drive_the_budget(service):
    service.pressure.set_budget(1234.0)
    assert abs(service.budget_mb() - 1234.0) < 1e-6
    assert abs(service.stats()["budget_mb"] - 1234.0) < 1e-6
    service.pressure.set_budget(float("inf"))
