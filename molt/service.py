"""Molt as a service: a streaming inference server with elastic tiering.

This is the deployment face of the prototype.  Everything below is a thin shell
around :class:`~molt.scheduler.QoSScheduler` — the point is that the elasticity
is *not* a benchmark artefact: a real client opens a request, tokens stream back,
and if the device comes under memory pressure mid-answer the stream keeps going
on a smaller model and says so in-band.

Endpoints (stdlib ``http.server``; no framework dependency)::

    GET  /health                      liveness + ladder + current budget
    GET  /stats                       live counters: tiers resident, footprint,
                                      migrations, kills (must stay 0)
    POST /v1/generate                 non-streaming; returns the whole answer
    POST /v1/generate/stream          server-sent events, one per token, with
                                      ``tier`` on every event and an explicit
                                      ``migration`` event at each switch
    POST /admin/pressure              set the memory budget (MiB) by hand, or
                                      start/stop a named trace — this is how a
                                      demo shows elasticity without waiting for
                                      a real app to launch

Design notes that matter for running it for real
------------------------------------------------
* **One inference worker thread.** Torch on CPU is not re-entrant across a shared
  KV cache and the whole memory argument depends on a single accountant.  HTTP
  handlers enqueue work; the worker owns every model.
* **Backpressure over rejection.** A request that does not fit *waits* in the
  scheduler's admission queue instead of being refused, and the client sees a
  ``queued`` event.  Refusing under pressure would reintroduce exactly the
  failure this project removes.
* **Nothing is ever killed.** ``/stats`` exposes ``kills``; it is the service's
  SLO and it is expected to be identically zero.

Run it::

    python -m molt.service --ladder qwen --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional

import torch

from .calibration import CalibrationConfig
from .config import (MigrationConfig, MoltConfig, TransplantConfig, get_ladder,
                     resolve_device, resolve_dtype)
from .kv_transplant import ProjectorRegistry
from .metrics import MB, EventLog, now, system_available_mb
from .model_zoo import ModelZoo, load_tokenizer
from .pressure import PressureSource, builtin_traces
from .runtime import (Generation, GenerationRequest, MoltRuntime, OOMKilled,
                      Policy)

__all__ = ["MoltService", "ManualPressure", "serve", "main"]


# --------------------------------------------------------------------------
# pressure that an operator (or a demo button) can drive
# --------------------------------------------------------------------------


class ManualPressure(PressureSource):
    """A budget an operator sets directly, or a trace they start on demand.

    In production this would be replaced by the platform's real signal
    (``memory_pressure`` on macOS, ``memory.pressure_level`` on Linux cgroups,
    ``onTrimMemory`` on Android).  Keeping it behind the same
    :class:`~molt.pressure.PressureSource` interface means swapping it in is a
    one-line change and the scheduler does not know the difference.
    """

    def __init__(self, budget_mb: float):
        self._budget = float(budget_mb)
        self._trace = None
        self._trace_t0 = 0.0
        self._t0 = now()
        self._lock = threading.Lock()

    def start(self) -> None:
        self._t0 = now()

    def elapsed(self) -> float:
        return now() - self._t0

    def set_budget(self, mb: float) -> None:
        with self._lock:
            self._budget = float(mb)
            self._trace = None

    def start_trace(self, trace) -> None:
        with self._lock:
            self._trace = trace
            self._trace_t0 = now()

    def stop_trace(self) -> None:
        with self._lock:
            self._trace = None

    def budget_mb(self) -> float:
        with self._lock:
            if self._trace is None:
                return self._budget
            return self._trace.budget_at(now() - self._trace_t0)

    def label(self) -> str:
        with self._lock:
            if self._trace is None:
                return "manual"
            return self._trace.label_at(now() - self._trace_t0)

    def finished(self) -> bool:
        return False


# --------------------------------------------------------------------------
# the service
# --------------------------------------------------------------------------


@dataclass
class _Job:
    """One in-flight request, owned by the worker thread."""

    request: GenerationRequest
    events: "queue.Queue[Optional[dict]]"
    priority: int = 0
    created: float = field(default_factory=now)
    cancelled: bool = False


class MoltService:
    """Single-worker elastic inference service.

    The worker owns every model and every KV cache.  HTTP threads only enqueue
    jobs and drain per-job event queues, which is what keeps the memory
    accounting (and therefore the no-kill guarantee) single-threaded and true.
    """

    def __init__(self, cfg: MoltConfig, pressure: ManualPressure,
                 max_queue: int = 64, verbose: bool = False):
        self.cfg = cfg
        self.pressure = pressure
        self.device = cfg.torch_device
        self.dtype = cfg.torch_dtype
        self.verbose = verbose

        self.zoo = ModelZoo(self.device, self.dtype, verbose=verbose)
        self.tok = load_tokenizer(cfg.ladder.tokenizer_id)
        self.registry = ProjectorRegistry(cfg.transplant.projector_dir,
                                          cfg.ladder.name, self.device)
        self.log = EventLog()
        self.rt = MoltRuntime(cfg, self.zoo, self.tok, self.registry, self.log)

        self.jobs: "queue.Queue[_Job]" = queue.Queue(maxsize=max_queue)
        self.started = now()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self.counters = dict(requests=0, completed=0, failed=0, kills=0,
                             tokens=0, migrations=0, up_shifts=0, down_shifts=0,
                             context_sheds=0, queued_now=0)
        self.recent_migrations: List[dict] = []
        self.current_tier: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------
    def warm(self) -> Dict[str, float]:
        """Measure every rung once so admission decisions use real numbers.

        Also surfaces a missing-projector problem at start-up rather than at the
        first pressure event, which is the worst possible time to discover it.
        """
        sizes: Dict[str, float] = {}
        for spec in self.cfg.ladder:
            sizes[spec.name] = self.zoo.measure_footprint(spec)
        self.zoo.evict_all()
        missing = self.registry.missing_routes(self.cfg.ladder)
        if missing:
            print(f"[molt] WARNING: no trained projector for {missing}. "
                  f"Migrations on those routes will fall back to an untrained map. "
                  f"Run scripts/train_projectors.py --ladder {self.cfg.ladder.name}",
                  file=sys.stderr)
        return sizes

    def start(self) -> None:
        self.pressure.start()
        self._worker = threading.Thread(target=self._run_worker, daemon=True,
                                        name="molt-worker")
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=10)
        self.zoo.evict_all()

    # -- submission --------------------------------------------------------
    def submit(self, prompt: str, max_new_tokens: int, priority: int = 0,
               app_id: Optional[str] = None) -> _Job:
        req = GenerationRequest(
            app_id=app_id or f"req-{self.counters['requests']}",
            prompt=prompt,
            max_new_tokens=max(1, min(max_new_tokens, 4096)),
            priority=priority)
        job = _Job(request=req, events=queue.Queue(), priority=priority)
        with self._lock:
            self.counters["requests"] += 1
        self.jobs.put(job, timeout=30)
        with self._lock:
            self.counters["queued_now"] = self.jobs.qsize()
        return job

    # -- the worker --------------------------------------------------------
    def _pressure_level(self) -> float:
        return self.rt.broker.pressure(self.budget_mb())

    def budget_mb(self) -> float:
        return self.pressure.budget_mb()

    def _run_worker(self) -> None:
        while not self._stop.is_set():
            try:
                job = self.jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._serve(job)
            except Exception:                    # pragma: no cover
                with self._lock:
                    self.counters["failed"] += 1
                job.events.put(dict(type="error", error=traceback.format_exc(limit=3)))
            finally:
                job.events.put(None)             # sentinel: stream complete
                with self._lock:
                    self.counters["queued_now"] = self.jobs.qsize()

    def _wait_for_room(self, job: _Job, tier) -> Optional[Any]:
        """Block until the requested rung fits, degrading the choice as needed.

        Backpressure, not rejection: the client is told it is queued and why.
        A service that returned 503 under memory pressure would have reproduced
        the very failure mode this project exists to remove.
        """
        deadline = now() + 120.0
        told = False
        while now() < deadline and not self._stop.is_set():
            budget = self.budget_mb()
            for candidate in [t for t in self.cfg.ladder if t.tier >= tier.tier]:
                need = self.zoo.incremental_mb(candidate)
                per_tok = self.zoo.kv_mb_per_token(candidate) or 0.05
                est = need + per_tok * (len(job.request.prompt) // 3
                                        + job.request.max_new_tokens)
                if self.rt.broker.would_fit(est, budget):
                    return candidate
            if not told:
                job.events.put(dict(type="queued",
                                    reason="waiting for memory",
                                    budget_mb=round(budget, 1),
                                    in_use_mb=round(self.rt.broker.usage_mb(), 1)))
                told = True
            time.sleep(0.2)
        return None

    def _serve(self, job: _Job) -> None:
        tier = self.cfg.ladder.top
        admitted = self._wait_for_room(job, tier)
        if admitted is None:
            job.events.put(dict(type="error", error="timed out waiting for memory"))
            with self._lock:
                self.counters["failed"] += 1
            return

        gen: Generation = self.rt.make_generation(
            job.request, Policy.MOLT, admitted,
            pressure_fn=self._pressure_level, budget_fn=self.budget_mb,
            enforce_oom=False)      # the service degrades; it never dies
        t0 = now()
        try:
            gen.prefill()
            job.events.put(dict(type="start", tier=gen.tier.name,
                                tier_label=gen.tier.label,
                                prompt_tokens=len(gen.prompt_ids),
                                ttft_ms=round((now() - t0) * 1000, 1)))
            self.current_tier = gen.tier.name
            last_tier = gen.tier.name
            n_migrations_before = len(gen.transplants)

            while not gen.done and not self._stop.is_set() and not job.cancelled:
                tid = gen.step()
                if tid is None:
                    break
                if gen.tier.name != last_tier:
                    rep = gen.transplants[-1] if gen.transplants else None
                    m = dict(type="migration", **{
                        "from": last_tier, "to": gen.tier.name,
                        "tier_label": gen.tier.label,
                        "at_token": len(gen.token_ids) - 1,
                        "method": rep.method if rep else "?",
                        "cost_ms": round(rep.wall_ms, 1) if rep else None,
                        "flops_saved": round(rep.flops_saving, 3) if rep else None,
                        "pressure": round(self._pressure_level(), 3),
                    })
                    job.events.put(m)
                    with self._lock:
                        self.counters["migrations"] += 1
                        key = ("up_shifts"
                               if gen.tier.tier < self.cfg.ladder.by_name(last_tier).tier
                               else "down_shifts")
                        self.counters[key] += 1
                        self.recent_migrations.append(m)
                        del self.recent_migrations[:-32]
                    last_tier = gen.tier.name
                    self.current_tier = last_tier

                piece = self.tok.decode([tid], skip_special_tokens=True)
                job.events.put(dict(type="token", text=piece, token_id=int(tid),
                                    tier=gen.tier.name, index=len(gen.token_ids) - 1))
                with self._lock:
                    self.counters["tokens"] += 1

            res = gen.result()
            sheds = len([e for e in self.log.of_kind("context_shed")])
            with self._lock:
                self.counters["completed"] += 1
                self.counters["context_sheds"] = sheds
                if res.killed:
                    self.counters["kills"] += 1
            job.events.put(dict(
                type="done", text=res.text, tokens=len(res.token_ids),
                tiers=sorted(set(res.tier_per_token)),
                migrations=len(res.transplants) - n_migrations_before,
                total_ms=round((now() - t0) * 1000, 1),
                killed=res.killed,
                migration_ms=round(sum(t.wall_ms for t in res.transplants), 1)))
        except OOMKilled as exc:                 # pragma: no cover - should not happen
            with self._lock:
                self.counters["kills"] += 1
            job.events.put(dict(type="error", error=str(exc)))
        finally:
            gen.close()
            self.current_tier = None

    # -- introspection -----------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            counters = dict(self.counters)
            recent = list(self.recent_migrations[-8:])
        resident = [dict(tier=lt.spec.name, label=lt.spec.label,
                         mb=round(lt.mb, 1), refs=lt.refcount)
                    for lt in self.zoo.resident()]
        return dict(
            uptime_s=round(now() - self.started, 1),
            ladder=self.cfg.ladder.name,
            budget_mb=round(self.budget_mb(), 1),
            pressure_label=self.pressure.label(),
            in_use_mb=round(self.rt.broker.usage_mb(), 1),
            peak_mb=round(self.rt.broker.peak_mb, 1),
            headroom_mb=round(self.budget_mb() - self.rt.broker.usage_mb(), 1),
            utilisation=round(self._pressure_level(), 3),
            system_available_mb=round(system_available_mb(), 1),
            current_tier=self.current_tier,
            resident=resident,
            counters=counters,
            recent_migrations=recent,
            slo_zero_kill=(counters["kills"] == 0),
        )

    def health(self) -> Dict[str, Any]:
        return dict(
            status="ok",
            ladder=self.cfg.ladder.name,
            tiers=[dict(name=t.name, label=t.label, tier=t.tier, quant=t.quant,
                        est_mb=t.est_weight_mb) for t in self.cfg.ladder],
            device=str(self.device), dtype=str(self.dtype),
            budget_mb=round(self.budget_mb(), 1),
            worker_alive=bool(self._worker and self._worker.is_alive()),
        )


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------


def _make_handler(service: MoltService, static_dir: Optional[str]):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Molt/0.1"
        protocol_version = "HTTP/1.1"

        # -- helpers ------------------------------------------------------
        def _json(self, obj, code: int = 200) -> None:
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            return json.loads(self.rfile.read(n) or b"{}")

        def log_message(self, fmt, *a):            # quieter default logging
            if service.verbose:
                super().log_message(fmt, *a)

        # -- routes -------------------------------------------------------
        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                return self._json(service.health())
            if path == "/stats":
                return self._json(service.stats())
            if path in ("/", "/index.html") and static_dir:
                return self._serve_static("index.html")
            if static_dir and path.startswith("/static/"):
                return self._serve_static(path[len("/static/"):])
            return self._json(dict(error="not found", path=path), 404)

        def _serve_static(self, name: str) -> None:
            safe = os.path.normpath(name).lstrip("/.")
            full = os.path.join(static_dir, safe)
            if not os.path.isfile(full):
                return self._json(dict(error="not found"), 404)
            with open(full, "rb") as fh:
                body = fh.read()
            ctype = ("text/html; charset=utf-8" if full.endswith(".html")
                     else "text/plain; charset=utf-8")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                payload = self._read_json()
            except Exception as exc:
                return self._json(dict(error=f"bad json: {exc}"), 400)

            if path == "/admin/pressure":
                return self._admin_pressure(payload)
            if path in ("/v1/generate", "/v1/generate/stream"):
                return self._generate(payload, stream=path.endswith("/stream"))
            return self._json(dict(error="not found", path=path), 404)

        def _admin_pressure(self, payload: Dict[str, Any]):
            if "budget_mb" in payload:
                service.pressure.set_budget(float(payload["budget_mb"]))
                return self._json(dict(ok=True, budget_mb=service.budget_mb()))
            if payload.get("trace"):
                top = max((t.est_weight_mb for t in service.cfg.ladder), default=6000.0)
                sizes = {lt.spec.name: lt.mb for lt in service.zoo.resident()}
                top = sizes.get(service.cfg.ladder.top.name, top)
                traces = builtin_traces(top_tier_mb=top)
                name = payload["trace"]
                if name == "stop":
                    service.pressure.stop_trace()
                    return self._json(dict(ok=True, trace=None))
                if name not in traces:
                    return self._json(dict(error=f"unknown trace {name}",
                                           have=sorted(traces)), 400)
                service.pressure.start_trace(traces[name])
                return self._json(dict(ok=True, trace=name,
                                       description=traces[name].description))
            return self._json(dict(error="send {budget_mb} or {trace}"), 400)

        def _generate(self, payload: Dict[str, Any], stream: bool):
            prompt = payload.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                return self._json(dict(error="'prompt' is required"), 400)
            job = service.submit(
                prompt=prompt,
                max_new_tokens=int(payload.get("max_new_tokens", 96)),
                priority=int(payload.get("priority", 0)),
                app_id=payload.get("app_id"))

            if not stream:
                out, events = None, []
                while True:
                    ev = job.events.get()
                    if ev is None:
                        break
                    events.append(ev)
                    if ev["type"] == "done":
                        out = ev
                if out is None:
                    err = next((e for e in events if e["type"] == "error"), None)
                    return self._json(dict(error=(err or {}).get("error", "failed")), 500)
                return self._json(dict(
                    text=out["text"], tokens=out["tokens"], tiers=out["tiers"],
                    migrations=[e for e in events if e["type"] == "migration"],
                    total_ms=out["total_ms"], migration_ms=out["migration_ms"]))

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    ev = job.events.get()
                    if ev is None:
                        self.wfile.write(b"data: {\"type\":\"end\"}\n\n")
                        self.wfile.flush()
                        break
                    self.wfile.write(f"data: {json.dumps(ev, default=str)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                job.cancelled = True

    return Handler


def serve(service: MoltService, host: str = "127.0.0.1", port: int = 8000,
          static_dir: Optional[str] = None) -> None:
    service.start()
    httpd = ThreadingHTTPServer((host, port), _make_handler(service, static_dir))
    banner = (
        f"\n  Molt service — elastic on-device inference\n"
        f"  ladder   : {service.cfg.ladder.name} "
        f"({' -> '.join(t.label for t in service.cfg.ladder)})\n"
        f"  device   : {service.device} / {service.dtype}\n"
        f"  budget   : {service.budget_mb():.0f} MiB "
        f"(POST /admin/pressure to change it)\n"
        f"  listening: http://{host}:{port}\n")
    if static_dir:
        banner += f"  demo UI  : http://{host}:{port}/\n"
    print(banner, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down…")
    finally:
        httpd.server_close()
        service.stop()


def build_service(args) -> MoltService:
    ladder = get_ladder(args.ladder)
    cfg = MoltConfig(
        ladder=ladder, device=args.device, dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        transplant=TransplantConfig(recompute_top_k=args.recompute_top_k,
                                    projector_dir=args.projector_dir),
        calibration=CalibrationConfig(mode=args.calibration,
                                      blend_steps=args.blend_steps),
        migration=MigrationConfig(down_threshold=args.down_threshold,
                                  up_threshold=args.up_threshold,
                                  cooldown_steps=args.cooldown,
                                  up_shift_patience=args.up_patience),
    )
    pressure = ManualPressure(args.budget_mb if args.budget_mb > 0 else float("inf"))
    svc = MoltService(cfg, pressure, verbose=args.verbose)
    print("[molt] measuring tier footprints…", flush=True)
    sizes = svc.warm()
    for name, mb in sizes.items():
        print(f"       {name:6s} {mb:8.0f} MiB", flush=True)
    if args.budget_mb <= 0:
        # default to something that comfortably holds the top rung, so the
        # operator sees the interesting behaviour only when they ask for it
        pressure.set_budget(max(sizes.values()) * 1.6)
    return svc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Molt elastic inference service")
    p.add_argument("--ladder", default="qwen", choices=["synthetic", "qwen", "qwen-3b"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--budget-mb", type=float, default=0.0,
                   help="initial memory budget; 0 = derive from the ladder")
    p.add_argument("--projector-dir", default="artifacts/projectors")
    p.add_argument("--recompute-top-k", type=int, default=6)
    p.add_argument("--calibration", default="dual",
                   choices=["dual", "frozen_ref", "none"])
    p.add_argument("--blend-steps", type=int, default=6)
    p.add_argument("--down-threshold", type=float, default=0.85)
    p.add_argument("--up-threshold", type=float, default=0.55)
    p.add_argument("--cooldown", type=int, default=8)
    p.add_argument("--up-patience", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-ui", action="store_true", help="disable the demo page")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    static = None if args.no_ui else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "static")
    if static and not os.path.isdir(static):
        static = None
    serve(build_service(args), args.host, args.port, static)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
