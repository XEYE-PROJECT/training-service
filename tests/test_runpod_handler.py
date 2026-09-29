"""Entrypoint de RunPod Serverless. El módulo carga settings y modelos al importarse (para que
un worker caliente los reutilice), así que se importa *dentro* del test con las factorías ya
sustituidas: sin leer el .env real, sin cargar modelos y sin red."""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass, field

import pytest

import app.core.config as config_module
import app.core.logging as logging_module
import app.core.worker as worker_module
from tests.conftest import make_settings

MODULE = "app.entrypoints.runpod_handler"


def job_payload() -> dict:
    return {
        "training_id": 12,
        "list_id": 5,
        "user_id": 1,
        "callback_url": "http://backend:8000/webhooks/training-update",
        "list": {"id": 5, "name": "Museos"},
        "elements": [{"id": 1, "text": "Museo A"}],
        "options": [{"key": "train_all", "value": True}],
    }


class FakeWorker:
    def __init__(self) -> None:
        self.jobs: list = []
        self.error: Exception | None = None

    def run(self, job) -> dict:
        self.jobs.append(job)
        if self.error:
            raise self.error
        return {"status": "ok", "training_id": job.training_id, "elements": len(job.elements)}


@dataclass
class Handler:
    module: types.ModuleType
    worker: FakeWorker
    builds: int = 0
    flushes: int = 0
    unused: list = field(default_factory=list)


@pytest.fixture
def handler(monkeypatch) -> Handler:
    # El SDK se sustituye siempre por un stub: el módulo solo lo usa en __main__, importarlo de
    # verdad cuesta ~5 s y así el test tampoco depende de que esté instalado (venv mínimo).
    stub = types.SimpleNamespace(serverless=types.SimpleNamespace(start=lambda config: None))
    monkeypatch.setitem(sys.modules, "runpod", stub)

    worker = FakeWorker()
    state = {"builds": 0}

    def build_worker(settings=None):
        state["builds"] += 1
        return worker

    monkeypatch.setattr(config_module, "get_settings", lambda: make_settings())
    monkeypatch.setattr(logging_module, "configure_logging", lambda level: None)
    monkeypatch.setattr(worker_module, "build_worker", build_worker)

    sys.modules.pop(MODULE, None)
    module = importlib.import_module(MODULE)
    result = Handler(module=module, worker=worker, builds=state["builds"])
    monkeypatch.setattr(module, "flush_sentry", lambda: setattr(result, "flushes", result.flushes + 1))
    yield result
    sys.modules.pop(MODULE, None)  # que ningún otro test herede el módulo con los fakes


def test_importing_the_module_builds_the_worker_once(handler):
    assert handler.builds == 1


def test_handler_delegates_a_valid_job_to_the_worker_and_flushes_sentry(handler):
    outcome = handler.module.handler({"input": job_payload()})

    assert outcome == {"status": "ok", "training_id": 12, "elements": 1}
    assert [job.training_id for job in handler.worker.jobs] == [12]
    assert handler.flushes == 1


def test_a_warm_worker_serves_several_jobs_without_rebuilding(handler):
    handler.module.handler({"input": job_payload()})
    handler.module.handler({"input": {**job_payload(), "training_id": 13}})

    assert [job.training_id for job in handler.worker.jobs] == [12, 13]
    assert handler.flushes == 2


@pytest.mark.parametrize(
    "event",
    [{}, None, {"input": {"training_id": 1}}, {"input": "no es un objeto"}],
    ids=["empty", "none", "missing-fields", "not-an-object"],
)
def test_invalid_events_answer_an_error_without_raising(handler, event):
    outcome = handler.module.handler(event)

    assert outcome["status"] == "error"
    assert outcome["error"]
    assert handler.worker.jobs == []


def test_sentry_is_flushed_even_if_the_worker_raises(handler):
    handler.worker.error = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        handler.module.handler({"input": job_payload()})
    assert handler.flushes == 1
