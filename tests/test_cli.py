"""Entrypoint one-shot: de dónde lee el job y qué código de salida devuelve al backend."""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass, field

import pytest

from app.entrypoints import cli
from app.infrastructure.job_loader import InvalidJobError
from tests.conftest import make_settings


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


class PipedStdin(io.StringIO):
    def isatty(self) -> bool:
        return False


class TerminalStdin(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # Ni el entorno del desarrollador ni el terminal deciden qué job se lee.
    monkeypatch.delenv("TRAINING_JOB", raising=False)
    monkeypatch.delenv("TRAINING_DATA_PATH", raising=False)
    monkeypatch.setattr(sys, "stdin", TerminalStdin())


# --- _read_payload --------------------------------------------------------------------


def test_read_payload_prefers_the_inline_env(monkeypatch, tmp_path):
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"training_id": 2}), encoding="utf-8")
    monkeypatch.setenv("TRAINING_DATA_PATH", str(path))
    monkeypatch.setenv("TRAINING_JOB", json.dumps({"training_id": 1}))

    assert cli._read_payload() == {"training_id": 1}


def test_read_payload_from_the_file_the_backend_mounts(monkeypatch, tmp_path):
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job_payload()), encoding="utf-8")
    monkeypatch.setenv("TRAINING_DATA_PATH", str(path))

    assert cli._read_payload() == job_payload()


def test_read_payload_from_stdin_when_piped(monkeypatch):
    monkeypatch.setattr(sys, "stdin", PipedStdin(json.dumps(job_payload()) + "\n"))

    assert cli._read_payload() == job_payload()


@pytest.mark.parametrize("stdin", [TerminalStdin, lambda: PipedStdin("   \n")])
def test_read_payload_without_any_source_is_an_invalid_job(monkeypatch, stdin):
    monkeypatch.setattr(sys, "stdin", stdin())

    with pytest.raises(InvalidJobError, match="No job"):
        cli._read_payload()


# --- main ------------------------------------------------------------------------------


@dataclass
class FakeWorker:
    outcome: dict | None = None
    error: Exception | None = None
    jobs: list = field(default_factory=list)

    def run(self, job) -> dict:
        self.jobs.append(job)
        if self.error:
            raise self.error
        return self.outcome or {}


@dataclass
class Harness:
    worker: FakeWorker
    builds: int = 0
    flushes: int = 0


@pytest.fixture
def harness(monkeypatch) -> Harness:
    """Aísla ``main()``: settings de test (sin leer el .env real), sin logging global, sin
    Sentry y sin cargar modelos (worker fake)."""
    state = Harness(worker=FakeWorker(outcome={"status": "ok"}))

    def build_worker(settings):
        state.builds += 1
        return state.worker

    def flush_sentry():
        state.flushes += 1

    monkeypatch.setattr(cli, "get_settings", lambda: make_settings())
    monkeypatch.setattr(cli, "configure_logging", lambda level: None)
    monkeypatch.setattr(cli, "init_sentry", lambda settings: False)
    monkeypatch.setattr(cli, "build_worker", build_worker)
    monkeypatch.setattr(cli, "flush_sentry", flush_sentry)
    return state


def test_main_exits_0_when_the_backend_confirmed_the_callback(monkeypatch, harness):
    monkeypatch.setenv("TRAINING_JOB", json.dumps(job_payload()))

    assert cli.main() == 0
    assert [job.training_id for job in harness.worker.jobs] == [12]
    assert harness.flushes == 1


def test_main_exits_1_when_the_training_failed(monkeypatch, harness):
    monkeypatch.setenv("TRAINING_JOB", json.dumps(job_payload()))
    harness.worker.outcome = {"status": "error", "error": "callback_failed"}

    assert cli.main() == 1
    assert harness.flushes == 1


@pytest.mark.parametrize("raw", ['{"training_id": 1}', "{not json"])
def test_main_exits_2_on_an_invalid_job_without_loading_models(monkeypatch, harness, raw):
    monkeypatch.setenv("TRAINING_JOB", raw)

    assert cli.main() == 2
    assert harness.builds == 0


def test_main_flushes_sentry_even_if_the_worker_raises(monkeypatch, harness):
    # Worker.run no debería lanzar nunca; si lo hiciera, los eventos de Sentry deben salir igualmente.
    monkeypatch.setenv("TRAINING_JOB", json.dumps(job_payload()))
    harness.worker.error = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        cli.main()
    assert harness.flushes == 1
