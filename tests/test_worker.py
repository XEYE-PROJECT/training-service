"""Composición del worker: la opción ``embedding_model`` del job elige el embedder, y ``Worker.run``
de extremo a extremo con los fakes (sin red ni modelos)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.core.worker import Worker
from app.domain.wire import decode_matrix
from app.infrastructure.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from tests.conftest import DIM, FakeEmbedder, FakeEnricher, make_job, make_settings


def make_worker() -> Worker:
    return Worker(settings=make_settings(), embedder=FakeEmbedder(), enricher=None)


def test_job_without_model_option_uses_the_default_embedder():
    worker = make_worker()
    assert worker._embedder_for(make_job()) is worker.embedder


def test_job_naming_the_default_model_reuses_the_default_embedder():
    worker = make_worker()
    job = make_job(options={"embedding_model": "fake-model"})
    assert worker._embedder_for(job) is worker.embedder


def test_job_option_selects_another_model_and_caches_it():
    worker = make_worker()
    job = make_job(options={"embedding_model": "paraphrase-multilingual-mpnet-base-v2"})

    embedder = worker._embedder_for(job)

    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert embedder.model_name == "paraphrase-multilingual-mpnet-base-v2"
    # Un worker caliente alterna modelos entre jobs; la instancia debe reutilizarse.
    assert worker._embedder_for(job) is embedder


class RecordingReporter:
    """Sustituye a WebhookReporter: captura el `failed` sin tocar la red."""

    instances: list[RecordingReporter] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.failures: list[str] = []
        RecordingReporter.instances.append(self)

    def phase(self, status: str) -> None:  # pragma: no cover - no debe llamarse
        raise AssertionError("training must not start")

    def failed(self, error: str) -> bool:
        self.failures.append(error)
        return True

    def completed(self, payload) -> bool:  # pragma: no cover - no debe llamarse
        raise AssertionError("training must not start")


def test_missing_webhook_secret_fails_before_any_compute(monkeypatch):
    import app.core.worker as worker_module

    RecordingReporter.instances.clear()
    monkeypatch.setattr(worker_module, "WebhookReporter", RecordingReporter)
    worker = make_worker()
    job = make_job(webhook_secret=None, callback_url="https://backend.xeye.es/webhooks/training-update")

    outcome = worker.run(job)

    assert outcome["status"] == "error"
    assert "WEBHOOK_SECRET" in outcome["error"]
    assert RecordingReporter.instances[0].failures == [outcome["error"]]


def test_dev_webhook_secret_against_a_production_backend_fails_fast(monkeypatch):
    import app.core.worker as worker_module

    RecordingReporter.instances.clear()
    monkeypatch.setattr(worker_module, "WebhookReporter", RecordingReporter)
    worker = Worker(settings=make_settings(webhook_secret="dev-webhook-secret"), embedder=FakeEmbedder(), enricher=None)
    job = make_job(webhook_secret=None, callback_url="https://backend.xeye.es/webhooks/training-update")

    outcome = worker.run(job)

    assert outcome["status"] == "error"
    assert "development" in outcome["error"]
    # El secreto de dev sí vale contra un backend local (http): no se bloquea el flujo de desarrollo.
    assert worker.settings.webhook_secret_problem("http://xeye-java-backend:8000/webhooks") is None


# --- Worker.run de extremo a extremo ------------------------------------------------------


class CapturingReporter:
    """Sustituye a WebhookReporter: captura fases, el payload de `completed` y los `failed`."""

    instances: list[CapturingReporter] = []
    deliver = True  # lo que devuelve `completed` (el backend confirmó o no el callback)

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.phases: list[str] = []
        self.completed_payload: dict | None = None
        self.failures: list[str] = []
        CapturingReporter.instances.append(self)

    def phase(self, status: str) -> None:
        self.phases.append(status)

    def completed(self, payload: dict) -> bool:
        self.completed_payload = payload
        return self.deliver

    def failed(self, error: str) -> bool:
        self.failures.append(error)
        return True


@pytest.fixture
def reporter(monkeypatch) -> type[CapturingReporter]:
    import app.core.worker as worker_module

    CapturingReporter.instances.clear()
    monkeypatch.setattr(CapturingReporter, "deliver", True)
    monkeypatch.setattr(worker_module, "WebhookReporter", CapturingReporter)
    return CapturingReporter


def test_run_trains_end_to_end_and_delivers_the_completed_payload(reporter):
    settings = make_settings(callback_timeout_seconds=12.5, callback_retries=4)
    worker = Worker(settings=settings, embedder=FakeEmbedder(), enricher=FakeEnricher())
    job = make_job()  # elementos 2 y 1 (en ese orden), secreto en el job, callback http

    outcome = worker.run(job)

    assert outcome["status"] == "ok"
    assert (outcome["elements"], outcome["enriched"], outcome["cached"]) == (2, 2, 0)
    assert isinstance(outcome["elapsed_seconds"], int)

    sent = reporter.instances[0]
    assert sent.kwargs == {
        "callback_url": job.callback_url,
        "training_id": 7,
        "list_id": 3,
        "secret": "s3cret",
        "timeout_seconds": 12.5,
        "retries": 4,
    }
    assert sent.phases == ["optimizing", "training"]
    assert sent.failures == []

    payload = sent.completed_payload
    assert (payload["training_id"], payload["list_id"], payload["status"]) == (7, 3, "completed")
    matrix = decode_matrix(payload["embeddings_data"])
    assert matrix.shape == (2, DIM) and matrix.dtype == np.float32
    assert payload["element_ids"] == [1, 2]  # id ASC, no el orden del job
    assert json.loads(payload["model"])["embedding_model"] == "fake-model"
    assert set(payload["generated_descriptions"]) == {"1", "2"}
    assert isinstance(payload["time"]["total_seconds"], int)
    assert payload["cost"] == {"runpod": 0.0, "total": 0.0}  # COMPUTE_PRICE_PER_HOUR=0


def test_run_reports_failed_and_returns_an_error_when_the_use_case_raises(reporter, monkeypatch):
    import app.core.worker as worker_module

    class ExplodingUseCase:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, job, reporter=None):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(worker_module, "RunTraining", ExplodingUseCase)

    outcome = make_worker().run(make_job())

    assert outcome == {"status": "error", "training_id": 7, "error": "CUDA out of memory"}
    assert reporter.instances[0].failures == ["CUDA out of memory"]
    assert reporter.instances[0].completed_payload is None


def test_run_returns_an_error_when_the_backend_does_not_confirm_completed(reporter, monkeypatch):
    # El cómputo se hizo, pero el callback no llegó: el entrypoint debe salir con código != 0.
    monkeypatch.setattr(reporter, "deliver", False)

    outcome = make_worker().run(make_job())

    assert outcome == {"status": "error", "training_id": 7, "error": "callback_failed"}
    assert reporter.instances[0].completed_payload is not None
