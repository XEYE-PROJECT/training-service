"""Composición del worker: la opción ``embedding_model`` del job elige el embedder."""

from __future__ import annotations

from app.core.worker import Worker
from app.infrastructure.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from tests.conftest import FakeEmbedder, make_job, make_settings


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

    instances: list["RecordingReporter"] = []

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
