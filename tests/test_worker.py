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
