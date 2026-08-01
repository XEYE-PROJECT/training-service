from __future__ import annotations

import hashlib

import numpy as np
import pytest

from app.core.config import Settings
from app.domain.models import ElementInput, Enrichment, ListInput, TrainingJob

DIM = 16


def make_settings(**overrides) -> Settings:
    defaults = dict(
        strategy="default",
        embedding_model="fake-model",
        enricher="none",
        enrich_max_elements=0,
        query_variant_weight=0.35,
    )
    defaults.update(overrides)
    return Settings(**defaults)


class FakeEmbedder:
    """Embeddings deterministas por hash: mismo texto -> mismo vector; distinto -> distinto."""

    def __init__(self, name: str = "fake-model") -> None:
        self._name = name
        self.calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return self._name

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            rows.append(np.random.default_rng(seed).normal(size=DIM))
        return np.asarray(rows, dtype=np.float32)


class FakeEnricher:
    """Registra qué se le pidió enriquecer; devuelve una línea de resumen y dos consultas."""

    def __init__(self, name: str = "fake-llm", fail_on: set[int] | None = None) -> None:
        self._name = name
        self._fail_on = fail_on or set()
        self.seen: list[int] = []

    @property
    def model_name(self) -> str:
        return self._name

    def enrich(self, element: ElementInput, list_context: ListInput) -> Enrichment | None:
        self.seen.append(element.id)
        if element.id in self._fail_on:
            raise RuntimeError("boom")
        return Enrichment(
            summary=[f"{element.text} es un elemento"],
            queries=[f"donde comprar {element.text}", f"{element.text} barato"],
            model=self._name,
        )


class FakeBatchEnricher(FakeEnricher):
    """Como FakeEnricher pero con el camino por lotes: registra los lotes que recibe."""

    def __init__(self, name: str = "fake-llm", fail_on: set[int] | None = None) -> None:
        super().__init__(name, fail_on)
        self.batches: list[list[int]] = []

    def enrich_many(self, elements, list_context, heartbeat=None):
        self.batches.append([e.id for e in elements])
        results = {}
        for element in elements:
            try:
                enrichment = self.enrich(element, list_context)
            except RuntimeError:
                continue
            results[element.id] = enrichment
        return results


class FakeReporter:
    def __init__(self) -> None:
        self.phases: list[str] = []
        self.completed_payload: dict | None = None
        self.error: str | None = None

    def phase(self, status: str) -> None:
        self.phases.append(status)

    def completed(self, payload: dict) -> bool:
        self.completed_payload = payload
        return True

    def failed(self, error: str) -> bool:
        self.error = error
        return True


def make_job(elements: list[ElementInput] | None = None, **overrides) -> TrainingJob:
    defaults = dict(
        training_id=7,
        list_id=3,
        callback_url="http://backend/webhooks/training-update",
        webhook_secret="s3cret",
        user_id=1,
        list=ListInput(id=3, name="Herramientas", description="Catálogo de ferretería"),
        elements=elements
        if elements is not None
        else [
            ElementInput(id=2, text="martillo"),
            ElementInput(id=1, text="destornillador"),
        ],
        options={},
    )
    defaults.update(overrides)
    return TrainingJob(**defaults)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()
