"""Puertos de salida como Protocol: los adaptadores no necesitan clase base y los fakes son triviales."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from app.domain.models import ElementInput, Enrichment, ListInput


class Embedder(Protocol):
    """Convierte textos en vectores; las filas vuelven en el mismo orden que ``texts``."""

    @property
    def model_name(self) -> str: ...

    def encode(self, texts: list[str]) -> np.ndarray:
        """Matriz float32 de forma (len(texts), dim)."""
        ...


class Enricher(Protocol):
    """Genera el enriquecimiento LLM de un elemento. ``None`` = sin enriquecimiento disponible."""

    @property
    def model_name(self) -> str | None: ...

    def enrich(self, element: ElementInput, list_context: ListInput) -> Enrichment | None: ...


class ProgressReporter(Protocol):
    """Comunica al backend los cambios de fase y el resultado final."""

    def phase(self, status: str) -> None: ...

    def completed(self, payload: dict[str, Any]) -> bool: ...

    def failed(self, error: str) -> bool: ...
