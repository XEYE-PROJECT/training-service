"""El algoritmo de entrenamiento como pipeline de pasos sobre un contexto compartido.

Punto de extensión: un paso lee/escribe :class:`TrainingContext` y una estrategia es solo
una lista ordenada de pasos (``strategies.py``) — cambiar el algoritmo no toca entrypoints
ni backend, y el contrato con el search-service (una fila float32 por elemento, id ASC)
se aplica una sola vez, en el paso de embedding.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from app.application.ports import Embedder, Enricher, ProgressReporter
from app.core.config import Settings
from app.domain.models import ElementInput, Enrichment, TrainingJob

logger = logging.getLogger(__name__)


@dataclass
class TrainingContext:
    """Todo lo que un paso puede leer o escribir; los pasos lo mutan in situ."""

    job: TrainingJob
    settings: Settings
    embedder: Embedder
    enricher: Enricher | None = None
    reporter: ProgressReporter | None = None

    #: Elementos en orden id ASC — el orden de filas de todas las matrices. Lo fija el pipeline.
    elements: list[ElementInput] = field(default_factory=list)
    #: id de elemento -> enriquecimiento (de caché o recién generado).
    enrichments: dict[int, Enrichment] = field(default_factory=dict)
    #: id de elemento -> JSON del enriquecimiento, solo los generados *ahora* (vuelven al backend).
    fresh_enrichments: dict[int, str] = field(default_factory=dict)
    #: Matriz final (n, dim) float32; la fila i corresponde a elements[i].
    vectors: np.ndarray | None = None
    #: Segundos por fase, reportados al backend.
    timings: dict[str, int] = field(default_factory=dict)

    def report(self, status: str) -> None:
        if self.reporter:
            self.reporter.phase(status)


class Step(Protocol):
    name: str

    def run(self, ctx: TrainingContext) -> None: ...


@dataclass
class TrainingPipeline:
    """Ejecuta los pasos en orden, cronometrando cada uno."""

    steps: list[Step]

    def run(self, ctx: TrainingContext) -> TrainingContext:
        ctx.elements = ctx.job.sorted_elements()
        for step in self.steps:
            started = time.monotonic()
            step.run(ctx)
            elapsed = int(time.monotonic() - started)
            ctx.timings[f"{step.name}_seconds"] = elapsed
            logger.info("Step '%s' finished in %ds", step.name, elapsed)
        return ctx
