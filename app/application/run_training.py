"""Caso de uso: ejecutar un entrenamiento de principio a fin.

El mismo código para todos los entrypoints (contenedor one-shot, Lambda, RunPod): construye
el pipeline de la estrategia del job, lo ejecuta y entrega el resultado al reporter — lo
*único* que habla con el backend.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.application.pipeline import TrainingContext
from app.application.ports import Embedder, Enricher, ProgressReporter
from app.application.strategies import build_pipeline
from app.core.config import Settings
from app.domain.models import TrainingJob, TrainingResult
from app.domain.wire import encode_matrix

logger = logging.getLogger(__name__)


class RunTraining:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        enricher: Enricher | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._enricher = enricher

    def execute(self, job: TrainingJob, reporter: ProgressReporter | None = None) -> TrainingResult:
        strategy = job.option("strategy") or self._settings.strategy
        pipeline = build_pipeline(strategy)
        ctx = TrainingContext(
            job=job,
            settings=self._settings,
            embedder=self._embedder,
            enricher=self._enricher,
            reporter=reporter,
        )

        started = time.monotonic()
        pipeline.run(ctx)
        total = int(time.monotonic() - started)

        if ctx.vectors is None:
            raise RuntimeError(f"Strategy '{strategy}' produced no embeddings for list {job.list_id}")

        timings = dict(ctx.timings)
        timings["total_seconds"] = max(total, sum(timings.values()))

        return TrainingResult(
            embeddings_b64=encode_matrix(ctx.vectors),
            model=self._model_metadata(job, strategy, int(ctx.vectors.shape[1])),
            dimension=int(ctx.vectors.shape[1]),
            element_ids=[element.id for element in ctx.elements],
            generated_descriptions=dict(ctx.fresh_enrichments),
            time=timings,
            enriched_count=len(ctx.fresh_enrichments),
            cached_count=len(ctx.enrichments) - len(ctx.fresh_enrichments),
        )

    def _model_metadata(self, job: TrainingJob, strategy: str, dimension: int) -> str:
        """Cadena ``model`` opaca. El search-service lee de ella ``embedding_model`` para
        embeber las consultas en el mismo espacio: debe ser un nombre de modelo *real* y cargable."""
        return json.dumps(
            {
                "embedding_model": self._embedder.model_name,
                "llm_model": self._enricher.model_name if self._enricher else None,
                "strategy": strategy,
                "dimension": dimension,
                "list_id": job.list_id,
            },
            ensure_ascii=False,
        )


def completion_payload(job: TrainingJob, result: TrainingResult, cost: dict[str, Any]) -> dict[str, Any]:
    """Cuerpo del webhook ``completed`` (los nombres de campo los fija el backend)."""
    return {
        "training_id": job.training_id,
        "list_id": job.list_id,
        "status": "completed",
        "embeddings_data": result.embeddings_b64,
        "model": result.model,
        "element_ids": result.element_ids,
        "generated_descriptions": {str(k): v for k, v in result.generated_descriptions.items()},
        "time": result.time,
        "cost": cost,
    }


def compute_cost(seconds: int, price_per_hour: float) -> dict[str, float]:
    if price_per_hour <= 0 or seconds <= 0:
        return {"runpod": 0.0, "total": 0.0}
    amount = round(seconds / 3600.0 * price_per_hour, 6)
    return {"runpod": amount, "total": amount}
