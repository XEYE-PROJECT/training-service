"""Raíz de composición + la función que llama todo entrypoint.

Los objetos pesados (modelo de embeddings, LLM) se construyen una vez por *proceso*: gratis
para el contenedor one-shot, y en RunPod el segundo job de un worker caliente se
ahorra la carga del modelo por completo.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.application.ports import Embedder, Enricher
from app.application.run_training import RunTraining, completion_payload, compute_cost
from app.core.config import Settings, get_settings
from app.domain.models import TrainingJob
from app.infrastructure.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.infrastructure.enrichment.factory import build_enricher
from app.infrastructure.notification.webhook_reporter import WebhookReporter

logger = logging.getLogger(__name__)


@dataclass
class Worker:
    settings: Settings
    embedder: Embedder
    enricher: Enricher | None
    _embedders: dict[str, Embedder] = field(default_factory=dict)

    def run(self, job: TrainingJob) -> dict[str, Any]:
        """Ejecuta el job y reporta el resultado; devuelve un resumen breve al llamante.

        Nunca lanza: un fallo se reporta al backend (estado ``failed``) y se devuelve como
        ``{"status": "error", ...}`` — una excepción que escapara de aquí dejaría el
        entrenamiento clavado en ``initialized`` en los reintentos de RunPod.
        """
        reporter = WebhookReporter(
            callback_url=job.callback_url,
            training_id=job.training_id,
            list_id=job.list_id,
            secret=job.webhook_secret or self.settings.webhook_secret or None,
            timeout_seconds=self.settings.callback_timeout_seconds,
            retries=self.settings.callback_retries,
        )
        use_case = RunTraining(self.settings, self._embedder_for(job), self.enricher)

        started = time.monotonic()
        try:
            result = use_case.execute(job, reporter=reporter)
        except Exception as exc:
            logger.exception("Training %d failed", job.training_id)
            reporter.failed(str(exc))
            return {"status": "error", "training_id": job.training_id, "error": str(exc)}

        elapsed = int(time.monotonic() - started)
        result.time["total_seconds"] = max(result.time.get("total_seconds", 0), elapsed)
        cost = compute_cost(result.time["total_seconds"], self.settings.compute_price_per_hour)

        delivered = reporter.completed(completion_payload(job, result, cost))
        logger.info(
            "Training %d (list %d): %d elements, %d enriched, %d cached, %ds, callback=%s",
            job.training_id, job.list_id, len(result.element_ids), result.enriched_count,
            result.cached_count, elapsed, "ok" if delivered else "FAILED",
        )
        if not delivered:
            return {"status": "error", "training_id": job.training_id, "error": "callback_failed"}
        return {
            "status": "ok",
            "training_id": job.training_id,
            "elements": len(result.element_ids),
            "enriched": result.enriched_count,
            "cached": result.cached_count,
            "elapsed_seconds": elapsed,
        }

    def _embedder_for(self, job: TrainingJob) -> Embedder:
        """La opción ``embedding_model`` del job elige el modelo; por defecto, el de settings.

        Se cachea por nombre para que un worker caliente de RunPod que alterna
        modelos no los recargue en cada job.
        """
        name = str(job.option("embedding_model") or "").strip()
        if not name or name == self.embedder.model_name:
            return self.embedder
        if name not in self._embedders:
            logger.info("Job %d requests embedding model %s", job.training_id, name)
            self._embedders[name] = SentenceTransformerEmbedder(
                name, self.settings.embedding_batch_size
            )
        return self._embedders[name]


def build_worker(settings: Settings | None = None) -> Worker:
    settings = settings or get_settings()
    return Worker(
        settings=settings,
        embedder=SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_batch_size),
        enricher=build_enricher(settings),
    )
