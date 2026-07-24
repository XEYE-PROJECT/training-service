"""Entrypoint de RunPod Serverless (GPU).

El destino recomendado para el LLM local: la GPU baja de ~7 s/elemento a menos de un
segundo y escala a cero (se paga por segundo de ejecución). El job llega en
``event["input"]`` — el mismo objeto que la CLI lee de un fichero; los modelos se cargan
a nivel de módulo para que un worker caliente los reutilice entre jobs.
"""

from __future__ import annotations

import logging
from typing import Any

import runpod  # type: ignore[import-not-found]

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.worker import build_worker
from app.infrastructure.job_loader import InvalidJobError, parse_job

_settings = get_settings()
configure_logging(_settings.log_level)
logger = logging.getLogger(__name__)

_worker = build_worker(_settings)


def handler(event: dict[str, Any]) -> dict[str, Any]:
    try:
        job = parse_job((event or {}).get("input") or {})
    except (InvalidJobError, ValueError) as exc:
        logger.error("Invalid job: %s", exc)
        return {"status": "error", "error": str(exc)}

    logger.info("RunPod: training %d for list %d (%d elements)",
                job.training_id, job.list_id, len(job.elements))
    return _worker.run(job)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
