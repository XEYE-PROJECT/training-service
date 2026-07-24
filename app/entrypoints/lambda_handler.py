"""Entrypoint de AWS Lambda (imagen de contenedor).

El backend invoca la función **asíncronamente** (``InvocationType=Event``): nada espera al
muro de 15 minutos y el resultado vuelve por el webhook, como en el resto de proveedores.
Lambda no tiene GPU, así que un LLM local reventaría los 900 s con ~100-200 elementos:
despliega con ``ENRICHER=groq``/``gemini``/``none``; para LLM local usa RunPod o AWS Batch.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.worker import build_worker
from app.infrastructure.job_loader import InvalidJobError, parse_job

_settings = get_settings()
configure_logging(_settings.log_level)
logger = logging.getLogger(__name__)

# Init a nivel de módulo a propósito: corre una vez por cold start y lo reutilizan las calientes.
_worker = build_worker(_settings)


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    # Acepta el job pelado y el sobre {"input": {...}}: el mismo payload que construye el
    # backend sirve en Lambda, RunPod y la CLI.
    payload = event.get("input") if isinstance(event, dict) and "input" in event else event
    try:
        job = parse_job(payload or {})
    except (InvalidJobError, ValueError) as exc:
        logger.error("Invalid job: %s", exc)
        return {"status": "error", "error": str(exc)}

    logger.info("Lambda: training %d for list %d (%d elements)",
                job.training_id, job.list_id, len(job.elements))
    return _worker.run(job)
