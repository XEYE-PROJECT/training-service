"""Entrypoint one-shot: un contenedor = un entrenamiento. CMD por defecto de la imagen.

Lo usa el proveedor ``docker`` del backend en local y, sin cambios, AWS Batch / ECS RunTask
en la nube. El job llega como JSON por, en orden de preferencia: la env ``TRAINING_JOB``
(el propio JSON), la env ``TRAINING_DATA_PATH`` (ruta a un fichero, normalmente un volumen
montado) o stdin. Código de salida 0 = el backend confirmó el callback ``completed``.
"""

from __future__ import annotations

import json
import logging
import os
import sys

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.worker import build_worker
from app.infrastructure.job_loader import InvalidJobError, load_job_file, parse_job

logger = logging.getLogger(__name__)


def _read_payload() -> dict:
    inline = os.environ.get("TRAINING_JOB")
    if inline:
        return json.loads(inline)

    path = os.environ.get("TRAINING_DATA_PATH")
    if path:
        return json.loads(open(path, encoding="utf-8").read())

    if not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        if raw:
            return json.loads(raw)

    raise InvalidJobError("No job: set TRAINING_JOB or TRAINING_DATA_PATH, or pipe JSON on stdin")


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        job = parse_job(_read_payload())
    except (InvalidJobError, ValueError) as exc:
        logger.error("Invalid job: %s", exc)
        return 2

    logger.info("Training %d for list %d (%d elements, enricher=%s)",
                job.training_id, job.list_id, len(job.elements), settings.enricher)
    outcome = build_worker(settings).run(job)
    return 0 if outcome["status"] == "ok" else 1


# Se conserva por simetría con los otros entrypoints (y para pruebas rápidas con `python -c`).
load_job = load_job_file

if __name__ == "__main__":
    sys.exit(main())
