"""Error tracking opcional (Sentry). Solo activo con SENTRY_DSN; el SDK se importa perezoso
para que dev/tests no lo necesiten. Un contenedor one-shot muere nada más terminar, de ahí
``flush`` antes de salir: sin él los eventos se perderían."""

from __future__ import annotations

import logging

from app.core.config import Settings

logger = logging.getLogger(__name__)

_enabled = False


def init_sentry(settings: Settings) -> bool:
    global _enabled
    if not settings.sentry_dsn:
        return False
    import sentry_sdk

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        release=settings.sentry_release or None,
        traces_sample_rate=0.0,
        send_default_pii=False,
    )
    _enabled = True
    logger.info(
        "Sentry enabled (environment=%s, release=%s)", settings.sentry_environment, settings.sentry_release or "-"
    )
    return True


def capture_exception(exc: BaseException) -> None:
    if not _enabled:
        return
    import sentry_sdk

    sentry_sdk.capture_exception(exc)


def flush(timeout: float = 5.0) -> None:
    if not _enabled:
        return
    import sentry_sdk

    sentry_sdk.flush(timeout=timeout)
