"""Lo único que habla con el backend: ``POST {callback_url}`` con ``X-Webhook-Token``.

Los callbacks de progreso (``optimizing``/``training``) son fire-and-forget: perder uno
solo desactualiza el estado en la UI. El final (``completed``/``failed``) se reintenta con
backoff: perderlo deja el entrenamiento clavado en ``initialized`` para siempre y, en
``completed``, tira todo el cómputo recién pagado.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class WebhookReporter:
    def __init__(
        self,
        callback_url: str,
        training_id: int,
        list_id: int,
        secret: str | None = None,
        timeout_seconds: float = 60.0,
        retries: int = 3,
    ) -> None:
        self._url = callback_url
        self._training_id = training_id
        self._list_id = list_id
        self._headers = {"Content-Type": "application/json"}
        if secret:
            self._headers["X-Webhook-Token"] = secret
        self._timeout = timeout_seconds
        self._retries = max(1, retries)

    def phase(self, status: str) -> None:
        self._post({"training_id": self._training_id, "list_id": self._list_id, "status": status},
                   attempts=1, timeout=10.0)

    def completed(self, payload: dict[str, Any]) -> bool:
        body = {**payload, "training_id": self._training_id, "list_id": self._list_id,
                "status": "completed"}
        return self._post(body, attempts=self._retries, timeout=self._timeout)

    def failed(self, error: str) -> bool:
        body = {"training_id": self._training_id, "list_id": self._list_id,
                "status": "failed", "error": error[:2000]}
        return self._post(body, attempts=self._retries, timeout=self._timeout)

    def _post(self, body: dict[str, Any], attempts: int, timeout: float) -> bool:
        for attempt in range(1, attempts + 1):
            try:
                response = httpx.post(self._url, json=body, headers=self._headers, timeout=timeout)
                if response.is_success:
                    return True
                logger.error(
                    "Callback %s for training %d returned %d: %.200s",
                    body["status"], self._training_id, response.status_code, response.text,
                )
                if 400 <= response.status_code < 500:
                    return False  # un token o un cuerpo inválidos no se arreglan solos
            except Exception as exc:
                logger.error("Callback %s for training %d failed: %s",
                             body["status"], self._training_id, exc)
            if attempt < attempts:
                time.sleep(2 ** attempt)
        return False
