"""Enriquecedores LLM remotos: Groq (compatible con OpenAI) y Gemini.

Sustitutos directos del local — mismo prompt, mismo parseo, misma caché. Hacen viables los
despliegues solo-CPU: la parte lenta pasa a un proveedor que responde en ~1 s
y el contenedor se mantiene pequeño.

A escala el elemento-a-elemento secuencial no sirve (10.000 elementos a ~1 s serían ~3 h),
así que los remotos enriquecen en lotes:

- **Concurrencia** (``LLM_CONCURRENCY``): ambos proveedores lanzan varias peticiones a la
  vez, con reintentos y backoff ante 429/5xx — a 8 hilos, 10.000 elementos caben en ~20 min.
- **Batch API de Gemini** (``LLM_BATCH_THRESHOLD``): a partir de ese tamaño el job entero se
  envía a ``:batchGenerateContent`` (50% del precio estándar), troceado en jobs de
  ``LLM_BATCH_CHUNK_SIZE`` para respetar el límite inline de ~20 MB, y se sondea hasta que
  termina. Cualquier fallo o timeout del batch cae de vuelta al camino concurrente: el
  batch abarata, nunca es motivo de perder un entrenamiento. Groq no tiene camino batch
  aquí (su Batch API es de ficheros con ventana de 24 h-7 d); escala solo por concurrencia.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import httpx

from app.core.config import Settings
from app.domain.models import ElementInput, Enrichment, ListInput
from app.infrastructure.enrichment.prompt import SYSTEM_PROMPT, build_user_prompt, parse_response

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_BATCH_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchGenerateContent"
)
GEMINI_POLL_URL = "https://generativelanguage.googleapis.com/v1beta/{name}"

_RETRIABLE_STATUSES = {429, 500, 502, 503, 504}
_BATCH_BAD_STATES = ("FAILED", "CANCELLED", "EXPIRED")

Heartbeat = Callable[[], None]


class _RemoteEnricher:
    """Base común: cliente httpx compartido (thread-safe), reintentos y camino concurrente."""

    def __init__(self, settings: Settings, headers: dict[str, str]) -> None:
        self._settings = settings
        self._client = httpx.Client(timeout=settings.llm_api_timeout_seconds, headers=headers)

    def enrich(self, element: ElementInput, list_context: ListInput) -> Enrichment | None:
        raise NotImplementedError

    def enrich_many(
        self,
        elements: list[ElementInput],
        list_context: ListInput,
        heartbeat: Heartbeat | None = None,
    ) -> dict[int, Enrichment]:
        """Enriquece en paralelo. Un elemento fallido se omite; nunca tumba a los demás."""
        workers = max(1, self._settings.llm_concurrency)
        results: dict[int, Enrichment] = {}

        def one(element: ElementInput) -> tuple[int, Enrichment | None]:
            try:
                return element.id, self.enrich(element, list_context)
            except Exception:
                logger.exception("Enrichment failed for element %d", element.id)
                return element.id, None

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for done, (element_id, enrichment) in enumerate(pool.map(one, elements), start=1):
                if enrichment is not None and not enrichment.is_empty():
                    results[element_id] = enrichment
                if heartbeat:
                    heartbeat()
                if done % 100 == 0:
                    logger.info("Enriched %d/%d", done, len(elements))
        return results

    def _post_with_retry(self, url: str, payload: dict[str, Any]) -> httpx.Response:
        """POST con reintentos ante 429/5xx y errores de red; respeta ``Retry-After``."""
        attempts = max(1, self._settings.llm_retry_attempts)
        for attempt in range(1, attempts + 1):
            response: httpx.Response | None = None
            try:
                response = self._client.post(url, json=payload)
                if response.status_code not in _RETRIABLE_STATUSES:
                    response.raise_for_status()
                    return response
                if attempt == attempts:
                    response.raise_for_status()
            except httpx.HTTPStatusError:
                raise
            except httpx.HTTPError:
                if attempt == attempts:
                    raise
            time.sleep(self._retry_delay(attempt, response))
        raise AssertionError("unreachable")

    def _retry_delay(self, attempt: int, response: httpx.Response | None) -> float:
        delay = self._settings.llm_retry_backoff_seconds * (2 ** (attempt - 1))
        if response is not None:
            try:
                delay = max(delay, float(response.headers.get("retry-after", "")))
            except ValueError:
                pass
        return delay


class GroqEnricher(_RemoteEnricher):
    """En realidad vale cualquier endpoint de chat compatible con OpenAI: solo cambian URL base y clave."""

    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise ValueError("ENRICHER=groq requires GROQ_API_KEY")
        self._model = settings.groq_model
        super().__init__(settings, headers={"Authorization": f"Bearer {settings.groq_api_key}"})

    @property
    def model_name(self) -> str:
        return self._model

    def enrich(self, element: ElementInput, list_context: ListInput) -> Enrichment | None:
        response = self._post_with_retry(
            GROQ_URL,
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(element, list_context)},
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
        )
        content = response.json()["choices"][0]["message"].get("content")
        return parse_response(content, self._model)


class GeminiEnricher(_RemoteEnricher):
    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise ValueError("ENRICHER=gemini requires GEMINI_API_KEY")
        self._model = settings.gemini_model
        super().__init__(settings, headers={"x-goog-api-key": settings.gemini_api_key})

    @property
    def model_name(self) -> str:
        return self._model

    def enrich(self, element: ElementInput, list_context: ListInput) -> Enrichment | None:
        response = self._post_with_retry(
            GEMINI_URL.format(model=self._model), self._request_body(element, list_context)
        )
        return self._parse_generate_response(response.json())

    def enrich_many(
        self,
        elements: list[ElementInput],
        list_context: ListInput,
        heartbeat: Heartbeat | None = None,
    ) -> dict[int, Enrichment]:
        threshold = self._settings.llm_batch_threshold
        if threshold <= 0 or len(elements) < threshold:
            return super().enrich_many(elements, list_context, heartbeat)
        try:
            results, unresolved = self._enrich_via_batch_api(elements, list_context, heartbeat)
        except Exception:
            logger.exception("Gemini Batch API failed; falling back to concurrent requests")
            return super().enrich_many(elements, list_context, heartbeat)
        if unresolved:
            logger.warning(
                "Batch API left %d element(s) unresolved; finishing them with concurrent requests",
                len(unresolved),
            )
            results.update(super().enrich_many(unresolved, list_context, heartbeat))
        return results

    # --- Batch API ------------------------------------------------------------------

    def _enrich_via_batch_api(
        self,
        elements: list[ElementInput],
        list_context: ListInput,
        heartbeat: Heartbeat | None,
    ) -> tuple[dict[int, Enrichment], list[ElementInput]]:
        chunk = max(1, self._settings.llm_batch_chunk_size)
        jobs: list[tuple[str, dict[str, ElementInput]]] = []
        for start in range(0, len(elements), chunk):
            piece = elements[start : start + chunk]
            body = {
                "batch": {
                    "display_name": f"xeye-enrich-{start // chunk}",
                    "input_config": {
                        "requests": {
                            "requests": [
                                {
                                    "request": self._request_body(element, list_context),
                                    "metadata": {"key": str(element.id)},
                                }
                                for element in piece
                            ]
                        }
                    },
                }
            }
            response = self._post_with_retry(GEMINI_BATCH_URL.format(model=self._model), body)
            name = response.json().get("name")
            if not name:
                raise ValueError("Gemini Batch API did not return a batch name")
            jobs.append((name, {str(element.id): element for element in piece}))
        logger.info("Submitted %d Gemini batch job(s) for %d elements", len(jobs), len(elements))

        results: dict[int, Enrichment] = {}
        unresolved: list[ElementInput] = []
        deadline = time.monotonic() + self._settings.llm_batch_wait_minutes * 60
        for name, by_key in jobs:
            payload = self._poll_batch(name, deadline, heartbeat)
            if payload is None:  # job fallido o fuera de plazo: sus elementos van al plan B
                unresolved.extend(by_key.values())
                continue
            answered: set[int] = set()
            for item in self._inlined_responses(payload):
                element = by_key.get(str((item.get("metadata") or {}).get("key")))
                if element is None:
                    continue
                answered.add(element.id)
                enrichment = self._parse_generate_response(item.get("response") or {})
                if enrichment is not None and not enrichment.is_empty():
                    results[element.id] = enrichment
            unresolved.extend(e for e in by_key.values() if e.id not in answered)
        return results, unresolved

    def _poll_batch(
        self, name: str, deadline: float, heartbeat: Heartbeat | None
    ) -> dict[str, Any] | None:
        """Sondea el job hasta que acabe. ``None`` = fallo o timeout (no lanza)."""
        url = GEMINI_POLL_URL.format(name=name)
        interval = max(1.0, self._settings.llm_batch_poll_seconds)
        failures = 0
        while time.monotonic() < deadline:
            payload: dict[str, Any] | None = None
            try:
                response = self._client.get(url)
                response.raise_for_status()
                payload = response.json()
                failures = 0
            except Exception:
                failures += 1
                if failures >= max(1, self._settings.llm_retry_attempts):
                    logger.exception("Polling Gemini batch %s keeps failing; giving up on it", name)
                    return None
            if payload is not None:
                state = self._batch_state(payload)
                if "error" in payload or any(state.endswith(bad) for bad in _BATCH_BAD_STATES):
                    logger.warning("Gemini batch %s ended in state '%s'", name, state or "error")
                    return None
                if state.endswith("SUCCEEDED") or payload.get("done") is True:
                    return payload
            if heartbeat:
                heartbeat()
            time.sleep(interval)
        logger.warning("Gemini batch %s did not finish within LLM_BATCH_WAIT_MINUTES", name)
        return None

    def _request_body(self, element: ElementInput, list_context: ListInput) -> dict[str, Any]:
        prompt = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(element, list_context)}"
        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "responseMimeType": "application/json",
            },
        }

    def _parse_generate_response(self, payload: dict[str, Any]) -> Enrichment | None:
        candidates = payload.get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts") or []
        content = "".join(part.get("text", "") for part in parts)
        return parse_response(content, self._model)

    @staticmethod
    def _batch_state(payload: dict[str, Any]) -> str:
        """El estado del job, esté donde esté según la versión de la API ("" si no viene)."""
        for container in (payload, payload.get("metadata"), payload.get("batch")):
            if isinstance(container, dict) and isinstance(container.get("state"), str):
                return container["state"]
        return ""

    @staticmethod
    def _inlined_responses(payload: Any) -> list[dict[str, Any]]:
        """Busca ``inlinedResponses`` tolerando dónde lo anide cada versión de la API."""
        if isinstance(payload, dict):
            value = payload.get("inlinedResponses")
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                return GeminiEnricher._inlined_responses(value)
            for child in payload.values():
                if isinstance(child, dict):
                    found = GeminiEnricher._inlined_responses(child)
                    if found:
                        return found
        return []
