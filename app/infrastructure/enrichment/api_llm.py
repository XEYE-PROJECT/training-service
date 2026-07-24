"""Enriquecedores LLM remotos: Groq (compatible con OpenAI) y Gemini.

Sustitutos directos del local — mismo prompt, mismo parseo, misma caché. Hacen viables los
despliegues solo-CPU (AWS Lambda): la parte lenta pasa a un proveedor que responde en ~1 s
y el contenedor se mantiene pequeño.
"""

from __future__ import annotations

import logging

import httpx

from app.core.config import Settings
from app.domain.models import ElementInput, Enrichment, ListInput
from app.infrastructure.enrichment.prompt import SYSTEM_PROMPT, build_user_prompt, parse_response

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GroqEnricher:
    """En realidad vale cualquier endpoint de chat compatible con OpenAI: solo cambian URL base y clave."""

    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise ValueError("ENRICHER=groq requires GROQ_API_KEY")
        self._model = settings.groq_model
        self._client = httpx.Client(
            timeout=settings.llm_api_timeout_seconds,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        )

    @property
    def model_name(self) -> str:
        return self._model

    def enrich(self, element: ElementInput, list_context: ListInput) -> Enrichment | None:
        response = self._client.post(
            GROQ_URL,
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(element, list_context)},
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"].get("content")
        return parse_response(content, self._model)


class GeminiEnricher:
    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise ValueError("ENRICHER=gemini requires GEMINI_API_KEY")
        self._model = settings.gemini_model
        self._client = httpx.Client(
            timeout=settings.llm_api_timeout_seconds,
            headers={"x-goog-api-key": settings.gemini_api_key},
        )

    @property
    def model_name(self) -> str:
        return self._model

    def enrich(self, element: ElementInput, list_context: ListInput) -> Enrichment | None:
        prompt = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(element, list_context)}"
        response = self._client.post(
            GEMINI_URL.format(model=self._model),
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "responseMimeType": "application/json",
                },
            },
        )
        response.raise_for_status()
        candidates = response.json().get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts") or []
        content = "".join(part.get("text", "") for part in parts)
        return parse_response(content, self._model)
