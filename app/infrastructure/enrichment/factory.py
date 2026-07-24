"""Elige el enriquecedor según ``ENRICHER``. Añadir un proveedor = una rama + una clase."""

from __future__ import annotations

import logging

from app.application.ports import Enricher
from app.core.config import Settings

logger = logging.getLogger(__name__)


def build_enricher(settings: Settings) -> Enricher | None:
    provider = (settings.enricher or "none").strip().lower()

    if provider in {"none", "off", ""}:
        logger.info("Enricher disabled; elements are embedded from their own text")
        return None

    if provider == "local":
        from app.infrastructure.enrichment.local_llm import LocalLlmEnricher

        return LocalLlmEnricher(settings)

    if provider == "groq":
        from app.infrastructure.enrichment.api_llm import GroqEnricher

        return GroqEnricher(settings)

    if provider == "gemini":
        from app.infrastructure.enrichment.api_llm import GeminiEnricher

        return GeminiEnricher(settings)

    raise ValueError(f"Unknown ENRICHER '{provider}' (expected: none | local | groq | gemini)")
