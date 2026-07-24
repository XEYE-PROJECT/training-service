"""Registro de estrategias: nombre -> lista ordenada de pasos del algoritmo.

El backend elige una con la opción ``strategy`` del entrenamiento (por defecto
``"default"``), así que un algoritmo nuevo se activa por lista sin redesplegar nada más:
basta registrar una factoría con ``@register("nombre")``.
"""

from __future__ import annotations

from typing import Callable

from app.application.pipeline import Step, TrainingPipeline
from app.application.steps.embed import EmbedStep
from app.application.steps.enrich import EnrichStep

DEFAULT_STRATEGY = "default"

_REGISTRY: dict[str, Callable[[], list[Step]]] = {}


def register(name: str) -> Callable[[Callable[[], list[Step]]], Callable[[], list[Step]]]:
    def decorator(factory: Callable[[], list[Step]]) -> Callable[[], list[Step]]:
        _REGISTRY[name] = factory
        return factory

    return decorator


@register(DEFAULT_STRATEGY)
def _default() -> list[Step]:
    """Enriquecimiento LLM (con caché) -> embedding centroide de documento + consultas representativas."""
    return [EnrichStep(), EmbedStep()]


@register("embeddings_only")
def _embeddings_only() -> list[Step]:
    """Sin LLM: embebe el texto del elemento + la descripción del usuario. Rápido, apto para CPU."""
    return [EmbedStep()]


def available() -> list[str]:
    return sorted(_REGISTRY)


def build_pipeline(name: str | None) -> TrainingPipeline:
    key = (name or DEFAULT_STRATEGY).strip() or DEFAULT_STRATEGY
    factory = _REGISTRY.get(key)
    if factory is None:
        raise ValueError(f"Unknown training strategy '{key}'. Available: {', '.join(available())}")
    return TrainingPipeline(steps=factory())
