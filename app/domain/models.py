"""Modelo de dominio de un entrenamiento. Python puro: sin E/S ni frameworks.

Las formas de cable las fijan los dos extremos: la entrada es lo que serializa el
``TrainingLaunchCommand`` del backend (el mismo JSON para los tres entrypoints) y la salida
es el cuerpo del webhook más la matriz de embeddings. Invariantes: filas en orden de id de
elemento ASC y ``model`` con un nombre real (el search-service embebe las consultas con él).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

ENRICHMENT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ListInput:
    """La lista que se entrena: su nombre/descripción son el contexto de dominio para el LLM."""

    id: int
    name: str | None = None
    description: str | None = None

    @property
    def context(self) -> str | None:
        parts = [p.strip() for p in (self.name, self.description) if p and p.strip()]
        return ". ".join(parts) if parts else None


@dataclass(frozen=True)
class Enrichment:
    """Lo que el LLM añade a un elemento.

    ``summary`` son líneas descriptivas; ``queries``, las búsquedas que un usuario
    teclearía para encontrarlo. Se embeben de forma distinta (ver el paso de embedding),
    por eso siguen separadas en vez de aplanarse en un solo texto.
    """

    summary: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    model: str | None = None

    def is_empty(self) -> bool:
        return not self.summary and not self.queries

    def to_json(self) -> str:
        return json.dumps(
            {
                "v": ENRICHMENT_FORMAT_VERSION,
                "summary": self.summary,
                "queries": self.queries,
                "model": self.model,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def from_json(raw: str | None) -> "Enrichment | None":
        """Parsea un enriquecimiento cacheado. Devuelve None ante cualquier cosa no fiable."""
        if not raw or not raw.strip():
            return None
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return None
        if not isinstance(parsed, dict) or parsed.get("v") != ENRICHMENT_FORMAT_VERSION:
            return None
        summary = [str(x).strip() for x in parsed.get("summary") or [] if str(x).strip()]
        queries = [str(x).strip() for x in parsed.get("queries") or [] if str(x).strip()]
        if not summary and not queries:
            return None
        model = parsed.get("model")
        return Enrichment(summary=summary, queries=queries, model=str(model) if model else None)


@dataclass
class ElementInput:
    """Un elemento de la lista tal como lo envía el backend.

    ``generated_description`` es la *caché*: el enriquecimiento de un entrenamiento previo,
    que el backend vació si cambió el texto/descripción del elemento. Reutilizarla hace que
    reentrenar una lista de 3.000 elementos cueste una llamada al LLM, no 3.000.
    """

    id: int
    text: str
    description: str | None = None
    generated_description: str | None = None
    trained: bool = False

    def cached_enrichment(self) -> Enrichment | None:
        return Enrichment.from_json(self.generated_description)


@dataclass(frozen=True)
class TrainingJob:
    """Una petición de entrenamiento completa, llegue por el entrypoint que llegue."""

    training_id: int
    list_id: int
    callback_url: str
    webhook_secret: str | None = None
    user_id: int | None = None
    list: ListInput = field(default_factory=lambda: ListInput(id=0))
    elements: list[ElementInput] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)

    def option(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    def sorted_elements(self) -> list[ElementInput]:
        """Elementos por id ASC — el orden de filas de la matriz de embeddings. Crítico."""
        return sorted(self.elements, key=lambda e: e.id)


@dataclass
class TrainingResult:
    """Lo que el worker devuelve en ``completed``."""

    embeddings_b64: str | None = None
    model: str | None = None
    dimension: int | None = None
    element_ids: list[int] = field(default_factory=list)
    generated_descriptions: dict[int, str] = field(default_factory=dict)
    time: dict[str, int] = field(default_factory=dict)
    enriched_count: int = 0
    cached_count: int = 0
