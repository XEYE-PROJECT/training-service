"""Composición de los textos que se embeben. Funciones puras — el corazón del recall.

El texto documento (lo que el elemento *es*) y las variantes de consulta (lo que un usuario
*teclearía*) se embeben por separado y se promedian (ver ``steps/embed.py``). El contexto de
la lista NO se antepone al documento a propósito: el search-service embebe la consulta sin
él y mezclarlos degradaba la precisión — ese contexto se le da al LLM, que es donde aporta.
"""

from __future__ import annotations

from app.domain.models import ElementInput, Enrichment

_MAX_QUERY_VARIANTS = 8


def document_text(element: ElementInput, enrichment: Enrichment | None) -> str:
    """Texto principal que se embebe para un elemento."""
    parts: list[str] = [element.text.strip()]
    if element.description and element.description.strip():
        parts.append(element.description.strip())
    if enrichment:
        parts.extend(line for line in enrichment.summary if line)
    return ". ".join(_dedupe(parts))


def query_variants(enrichment: Enrichment | None) -> list[str]:
    """Consultas representativas del elemento, con tope (se promedian: más no es mejor)."""
    if not enrichment:
        return []
    return _dedupe(q for q in enrichment.queries if q)[:_MAX_QUERY_VARIANTS]


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip().rstrip(".")
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out
