"""El prompt de enriquecimiento y el parseo de la respuesta del modelo.

Compartido por todos los enriquecedores (llama.cpp local, Groq, Gemini) para que cambiar de
proveedor no altere en silencio los datos de entrenamiento: mismas instrucciones, misma
forma JSON, mismo parseo.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.domain.models import ElementInput, Enrichment, ListInput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Eres un asistente que prepara datos para un buscador semántico. "
    "Respondes SIEMPRE con un único objeto JSON válido, sin texto alrededor, "
    "en el mismo idioma que el elemento."
)

_USER_TEMPLATE = """\
Contexto de la lista: {list_context}

Elemento: {element_text}
{element_description}
Devuelve un JSON con esta forma exacta:
{{
  "summary": ["2-4 frases cortas que describan qué es el elemento, para qué sirve y en qué \
categoría entra"],
  "queries": ["5-8 búsquedas distintas y realistas que un usuario escribiría para encontrar \
este elemento; incluye sinónimos, coloquialismos y errores comunes"]
}}
No inventes datos que contradigan el elemento. No repitas literalmente el nombre en todas \
las consultas.
El contexto de la lista es solo para entender el dominio: no uses sus palabras (ni el \
nombre de la lista) en las consultas — las consultas hablan únicamente del elemento.
Si el elemento contiene una exclusión ("excluyendo", "excluido", "salvo", "sin incluir", \
"no ..."), describe y genera consultas SOLO sobre lo que el elemento sí es; nunca sobre \
lo excluido."""

_MAX_SUMMARY = 6
_MAX_QUERIES = 10


def build_user_prompt(element: ElementInput, list_context: ListInput) -> str:
    description = (
        f"Descripción del usuario: {element.description.strip()}\n"
        if element.description and element.description.strip()
        else ""
    )
    return _USER_TEMPLATE.format(
        list_context=list_context.context or "(sin contexto)",
        element_text=element.text.strip(),
        element_description=description,
    )


def parse_response(raw: str | None, model_name: str | None) -> Enrichment | None:
    """Salida del modelo -> Enrichment. Tolera vallas de código y texto alrededor."""
    payload = _extract_json_object(raw)
    if payload is None:
        if raw and raw.strip():
            logger.warning("Enricher returned no usable JSON: %.120s", raw.strip())
        return None

    summary = _string_list(payload.get("summary"))[:_MAX_SUMMARY]
    queries = _string_list(payload.get("queries"))[:_MAX_QUERIES]
    if not summary and not queries:
        return None
    return Enrichment(summary=summary, queries=queries, model=model_name)


def _extract_json_object(raw: str | None) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except ValueError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
