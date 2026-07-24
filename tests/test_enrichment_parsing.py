"""Parseo de la salida del LLM y de la caché: la capa que aísla el desorden de los modelos
pequeños de los datos de entrenamiento."""

from __future__ import annotations

from app.domain.models import ElementInput, Enrichment, ListInput
from app.domain.text import document_text, query_variants
from app.infrastructure.enrichment.prompt import build_user_prompt, parse_response


def test_parses_clean_json():
    enrichment = parse_response(
        '{"summary": ["Un martillo"], "queries": ["martillo", "herramienta para clavos"]}', "llm"
    )
    assert enrichment.summary == ["Un martillo"]
    assert enrichment.queries == ["martillo", "herramienta para clavos"]
    assert enrichment.model == "llm"


def test_parses_json_inside_a_code_fence_with_chatter():
    raw = 'Claro:\n```json\n{"summary": ["A"], "queries": ["b"]}\n```\n¡Espero que ayude!'
    enrichment = parse_response(raw, "llm")
    assert enrichment.summary == ["A"] and enrichment.queries == ["b"]


def test_rejects_unusable_output():
    assert parse_response("no soy JSON", "llm") is None
    assert parse_response("", "llm") is None
    assert parse_response('{"summary": [], "queries": []}', "llm") is None


def test_prompt_carries_the_list_and_user_context():
    prompt = build_user_prompt(
        ElementInput(id=1, text="martillo", description="de carpintero"),
        ListInput(id=1, name="Ferretería", description="Catálogo"),
    )
    assert "Ferretería. Catálogo" in prompt
    assert "martillo" in prompt
    assert "de carpintero" in prompt


def test_cached_enrichment_roundtrips():
    original = Enrichment(summary=["a"], queries=["b"], model="llm")
    assert Enrichment.from_json(original.to_json()) == original


def test_cache_from_a_foreign_or_broken_value_is_ignored():
    # La columna guardaba texto libre del servicio antiguo; no hay que fiarse de ella.
    assert Enrichment.from_json("Nombre: martillo\nTipo: herramienta") is None
    assert Enrichment.from_json('{"v": 99, "summary": ["x"]}') is None
    assert Enrichment.from_json(None) is None


def test_document_text_excludes_query_variants_and_dedupes():
    element = ElementInput(id=1, text="martillo", description="martillo")
    enrichment = Enrichment(summary=["sirve para clavar"], queries=["comprar martillo"])

    assert document_text(element, enrichment) == "martillo. sirve para clavar"
    assert query_variants(enrichment) == ["comprar martillo"]
