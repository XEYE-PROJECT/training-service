"""Los enrichers remotos a escala: concurrencia, reintentos y la Batch API de Gemini.

Todo con ``httpx.MockTransport``: ninguna prueba toca la red. Los settings ponen el
backoff a 0 para que los reintentos no duerman.
"""

from __future__ import annotations

import json
import re

import httpx

from app.domain.models import ElementInput, ListInput
from app.infrastructure.enrichment.api_llm import GeminiEnricher, GroqEnricher
from tests.conftest import make_settings

LIST = ListInput(id=3, name="Herramientas", description="Catálogo de ferretería")


def make_elements(n: int) -> list[ElementInput]:
    return [ElementInput(id=i, text=f"item {i}") for i in range(1, n + 1)]


def enrichment_json(element_id: int) -> str:
    return json.dumps({"summary": [f"resumen {element_id}"], "queries": [f"consulta {element_id}"]})


def gemini_settings(**overrides):
    defaults = dict(
        enricher="gemini",
        gemini_api_key="k",
        llm_retry_attempts=2,
        llm_retry_backoff_seconds=0.0,
        llm_batch_poll_seconds=0.0,
        llm_concurrency=4,
    )
    defaults.update(overrides)
    return make_settings(**defaults)


def use_transport(enricher, handler) -> None:
    enricher._client = httpx.Client(transport=httpx.MockTransport(handler))


def element_id_from_prompt(request: httpx.Request) -> int:
    """Los prompts llevan "item {id}"; sirve para responder a cada elemento con lo suyo."""
    return int(re.search(r"item (\d+)", request.read().decode()).group(1))


def gemini_response_for(element_id: int) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": enrichment_json(element_id)}]}}]}


# --- Groq / camino concurrente ------------------------------------------------------


def test_concurrent_enrich_many_maps_results_by_element_id():
    def handler(request: httpx.Request) -> httpx.Response:
        element_id = element_id_from_prompt(request)
        content = enrichment_json(element_id)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    enricher = GroqEnricher(gemini_settings(groq_api_key="k"))
    use_transport(enricher, handler)
    results = enricher.enrich_many(make_elements(7), LIST)

    assert set(results) == {1, 2, 3, 4, 5, 6, 7}
    assert results[3].summary == ["resumen 3"]


def test_a_failing_element_is_skipped_without_sinking_the_batch():
    def handler(request: httpx.Request) -> httpx.Response:
        if element_id_from_prompt(request) == 2:
            return httpx.Response(500)
        content = enrichment_json(element_id_from_prompt(request))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    enricher = GroqEnricher(gemini_settings(groq_api_key="k"))
    use_transport(enricher, handler)
    results = enricher.enrich_many(make_elements(3), LIST)

    assert set(results) == {1, 3}


def test_retriable_statuses_are_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"choices": [{"message": {"content": enrichment_json(1)}}]})

    enricher = GroqEnricher(gemini_settings(groq_api_key="k"))
    use_transport(enricher, handler)

    assert enricher.enrich(ElementInput(id=1, text="item 1"), LIST) is not None
    assert calls["n"] == 2


# --- Gemini / Batch API -------------------------------------------------------------


def batch_handler(seen: dict):
    """Simula la Batch API: acepta jobs troceados y los da por terminados al primer sondeo."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith(":batchGenerateContent"):
            body = json.loads(request.read())
            requests = body["batch"]["input_config"]["requests"]["requests"]
            name = f"batches/{len(seen)}"
            seen[name] = [item["metadata"]["key"] for item in requests]
            return httpx.Response(200, json={"name": name})
        if "/batches/" in url:
            name = "batches/" + url.rsplit("/", 1)[1]
            inlined = [
                {"metadata": {"key": key}, "response": gemini_response_for(int(key))}
                for key in seen[name]
            ]
            return httpx.Response(200, json={
                "name": name,
                "done": True,
                "metadata": {"state": "BATCH_STATE_SUCCEEDED"},
                "response": {"inlinedResponses": {"inlinedResponses": inlined}},
            })
        if url.endswith(":generateContent"):
            seen.setdefault("single", []).append(1)
            return httpx.Response(200, json=gemini_response_for(element_id_from_prompt(request)))
        raise AssertionError(f"unexpected URL {url}")

    return handler


def test_gemini_uses_the_batch_api_above_the_threshold_and_chunks_the_jobs():
    seen: dict = {}
    enricher = GeminiEnricher(gemini_settings(llm_batch_threshold=3, llm_batch_chunk_size=2))
    use_transport(enricher, batch_handler(seen))

    results = enricher.enrich_many(make_elements(5), LIST)

    assert set(results) == {1, 2, 3, 4, 5}
    assert results[4].queries == ["consulta 4"]
    jobs = {k: v for k, v in seen.items() if k.startswith("batches/")}
    assert [len(keys) for keys in jobs.values()] == [2, 2, 1]  # 5 elementos, jobs de 2
    assert "single" not in seen  # ninguna petición unitaria: todo fue por batch


def test_gemini_below_the_threshold_stays_concurrent():
    seen: dict = {}
    enricher = GeminiEnricher(gemini_settings(llm_batch_threshold=100))
    use_transport(enricher, batch_handler(seen))

    results = enricher.enrich_many(make_elements(3), LIST)

    assert set(results) == {1, 2, 3}
    assert not any(k.startswith("batches/") for k in seen)


def test_a_broken_batch_api_falls_back_to_concurrent_requests():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith(":batchGenerateContent"):
            return httpx.Response(500)
        return httpx.Response(200, json=gemini_response_for(element_id_from_prompt(request)))

    enricher = GeminiEnricher(gemini_settings(llm_batch_threshold=2, llm_retry_attempts=1))
    use_transport(enricher, handler)

    results = enricher.enrich_many(make_elements(3), LIST)
    assert set(results) == {1, 2, 3}


def test_a_failed_batch_job_is_finished_concurrently():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith(":batchGenerateContent"):
            return httpx.Response(200, json={"name": "batches/1"})
        if "/batches/" in url:
            return httpx.Response(200, json={"done": True, "metadata": {"state": "JOB_STATE_FAILED"}})
        return httpx.Response(200, json=gemini_response_for(element_id_from_prompt(request)))

    enricher = GeminiEnricher(gemini_settings(llm_batch_threshold=2))
    use_transport(enricher, handler)

    results = enricher.enrich_many(make_elements(3), LIST)
    assert set(results) == {1, 2, 3}


def test_inlined_responses_are_found_in_either_nesting():
    inlined = [{"metadata": {"key": "1"}, "response": {}}]
    nested = {"response": {"inlinedResponses": {"inlinedResponses": inlined}}}
    flat = {"dest": {"inlinedResponses": inlined}}

    assert GeminiEnricher._inlined_responses(nested) == inlined
    assert GeminiEnricher._inlined_responses(flat) == inlined
    assert GeminiEnricher._inlined_responses({"done": True}) == []
