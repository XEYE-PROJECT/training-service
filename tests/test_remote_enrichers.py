"""Los enrichers remotos a escala: concurrencia, reintentos y la Batch API de Gemini.

Todo con ``httpx.MockTransport``: ninguna prueba toca la red. Los settings ponen el
backoff a 0 para que los reintentos no duerman.
"""

from __future__ import annotations

import json
import re
import time

import httpx

from app.domain.models import ElementInput, ListInput
from app.domain.spend import SpendMeter
from app.infrastructure.enrichment.api_llm import GeminiEnricher, GroqEnricher, _RateLimiter
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
        llm_retry_round_wait_seconds=0.0,
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


# --- Tokens de salida, temperatura y gasto --------------------------------------------


def test_groq_requests_carry_the_configured_output_cap_and_temperature_and_report_usage():
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.read()))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": enrichment_json(1)}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 45},
            },
        )

    enricher = GroqEnricher(gemini_settings(groq_api_key="k", llm_max_tokens=256, llm_temperature=0.7))
    use_transport(enricher, handler)
    meter = SpendMeter(price_per_million_output=2.0)

    assert enricher.enrich(ElementInput(id=1, text="item 1"), LIST, meter=meter) is not None
    assert bodies[0]["max_tokens"] == 256 and bodies[0]["temperature"] == 0.7
    assert (meter.usage.input_tokens, meter.usage.output_tokens, meter.usage.requests) == (120, 45, 1)
    assert meter.cost == 0.00009


def test_gemini_requests_carry_the_configured_output_cap_and_temperature_and_report_usage():
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.read()))
        return httpx.Response(
            200,
            json={**gemini_response_for(1), "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 50}},
        )

    enricher = GeminiEnricher(gemini_settings(llm_max_tokens=200, llm_temperature=0.1))
    use_transport(enricher, handler)
    meter = SpendMeter(price_per_million_input=1.0)

    assert enricher.enrich(ElementInput(id=1, text="item 1"), LIST, meter=meter) is not None
    config = bodies[0]["generationConfig"]
    assert config["maxOutputTokens"] == 200 and config["temperature"] == 0.1
    assert meter.usage == meter.usage.__class__(300, 50, 1)
    assert meter.cost == 0.0003
    # Por defecto el razonamiento va desactivado: si no, consume el tope entero y no hay respuesta.
    assert config["thinkingConfig"] == {"thinkingBudget": 0}


def test_gemini_thinking_budget_is_configurable_and_thought_tokens_count_as_output():
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.read()))
        return httpx.Response(
            200,
            json={
                **gemini_response_for(1),
                "usageMetadata": {"promptTokenCount": 57, "candidatesTokenCount": 4, "thoughtsTokenCount": 365},
            },
        )

    meter = SpendMeter()
    enricher = GeminiEnricher(gemini_settings(gemini_thinking_budget=1024))
    use_transport(enricher, handler)
    assert enricher.enrich(ElementInput(id=1, text="item 1"), LIST, meter=meter) is not None
    assert bodies[0]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 1024}
    assert meter.usage.output_tokens == 369  # candidatos + razonamiento

    enricher = GeminiEnricher(gemini_settings(gemini_thinking_budget=-1))
    use_transport(enricher, handler)
    assert enricher.enrich(ElementInput(id=1, text="item 1"), LIST) is not None
    assert "thinkingConfig" not in bodies[1]["generationConfig"]  # modelos que no lo admiten


def test_the_spending_cap_stops_new_requests_and_skips_rescue_rounds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        element_id = element_id_from_prompt(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": enrichment_json(element_id)}}],
                "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0},  # 1.0 por petición
            },
        )

    # Concurrencia 1 para que el tope se evalúe petición a petición.
    enricher = GroqEnricher(gemini_settings(groq_api_key="k", llm_concurrency=1, llm_retry_rounds=3))
    use_transport(enricher, handler)
    meter = SpendMeter(price_per_million_input=1.0, max_cost=2.0)

    results = enricher.enrich_many(make_elements(6), LIST, meter=meter)

    assert calls["n"] == 2  # la segunda alcanza el tope; ni una más (tampoco pasadas de rescate)
    assert set(results) == {1, 2}
    assert meter.exhausted is True


def test_gemini_batch_usage_is_metered_too():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        response = batch_handler(seen)(request)
        if "/batches/" in str(request.url):
            payload = response.json()
            for item in payload["response"]["inlinedResponses"]["inlinedResponses"]:
                item["response"]["usageMetadata"] = {"promptTokenCount": 10, "candidatesTokenCount": 5}
            return httpx.Response(200, json=payload)
        return response

    enricher = GeminiEnricher(gemini_settings(llm_batch_threshold=2, llm_batch_chunk_size=10))
    use_transport(enricher, handler)
    meter = SpendMeter()

    results = enricher.enrich_many(make_elements(3), LIST, meter=meter)

    assert set(results) == {1, 2, 3}
    assert (meter.usage.input_tokens, meter.usage.output_tokens, meter.usage.requests) == (30, 15, 3)


# --- Límites del proveedor ----------------------------------------------------------


def test_an_element_rate_limited_past_its_retries_is_rescued_in_a_later_round():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        element_id = element_id_from_prompt(request)
        if element_id == 2:
            calls["n"] += 1
            if calls["n"] == 1:  # el primer intento agota el único reintento por petición
                return httpx.Response(429, headers={"retry-after": "0"})
        content = enrichment_json(element_id)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    enricher = GroqEnricher(gemini_settings(groq_api_key="k", llm_retry_attempts=1, llm_retry_rounds=1))
    use_transport(enricher, handler)
    results = enricher.enrich_many(make_elements(3), LIST)

    assert set(results) == {1, 2, 3}  # la pasada de rescate lo recupera


def test_an_element_that_keeps_failing_is_dropped_after_all_rounds():
    def handler(request: httpx.Request) -> httpx.Response:
        if element_id_from_prompt(request) == 2:
            return httpx.Response(429)
        content = enrichment_json(element_id_from_prompt(request))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    enricher = GroqEnricher(gemini_settings(groq_api_key="k", llm_retry_attempts=1, llm_retry_rounds=1))
    use_transport(enricher, handler)
    results = enricher.enrich_many(make_elements(3), LIST)

    assert set(results) == {1, 3}


def test_gemini_retry_info_in_the_error_body_is_honored():
    response = httpx.Response(
        429,
        json={"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}]}},
    )
    enricher = GeminiEnricher(gemini_settings())

    assert enricher._retry_delay(attempt=1, response=response) == 7.0


def test_the_rate_limiter_spaces_requests_and_pause_holds_everyone():
    limiter = _RateLimiter(requests_per_minute=1200)  # 0,05 s entre peticiones
    start = time.monotonic()
    for _ in range(3):
        limiter.acquire()
    assert time.monotonic() - start >= 0.1

    limiter = _RateLimiter(requests_per_minute=0)  # sin RPM, la pausa manda igualmente
    limiter.pause(0.05)
    start = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - start >= 0.05


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
            inlined = [{"metadata": {"key": key}, "response": gemini_response_for(int(key))} for key in seen[name]]
            return httpx.Response(
                200,
                json={
                    "name": name,
                    "done": True,
                    "metadata": {"state": "BATCH_STATE_SUCCEEDED"},
                    "response": {"inlinedResponses": {"inlinedResponses": inlined}},
                },
            )
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
