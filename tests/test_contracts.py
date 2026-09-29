"""Contratos con el backend. Las fixtures de ``tests/contracts/`` son copias byte a byte de las
canónicas (``backend/src/test/resources/contracts/``; ``contracts-check.sh`` en xeye-infra lo
comprueba). Aquí el código real *consume* el job y *produce* los tres webhooks: un campo
renombrado en cualquiera de los dos lados rompe en CI, no en producción."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.application.run_training import completion_payload, compute_cost
from app.domain.models import TrainingResult
from app.domain.wire import decode_matrix, encode_matrix
from app.infrastructure.job_loader import parse_job
from app.infrastructure.notification import webhook_reporter
from app.infrastructure.notification.webhook_reporter import WebhookReporter

CONTRACTS = Path(__file__).parent / "contracts"


def fixture(name: str) -> dict:
    return json.loads((CONTRACTS / name).read_text(encoding="utf-8"))


@pytest.fixture
def sent(monkeypatch) -> list[dict]:
    """Cuerpos que WebhookReporter intenta enviar (httpx.post sustituido; nada sale a la red)."""
    bodies: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        bodies.append(json)
        import httpx

        return httpx.Response(200)

    monkeypatch.setattr(webhook_reporter.httpx, "post", fake_post)
    return bodies


def make_reporter() -> WebhookReporter:
    job = parse_job(fixture("training-job.json"))
    return WebhookReporter(callback_url=job.callback_url, training_id=job.training_id, list_id=job.list_id)


# --- backend -> worker: el job -----------------------------------------------------------


def test_the_canonical_job_is_parsed_field_by_field():
    job = parse_job(fixture("training-job.json"))

    assert (job.training_id, job.list_id, job.user_id) == (42, 7, 3)
    assert job.callback_url == "https://hooks.xeye.es/webhooks/training-update"
    # El secreto nunca viaja en el job: viaja un token por entrenamiento (id.hmac) que el backend verifica.
    assert job.webhook_token and job.webhook_token.startswith("42.")
    assert parse_job(fixture("training-job.json")).webhook_token == fixture("training-job.json")["webhook_token"]
    assert job.list.llm_enrichment is True
    assert job.list.context == "Herramientas. Catálogo de ferretería"
    assert [e.id for e in job.elements] == [101, 102]
    assert job.elements[0].description == "Mango de madera, 500 g"
    assert job.elements[0].cached_enrichment() is None
    assert job.options == {
        "train_all": True,
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "force_enrich": False,
    }


def test_the_generated_description_travels_untouched():
    # El backend devuelve tal cual lo que este worker le mandó; el parseo lo decide el worker.
    raw = fixture("training-job.json")["elements"][1]["generated_description"]
    job = parse_job(fixture("training-job.json"))

    assert job.elements[1].generated_description == raw


def test_the_cached_generated_description_of_the_job_is_a_usable_enrichment():
    job = parse_job(fixture("training-job.json"))
    cached = job.elements[1].cached_enrichment()

    assert cached is not None
    assert cached.queries == ["destornillador phillips", "destornillador estrella"]
    assert cached.model == "llama-3.3-70b-versatile"


# --- worker -> backend: los webhooks --------------------------------------------------------


def test_completed_payload_has_exactly_the_keys_of_the_canonical_webhook():
    expected = fixture("training-webhook-completed.json")
    job = parse_job(fixture("training-job.json"))
    result = TrainingResult(
        embeddings_b64=encode_matrix(np.eye(2, 4, dtype=np.float32)),
        model=json.dumps({"embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"}),
        dimension=4,
        element_ids=[101, 102],
        generated_descriptions={101: '{"v": 1, "summary": ["x"], "queries": ["y"], "model": "m"}'},
        # Nombres de fase reales del pipeline (EnrichStep/EmbedStep): la fixture usa
        # enrich_seconds/embed_seconds, por eso abajo solo se exige total_seconds en común.
        time={"optimizing_seconds": 3, "training_seconds": 2, "total_seconds": 5},
        enriched_count=1,
        cached_count=1,
    )

    payload = completion_payload(job, result, compute_cost(5, 0.5, llm_cost=0.00012))

    assert set(payload) == set(expected)
    assert (payload["training_id"], payload["list_id"], payload["status"]) == (42, 7, "completed")
    assert payload["element_ids"] == expected["element_ids"]
    assert all(isinstance(k, str) for k in payload["generated_descriptions"])
    assert isinstance(payload["described_count"], int)
    assert "total_seconds" in payload["time"] and "total_seconds" in expected["time"]
    assert all(isinstance(v, int) for v in payload["time"].values())
    assert set(payload["cost"]) == set(expected["cost"])
    assert all(isinstance(v, float) for v in payload["cost"].values())
    assert set(payload["usage"]) == set(expected["usage"])
    assert isinstance(payload["usage"]["llm_budget_exhausted"], bool)
    assert all(isinstance(payload["usage"][k], int) for k in ("llm_input_tokens", "llm_output_tokens", "llm_requests"))


def test_the_canonical_embeddings_data_decodes_with_the_wire_format():
    expected = fixture("training-webhook-completed.json")

    matrix = decode_matrix(expected["embeddings_data"])

    assert matrix.shape == (len(expected["element_ids"]), json.loads(expected["model"])["dimension"])
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(decode_matrix(encode_matrix(matrix)), matrix)


def test_the_reporter_sends_completed_with_the_canonical_keys(sent):
    expected = fixture("training-webhook-completed.json")
    body = {k: v for k, v in expected.items() if k not in ("training_id", "list_id", "status")}

    assert make_reporter().completed(body) is True
    assert set(sent[0]) == set(expected)
    assert (sent[0]["training_id"], sent[0]["list_id"], sent[0]["status"]) == (42, 7, "completed")


def test_the_reporter_sends_phase_with_the_canonical_keys(sent):
    expected = fixture("training-webhook-phase.json")

    make_reporter().phase("training")

    assert sent[0] == expected


def test_the_reporter_sends_failed_with_the_canonical_keys(sent):
    expected = fixture("training-webhook-failed.json")

    assert make_reporter().failed(expected["error"]) is True
    assert sent[0] == expected
