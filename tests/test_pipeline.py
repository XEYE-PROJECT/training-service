"""El pipeline de entrenamiento: orden, caché de enriquecimiento, centroide y contrato de cable."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.application.run_training import RunTraining, completion_payload, compute_cost
from app.domain.models import ElementInput, Enrichment
from app.domain.wire import decode_matrix
from tests.conftest import (
    DIM,
    FakeBatchEnricher,
    FakeEmbedder,
    FakeEnricher,
    FakeReporter,
    make_job,
    make_settings,
)


def run(job, enricher=None, settings=None, embedder=None):
    embedder = embedder or FakeEmbedder()
    reporter = FakeReporter()
    result = RunTraining(settings or make_settings(), embedder, enricher).execute(job, reporter)
    return result, embedder, reporter


def test_rows_are_ordered_by_element_id_ascending():
    # El search-service alinea filas con los ids registrados al lanzar (ASC): emitirlas en
    # orden del payload asignaría mal todos los vectores en silencio.
    job = make_job([ElementInput(id=9, text="sierra"), ElementInput(id=4, text="taladro")])
    result, _, _ = run(job)

    assert result.element_ids == [4, 9]
    matrix = decode_matrix(result.embeddings_b64)
    assert matrix.shape == (2, DIM)
    assert matrix.dtype == np.float32


def test_every_row_is_a_unit_vector():
    result, _, _ = run(make_job(), enricher=FakeEnricher())
    matrix = decode_matrix(result.embeddings_b64)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-5)


def test_model_metadata_carries_a_real_embedding_model_name():
    # El search-service carga este nombre para embeber consultas; con "mock" o sin nombre,
    # la búsqueda semántica degrada en silencio a solo texto.
    result, _, _ = run(make_job(), enricher=FakeEnricher())
    model = json.loads(result.model)

    assert model["embedding_model"] == "fake-model"
    assert model["llm_model"] == "fake-llm"
    assert model["strategy"] == "default"
    assert model["dimension"] == DIM


def test_enrichment_adds_query_variants_to_the_embedded_texts():
    _, embedder, _ = run(make_job(), enricher=FakeEnricher())
    texts = embedder.calls[0]

    # por elemento: 1 documento + 2 variantes de consulta
    assert len(texts) == 6
    assert "donde comprar martillo" in texts
    assert any(t.startswith("martillo") and "elemento" in t for t in texts)


def test_cached_enrichment_is_reused_and_not_regenerated():
    cached = Enrichment(summary=["ya descrito"], queries=["consulta cacheada"], model="old-llm")
    job = make_job([
        ElementInput(id=1, text="martillo", generated_description=cached.to_json()),
        ElementInput(id=2, text="destornillador"),
    ])
    enricher = FakeEnricher()
    result, embedder, _ = run(job, enricher=enricher)

    assert enricher.seen == [2]  # el elemento 1 salió de la caché
    assert result.enriched_count == 1
    assert result.cached_count == 1
    assert list(result.generated_descriptions) == [2]  # al backend solo vuelven los nuevos
    assert "consulta cacheada" in embedder.calls[0]


def test_force_enrich_option_bypasses_the_cache():
    cached = Enrichment(summary=["ya descrito"], queries=["consulta cacheada"], model="old-llm")
    job = make_job(
        [ElementInput(id=1, text="martillo", generated_description=cached.to_json())],
        options={"force_enrich": True},
    )
    enricher = FakeEnricher()
    run(job, enricher=enricher)

    assert enricher.seen == [1]


def test_enrich_budget_caps_llm_calls_and_still_embeds_everything():
    job = make_job([ElementInput(id=i, text=f"item {i}") for i in range(1, 6)])
    enricher = FakeEnricher()
    result, _, _ = run(job, enricher=enricher, settings=make_settings(enrich_max_elements=2))

    assert len(enricher.seen) == 2
    assert decode_matrix(result.embeddings_b64).shape == (5, DIM)


def test_a_batch_capable_enricher_receives_all_pending_elements_at_once():
    # Los remotos (groq/gemini) exponen enrich_many; el paso debe preferirlo al bucle
    # elemento a elemento y respetar caché y presupuesto igualmente.
    cached = Enrichment(summary=["ya descrito"], queries=["consulta cacheada"], model="old-llm")
    job = make_job([
        ElementInput(id=1, text="martillo", generated_description=cached.to_json()),
        ElementInput(id=2, text="destornillador"),
        ElementInput(id=3, text="sierra"),
    ])
    enricher = FakeBatchEnricher()
    result, _, _ = run(job, enricher=enricher)

    assert enricher.batches == [[2, 3]]  # una sola llamada, sin el cacheado
    assert result.enriched_count == 2
    assert result.cached_count == 1
    assert set(result.generated_descriptions) == {2, 3}


def test_a_failing_element_does_not_sink_the_training():
    job = make_job([ElementInput(id=1, text="martillo"), ElementInput(id=2, text="sierra")])
    result, _, _ = run(job, enricher=FakeEnricher(fail_on={1}))

    assert result.enriched_count == 1
    assert decode_matrix(result.embeddings_b64).shape == (2, DIM)


def test_phases_are_reported_in_order():
    _, _, reporter = run(make_job(), enricher=FakeEnricher())
    assert reporter.phases == ["optimizing", "training"]


def test_embeddings_only_strategy_skips_the_llm():
    job = make_job(options={"strategy": "embeddings_only"})
    enricher = FakeEnricher()
    result, embedder, reporter = run(job, enricher=enricher)

    assert enricher.seen == []
    assert reporter.phases == ["training"]
    assert len(embedder.calls[0]) == 2  # un documento por elemento, sin variantes
    assert json.loads(result.model)["strategy"] == "embeddings_only"


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unknown training strategy"):
        run(make_job(options={"strategy": "does-not-exist"}))


def test_completion_payload_matches_the_backend_contract():
    job = make_job()
    result, _, _ = run(job, enricher=FakeEnricher())
    payload = completion_payload(job, result, compute_cost(10, 1.10))

    assert payload["training_id"] == 7
    assert payload["list_id"] == 3
    assert payload["status"] == "completed"
    assert payload["model"] and payload["embeddings_data"]
    assert set(payload["generated_descriptions"]) == {"1", "2"}  # ids como cadenas: claves de objeto JSON
    assert payload["time"]["total_seconds"] >= 0
    assert payload["cost"]["total"] == pytest.approx(10 / 3600 * 1.10, rel=1e-3)
