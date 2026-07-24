from __future__ import annotations

import json

import pytest

from app.infrastructure.job_loader import InvalidJobError, load_job_file, parse_job

BACKEND_PAYLOAD = {
    "training_id": 12,
    "list_id": 5,
    "user_id": 1,
    "callback_url": "http://backend:8000/webhooks/training-update",
    "webhook_secret": "s3cret",
    "list": {"id": 5, "name": "Museos", "description": "Museos de Baleares"},
    "elements": [
        {"id": 2, "text": "Museo B", "description": None, "generated_description": None,
         "trained": False},
        {"id": 1, "text": "Museo A", "description": "arte moderno", "trained": True},
    ],
    "options": [{"key": "train_all", "value": True}, {"key": "strategy", "value": "default"}],
}


def test_parses_the_exact_payload_the_backend_sends():
    job = parse_job(BACKEND_PAYLOAD)

    assert (job.training_id, job.list_id, job.user_id) == (12, 5, 1)
    assert job.webhook_secret == "s3cret"
    assert job.list.context == "Museos. Museos de Baleares"
    assert job.option("train_all") is True
    assert job.option("strategy") == "default"
    assert [e.id for e in job.sorted_elements()] == [1, 2]


def test_options_also_accept_a_plain_object():
    job = parse_job({**BACKEND_PAYLOAD, "options": {"strategy": "embeddings_only"}})
    assert job.option("strategy") == "embeddings_only"


def test_elements_without_id_or_text_are_dropped():
    job = parse_job({**BACKEND_PAYLOAD, "elements": [
        {"id": 1, "text": "ok"},
        {"id": None, "text": "sin id"},
        {"id": 3, "text": "   "},
    ]})
    assert [e.id for e in job.elements] == [1]


@pytest.mark.parametrize("missing", ["training_id", "list_id", "callback_url"])
def test_missing_required_field_is_rejected(missing):
    payload = {k: v for k, v in BACKEND_PAYLOAD.items() if k != missing}
    with pytest.raises(InvalidJobError):
        parse_job(payload)


def test_load_job_file(tmp_path):
    path = tmp_path / "job.json"
    path.write_text(json.dumps(BACKEND_PAYLOAD), encoding="utf-8")
    assert load_job_file(path).training_id == 12


def test_load_job_file_missing(tmp_path):
    with pytest.raises(InvalidJobError):
        load_job_file(tmp_path / "nope.json")
