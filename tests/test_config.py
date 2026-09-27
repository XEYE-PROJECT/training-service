"""Fail fast: lo que haría fallar un entrenamiento a mitad falla al construir Settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import MIN_WEBHOOK_SECRET_LENGTH, Settings
from tests.conftest import make_settings


@pytest.mark.parametrize("value", ["", "off", "none", "NONE", " None "])
def test_enricher_aliases_normalize_to_none(value):
    assert make_settings(enricher=value).enricher == "none"


def test_unknown_enricher_is_rejected():
    with pytest.raises(ValidationError, match="enricher"):
        make_settings(enricher="openai")


@pytest.mark.parametrize(("enricher", "variable"), [("groq", "GROQ_API_KEY"), ("gemini", "GEMINI_API_KEY")])
def test_remote_enricher_requires_its_api_key(enricher, variable):
    with pytest.raises(ValidationError, match=variable):
        make_settings(enricher=enricher)
    assert make_settings(enricher=enricher, **{variable.lower(): "k"}).enricher == enricher


def test_secrets_never_appear_in_repr():
    cfg = make_settings(enricher="groq", groq_api_key="gsk_very_secret", webhook_secret="hook_very_secret")
    text = repr(cfg) + str(cfg) + str(cfg.model_dump())
    assert "gsk_very_secret" not in text
    assert "hook_very_secret" not in text
    assert cfg.groq_api_key.get_secret_value() == "gsk_very_secret"


def test_all_problems_are_reported_at_once():
    with pytest.raises(ValidationError) as info:
        make_settings(enricher="groq", embedding_batch_size=0, callback_retries=0)
    message = str(info.value)
    assert "GROQ_API_KEY" in message and "EMBEDDING_BATCH_SIZE" in message and "CALLBACK_RETRIES" in message


def test_webhook_secret_problem_depends_on_the_backend_scheme():
    strong = "x" * MIN_WEBHOOK_SECRET_LENGTH
    local, prod = "http://xeye-java-backend:8000/webhooks", "https://backend.xeye.es/webhooks"

    assert "not set" in make_settings().webhook_secret_problem(local)
    # Contra un backend local (red docker) basta con que exista, aunque sea el de dev.
    assert make_settings(webhook_secret="dev-webhook-secret").webhook_secret_problem(local) is None
    # Contra producción (https) no valen los de desarrollo ni los cortos.
    assert "development" in make_settings(webhook_secret="dev-webhook-secret").webhook_secret_problem(prod)
    assert "at least" in make_settings(webhook_secret="short").webhook_secret_problem(prod)
    assert make_settings(webhook_secret=strong).webhook_secret_problem(prod) is None
    # El secreto que viniera en el job (formato antiguo) tiene prioridad sobre el del entorno.
    assert make_settings().webhook_secret_problem(prod, job_secret=strong) is None


def test_settings_reads_secrets_from_the_environment(monkeypatch):
    monkeypatch.setenv("ENRICHER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    monkeypatch.setenv("WEBHOOK_SECRET", "hook-from-env")
    cfg = Settings(_env_file=None)
    assert cfg.enricher == "gemini"
    assert cfg.gemini_api_key.get_secret_value() == "from-env"
    assert cfg.webhook_secret.get_secret_value() == "hook-from-env"
