"""Fail fast: lo que haría fallar un entrenamiento a mitad falla al construir Settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from tests.conftest import JOB_TOKEN, make_settings


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
    cfg = make_settings(enricher="groq", groq_api_key="gsk_very_secret", gemini_api_key="gem_very_secret")
    text = repr(cfg) + str(cfg) + str(cfg.model_dump())
    assert "gsk_very_secret" not in text
    assert "gem_very_secret" not in text
    assert cfg.groq_api_key.get_secret_value() == "gsk_very_secret"


def test_all_problems_are_reported_at_once():
    with pytest.raises(ValidationError) as info:
        make_settings(enricher="groq", embedding_batch_size=0, callback_retries=0)
    message = str(info.value)
    assert "GROQ_API_KEY" in message and "EMBEDDING_BATCH_SIZE" in message and "CALLBACK_RETRIES" in message


def test_webhook_token_problem_checks_shape_and_training_id():
    problem = Settings.webhook_token_problem
    assert problem(JOB_TOKEN, 7) is None
    assert "no valid webhook_token" in problem(None, 7)
    assert "no valid webhook_token" in problem("secret-in-the-clear", 7)
    assert "no valid webhook_token" in problem("7.abc", 7)
    assert "belongs to training 7" in problem(JOB_TOKEN, 8)


def test_callback_problem_requires_https_unless_allowed_and_enforces_the_host_allowlist():
    prod, local = "https://hooks.xeye.es/webhooks/training-update", "http://xeye-java-backend:8000/webhooks"

    assert make_settings(callback_allow_http=False).callback_problem(prod) is None
    assert "https" in make_settings(callback_allow_http=False).callback_problem(local)
    assert make_settings(callback_allow_http=True).callback_problem(local) is None
    assert "not an absolute" in make_settings().callback_problem("ftp://x") and "not an absolute" in (
        make_settings().callback_problem("/webhooks")
    )
    allow = make_settings(callback_allow_http=False, callback_allowed_hosts="hooks.xeye.es,backend.xeye.es")
    assert allow.callback_problem(prod) is None
    assert "not in CALLBACK_ALLOWED_HOSTS" in allow.callback_problem("https://evil.example/webhooks")


def test_embedding_model_allowlist_includes_the_default_model():
    cfg = make_settings(embedding_model="a", embedding_models_allowed="b c, d")
    assert cfg.allowed_embedding_models() == {"a", "b", "c", "d"}
    assert cfg.embedding_model_problem(None) is None and cfg.embedding_model_problem("d") is None
    assert "not allowed" in cfg.embedding_model_problem("evil/model")


def test_a_spending_cap_without_prices_is_rejected():
    with pytest.raises(ValidationError, match="LLM_MAX_COST_PER_JOB"):
        make_settings(llm_max_cost_per_job=1.0)
    assert make_settings(llm_max_cost_per_job=1.0, llm_price_per_million_output_tokens=2.0).llm_max_cost_per_job == 1.0
    with pytest.raises(ValidationError, match="LLM_MAX_TOKENS"):
        make_settings(llm_max_tokens=0)


def test_settings_reads_secrets_from_the_environment(monkeypatch):
    monkeypatch.setenv("ENRICHER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    monkeypatch.setenv("CALLBACK_ALLOWED_HOSTS", "hooks.xeye.es")
    cfg = Settings(_env_file=None)
    assert cfg.enricher == "gemini"
    assert cfg.gemini_api_key.get_secret_value() == "from-env"
    assert cfg.callback_allowed_hosts == "hooks.xeye.es"
