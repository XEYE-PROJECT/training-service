"""WebhookReporter: lo único que habla con el backend. Sin red: ``httpx.post`` se sustituye por
un fake que registra cada llamada y ``time.sleep`` por un contador, así los reintentos no duermen."""

from __future__ import annotations

import httpx
import pytest

from app.infrastructure.notification import webhook_reporter
from app.infrastructure.notification.webhook_reporter import WebhookReporter

URL = "http://backend:8000/webhooks/training-update"


class FakePost:
    """Sustituye a ``httpx.post``: responde con los ``outcomes`` en orden (código HTTP o excepción
    a lanzar); agotados, responde 200. Registra url, json, headers y timeout de cada llamada."""

    def __init__(self, *outcomes: int | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def __call__(self, url, json=None, headers=None, timeout=None) -> httpx.Response:
        self.calls.append({"url": url, "json": json, "headers": dict(headers or {}), "timeout": timeout})
        outcome = self.outcomes.pop(0) if self.outcomes else 200
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome, text="nope" if outcome >= 400 else "")


@pytest.fixture
def post(monkeypatch) -> FakePost:
    fake = FakePost()
    monkeypatch.setattr(webhook_reporter.httpx, "post", fake)
    return fake


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(webhook_reporter.time, "sleep", slept.append)
    return slept


TOKEN = "42." + "0" * 64


def make_reporter(token: str | None = TOKEN, retries: int = 3, timeout: float = 60.0) -> WebhookReporter:
    return WebhookReporter(
        callback_url=URL, training_id=42, list_id=7, token=token, timeout_seconds=timeout, retries=retries
    )


# --- phase: fire-and-forget --------------------------------------------------------


def test_phase_posts_the_minimal_body_with_a_short_timeout(post):
    make_reporter().phase("training")

    assert len(post.calls) == 1
    call = post.calls[0]
    assert call["url"] == URL
    assert call["json"] == {"training_id": 42, "list_id": 7, "status": "training"}
    assert call["timeout"] == 10.0
    assert call["headers"]["X-Webhook-Token"] == TOKEN


@pytest.mark.parametrize("outcome", [500, httpx.ConnectError("down")])
def test_phase_never_retries(post, sleeps, outcome):
    # Perder un callback de progreso solo desactualiza la UI: un único intento y sin dormir.
    post.outcomes = [outcome]

    make_reporter().phase("optimizing")

    assert len(post.calls) == 1
    assert sleeps == []


# --- completed: el callback que no se puede perder ---------------------------------


def test_completed_adds_ids_and_status_to_the_payload(post):
    delivered = make_reporter(timeout=45.0).completed({"embeddings_data": "abc", "model": "{}"})

    assert delivered is True
    call = post.calls[0]
    assert call["json"] == {
        "embeddings_data": "abc",
        "model": "{}",
        "training_id": 42,
        "list_id": 7,
        "status": "completed",
    }
    assert call["timeout"] == 45.0
    assert call["headers"]["Content-Type"] == "application/json"


@pytest.mark.parametrize("token", [None, ""])
def test_token_header_only_when_there_is_a_token(post, token):
    make_reporter(token=token).completed({})
    assert "X-Webhook-Token" not in post.calls[0]["headers"]


def test_completed_retries_on_5xx_and_exceptions_with_exponential_backoff(post, sleeps):
    post.outcomes = [503, httpx.ReadTimeout("slow"), 200]

    assert make_reporter(retries=3).completed({}) is True
    assert len(post.calls) == 3
    assert sleeps == [2, 4]


def test_completed_gives_up_after_the_configured_retries(post, sleeps):
    post.outcomes = [500, 500, 500, 500]

    assert make_reporter(retries=3).completed({}) is False
    assert len(post.calls) == 3
    assert sleeps == [2, 4]  # tras el último intento no se duerme


def test_a_4xx_is_not_retried(post, sleeps):
    # Un token o un cuerpo inválidos no se arreglan reintentando.
    post.outcomes = [403, 200]

    assert make_reporter(retries=3).completed({}) is False
    assert len(post.calls) == 1
    assert sleeps == []


def test_retries_are_at_least_one(post):
    assert make_reporter(retries=0).completed({}) is True
    assert len(post.calls) == 1


# --- failed --------------------------------------------------------------------------


def test_failed_truncates_the_error_and_reports_status_failed(post):
    assert make_reporter().failed("x" * 5000) is True

    body = post.calls[0]["json"]
    assert body["status"] == "failed"
    assert (body["training_id"], body["list_id"]) == (42, 7)
    assert body["error"] == "x" * 2000


def test_failed_is_retried_like_completed(post, sleeps):
    post.outcomes = [502, 200]

    assert make_reporter(retries=3).failed("boom") is True
    assert len(post.calls) == 2
    assert sleeps == [2]
