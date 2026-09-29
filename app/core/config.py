"""Configuración. Cada ajuste es una variable de entorno: el contenedor se configura igual
lo arranque ``docker run`` o RunPod."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Valores de .env.example / compose de desarrollo: nunca válidos contra un backend https.
INSECURE_WEBHOOK_SECRETS = frozenset({"dev-webhook-secret", "changeme", "change-me", "secret", "password"})
MIN_WEBHOOK_SECRET_LENGTH = 32

Enricher = Literal["none", "local", "groq", "gemini"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Algoritmo ------------------------------------------------------------------
    #: Estrategia registrada (ver application/strategies.py). Un job puede sobreescribirla.
    strategy: str = "default"
    #: Modelo de embeddings. Debe poder cargarlo también el search-service (embebe las
    #: consultas con el nombre que reportamos), así que mantenerlos en la misma imagen.
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_batch_size: int = 32
    #: Peso de cada consulta generada por el LLM en el centroide del elemento (documento = 1.0).
    query_variant_weight: float = 0.35

    # --- Enriquecimiento (LLM) ------------------------------------------------------
    #: local | groq | gemini | none (también off/vacío = none). Valor desconocido = no arranca.
    enricher: Enricher = "local"
    enrich_max_elements: int = 0  # 0 = sin tope; un tope mantiene las ejecuciones solo-CPU en el timeout
    llm_model_path: str = "/app/models/qwen2.5-3b-instruct-q4_k_m.gguf"
    llm_context_size: int = 2048
    llm_max_tokens: int = 384
    llm_temperature: float = 0.3
    llm_threads: int = 0  # 0 = decide llama.cpp
    llm_gpu_layers: int = -1  # -1 = descargar todo a la GPU si la hay (los builds de CPU lo ignoran)
    #: Secretos como SecretStr: no salen en repr()/logs; leer con .get_secret_value().
    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-2.0-flash"
    llm_api_timeout_seconds: float = 60.0
    #: Peticiones simultáneas contra el LLM remoto (groq/gemini). 1 = secuencial.
    llm_concurrency: int = 8
    #: Reintentos por petición ante 429/5xx/errores de red, con backoff exponencial.
    llm_retry_attempts: int = 3
    llm_retry_backoff_seconds: float = 2.0
    #: Peticiones/minuto máximas contra el proveedor, entre todos los hilos (0 = sin tope).
    #: Ponlo por debajo del RPM del plan (Groq free ~30, Gemini free ~10-15) para no
    #: llegar nunca al 429 en vez de reaccionar a él.
    llm_requests_per_minute: int = 0
    #: Pasadas extra al final sobre los elementos cuyo enriquecimiento falló (cuota
    #: agotada, red...): nadie se queda sin descripción por un pico de 429s.
    llm_retry_rounds: int = 2
    #: Respiro antes de cada pasada extra, para que la cuota por minuto se recupere.
    llm_retry_round_wait_seconds: float = 30.0
    #: A partir de cuántos elementos pendientes usar la Batch API del proveedor (Gemini:
    #: 50% del precio estándar). 0 = nunca.
    llm_batch_threshold: int = 500
    #: Peticiones por job de batch: trocea listas grandes para no pasar el límite inline
    #: de ~20 MB de Gemini (10.000 elementos = 5 jobs con el valor por defecto).
    llm_batch_chunk_size: int = 2000
    llm_batch_poll_seconds: float = 15.0
    #: Espera máxima a la Batch API antes de rematar lo que falte con peticiones normales.
    llm_batch_wait_minutes: float = 60.0

    # --- Callback al backend --------------------------------------------------------
    #: Secreto de la cabecera X-Webhook-Token (= TRAINING_WEBHOOK_SECRET del backend). Llega
    #: por el entorno del contenedor / endpoint de RunPod: el backend ya no lo mete en el job.
    webhook_secret: SecretStr = SecretStr("")
    callback_timeout_seconds: float = 60.0
    callback_retries: int = 3

    # --- Coste / varios -------------------------------------------------------------
    compute_price_per_hour: float = 0.0  # se reporta como coste del entrenamiento
    log_level: str = "INFO"

    # --- Error tracking (Sentry) — vacío = desactivado -------------------------------
    sentry_dsn: str = ""
    sentry_environment: str = "local"
    sentry_release: str = ""  # commit desplegado (SENTRY_RELEASE, lo fija el Dockerfile)

    @field_validator("enricher", mode="before")
    @classmethod
    def _normalize_enricher(cls, value: object) -> object:
        text = str(value or "").strip().lower()
        return "none" if text in {"", "off", "none"} else text

    @model_validator(mode="after")
    def _fail_fast(self) -> Settings:
        """Lo que fallaría a mitad de un entrenamiento (tras pagar embeddings) falla al arrancar.

        El secreto del webhook no se valida aquí porque depende del job (ver
        ``Worker.check_webhook_secret``): la misma imagen sirve para local y RunPod.
        """
        problems: list[str] = []
        if self.enricher == "groq" and not self.groq_api_key.get_secret_value().strip():
            problems.append("ENRICHER=groq requires GROQ_API_KEY")
        if self.enricher == "gemini" and not self.gemini_api_key.get_secret_value().strip():
            problems.append("ENRICHER=gemini requires GEMINI_API_KEY")
        if self.embedding_batch_size <= 0:
            problems.append("EMBEDDING_BATCH_SIZE must be > 0")
        if self.callback_retries < 1:
            problems.append("CALLBACK_RETRIES must be >= 1 (the final callback must be retried)")
        if problems:
            raise ValueError("Unsafe configuration:\n - " + "\n - ".join(problems))
        return self

    def webhook_secret_problem(self, callback_url: str, job_secret: str | None = None) -> str | None:
        """Motivo por el que el callback fallaría con 403, o ``None`` si el secreto vale.

        Contra un backend ``https://`` (producción) se exige un secreto fuerte y distinto de los
        de desarrollo; contra ``http://`` (backend local en la red docker) basta con que exista.
        """
        secret = (job_secret or self.webhook_secret.get_secret_value()).strip()
        if not secret:
            return "WEBHOOK_SECRET is not set: the backend would reject the callback with 403"
        if callback_url.lower().startswith("https://"):
            if secret.lower() in INSECURE_WEBHOOK_SECRETS:
                return "WEBHOOK_SECRET is a known development value; a production backend rejects it"
            if len(secret) < MIN_WEBHOOK_SECRET_LENGTH:
                return (
                    f"WEBHOOK_SECRET must be at least {MIN_WEBHOOK_SECRET_LENGTH} characters "
                    "against a production backend"
                )
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
