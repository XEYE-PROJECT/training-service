"""Configuración. Cada ajuste es una variable de entorno: el contenedor se configura igual
lo arranque ``docker run`` o RunPod."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal
from urllib.parse import urlsplit

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Enricher = Literal["none", "local", "groq", "gemini"]

#: Forma del token por entrenamiento que emite el backend: ``<training_id>.<hex HMAC-SHA256>``.
WEBHOOK_TOKEN_PATTERN = re.compile(r"^(\d+)\.([0-9a-fA-F]{64})$")


def _split_names(raw: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,\s]+", raw or "") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Algoritmo ------------------------------------------------------------------
    #: Estrategia registrada (ver application/strategies.py). Un job puede sobreescribirla.
    strategy: str = "default"
    #: Modelo de embeddings. Debe poder cargarlo también el search-service (embebe las
    #: consultas con el nombre que reportamos), así que mantenerlos en la misma imagen.
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    #: Allowlist de la opción ``embedding_model`` del job (nombres separados por espacios o
    #: comas), además del modelo por defecto. Un job que pida otro se rechaza antes de tocar la
    #: red: sin ella cualquier nombre haría descargar un modelo arbitrario de Hugging Face. Las
    #: imágenes la fijan a los modelos que hornean.
    embedding_models_allowed: str = ""
    embedding_batch_size: int = 32
    #: Peso de cada consulta generada por el LLM en el centroide del elemento (documento = 1.0).
    query_variant_weight: float = 0.35

    # --- Enriquecimiento (LLM) ------------------------------------------------------
    #: local | groq | gemini | none (también off/vacío = none). Valor desconocido = no arranca.
    enricher: Enricher = "local"
    enrich_max_elements: int = 0  # 0 = sin tope; un tope mantiene las ejecuciones solo-CPU en el timeout
    llm_model_path: str = "/app/models/qwen2.5-3b-instruct-q4_k_m.gguf"
    llm_context_size: int = 2048
    #: Tope de tokens de salida por petición (local y remotos) y temperatura, para todos los proveedores.
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
    #: Tarifa del proveedor por millón de tokens de entrada/salida (0 = no se tarifica) y tope
    #: de gasto por job en esas unidades (0 = sin tope). Alcanzado el tope no se lanza ninguna
    #: petición más: lo que queda conserva su texto y lo recoge el siguiente entrenamiento.
    llm_price_per_million_input_tokens: float = 0.0
    llm_price_per_million_output_tokens: float = 0.0
    llm_max_cost_per_job: float = 0.0

    # --- Callback al backend --------------------------------------------------------
    #: El token de X-Webhook-Token viaja en el job (uno por entrenamiento, derivado del secreto
    #: del backend con HMAC): aquí no hay secreto. Lo que sí se valida es a dónde se envía:
    #: solo https salvo CALLBACK_ALLOW_HTTP (backend local en la red docker), y solo a los hosts
    #: de CALLBACK_ALLOWED_HOSTS si se fija (en RunPod: hooks.xeye.es). Sin allowlist, un job
    #: manipulado podría hacer que el worker enviara los textos de la lista a cualquier URL.
    callback_allowed_hosts: str = ""
    callback_allow_http: bool = False
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

        Lo que depende del job (token del webhook, destino del callback, modelo pedido) se
        comprueba en ``Worker.run`` antes de cargar nada: la misma imagen sirve para local y RunPod.
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
        if self.llm_max_tokens <= 0:
            problems.append("LLM_MAX_TOKENS must be > 0")
        if self.llm_max_cost_per_job > 0 and not (
            self.llm_price_per_million_input_tokens > 0 or self.llm_price_per_million_output_tokens > 0
        ):
            problems.append(
                "LLM_MAX_COST_PER_JOB needs LLM_PRICE_PER_MILLION_INPUT_TOKENS/OUTPUT_TOKENS "
                "(a cap without prices would never trigger)"
            )
        if problems:
            raise ValueError("Unsafe configuration:\n - " + "\n - ".join(problems))
        return self

    def allowed_embedding_models(self) -> set[str]:
        return {self.embedding_model, *_split_names(self.embedding_models_allowed)}

    def embedding_model_problem(self, requested: str | None) -> str | None:
        """Motivo por el que la opción ``embedding_model`` del job se rechaza, o ``None``."""
        name = str(requested or "").strip()
        if not name or name in self.allowed_embedding_models():
            return None
        allowed = ", ".join(sorted(self.allowed_embedding_models()))
        return f"embedding model '{name}' is not allowed on this worker (allowed: {allowed})"

    def callback_problem(self, callback_url: str) -> str | None:
        """Motivo por el que NO se debe llamar a ``callback_url``, o ``None`` si es un destino válido."""
        parts = urlsplit(str(callback_url or "").strip())
        host = (parts.hostname or "").lower()
        if parts.scheme not in {"http", "https"} or not host:
            return f"callback_url '{callback_url}' is not an absolute http(s) URL"
        if parts.scheme == "http" and not self.callback_allow_http:
            return "callback_url must use https (set CALLBACK_ALLOW_HTTP=true only for a local backend)"
        allowed = {h.lower() for h in _split_names(self.callback_allowed_hosts)}
        if allowed and host not in allowed:
            return f"callback host '{host}' is not in CALLBACK_ALLOWED_HOSTS ({', '.join(sorted(allowed))})"
        return None

    @staticmethod
    def webhook_token_problem(token: str | None, training_id: int) -> str | None:
        """Motivo por el que el callback fallaría con 403, o ``None`` si el token tiene sentido.

        Solo forma e id (el HMAC lo verifica el backend): un job sin token, o con el de otro
        entrenamiento, no debe pagar un entrenamiento cuyo resultado va a ser rechazado.
        """
        match = WEBHOOK_TOKEN_PATTERN.match(str(token or "").strip())
        if not match:
            return "the job has no valid webhook_token: the backend would reject the callback with 403"
        if int(match.group(1)) != int(training_id):
            return f"webhook_token belongs to training {match.group(1)}, not {training_id}"
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
