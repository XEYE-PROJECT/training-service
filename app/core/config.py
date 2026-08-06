"""Configuración. Cada ajuste es una variable de entorno: el contenedor se configura igual
lo arranque ``docker run`` o RunPod."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    #: local | groq | gemini | none
    enricher: str = "local"
    enrich_max_elements: int = 0  # 0 = sin tope; un tope mantiene las ejecuciones solo-CPU en el timeout
    llm_model_path: str = "/app/models/qwen2.5-3b-instruct-q4_k_m.gguf"
    llm_context_size: int = 2048
    llm_max_tokens: int = 384
    llm_temperature: float = 0.3
    llm_threads: int = 0  # 0 = decide llama.cpp
    llm_gpu_layers: int = -1  # -1 = descargar todo a la GPU si la hay (los builds de CPU lo ignoran)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str = ""
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
    #: Respaldo cuando el job no trae secreto (el launcher normalmente envía uno).
    webhook_secret: str = ""
    callback_timeout_seconds: float = 60.0
    callback_retries: int = 3

    # --- Coste / varios -------------------------------------------------------------
    compute_price_per_hour: float = 0.0  # se reporta como coste del entrenamiento
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
