# Referencia de configuración — training-service

Todo se lee del entorno (pydantic-settings, `app/core/config.py`): la misma imagen se comporta
igual la arranque el backend con `docker run -e …` (provider `docker`) o RunPod (variables del
endpoint). No hay `ENVIRONMENT`: lo que distingue producción es que el callback del job es
`https://` (backend público), y con ese callback el secreto del webhook debe ser fuerte.

Leyenda: **Prod** = obligatorio en producción · 🔑 = secreto (`SecretStr`: nunca en git, logs ni `repr`) ·
Valida = lo comprueba `Settings` al arrancar (o `Worker` antes de entrenar, en el caso del webhook).

## Callback al backend

| Variable | Descripción | Default | Prod | Valida |
|---|---|---|---|---|
| `WEBHOOK_SECRET` 🔑 | `X-Webhook-Token` = `TRAINING_WEBHOOK_SECRET` del backend. Llega por el entorno, nunca en el job | vacío | **sí** | contra `https://`: ≥ 32 chars y no de dev; contra `http://`: no vacío. Si falla, el job se reporta `failed` antes de entrenar |
| `CALLBACK_TIMEOUT_SECONDS` | Timeout del callback final | `60` | opcional | — |
| `CALLBACK_RETRIES` | Reintentos del callback final (`completed`/`failed`) | `3` | opcional | ≥ 1 |

## Algoritmo

| Variable | Descripción | Default | Prod | Valida |
|---|---|---|---|---|
| `STRATEGY` | `default` (LLM + embeddings) o `embeddings_only`; un job puede sobreescribirla | `default` | opcional | — |
| `EMBEDDING_MODEL` | Modelo sentence-transformers; el search-service debe poder cargarlo | MiniLM | debe coincidir con backend/search | — |
| `EMBEDDING_BATCH_SIZE` | Lote de embeddings | `32` | opcional | > 0 |
| `QUERY_VARIANT_WEIGHT` | Peso de cada consulta generada en el centroide | `0.35` | opcional | — |

## Enriquecimiento (LLM)

| Variable | Descripción | Default | Prod | Valida |
|---|---|---|---|---|
| `ENRICHER` | `local` (llama.cpp), `groq`, `gemini` o `none` (`off`/vacío = none) | `local` | elegir uno | valor desconocido = no arranca |
| `ENRICH_MAX_ELEMENTS` | Tope de elementos enriquecidos por ejecución (0 = sin tope) | `0` | opcional | — |
| `GROQ_API_KEY` 🔑 / `GROQ_MODEL` | Solo `groq` | vacío / `llama-3.3-70b-versatile` | si groq | clave obligatoria al arrancar |
| `GEMINI_API_KEY` 🔑 / `GEMINI_MODEL` | Solo `gemini` | vacío / `gemini-2.0-flash` | si gemini | clave obligatoria al arrancar |
| `LLM_MODEL_PATH` / `LLM_CONTEXT_SIZE` / `LLM_MAX_TOKENS` / `LLM_TEMPERATURE` / `LLM_THREADS` / `LLM_GPU_LAYERS` | Solo `local` | ver `.env.example` | si local | — |
| `LLM_API_TIMEOUT_SECONDS` / `LLM_CONCURRENCY` / `LLM_REQUESTS_PER_MINUTE` | Llamadas a Groq/Gemini | `60` / `8` / `0` | opcional | — |
| `LLM_RETRY_ATTEMPTS` / `LLM_RETRY_BACKOFF_SECONDS` / `LLM_RETRY_ROUNDS` / `LLM_RETRY_ROUND_WAIT_SECONDS` | Reintentos y pasadas de rescate | `3` / `2` / `2` / `30` | opcional | — |
| `LLM_BATCH_THRESHOLD` / `LLM_BATCH_CHUNK_SIZE` / `LLM_BATCH_POLL_SECONDS` / `LLM_BATCH_WAIT_MINUTES` | Batch API de Gemini | `500` / `2000` / `15` / `60` | opcional | — |

## Coste, logs y observabilidad

| Variable | Descripción | Default | Prod | Valida |
|---|---|---|---|---|
| `COMPUTE_PRICE_PER_HOUR` | Se reporta como coste del entrenamiento | `0` | recomendado | — |
| `LOG_LEVEL` | Nivel de log | `INFO` | opcional | — |
| `SENTRY_DSN` | DSN del proyecto `xeye-training-service` (vacío = desactivado) | vacío | recomendado | — |
| `SENTRY_ENVIRONMENT` | Etiqueta de entorno | `local` | `production` | — |
| `SENTRY_RELEASE` | Commit; lo fija el Dockerfile (`GIT_SHA`) | vacío | automático | — |

## Entrada del job (solo entrypoint CLI)

| Variable | Descripción |
|---|---|
| `TRAINING_JOB` | JSON del job en línea |
| `TRAINING_DATA_PATH` | Ruta a un fichero JSON (el provider `docker` del backend monta el directorio de jobs) |

Sin ninguna de las dos se lee stdin. En RunPod el job llega en `event["input"]`.

## Dónde se configura en producción

- **RunPod** (provider `runpod` del backend): variables del endpoint → `WEBHOOK_SECRET`,
  `ENRICHER` + su API key, `SENTRY_DSN`, `SENTRY_ENVIRONMENT=production`, `COMPUTE_PRICE_PER_HOUR`.
- **Docker local** (provider `docker`): el backend pasa `-e WEBHOOK_SECRET=…` y lo que haya en
  `TRAINING_DOCKER_ENV` (redactado en sus logs).

## Qué NO sale nunca en los logs

- `WEBHOOK_SECRET`, `GROQ_API_KEY` y `GEMINI_API_KEY` son `SecretStr`: `repr(settings)` los enmascara.
- Las claves viajan en cabeceras (`Authorization`, `x-goog-api-key`), nunca en la URL, así que un
  error de `httpx` no las incluye.
- El job JSON no contiene el secreto del webhook (el backend dejó de escribirlo).
