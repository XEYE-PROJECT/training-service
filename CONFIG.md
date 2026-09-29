# Referencia de configuración — training-service

Todo se lee del entorno (pydantic-settings, `app/core/config.py`): la misma imagen se comporta
igual la arranque el backend con `docker run -e …` (provider `docker`) o RunPod (variables del
endpoint). No hay `ENVIRONMENT`: el worker no guarda ningún secreto del backend (el token del
webhook viaja en el job, uno por entrenamiento) y lo que distingue producción es que solo llama a
callbacks `https://` de los hosts permitidos.

Leyenda: **Prod** = obligatorio en producción · 🔑 = secreto (`SecretStr`: nunca en git, logs ni `repr`) ·
Valida = lo comprueba `Settings` al arrancar (o `Worker` antes de entrenar, cuando depende del job).

## Callback al backend

| Variable | Descripción | Default | Prod | Valida |
|---|---|---|---|---|
| `CALLBACK_ALLOWED_HOSTS` | Hosts (coma/espacio) a los que se permite llamar; vacío = cualquiera. Un job con otro `callback_url` no se ejecuta ni se le responde | vacío | **`hooks.xeye.es`** | `Worker`, antes de entrenar |
| `CALLBACK_ALLOW_HTTP` | Permite un `callback_url` `http://` (backend local en la red docker; el provider `docker` del backend lo pasa solo). Con `false` se exige `https://` | `false` | `false` | `Worker`, antes de entrenar |
| `CALLBACK_TIMEOUT_SECONDS` | Timeout del callback final | `60` | opcional | — |
| `CALLBACK_RETRIES` | Reintentos del callback final (`completed`/`failed`) | `3` | opcional | ≥ 1 |

El token de `X-Webhook-Token` no se configura: es el `webhook_token` del job
(`<training_id>.<HMAC-SHA256>` derivado del `TRAINING_WEBHOOK_SECRET` del backend). `Worker` comprueba
su forma y que el id sea el del job antes de entrenar; el HMAC lo verifica el backend.

## Algoritmo

| Variable | Descripción | Default | Prod | Valida |
|---|---|---|---|---|
| `STRATEGY` | `default` (LLM + embeddings) o `embeddings_only`; un job puede sobreescribirla | `default` | opcional | — |
| `EMBEDDING_MODEL` | Modelo sentence-transformers; el search-service debe poder cargarlo | MiniLM | debe coincidir con backend/search | — |
| `EMBEDDING_MODELS_ALLOWED` | Allowlist (coma/espacio) de la opción `embedding_model` del job, además del por defecto. Las imágenes la fijan a los modelos horneados | mpnet (Dockerfile) | automático | `Worker`: un job con otro modelo se reporta `failed` sin tocar la red |
| `EMBEDDING_BATCH_SIZE` | Lote de embeddings | `32` | opcional | > 0 |
| `QUERY_VARIANT_WEIGHT` | Peso de cada consulta generada en el centroide | `0.35` | opcional | — |

## Enriquecimiento (LLM)

| Variable | Descripción | Default | Prod | Valida |
|---|---|---|---|---|
| `ENRICHER` | `local` (llama.cpp), `groq`, `gemini` o `none` (`off`/vacío = none) | `local` | elegir uno | valor desconocido = no arranca |
| `ENRICH_MAX_ELEMENTS` | Tope de elementos enriquecidos por ejecución (0 = sin tope) | `0` | opcional | — |
| `GROQ_API_KEY` 🔑 / `GROQ_MODEL` | Solo `groq` | vacío / `llama-3.3-70b-versatile` | si groq | clave obligatoria al arrancar |
| `GEMINI_API_KEY` 🔑 / `GEMINI_MODEL` | Solo `gemini` | vacío / `gemini-2.0-flash` | si gemini | clave obligatoria al arrancar |
| `GEMINI_THINKING_BUDGET` | Tokens de razonamiento por petición en los modelos pensantes (Gemini 2.5/3.x). `0` los desactiva: describir un elemento no lo necesita y, si no, el razonamiento agota `LLM_MAX_TOKENS` y la respuesta llega vacía. `-1` = no enviar `thinkingConfig` (modelos que no lo admiten, p. ej. `gemini-2.0-*`) | `0` | opcional | — |
| `LLM_MAX_TOKENS` / `LLM_TEMPERATURE` | Tope de tokens de salida por petición y temperatura, para **todos** los proveedores | `384` / `0.3` | opcional | `LLM_MAX_TOKENS` > 0 |
| `LLM_PRICE_PER_MILLION_INPUT_TOKENS` / `_OUTPUT_TOKENS` | Tarifa del proveedor (unidades monetarias por millón de tokens); 0 = solo se cuentan tokens | `0` / `0` | recomendado con groq/gemini | — |
| `LLM_MAX_COST_PER_JOB` | Tope de gasto en LLM por entrenamiento (0 = sin tope). Alcanzado, no se lanza ninguna petición más; el webhook lo marca con `usage.llm_budget_exhausted` | `0` | recomendado | exige tarifas > 0 |
| `LLM_MODEL_PATH` / `LLM_CONTEXT_SIZE` / `LLM_THREADS` / `LLM_GPU_LAYERS` | Solo `local` | ver `.env.example` | si local | — |
| `LLM_API_TIMEOUT_SECONDS` / `LLM_CONCURRENCY` / `LLM_REQUESTS_PER_MINUTE` | Llamadas a Groq/Gemini | `60` / `8` / `0` | opcional | — |
| `LLM_RETRY_ATTEMPTS` / `LLM_RETRY_BACKOFF_SECONDS` / `LLM_RETRY_ROUNDS` / `LLM_RETRY_ROUND_WAIT_SECONDS` | Reintentos y pasadas de rescate | `3` / `2` / `2` / `30` | opcional | — |
| `LLM_BATCH_THRESHOLD` / `LLM_BATCH_CHUNK_SIZE` / `LLM_BATCH_POLL_SECONDS` / `LLM_BATCH_WAIT_MINUTES` | Batch API de Gemini | `500` / `2000` / `15` / `60` | opcional | — |

## Coste, logs y observabilidad

| Variable | Descripción | Default | Prod | Valida |
|---|---|---|---|---|
| `COMPUTE_PRICE_PER_HOUR` | Precio/hora de la máquina: `cost.runpod` del webhook = tiempo × precio (coste real, informativo) | `0` | recomendado | — |
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

- **RunPod** (provider `runpod` del backend): variables del endpoint → `CALLBACK_ALLOWED_HOSTS=hooks.xeye.es`,
  `ENRICHER` + su API key, tarifas y tope del LLM, `SENTRY_DSN`, `SENTRY_ENVIRONMENT=production`,
  `COMPUTE_PRICE_PER_HOUR`. Ningún secreto del backend.
- **Docker local** (provider `docker`): el backend pasa `-e CALLBACK_ALLOW_HTTP=true` cuando su
  URL es `http://` y lo que haya en `TRAINING_DOCKER_ENV` (redactado en sus logs).

## Qué NO sale nunca en los logs

- `GROQ_API_KEY` y `GEMINI_API_KEY` son `SecretStr`: `repr(settings)` los enmascara.
- Las claves viajan en cabeceras (`Authorization`, `x-goog-api-key`), nunca en la URL, así que un
  error de `httpx` no las incluye.
- El job JSON no contiene ningún secreto: solo el token por entrenamiento, inútil para otro run.
