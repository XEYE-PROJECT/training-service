# XEYE training-service

Servicio de entrenamiento de listas, reescrito desde cero (sustituye a
`../XEYE-training-service`). Python 3.11, **sin API ni servidor**: un entrenamiento = **un
contenedor** que arranca, entrena la lista, envía el resultado al webhook del backend y
muere.

El mismo código corre en local (`docker run`) y en **RunPod Serverless**. Solo cambia el
*entrypoint*, que es un fichero de 40 líneas.

## Qué produce

El backend recibe por webhook:

- `embeddings_data`: **base64 de un `.npy` float32 `(n, dim)`**, una fila por elemento,
  **ordenadas por id de elemento ASC** (el backend guardó esos ids al lanzar, en
  `trainings.element_ids`, y el search-service alinea filas↔elementos por ellos). Cada fila
  es un vector unitario.
- `model`: JSON opaco con `embedding_model` — el search-service **carga ese modelo** para
  embeber las queries, así que tiene que ser un nombre real de sentence-transformers.
- `generated_descriptions`: `{element_id: enriquecido}` **solo de los elementos enriquecidos
  en esta ejecución**. El backend los guarda en `elements.generated_description` y los
  devuelve en el siguiente entrenamiento → **el LLM no se vuelve a pagar** salvo para los
  elementos nuevos o modificados (el backend borra el cache cuando cambia el texto o la
  descripción del elemento). Esto es lo que hace viable la IA local.
- `time` / `cost`.

## El algoritmo (y cómo cambiarlo)

Un entrenamiento es una **lista de pasos** sobre un contexto compartido
(`application/pipeline.py`). Una *estrategia* es simplemente una lista de pasos registrada
por nombre (`application/strategies.py`), y el backend puede elegirla por lista con la
opción de entrenamiento `strategy`.

Estrategia `default`:

1. **`EnrichStep` (fase `optimizing`)** — el LLM genera, por elemento, `summary` (qué es) y
   `queries` (5-8 búsquedas realistas que alguien escribiría para encontrarlo). Reutiliza el
   cache; respeta `ENRICH_MAX_ELEMENTS`; un elemento que falle no tumba el entrenamiento.
2. **`EmbedStep` (fase `training`)** — el vector del elemento es el **centroide ponderado** de
   su documento (texto + descripción + summary) y de sus queries representativas, todos
   normalizados antes de promediar. Un centroide de consultas reales cae mucho más cerca de
   las consultas reales que un párrafo descriptivo largo, que es exactamente lo que mide el
   search-service.

Estrategia `embeddings_only`: solo `EmbedStep` (sin LLM).

Para cambiar el algoritmo: escribe un paso, regístralo en una estrategia. No hay que tocar
entrypoints, adaptadores ni el backend:

```python
@register("mi_algo")
def _mi_algo():
    return [EnrichStep(), MiPasoNuevo(), EmbedStep()]
```

Dos decisiones frente al servicio viejo:

- **No se prefija el contexto de la lista al texto que se embebe.** El viejo embebía
  `"{lista} | {texto}"` pero el search-service embebe la query *sin* ese prefijo: documento y
  query quedaban en zonas distintas del espacio. El contexto de la lista ahora se le pasa al
  LLM, que es donde aporta.
- El enriquecido es **JSON estructurado y versionado**, no texto libre: por eso se puede
  cachear y validar.

## Arquitectura

```
app/
  domain/          # puro: modelos, composición de textos, formato de los embeddings
  application/     # puertos (Protocols), pipeline, pasos, estrategias, caso de uso
  infrastructure/  # sentence-transformers, enrichers (llama.cpp / Groq / Gemini), webhook
  entrypoints/     # cli (one-shot) · runpod_handler
  core/            # settings + composición (worker.py)
```

## Enrichers (`ENRICHER`)

| Valor | Qué usa | Dónde encaja |
|---|---|---|
| `local` (por defecto) | llama.cpp + GGUF horneado en la imagen (Qwen2.5-3B Q4) | Local y **RunPod/GPU**. En CPU son ~7 s por elemento |
| `groq` | API de Groq (OpenAI-compatible) | Despliegues solo-CPU: ~1 s por elemento, imagen pequeña |
| `gemini` | API de Gemini | igual que Groq, y con **Batch API** para listas grandes |
| `none` | ninguno | embeddings solo del texto + descripción; segundos |

### Escala: concurrencia y Batch API

El elemento-a-elemento secuencial no aguanta listas grandes (10.000 elementos a ~1 s
serían ~3 h), así que los enrichers remotos trabajan en lotes (`enrich_many`):

- **Concurrencia** (`LLM_CONCURRENCY`, por defecto 8): Groq y Gemini lanzan varias
  peticiones a la vez, con reintentos y backoff exponencial ante 429/5xx
  (`LLM_RETRY_ATTEMPTS`/`LLM_RETRY_BACKOFF_SECONDS`). 10.000 elementos ≈ 20 min.
- **Límites del proveedor** (que un pico de cuota no deje elementos sin descripción):
  - *Proactivo*: `LLM_REQUESTS_PER_MINUTE` (0 = sin tope) reparte un RPM máximo entre
    todos los hilos. Ponlo por debajo del límite del plan (Groq free ~30, Gemini free
    ~10-15) y el 429 no llega a producirse.
  - *Reactivo*: ante un 429 se respeta lo que pida esperar el proveedor — la cabecera
    `Retry-After` (Groq) o el `RetryInfo` del cuerpo del error (Gemini) — con tope de
    5 min, y la pausa frena a **todos** los hilos, no solo al que recibió el 429.
  - *Rescate*: los elementos que agoten sus reintentos se reintentan al final en
    `LLM_RETRY_ROUNDS` pasadas extra (por defecto 2), cada una tras un respiro de
    `LLM_RETRY_ROUND_WAIT_SECONDS` (30) para que la cuota por minuto se recupere. Solo
    lo que siga fallando se queda con su texto en crudo hasta el siguiente entrenamiento.
- **Batch API de Gemini** (**50% del precio estándar**): a partir de
  `LLM_BATCH_THRESHOLD` elementos pendientes (por defecto 500) el job entero va a
  `:batchGenerateContent`, troceado en jobs de `LLM_BATCH_CHUNK_SIZE` (2.000, para no
  pasar el límite inline de ~20 MB: 10.000 elementos = 5 jobs) y sondeado cada
  `LLM_BATCH_POLL_SECONDS`. Si un job falla o no acaba en `LLM_BATCH_WAIT_MINUTES`
  (60), *sus* elementos se rematan por el camino concurrente: el batch abarata, nunca
  pierde un entrenamiento. Groq no usa su Batch API (es de ficheros, con ventana de
  24 h-7 d); escala solo por concurrencia.
- Durante un enriquecimiento largo el paso re-reporta `optimizing` cada 5 min para que
  el sweeper de estancados del backend (`TRAINING_STALLED_AFTER_MINUTES`, 30 por
  defecto) no dé por muerto el run.

## Dónde desplegarlo

| Destino | Imagen | Entrypoint | Notas |
|---|---|---|---|
| **Local** | `Dockerfile` (CPU) o **`Dockerfile.gpu`** | `cli` | El backend con `TRAINING_PROVIDER=docker` hace `docker run --gpus all` por entrenamiento (ver "GPU en local") |
| **RunPod Serverless** | `Dockerfile.gpu` | `runpod_handler` | **Recomendado con `ENRICHER=local`**: GPU, escala a cero, ~0,06-0,15 $ por job grande |

## GPU en local

El contenedor usa la GPU **siempre que pueda**, y hacen falta las dos mitades:

1. **La imagen** tiene que traer las builds CUDA (torch cu121 + llama.cpp compilado con CUDA):
   eso es `Dockerfile.gpu`, no la CPU. Constrúyela con el mismo tag que usa el backend:

   ```bash
   docker compose --profile gpu build     # -> xeye-training-service:latest (CUDA)
   ```

2. **El daemon** tiene que poder ceder la GPU. El backend lanza el contenedor con
   `--gpus all` (`TRAINING_DOCKER_GPUS`, por defecto `all`) y **si el daemon no puede darla,
   reintenta en CPU** y lo avisa en el log — así el entrenamiento nunca falla por esto.

   En WSL2 hace falta el NVIDIA Container Toolkit (con `nvidia-smi` funcionando en el host no
   basta):

   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
     | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
     | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
     | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo service docker restart      # en WSL2; con systemd: sudo systemctl restart docker

   # comprobación
   docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi -L
   ```

Con una RTX 3060 (6 GB) entran de sobra Qwen2.5-3B Q4 (~2 GB) y el modelo de embeddings, y el
paso de LLM baja de ~7 s por elemento (CPU) a bastante menos de 1 s.

Los embeddings ya eligen `cuda` solos si torch la ve; el LLM se descarga entero a la GPU con
`LLM_GPU_LAYERS=-1` (por defecto; una build CPU lo ignora sin romperse).

## Ejecutar

```bash
# tests (no necesitan torch ni llama.cpp)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest

# imagen ligera (sin LLM) para probar rápido
docker build --build-arg INCLUDE_LLM=false -t xeye-training-service:latest .
# imagen completa con el GGUF horneado (~5-6 GB)
docker build -t xeye-training-service:latest .

# un entrenamiento a mano
docker run --rm --network xeye-network \
  -e ENRICHER=none -e TRAINING_DATA_PATH=/data/input/job.json \
  -v "$PWD/input:/data/input:ro" xeye-training-service:latest
```

Un job (lo genera el backend; los dos entrypoints leen el mismo objeto):

```json
{
  "training_id": 1, "list_id": 5, "user_id": 1,
  "callback_url": "http://xeye-java-backend:8000/webhooks/training-update",
  "webhook_secret": "…",
  "list": {"id": 5, "name": "Ferretería", "description": "Catálogo"},
  "elements": [{"id": 1, "text": "martillo", "description": null,
                "generated_description": null, "trained": false}],
  "options": [{"key": "train_all", "value": true}]
}
```

Variables en `.env.example`. En el backend: `TRAINING_PROVIDER=docker|runpod`,
`TRAINING_WEBHOOK_SECRET` = el `WEBHOOK_SECRET` de aquí, y `BACKEND_URL` con la URL que el
**contenedor** usa para llamar al webhook (no `localhost`).
