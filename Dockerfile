# Imagen one-shot de CPU: un contenedor = un entrenamiento.
#
# Es la imagen que el backend lanza en local (proveedor `docker`): corre el CMD por
# defecto con TRAINING_JOB/TRAINING_DATA_PATH en el entorno.
#
# INCLUDE_LLM=true hornea el modelo cuantizado (ENRICHER=local funciona sin red, ~2 GB más);
# con --build-arg INCLUDE_LLM=false sale una imagen ligera para Groq/Gemini o sin LLM.
# Base fijada por digest (Dependabot abre PR cuando cambia): builds reproducibles.
#
# Endurecimiento: el proceso corre como `xeye` (uid 1000, sin root); los modelos se hornean
# verificados (GGUF por SHA-256, sentence-transformers por revisión de Hugging Face) y en
# ejecución HF_HUB_OFFLINE=1 impide cualquier descarga: solo se puede usar lo horneado, que es
# exactamente la allowlist EMBEDDING_MODELS_ALLOWED.
FROM python:3.11-slim@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf-cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --user-group --home-dir /app --no-create-home --shell /usr/sbin/nologin xeye

# torch solo-CPU: la wheel por defecto arrastra ~2 GB de CUDA que esta imagen nunca usa.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY requirements.txt requirements-local-llm.txt ./
RUN pip install -r requirements.txt

ARG INCLUDE_LLM=true
RUN if [ "$INCLUDE_LLM" = "true" ]; then pip install -r requirements-local-llm.txt; fi

# A partir de aquí todo es del usuario del worker: los modelos se descargan ya como `xeye`
# (un chown -R posterior duplicaría gigabytes en una capa nueva).
RUN mkdir -p /app/models /app/.hf-cache && chown -R xeye:xeye /app
USER xeye

# Hornea el modelo de embeddings: debe cargar sin red (misma trampa que en el search-service).
ARG EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

# Hornea el GGUF y verifica su SHA-256 (el que publica Hugging Face para ese fichero): si
# upstream lo reemplaza, el build falla en vez de hornear otro modelo. Qwen2.5-3B Q4_K_M por
# defecto; el build 1.5B es ~3x más rápido en CPU a costa de algo de calidad (cambiar también el hash).
ARG LLM_MODEL_REPO=bartowski/Qwen2.5-3B-Instruct-GGUF
ARG LLM_MODEL_FILE=Qwen2.5-3B-Instruct-Q4_K_M.gguf
ARG LLM_MODEL_SHA256=9c9f56a391a3abbd5b89d0245bf6106081bcc3173119d4229235dd9d23253f94
ENV LLM_MODEL_PATH=/app/models/${LLM_MODEL_FILE}
RUN if [ "$INCLUDE_LLM" = "true" ]; then \
        python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='${LLM_MODEL_REPO}', filename='${LLM_MODEL_FILE}', local_dir='/app/models')" \
        && echo "${LLM_MODEL_SHA256}  /app/models/${LLM_MODEL_FILE}" | sha256sum -c - \
        && rm -rf /app/models/.cache; \
    fi

# Modelos de embeddings extra seleccionables por entrenamiento (opción `embedding_model`),
# separados por espacios. Capa aparte *después* del GGUF a propósito: añadir un modelo no
# re-descarga nada más. Mantener en sincronía con `xeye.training.embedding-models` del
# backend y con la imagen del search-service (que embebe las consultas). Son también la
# allowlist del worker: ningún job puede pedir otro modelo.
ARG EXTRA_EMBEDDING_MODELS="paraphrase-multilingual-mpnet-base-v2"
ENV EMBEDDING_MODELS_ALLOWED=${EXTRA_EMBEDDING_MODELS}
RUN for m in ${EXTRA_EMBEDDING_MODELS}; do \
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$m')"; \
    done

# Cada modelo horneado debe ser la revisión (commit de Hugging Face) que se auditó; se descarga
# `main` para que cargue sin red por su nombre y después se comprueba a qué commit apuntó.
ARG EMBEDDING_MODEL_REVISIONS="paraphrase-multilingual-MiniLM-L12-v2=e8f8c211226b894fcb81acc59f3b34ba3efd5f42 paraphrase-multilingual-mpnet-base-v2=4328cf26390c98c5e3c738b4460a05b95f4911f5"
RUN for pair in ${EMBEDDING_MODEL_REVISIONS}; do \
        name="${pair%%=*}"; rev="${pair#*=}"; \
        ref="$(cat "${HF_HOME}/hub/models--sentence-transformers--${name}/refs/main")"; \
        [ "$ref" = "$rev" ] || { echo "Model ${name} resolved to ${ref}, expected ${rev}: update EMBEDDING_MODEL_REVISIONS deliberately"; exit 1; }; \
    done

# Nada se descarga en ejecución: lo que no está horneado no existe.
ENV HF_HUB_OFFLINE=1

COPY --chown=xeye:xeye app ./app

# Commit desplegado, para etiquetar los eventos de Sentry (RunPod construye sin build-args:
# ahí queda "unknown"; se puede fijar SENTRY_RELEASE en las variables del endpoint).
ARG GIT_SHA=unknown
ENV SENTRY_RELEASE=${GIT_SHA}

CMD ["python", "-m", "app.entrypoints.cli"]
