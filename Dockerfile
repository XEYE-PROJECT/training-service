# Imagen one-shot de CPU: un contenedor = un entrenamiento.
#
# Es la imagen que el backend lanza en local (proveedor `docker`) y la misma que ejecutarían
# AWS Batch / ECS RunTask en la nube: solo corren el CMD por defecto con
# TRAINING_JOB/TRAINING_DATA_PATH en el entorno.
#
# INCLUDE_LLM=true hornea el modelo cuantizado (ENRICHER=local funciona sin red, ~2 GB más);
# con --build-arg INCLUDE_LLM=false sale una imagen ligera para Groq/Gemini o sin LLM.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf-cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

# torch solo-CPU: la wheel por defecto arrastra ~2 GB de CUDA que esta imagen nunca usa.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY requirements.txt requirements-local-llm.txt ./
RUN pip install -r requirements.txt

ARG INCLUDE_LLM=true
RUN if [ "$INCLUDE_LLM" = "true" ]; then pip install -r requirements-local-llm.txt; fi

# Hornea el modelo de embeddings: debe cargar sin red (misma trampa que en el search-service).
ARG EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

# Hornea el GGUF. Qwen2.5-3B Q4_K_M por defecto; el build 1.5B es ~3x más rápido en CPU a
# costa de algo de calidad.
ARG LLM_MODEL_REPO=bartowski/Qwen2.5-3B-Instruct-GGUF
ARG LLM_MODEL_FILE=Qwen2.5-3B-Instruct-Q4_K_M.gguf
ENV LLM_MODEL_PATH=/app/models/${LLM_MODEL_FILE}
RUN if [ "$INCLUDE_LLM" = "true" ]; then \
        mkdir -p /app/models && \
        python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='${LLM_MODEL_REPO}', filename='${LLM_MODEL_FILE}', local_dir='/app/models')"; \
    fi

# Modelos de embeddings extra seleccionables por entrenamiento (opción `embedding_model`),
# separados por espacios. Capa aparte *después* del GGUF a propósito: añadir un modelo no
# re-descarga nada más. Mantener en sincronía con `xeye.training.embedding-models` del
# backend y con la imagen del search-service (que embebe las consultas).
ARG EXTRA_EMBEDDING_MODELS="paraphrase-multilingual-mpnet-base-v2"
RUN for m in ${EXTRA_EMBEDDING_MODELS}; do \
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$m')"; \
    done

COPY app ./app

CMD ["python", "-m", "app.entrypoints.cli"]
