"""Enriquecedor LLM local (llama.cpp + un GGUF cuantizado, p. ej. Qwen2.5-3B-Instruct Q4_K_M).

El proveedor por defecto: sin clave de API y sin sacar datos del contenedor. También es con
diferencia lo más lento del entrenamiento (~7 s/elemento en CPU, ~0,5 s con GPU): de ahí la
caché en el backend y la existencia de ``ENRICH_MAX_ELEMENTS``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import Settings
from app.domain.models import ElementInput, Enrichment, ListInput
from app.infrastructure.enrichment.prompt import SYSTEM_PROMPT, build_user_prompt, parse_response

logger = logging.getLogger(__name__)


class LocalLlmEnricher:
    def __init__(self, settings: Settings) -> None:
        self._path = Path(settings.llm_model_path)
        self._context_size = settings.llm_context_size
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        self._threads = settings.llm_threads or None
        self._gpu_layers = settings.llm_gpu_layers
        self._llm = None

    @property
    def model_name(self) -> str:
        return self._path.stem

    def enrich(self, element: ElementInput, list_context: ListInput) -> Enrichment | None:
        llm = self._load()
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(element, list_context)},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
        )
        content = (response["choices"][0]["message"].get("content") or "").strip()
        return parse_response(content, self.model_name)

    def _load(self):
        if self._llm is None:
            if not self._path.is_file():
                raise FileNotFoundError(f"LLM model not found at {self._path}")
            import llama_cpp
            from llama_cpp import Llama

            # -1 = descargar todas las capas a la GPU. Es el valor por defecto y es seguro:
            # un llama.cpp compilado sin CUDA lo ignora y corre en CPU. Que la GPU se use de
            # verdad depende de la imagen (Dockerfile.gpu) y de que el contenedor reciba `--gpus`.
            supports_offload = bool(llama_cpp.llama_supports_gpu_offload())
            logger.info("Loading local LLM %s (n_gpu_layers=%d, cuda=%s, gpu_offload_support=%s)",
                        self._path, self._gpu_layers, _cuda_available(), supports_offload)
            if self._gpu_layers != 0 and not supports_offload:
                logger.warning(
                    "llama-cpp-python compilado SIN offload a GPU: el enriquecimiento correrá "
                    "en CPU (~8 s/elemento). Revisa el wheel instalado en la imagen (Dockerfile.gpu)."
                )
            kwargs = {
                "model_path": str(self._path),
                "n_ctx": self._context_size,
                "n_gpu_layers": self._gpu_layers,
                # verbose durante la carga: llama.cpp imprime "offloaded N/M layers to GPU",
                # la única prueba fiable de que el offload ocurre (supports_offload puede ser
                # True y aun así no haber kernels para la GPU asignada).
                "verbose": True,
            }
            if self._threads:
                kwargs["n_threads"] = self._threads
            self._llm = Llama(**kwargs)
            self._llm.verbose = False  # silenciar los timings por petición; la carga ya quedó logueada
        return self._llm


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
