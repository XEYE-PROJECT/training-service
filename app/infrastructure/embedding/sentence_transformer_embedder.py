"""Embeddings con sentence-transformers. Carga perezosa y reutilizada (Lambda/RunPod
mantienen el proceso caliente entre jobs, lo que amortiza el ~1 s de carga del modelo)."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = max(1, batch_size)
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def warm_up(self) -> None:
        self._load()

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = _device()
            logger.info("Loading embedding model %s on %s", self._model_name, device)
            try:
                # Primero la caché: la imagen lleva el modelo horneado y debe cargar sin red.
                self._model = SentenceTransformer(
                    self._model_name, device=device, local_files_only=True
                )
            except Exception:
                logger.info("Model %s is not cached; downloading it", self._model_name)
                self._model = SentenceTransformer(self._model_name, device=device)
        return self._model


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
