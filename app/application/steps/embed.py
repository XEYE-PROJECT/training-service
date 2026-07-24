"""Paso 2 — un vector por elemento: centroide ponderado del vector documento (lo que el
elemento *es*) y de las consultas del LLM (lo que la gente *teclea*), todo L2-normalizado
antes y después de promediar — acerca el elemento a la zona donde caen las consultas reales.

Contrato (no cambiar sin cambiar el search-service): matriz float32 con una fila por
elemento en orden de id ASC; el backend registró esos ids al lanzar
(``trainings.element_ids``) y el search-service alinea las filas por ellos.
"""

from __future__ import annotations

import logging

import numpy as np

from app.application.pipeline import TrainingContext
from app.domain.text import document_text, query_variants

logger = logging.getLogger(__name__)


class EmbedStep:
    name = "training"

    def run(self, ctx: TrainingContext) -> None:
        ctx.report("training")
        if not ctx.elements:
            logger.warning("List %d has no elements; nothing to embed", ctx.job.list_id)
            return

        query_weight = ctx.settings.query_variant_weight
        texts: list[str] = []
        # rows_of[i] = índices en `texts` que pertenecen al elemento i; weights_of[i], sus pesos.
        rows_of: list[list[int]] = []
        weights_of: list[list[float]] = []

        for element in ctx.elements:
            enrichment = ctx.enrichments.get(element.id)
            rows: list[int] = []
            weights: list[float] = []

            rows.append(len(texts))
            weights.append(1.0)
            texts.append(document_text(element, enrichment))

            for variant in query_variants(enrichment):
                rows.append(len(texts))
                weights.append(query_weight)
                texts.append(variant)

            rows_of.append(rows)
            weights_of.append(weights)

        matrix = np.asarray(ctx.embedder.encode(texts), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise ValueError(f"Embedder returned {matrix.shape}, expected ({len(texts)}, dim)")
        matrix = _l2_normalize(matrix)

        vectors = np.zeros((len(ctx.elements), matrix.shape[1]), dtype=np.float32)
        for i, (rows, weights) in enumerate(zip(rows_of, weights_of)):
            weighted = matrix[rows] * np.asarray(weights, dtype=np.float32)[:, None]
            vectors[i] = weighted.sum(axis=0)
        ctx.vectors = _l2_normalize(vectors)

        logger.info(
            "Embedded %d elements (%d texts) into %d dims with %s",
            len(ctx.elements), len(texts), matrix.shape[1], ctx.embedder.model_name,
        )


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)
