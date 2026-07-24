"""Paso 1 — enriquecer cada elemento con el LLM (resumen + consultas representativas).

Es el paso caro (segundos por elemento). Lo hacen viable la caché del backend
(``elements.generated_description``, vaciada al cambiar texto/descripción; la opción
``force_enrich`` la ignora) y el presupuesto ``ENRICH_MAX_ELEMENTS``: lo omitido se embebe
con su texto tal cual y lo recoge el siguiente entrenamiento (clave en el muro de 15 min de Lambda).
"""

from __future__ import annotations

import logging

from app.application.pipeline import TrainingContext

logger = logging.getLogger(__name__)


class EnrichStep:
    name = "optimizing"

    def run(self, ctx: TrainingContext) -> None:
        ctx.report("optimizing")
        if ctx.enricher is None:
            logger.info("No enricher configured; embedding raw element texts")
            return

        force = bool(ctx.job.option("force_enrich", False))
        budget = ctx.settings.enrich_max_elements
        pending = []

        for element in ctx.elements:
            cached = None if force else element.cached_enrichment()
            if cached is not None:
                ctx.enrichments[element.id] = cached
            else:
                pending.append(element)

        if budget > 0 and len(pending) > budget:
            logger.warning(
                "%d elements need enrichment but the budget is %d; the rest keep their raw "
                "text this run (they will be picked up by the next training)",
                len(pending), budget,
            )
            pending = pending[:budget]

        logger.info(
            "Enriching %d elements (%d reused from cache) with %s",
            len(pending), len(ctx.enrichments), ctx.enricher.model_name,
        )

        for done, element in enumerate(pending, start=1):
            try:
                enrichment = ctx.enricher.enrich(element, ctx.job.list)
            except Exception:  # un elemento fallido no debe tumbar el entrenamiento entero
                logger.exception("Enrichment failed for element %d", element.id)
                continue
            if enrichment is None or enrichment.is_empty():
                continue
            ctx.enrichments[element.id] = enrichment
            ctx.fresh_enrichments[element.id] = enrichment.to_json()
            if done % 25 == 0:
                logger.info("Enriched %d/%d", done, len(pending))
