"""Paso 1 — enriquecer cada elemento con el LLM (resumen + consultas representativas).

Es el paso caro (segundos por elemento). Lo hacen viable la caché del backend
(``elements.generated_description``, vaciada al cambiar texto/descripción; la opción
``force_enrich`` la ignora) y dos presupuestos: ``ENRICH_MAX_ELEMENTS`` (elementos) y
``LLM_MAX_COST_PER_JOB`` (gasto por tokens, ``ctx.spend``). Lo omitido se embebe con su texto
tal cual y lo recoge el siguiente entrenamiento. Una lista con ``llm_enrichment=false`` (opt-out
del dueño) se salta el paso entero: sus textos no salen hacia ningún modelo.

Los enrichers remotos exponen ``enrich_many`` (concurrencia y, en Gemini, su Batch API):
este paso lo usa si existe y solo cae al bucle secuencial con el LLM local. En ambos caminos
se re-reporta ``optimizing`` periódicamente para que el sweeper de estancados del backend
no dé por muerto un enriquecimiento largo (listas de miles de elementos).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from app.application.pipeline import TrainingContext
from app.application.ports import Enricher
from app.domain.models import ElementInput, Enrichment

logger = logging.getLogger(__name__)

#: Cada cuánto re-reportar la fase durante un enriquecimiento largo (< stalled-after-minutes).
HEARTBEAT_SECONDS = 300


class EnrichStep:
    name = "optimizing"

    def run(self, ctx: TrainingContext) -> None:
        ctx.report("optimizing")
        if not ctx.job.list.llm_enrichment:
            logger.info("List %d opted out of LLM enrichment; embedding raw element texts", ctx.job.list_id)
            return
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
                len(pending),
                budget,
            )
            pending = pending[:budget]

        logger.info(
            "Enriching %d elements (%d reused from cache) with %s",
            len(pending),
            len(ctx.enrichments),
            ctx.enricher.model_name,
        )
        if not pending:
            return

        heartbeat = self._make_heartbeat(ctx)
        if hasattr(ctx.enricher, "enrich_many"):
            fresh = ctx.enricher.enrich_many(pending, ctx.job.list, heartbeat=heartbeat, meter=ctx.spend)
        else:
            fresh = self._enrich_sequentially(ctx.enricher, ctx, pending, heartbeat)

        for element_id, enrichment in fresh.items():
            ctx.enrichments[element_id] = enrichment
            ctx.fresh_enrichments[element_id] = enrichment.to_json()
        if ctx.spend.exhausted:
            logger.warning(
                "LLM budget exhausted (%.6f of %.6f) after %d/%d elements; the rest keep their raw text this run",
                ctx.spend.cost,
                ctx.spend.max_cost,
                len(fresh),
                len(pending),
            )

    @staticmethod
    def _enrich_sequentially(
        enricher: Enricher, ctx: TrainingContext, pending: list[ElementInput], heartbeat: Callable[[], None]
    ) -> dict[int, Enrichment]:
        results: dict[int, Enrichment] = {}
        for done, element in enumerate(pending, start=1):
            if ctx.spend.exhausted:
                break
            try:
                enrichment = enricher.enrich(element, ctx.job.list, meter=ctx.spend)
            except Exception:  # un elemento fallido no debe tumbar el entrenamiento entero
                logger.exception("Enrichment failed for element %d", element.id)
                continue
            finally:
                heartbeat()
            if enrichment is None or enrichment.is_empty():
                continue
            results[element.id] = enrichment
            if done % 25 == 0:
                logger.info("Enriched %d/%d", done, len(pending))
        return results

    @staticmethod
    def _make_heartbeat(ctx: TrainingContext) -> Callable[[], None]:
        last = time.monotonic()

        def beat() -> None:
            nonlocal last
            now = time.monotonic()
            if now - last >= HEARTBEAT_SECONDS:
                last = now
                ctx.report("optimizing")

        return beat
