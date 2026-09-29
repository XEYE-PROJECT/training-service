"""Consumo y gasto del LLM en un job. Python puro, seguro entre hilos.

El worker paga el LLM por tokens: el medidor los suma (cada petición aporta lo que reporta su
proveedor), los convierte en coste con la tarifa configurada y, si hay tope por job, avisa de que
está agotado para que ningún hilo lance más peticiones. Lo que queda sin enriquecer conserva su
texto tal cual (como con ``ENRICH_MAX_ELEMENTS``) y lo recoge el siguiente entrenamiento.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class LlmUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    def __add__(self, other: LlmUsage) -> LlmUsage:
        return LlmUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.requests + other.requests,
        )


class SpendMeter:
    """Tokens consumidos -> coste, con tope opcional por job (0 = sin tope).

    Precios en unidades monetarias por millón de tokens (la forma en que los publican los
    proveedores). Sin precios, el coste es 0 y el tope no puede agotarse: se cuenta igualmente.
    """

    def __init__(
        self,
        price_per_million_input: float = 0.0,
        price_per_million_output: float = 0.0,
        max_cost: float = 0.0,
    ) -> None:
        self._price_in = max(0.0, price_per_million_input)
        self._price_out = max(0.0, price_per_million_output)
        self._max_cost = max(0.0, max_cost)
        self._lock = threading.Lock()
        self._usage = LlmUsage()

    def add(self, input_tokens: int | None, output_tokens: int | None, requests: int = 1) -> None:
        with self._lock:
            self._usage = self._usage + LlmUsage(int(input_tokens or 0), int(output_tokens or 0), requests)

    @property
    def usage(self) -> LlmUsage:
        with self._lock:
            return self._usage

    @property
    def cost(self) -> float:
        usage = self.usage
        amount = usage.input_tokens * self._price_in + usage.output_tokens * self._price_out
        return round(amount / 1_000_000, 6)

    @property
    def max_cost(self) -> float:
        return self._max_cost

    @property
    def exhausted(self) -> bool:
        """True si hay tope y el gasto acumulado ya lo alcanza: no lanzar más peticiones."""
        return self._max_cost > 0 and self.cost >= self._max_cost
