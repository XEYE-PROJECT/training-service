"""El medidor de gasto: tokens -> coste, tope por job y seguridad entre hilos."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from app.domain.spend import LlmUsage, SpendMeter


def test_usage_adds_up_and_cost_follows_the_prices_per_million_tokens():
    meter = SpendMeter(price_per_million_input=0.5, price_per_million_output=2.0, max_cost=0)
    meter.add(1_000_000, 0)
    meter.add(0, 500_000)
    meter.add(None, None)  # un proveedor que no reporta tokens no rompe el conteo

    assert meter.usage == LlmUsage(input_tokens=1_000_000, output_tokens=500_000, requests=3)
    assert meter.cost == 1.5
    assert meter.exhausted is False  # sin tope nunca se agota


def test_without_prices_tokens_are_counted_but_nothing_is_charged():
    meter = SpendMeter()
    meter.add(10, 5)
    assert meter.usage.requests == 1 and meter.cost == 0.0 and meter.exhausted is False


def test_the_cap_trips_once_the_cost_reaches_it():
    meter = SpendMeter(price_per_million_output=1.0, max_cost=0.000002)
    meter.add(0, 1)
    assert meter.exhausted is False
    meter.add(0, 1)
    assert meter.exhausted is True


def test_concurrent_adds_are_not_lost():
    meter = SpendMeter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: meter.add(1, 1), range(2000)))
    assert meter.usage == LlmUsage(2000, 2000, 2000)
