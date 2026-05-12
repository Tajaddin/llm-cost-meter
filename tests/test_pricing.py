"""Pricing math tests."""

from __future__ import annotations

import pytest

from llm_cost_meter import Pricing, cost_for, get_pricing, register_pricing


def test_anthropic_helper_applies_multipliers() -> None:
    p = Pricing.for_anthropic(input_per_mtok=0.80, output_per_mtok=4.00)
    assert p.input_per_mtok == 0.80
    assert p.output_per_mtok == 4.00
    assert p.cache_write_per_mtok == pytest.approx(1.0)  # 0.80 * 1.25
    assert p.cache_read_per_mtok == pytest.approx(0.08)  # 0.80 * 0.10


def test_cost_for_haiku_input_output_only() -> None:
    # 1_000_000 input × $0.80 + 500_000 output × $4.00 = $0.80 + $2.00 = $2.80
    cost = cost_for("claude-haiku-4-5-20251001", input_tokens=1_000_000, output_tokens=500_000)
    assert cost == pytest.approx(2.80, rel=1e-9)


def test_cost_for_haiku_with_cache_read_is_much_cheaper() -> None:
    # 1M cache read at $0.08/Mtok = $0.08 vs $0.80 if it were fresh input.
    cache_cost = cost_for("claude-haiku-4-5-20251001", cache_read_tokens=1_000_000)
    fresh_cost = cost_for("claude-haiku-4-5-20251001", input_tokens=1_000_000)
    assert cache_cost == pytest.approx(0.08, rel=1e-9)
    assert fresh_cost / cache_cost == pytest.approx(10.0, rel=1e-9)


def test_cost_for_unknown_model_returns_zero() -> None:
    assert cost_for("not-a-real-model", input_tokens=1_000_000) == 0.0


def test_cost_for_falls_back_on_dated_suffix() -> None:
    # ``claude-haiku-4-5-20251001`` and ``claude-haiku-4-5`` both registered;
    # ``claude-haiku-4-5-99999999`` is not, but the dated-suffix fallback should
    # find ``claude-haiku-4-5``.
    cost = cost_for("claude-haiku-4-5-99999999", input_tokens=1_000_000)
    assert cost == pytest.approx(0.80, rel=1e-9)


def test_register_pricing_overrides() -> None:
    register_pricing("custom-model", Pricing(input_per_mtok=10.0, output_per_mtok=20.0))
    assert get_pricing("custom-model").input_per_mtok == 10.0
    cost = cost_for("custom-model", input_tokens=1_000_000)
    assert cost == pytest.approx(10.0)


def test_openai_cached_tokens_cheaper() -> None:
    fresh = cost_for("gpt-4o-mini", input_tokens=1_000_000)
    cached = cost_for("gpt-4o-mini", cache_read_tokens=1_000_000)
    assert fresh > cached
    assert cached == pytest.approx(0.075, rel=1e-9)
