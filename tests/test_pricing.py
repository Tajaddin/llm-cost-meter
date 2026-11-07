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


@pytest.mark.parametrize(
    ("model", "input_rate", "output_rate", "cache_write_rate", "cache_read_rate"),
    [
        # https://platform.claude.com/docs/en/about-claude/pricing, checked 2026-11-06
        ("claude-haiku-4-5", 1.00, 5.00, 1.25, 0.10),
        ("claude-haiku-4-5-20251001", 1.00, 5.00, 1.25, 0.10),
        ("claude-sonnet-4-6", 3.00, 15.00, 3.75, 0.30),
        ("claude-opus-4-7", 5.00, 25.00, 6.25, 0.50),
    ],
)
def test_anthropic_rates_match_published_list_prices(
    model: str,
    input_rate: float,
    output_rate: float,
    cache_write_rate: float,
    cache_read_rate: float,
) -> None:
    p = get_pricing(model)
    assert p is not None
    assert p.input_per_mtok == pytest.approx(input_rate)
    assert p.output_per_mtok == pytest.approx(output_rate)
    assert p.cache_write_per_mtok == pytest.approx(cache_write_rate)
    assert p.cache_read_per_mtok == pytest.approx(cache_read_rate)


def test_cost_for_haiku_input_output_only() -> None:
    # 1_000_000 input × $1.00 + 500_000 output × $5.00 = $1.00 + $2.50 = $3.50
    cost = cost_for("claude-haiku-4-5-20251001", input_tokens=1_000_000, output_tokens=500_000)
    assert cost == pytest.approx(3.50, rel=1e-9)


def test_cost_for_haiku_with_cache_read_is_much_cheaper() -> None:
    # 1M cache read at $0.10/Mtok = $0.10 vs $1.00 if it were fresh input.
    cache_cost = cost_for("claude-haiku-4-5-20251001", cache_read_tokens=1_000_000)
    fresh_cost = cost_for("claude-haiku-4-5-20251001", input_tokens=1_000_000)
    assert cache_cost == pytest.approx(0.10, rel=1e-9)
    assert fresh_cost / cache_cost == pytest.approx(10.0, rel=1e-9)


def test_cost_for_unknown_model_returns_zero() -> None:
    assert cost_for("not-a-real-model", input_tokens=1_000_000) == 0.0


def test_cost_for_falls_back_on_dated_suffix() -> None:
    # ``claude-haiku-4-5-20251001`` and ``claude-haiku-4-5`` both registered;
    # ``claude-haiku-4-5-99999999`` is not, but the dated-suffix fallback should
    # find ``claude-haiku-4-5``.
    cost = cost_for("claude-haiku-4-5-99999999", input_tokens=1_000_000)
    assert cost == pytest.approx(1.00, rel=1e-9)


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
