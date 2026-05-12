"""Wrapper integration tests."""

from __future__ import annotations

import pytest

from llm_cost_meter import (
    AnthropicMeter,
    OpenAIMeter,
    metrics_text,
    wrap,
)

from tests.conftest import make_fake_anthropic, make_fake_openai


def _metric_value(body: bytes, line_prefix: str) -> float | None:
    for line in body.decode().splitlines():
        if line.startswith(line_prefix):
            return float(line.split()[-1])
    return None


def test_anthropic_wrap_records_tokens_and_cost() -> None:
    client = wrap(make_fake_anthropic(input_tokens=100, output_tokens=50))
    resp = client.messages.create(model="claude-haiku-4-5-20251001", messages=[])
    assert resp.usage.input_tokens == 100  # passthrough preserved
    body, _ = metrics_text()
    assert _metric_value(body, 'llm_calls_total{model="claude-haiku-4-5-20251001",provider="anthropic",status="ok"}') == 1.0
    assert _metric_value(body, 'llm_tokens_total{model="claude-haiku-4-5-20251001",provider="anthropic",token_type="input"}') == 100.0
    assert _metric_value(body, 'llm_tokens_total{model="claude-haiku-4-5-20251001",provider="anthropic",token_type="output"}') == 50.0
    cost = _metric_value(body, 'llm_cost_usd_total{model="claude-haiku-4-5-20251001",provider="anthropic"}')
    # 100 × 0.80/M + 50 × 4.00/M = 0.00008 + 0.0002 = 0.00028
    assert cost == pytest.approx(0.00028, rel=1e-6)


def test_anthropic_wrap_records_cache_tokens() -> None:
    client = wrap(make_fake_anthropic(input_tokens=200, output_tokens=20, cache_write=500, cache_read=1000))
    client.messages.create(model="claude-haiku-4-5-20251001", messages=[])
    body, _ = metrics_text()
    assert _metric_value(body, 'llm_tokens_total{model="claude-haiku-4-5-20251001",provider="anthropic",token_type="cache_creation"}') == 500.0
    assert _metric_value(body, 'llm_tokens_total{model="claude-haiku-4-5-20251001",provider="anthropic",token_type="cache_read"}') == 1000.0
    assert _metric_value(body, 'llm_cache_hits_total{model="claude-haiku-4-5-20251001",provider="anthropic"}') == 1000.0


def test_anthropic_error_still_records_call() -> None:
    client = wrap(make_fake_anthropic(raise_exc=RuntimeError))
    with pytest.raises(RuntimeError):
        client.messages.create(model="claude-haiku-4-5-20251001", messages=[])
    body, _ = metrics_text()
    assert _metric_value(body, 'llm_calls_total{model="claude-haiku-4-5-20251001",provider="anthropic",status="error"}') == 1.0
    # No token counts recorded on error.
    assert _metric_value(body, 'llm_tokens_total{model="claude-haiku-4-5-20251001",provider="anthropic",token_type="input"}') is None


def test_openai_wrap_records_prompt_and_completion_tokens() -> None:
    client = wrap(make_fake_openai(prompt_tokens=200, completion_tokens=80, cached_tokens=50))
    client.chat.completions.create(model="gpt-4o-mini", messages=[])
    body, _ = metrics_text()
    # fresh_input = 200 - 50 = 150
    assert _metric_value(body, 'llm_tokens_total{model="gpt-4o-mini",provider="openai",token_type="input"}') == 150.0
    assert _metric_value(body, 'llm_tokens_total{model="gpt-4o-mini",provider="openai",token_type="output"}') == 80.0
    assert _metric_value(body, 'llm_tokens_total{model="gpt-4o-mini",provider="openai",token_type="cache_read"}') == 50.0


def test_openai_wrap_cost_uses_cache_discount() -> None:
    client = wrap(make_fake_openai(prompt_tokens=1_000_050, completion_tokens=0, cached_tokens=50))
    client.chat.completions.create(model="gpt-4o-mini", messages=[])
    body, _ = metrics_text()
    cost = _metric_value(body, 'llm_cost_usd_total{model="gpt-4o-mini",provider="openai"}')
    # 1_000_000 fresh × $0.15/M + 50 cached × $0.075/M ≈ $0.15 + tiny
    assert 0.149 < cost < 0.151


def test_wrap_explicit_constructors() -> None:
    a = AnthropicMeter(make_fake_anthropic())
    a.messages.create(model="claude-haiku-4-5-20251001", messages=[])
    o = OpenAIMeter(make_fake_openai())
    o.chat.completions.create(model="gpt-4o-mini", messages=[])
    body, _ = metrics_text()
    assert _metric_value(body, 'llm_calls_total{model="claude-haiku-4-5-20251001",provider="anthropic",status="ok"}') == 1.0
    assert _metric_value(body, 'llm_calls_total{model="gpt-4o-mini",provider="openai",status="ok"}') == 1.0


def test_wrap_rejects_unknown_client() -> None:
    class NotAnLLM:
        pass

    with pytest.raises(TypeError):
        wrap(NotAnLLM())


def test_wrap_passthrough_for_unknown_attr() -> None:
    inner = make_fake_anthropic()
    inner.extra = "hello"  # attribute that doesn't exist on the wrapper
    wrapped = wrap(inner)
    assert wrapped.extra == "hello"


def test_latency_histogram_observes_value() -> None:
    client = wrap(make_fake_anthropic())
    client.messages.create(model="claude-haiku-4-5-20251001", messages=[])
    body, _ = metrics_text()
    # Histogram has a _count series.
    count = _metric_value(body, 'llm_latency_seconds_count{model="claude-haiku-4-5-20251001",provider="anthropic"}')
    assert count == 1.0


def test_multiple_calls_accumulate() -> None:
    client = wrap(make_fake_anthropic(input_tokens=10, output_tokens=5))
    for _ in range(5):
        client.messages.create(model="claude-haiku-4-5-20251001", messages=[])
    body, _ = metrics_text()
    assert _metric_value(body, 'llm_calls_total{model="claude-haiku-4-5-20251001",provider="anthropic",status="ok"}') == 5.0
    assert _metric_value(body, 'llm_tokens_total{model="claude-haiku-4-5-20251001",provider="anthropic",token_type="input"}') == 50.0
