"""Prometheus metric definitions and helpers.

Six metrics are exported by the wrapper:

* ``llm_calls_total`` (Counter)        — calls made, labeled by provider / model / status
* ``llm_tokens_total`` (Counter)       — tokens consumed, labeled by provider / model / token_type
* ``llm_cost_usd_total`` (Counter)     — USD spend, labeled by provider / model
* ``llm_latency_seconds`` (Histogram)  — call latency, labeled by provider / model
* ``llm_cache_hits_total`` (Counter)   — cached input tokens hit, labeled by provider / model
* ``llm_cache_misses_total`` (Counter) — non-cached input tokens, labeled by provider / model

Cache hit ratio is computed in Prometheus as
``rate(llm_cache_hits_total[5m]) / (rate(llm_cache_hits_total[5m]) + rate(llm_cache_misses_total[5m]))``.

For tests and benchmarks, :func:`reset_metrics` clears every counter — it
replaces the underlying samples but keeps the registry stable.
"""

from __future__ import annotations

from typing import Iterable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    start_http_server,
)

_LABELS = ("provider", "model")
_STATUS_LABELS = ("provider", "model", "status")
_TOKEN_LABELS = ("provider", "model", "token_type")

CALLS = Counter(
    "llm_calls_total",
    "Total LLM SDK calls observed by llm-cost-meter.",
    labelnames=_STATUS_LABELS,
)

TOKENS = Counter(
    "llm_tokens_total",
    "Total tokens consumed (token_type in {input,output,cache_read,cache_creation}).",
    labelnames=_TOKEN_LABELS,
)

COST = Counter(
    "llm_cost_usd_total",
    "Total USD cost accumulated, using public list pricing.",
    labelnames=_LABELS,
)

LATENCY = Histogram(
    "llm_latency_seconds",
    "Wall-clock latency per LLM SDK call.",
    labelnames=_LABELS,
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

CACHE_HITS = Counter(
    "llm_cache_hits_total",
    "Input tokens served from the prompt cache (Anthropic cache_read, OpenAI cached_tokens).",
    labelnames=_LABELS,
)

CACHE_MISSES = Counter(
    "llm_cache_misses_total",
    "Input tokens NOT served from cache (fresh input tokens billed at full rate).",
    labelnames=_LABELS,
)


_ALL_METRICS = (CALLS, TOKENS, COST, LATENCY, CACHE_HITS, CACHE_MISSES)


def metrics_text(registry: CollectorRegistry = REGISTRY) -> tuple[bytes, str]:
    """Return ``(body, content_type)`` for serving the Prometheus exposition format."""
    return generate_latest(registry), CONTENT_TYPE_LATEST


def start_metrics_server(port: int = 9000, addr: str = "0.0.0.0") -> None:
    """Start a Prometheus HTTP exposition server on ``addr:port``.

    Wraps ``prometheus_client.start_http_server``. Calling this is optional —
    if the host process already has a Prometheus endpoint, the metrics
    registered in this module will be picked up automatically.
    """
    start_http_server(port, addr)


def reset_metrics() -> None:
    """Reset every counter / histogram to zero. Intended for tests + benchmarks.

    ``prometheus_client`` does not expose a public reset, but the labels-only
    metrics it produces are safe to clear via the private ``_metrics`` dict.
    """
    for m in _ALL_METRICS:
        # ``_metrics`` is the per-label child store; clearing it drops every
        # observed series. New observations re-create children on demand.
        children: dict = getattr(m, "_metrics", {})  # type: ignore[assignment]
        children.clear()


def iter_all_metrics() -> Iterable:
    """Yield each metric object defined by this module. Useful in tests."""
    yield from _ALL_METRICS
