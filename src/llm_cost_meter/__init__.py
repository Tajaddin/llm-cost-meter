"""Drop-in Prometheus middleware for LLM SDK calls.

Wraps an Anthropic or OpenAI client and records per-call token usage, cost
(via public pricing), latency, and Anthropic prompt-cache hit metrics.
"""

from llm_cost_meter.metrics import (
    CALLS,
    CACHE_HITS,
    COST,
    LATENCY,
    TOKENS,
    metrics_text,
    reset_metrics,
    start_metrics_server,
)
from llm_cost_meter.pricing import (
    Pricing,
    cost_for,
    get_pricing,
    register_pricing,
)
from llm_cost_meter.wrap import (
    AnthropicMeter,
    OpenAIMeter,
    wrap,
)

__version__ = "0.1.0"

__all__ = [
    # wrappers
    "wrap",
    "AnthropicMeter",
    "OpenAIMeter",
    # metrics
    "CALLS",
    "TOKENS",
    "COST",
    "LATENCY",
    "CACHE_HITS",
    "metrics_text",
    "reset_metrics",
    "start_metrics_server",
    # pricing
    "Pricing",
    "cost_for",
    "get_pricing",
    "register_pricing",
]
