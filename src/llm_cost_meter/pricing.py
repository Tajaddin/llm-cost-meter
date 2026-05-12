"""Per-model pricing (USD per million tokens).

Pricing tables track the *public list price* for each model as of the date in
the entry. Production deployments with negotiated rates or batch APIs should
override via :func:`register_pricing`.

Anthropic prompt caching has two extra rates:

* ``cache_creation_input``: 1.25x the input rate (you pay extra to write to cache)
* ``cache_read_input``:     0.1x the input rate (you pay much less to read)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pricing:
    """Per-million-token pricing for one model.

    Use :func:`Pricing.for_anthropic` to apply the standard 1.25x / 0.1x
    cache multipliers automatically.
    """

    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float = 0.0
    cache_read_per_mtok: float = 0.0

    @classmethod
    def for_anthropic(cls, input_per_mtok: float, output_per_mtok: float) -> "Pricing":
        return cls(
            input_per_mtok=input_per_mtok,
            output_per_mtok=output_per_mtok,
            cache_write_per_mtok=input_per_mtok * 1.25,
            cache_read_per_mtok=input_per_mtok * 0.10,
        )


# ---------------------------------------------------------------------------
# Anthropic public pricing (USD per million tokens, as of 2026-01).
# ---------------------------------------------------------------------------
_ANTHROPIC_PRICING: dict[str, Pricing] = {
    "claude-haiku-4-5-20251001": Pricing.for_anthropic(0.80, 4.00),
    "claude-haiku-4-5": Pricing.for_anthropic(0.80, 4.00),
    "claude-3-5-haiku-latest": Pricing.for_anthropic(0.80, 4.00),
    "claude-sonnet-4-6": Pricing.for_anthropic(3.00, 15.00),
    "claude-3-5-sonnet-latest": Pricing.for_anthropic(3.00, 15.00),
    "claude-opus-4-7": Pricing.for_anthropic(15.00, 75.00),
    "claude-3-opus-latest": Pricing.for_anthropic(15.00, 75.00),
}

# ---------------------------------------------------------------------------
# OpenAI public pricing (USD per million tokens, as of 2026-01).
# OpenAI's prompt cache is included automatically when applicable (50% off on
# input). We expose it via ``cache_read_per_mtok`` for consistency with the
# Anthropic side; ``cache_write_per_mtok`` is left at 0 because OpenAI does not
# charge a creation premium.
# ---------------------------------------------------------------------------
_OPENAI_PRICING: dict[str, Pricing] = {
    "gpt-4o-mini": Pricing(0.15, 0.60, 0.0, 0.075),
    "gpt-4o": Pricing(2.50, 10.00, 0.0, 1.25),
    "gpt-4.1-mini": Pricing(0.40, 1.60, 0.0, 0.10),
    "gpt-4.1": Pricing(2.00, 8.00, 0.0, 0.50),
    "o4-mini": Pricing(1.10, 4.40, 0.0, 0.275),
}

_REGISTRY: dict[str, Pricing] = {**_ANTHROPIC_PRICING, **_OPENAI_PRICING}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def register_pricing(model: str, pricing: Pricing) -> None:
    """Add or override the pricing entry for ``model``."""
    _REGISTRY[model] = pricing


def get_pricing(model: str) -> Pricing | None:
    """Return the registered pricing for ``model`` or ``None``.

    Falls back to the model stripped of a trailing ``-NNNNNNNN`` date suffix
    when the exact ID is not registered.
    """
    if model in _REGISTRY:
        return _REGISTRY[model]
    import re

    # Strip a trailing date-like suffix: ``-YYYYMMDD`` or ``-NNNNNNNN``.
    stripped = re.sub(r"-\d{6,8}$", "", model)
    if stripped != model and stripped in _REGISTRY:
        return _REGISTRY[stripped]
    return None


def cost_for(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Compute the USD cost for one call given token counts.

    Returns 0.0 if no pricing entry is registered for the model. The Anthropic
    SDK reports cache tokens as ``cache_creation_input_tokens`` and
    ``cache_read_input_tokens`` on the ``usage`` object; the OpenAI SDK reports
    cache reads under ``usage.prompt_tokens_details.cached_tokens``.
    """
    p = get_pricing(model)
    if p is None:
        return 0.0
    return (
        input_tokens * p.input_per_mtok / 1_000_000
        + output_tokens * p.output_per_mtok / 1_000_000
        + cache_write_tokens * p.cache_write_per_mtok / 1_000_000
        + cache_read_tokens * p.cache_read_per_mtok / 1_000_000
    )
