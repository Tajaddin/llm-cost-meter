"""Drop-in wrappers for Anthropic and OpenAI SDK clients.

Both wrappers preserve the underlying client's API. Only the ``create`` method
on the relevant resource is intercepted; every other attribute is forwarded
verbatim via ``__getattr__``.

Usage::

    from anthropic import Anthropic
    from llm_cost_meter import wrap

    client = wrap(Anthropic())  # auto-detected
    resp = client.messages.create(model="claude-haiku-4-5-20251001", ...)
    # metrics recorded transparently; resp is the same object the SDK returns
"""

from __future__ import annotations

import time
from typing import Any

from llm_cost_meter.metrics import CACHE_HITS, CACHE_MISSES, CALLS, COST, LATENCY, TOKENS
from llm_cost_meter.pricing import cost_for


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _record(
    provider: str,
    model: str,
    status: str,
    elapsed: float,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> None:
    CALLS.labels(provider=provider, model=model, status=status).inc()
    LATENCY.labels(provider=provider, model=model).observe(elapsed)

    if input_tokens:
        TOKENS.labels(provider=provider, model=model, token_type="input").inc(input_tokens)
    if output_tokens:
        TOKENS.labels(provider=provider, model=model, token_type="output").inc(output_tokens)
    if cache_write_tokens:
        TOKENS.labels(provider=provider, model=model, token_type="cache_creation").inc(cache_write_tokens)
    if cache_read_tokens:
        TOKENS.labels(provider=provider, model=model, token_type="cache_read").inc(cache_read_tokens)

    if cache_read_tokens:
        CACHE_HITS.labels(provider=provider, model=model).inc(cache_read_tokens)
    if input_tokens:
        CACHE_MISSES.labels(provider=provider, model=model).inc(input_tokens)

    cost = cost_for(
        model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
    )
    if cost > 0:
        COST.labels(provider=provider, model=model).inc(cost)


def _safe_int(obj: Any, name: str, default: int = 0) -> int:
    try:
        value = getattr(obj, name, default)
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class _AnthropicMessagesProxy:
    """Wraps the ``messages`` resource. Only ``create`` is intercepted."""

    __slots__ = ("_inner",)

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model") or "unknown"
        start = time.perf_counter()
        try:
            resp = self._inner.create(**kwargs)
        except Exception:
            elapsed = time.perf_counter() - start
            _record(provider="anthropic", model=model, status="error", elapsed=elapsed)
            raise
        elapsed = time.perf_counter() - start

        usage = getattr(resp, "usage", None)
        input_tokens = _safe_int(usage, "input_tokens") if usage else 0
        output_tokens = _safe_int(usage, "output_tokens") if usage else 0
        cache_write = _safe_int(usage, "cache_creation_input_tokens") if usage else 0
        cache_read = _safe_int(usage, "cache_read_input_tokens") if usage else 0
        _record(
            provider="anthropic",
            model=model,
            status="ok",
            elapsed=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
        )
        return resp

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - trivial forwarder
        return getattr(self._inner, name)


class AnthropicMeter:
    """Drop-in wrapper for an Anthropic client.

    Forwards every attribute except ``messages``, which is wrapped by a proxy
    that intercepts ``create`` calls.
    """

    __slots__ = ("_inner", "_messages")

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._messages = _AnthropicMessagesProxy(inner.messages) if hasattr(inner, "messages") else None

    @property
    def messages(self) -> Any:
        return self._messages if self._messages is not None else self._inner.messages

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - trivial forwarder
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class _OpenAIChatCompletionsProxy:
    """Wraps ``chat.completions``. Intercepts ``create``."""

    __slots__ = ("_inner",)

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model") or "unknown"
        start = time.perf_counter()
        try:
            resp = self._inner.create(**kwargs)
        except Exception:
            elapsed = time.perf_counter() - start
            _record(provider="openai", model=model, status="error", elapsed=elapsed)
            raise
        elapsed = time.perf_counter() - start

        usage = getattr(resp, "usage", None)
        input_tokens = _safe_int(usage, "prompt_tokens") if usage else 0
        output_tokens = _safe_int(usage, "completion_tokens") if usage else 0
        cache_read = 0
        if usage is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cache_read = _safe_int(details, "cached_tokens")
        # Non-cached input tokens for accurate cache-hit ratio.
        fresh_input = max(0, input_tokens - cache_read)
        _record(
            provider="openai",
            model=model,
            status="ok",
            elapsed=elapsed,
            input_tokens=fresh_input,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
        )
        return resp

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - trivial forwarder
        return getattr(self._inner, name)


class _OpenAIChatProxy:
    """Wraps ``chat`` so that ``chat.completions.create`` is intercepted."""

    __slots__ = ("_inner", "_completions")

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._completions = _OpenAIChatCompletionsProxy(inner.completions) if hasattr(inner, "completions") else None

    @property
    def completions(self) -> Any:
        return self._completions if self._completions is not None else self._inner.completions

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - trivial forwarder
        return getattr(self._inner, name)


class OpenAIMeter:
    """Drop-in wrapper for an OpenAI client.

    Forwards every attribute except ``chat``, which is wrapped to intercept
    ``chat.completions.create``.
    """

    __slots__ = ("_inner", "_chat")

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._chat = _OpenAIChatProxy(inner.chat) if hasattr(inner, "chat") else None

    @property
    def chat(self) -> Any:
        return self._chat if self._chat is not None else self._inner.chat

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - trivial forwarder
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# auto-detect wrap()
# ---------------------------------------------------------------------------


def wrap(client: Any) -> Any:
    """Return a metered wrapper for an Anthropic or OpenAI SDK client.

    Detection is done by inspecting attribute shape, so it works even if the
    SDK package name differs from the typical import path (e.g. ``AsyncOpenAI``,
    Azure-flavored OpenAI clients, etc.).
    """
    class_name = type(client).__name__.lower()
    if "anthropic" in class_name:
        return AnthropicMeter(client)
    if "openai" in class_name:
        return OpenAIMeter(client)
    # Fallback: structural detection.
    if hasattr(client, "messages") and hasattr(client.messages, "create"):
        return AnthropicMeter(client)
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        return OpenAIMeter(client)
    raise TypeError(
        f"llm_cost_meter.wrap(): unrecognized client type {type(client)!r}. "
        "Pass an Anthropic or OpenAI client, or wrap manually with "
        "AnthropicMeter / OpenAIMeter."
    )
