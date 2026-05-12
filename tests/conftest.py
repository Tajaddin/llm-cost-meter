"""Shared fixtures: fake Anthropic / OpenAI clients that exercise the wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_cost_meter.metrics import reset_metrics


@pytest.fixture(autouse=True)
def _clear_metrics():
    reset_metrics()
    yield
    reset_metrics()


def _anthropic_usage(input_tokens=10, output_tokens=5, cache_write=0, cache_read=0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_write,
        cache_read_input_tokens=cache_read,
    )


def make_fake_anthropic(
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_write: int = 0,
    cache_read: int = 0,
    raise_exc: type[Exception] | None = None,
):
    class _Messages:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create(self, **kw):
            self.calls.append(kw)
            if raise_exc is not None:
                raise raise_exc("boom")
            return SimpleNamespace(
                usage=_anthropic_usage(input_tokens, output_tokens, cache_write, cache_read),
                content=[SimpleNamespace(type="text", text="ok")],
            )

    class FakeAnthropic:
        def __init__(self) -> None:
            self.messages = _Messages()

    return FakeAnthropic()


def make_fake_openai(
    *,
    prompt_tokens: int = 12,
    completion_tokens: int = 6,
    cached_tokens: int = 0,
    raise_exc: type[Exception] | None = None,
):
    details = SimpleNamespace(cached_tokens=cached_tokens)

    class _Completions:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create(self, **kw):
            self.calls.append(kw)
            if raise_exc is not None:
                raise raise_exc("boom")
            return SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    prompt_tokens_details=details,
                ),
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            )

    class _Chat:
        def __init__(self) -> None:
            self.completions = _Completions()

    class FakeOpenAI:
        def __init__(self) -> None:
            self.chat = _Chat()

    return FakeOpenAI()
