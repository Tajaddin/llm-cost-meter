"""End-to-end demo: synthesize traffic against a fake LLM client and expose
metrics on http://localhost:9000/metrics so the docker-compose Prometheus +
Grafana stack can scrape it.

Run from the project root::

    pip install -e ".[anthropic]"
    python examples/demo.py

Then in another shell::

    cd examples
    docker compose up -d
    open http://localhost:3000     # admin / admin
"""

from __future__ import annotations

import random
import time
from types import SimpleNamespace

from llm_cost_meter import AnthropicMeter, start_metrics_server


def _fake_anthropic_client():
    class _Messages:
        def create(self, **kw):
            # Simulate variable latency.
            time.sleep(random.uniform(0.05, 0.5))
            input_tokens = random.randint(80, 400)
            cache_read = random.choice([0, 0, 0, input_tokens // 2])  # 25% cache hit rate
            return SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=input_tokens - cache_read,
                    output_tokens=random.randint(30, 200),
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=cache_read,
                ),
                content=[SimpleNamespace(type="text", text="demo output")],
            )

    class _Client:
        messages = _Messages()

    return _Client()


def main() -> None:
    start_metrics_server(9000)
    print("Metrics exposed at http://localhost:9000/metrics")
    print("Generating synthetic traffic. Ctrl-C to stop.")
    client = AnthropicMeter(_fake_anthropic_client())
    models = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-7"]
    while True:
        try:
            client.messages.create(
                model=random.choices(models, weights=[6, 3, 1], k=1)[0],
                messages=[{"role": "user", "content": "demo"}],
                max_tokens=100,
            )
        except KeyboardInterrupt:
            print("Stopping.")
            break


if __name__ == "__main__":
    main()
