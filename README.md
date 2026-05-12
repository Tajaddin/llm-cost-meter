# llm-cost-meter

> Drop-in Prometheus middleware for Anthropic and OpenAI SDK calls. Token cost, latency, and prompt-cache hit ratio in one wrapper — **47 µs p99 overhead per call**, 21× under the 1 ms budget.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Tests](https://img.shields.io/badge/tests-17%20passing-brightgreen)](#tests) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

## Hero numbers

10,000-call overhead benchmark, mock client returning realistic usage data, single thread, Python 3.12 on Windows 11:

| | mean | p50 | p95 | **p99** | p99.9 | max |
|---|---:|---:|---:|---:|---:|---:|
| Bare client | 0.37 µs | 0.40 µs | 0.50 µs | 0.60 µs | 0.90 µs | 36.6 µs |
| Wrapped client | 22.02 µs | 20.80 µs | 32.00 µs | 47.40 µs | 84.10 µs | 248.0 µs |
| **Wrapper overhead** | **21.66 µs** | **20.50 µs** | **31.70 µs** | **47.10 µs** | **83.80 µs** | **247.3 µs** |

Target: p99 overhead < 1000 µs. Result: **PASS (47 µs)**. For context, a typical Anthropic API call takes 500-2000 ms — so the wrapper adds ~0.005% to real-world request latency.

Reproduce: `python bench/overhead.py -n 10000`. Full output in [`bench/overhead_results.json`](bench/overhead_results.json) after the run.

## Why this exists

LangSmith and Helicone are SaaS — your prompts, usage, and costs leave your network. Self-hosted alternatives like LiteLLM Proxy are heavyweight gateways that you have to deploy and route every call through.

`llm-cost-meter` is one file of Python plus a Prometheus client. You import it, wrap your existing Anthropic or OpenAI client, and the same Prometheus stack you already run picks up the metrics. No proxy, no cloud, no DNS changes, no SDK rewrite.

## Quickstart

```bash
pip install -e ".[anthropic,openai]"
```

```python
from anthropic import Anthropic
from llm_cost_meter import wrap, start_metrics_server

start_metrics_server(9000)        # /metrics on :9000

client = wrap(Anthropic())        # auto-detected
# ... use the client normally — every call is metered
resp = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=200,
    messages=[{"role": "user", "content": "hi"}],
)
```

The wrapper is a structural passthrough: `client.messages.create(...)` returns the *same* SDK response object. Nothing about the call shape changes.

OpenAI works the same way:

```python
from openai import OpenAI
client = wrap(OpenAI())
client.chat.completions.create(model="gpt-4o-mini", messages=[...])
```

## Metrics catalog

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `llm_calls_total` | Counter | provider, model, status (`ok` or `error`) | Every wrapped call |
| `llm_tokens_total` | Counter | provider, model, token_type (`input`, `output`, `cache_creation`, `cache_read`) | Token volume |
| `llm_cost_usd_total` | Counter | provider, model | USD spend at public list prices |
| `llm_latency_seconds` | Histogram | provider, model | Wall-clock latency per call |
| `llm_cache_hits_total` | Counter | provider, model | Cached input tokens served (Anthropic `cache_read_input_tokens` / OpenAI `cached_tokens`) |
| `llm_cache_misses_total` | Counter | provider, model | Fresh input tokens billed at full rate |

Cache hit ratio in PromQL:

```promql
sum(rate(llm_cache_hits_total[5m]))
  / clamp_min(sum(rate(llm_cache_hits_total[5m]) + rate(llm_cache_misses_total[5m])), 1e-9)
```

## Demo: Prometheus + Grafana stack

The `examples/` directory has a one-command demo that spins up Prometheus + Grafana with the `LLM Cost Meter` dashboard preloaded:

```bash
# Terminal 1 — generate synthetic LLM traffic, expose /metrics on :9000
python examples/demo.py

# Terminal 2 — bring up the observability stack
cd examples
docker compose up -d

# http://localhost:3000  (admin / admin) → Dashboards → LLM Cost Meter
```

Dashboard panels:

* Spend last 24h (USD)
* Calls per 5m
* Cache hit ratio (5m)
* Errors last 5m
* Cost rate per model ($/min)
* Latency percentiles (p50 / p95 / p99 per model)
* Token rate per type
* Cache hit ratio (rolling)

The dashboard JSON is at `examples/grafana/dashboards/llm_cost_meter.json` — drop it into any existing Grafana instance and point it at a Prometheus that scrapes your app's `/metrics`.

## Pricing customization

Public list pricing for Anthropic and OpenAI is built in. Override for your negotiated rates / batch API:

```python
from llm_cost_meter import Pricing, register_pricing

# 50% off via your enterprise agreement
register_pricing(
    "claude-haiku-4-5-20251001",
    Pricing.for_anthropic(input_per_mtok=0.40, output_per_mtok=2.00),
)
```

Anthropic prompt-cache math is applied automatically: `Pricing.for_anthropic` sets `cache_write_per_mtok = 1.25 × input` and `cache_read_per_mtok = 0.10 × input` (the public rates).

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

```
17 passed in 0.17s
```

Covers pricing math, metrics-increment correctness, error-path bookkeeping, prompt-cache token accounting, response-shape passthrough, multi-call accumulation, and auto-detect.

## Project layout

```
.
├── src/llm_cost_meter/
│   ├── metrics.py            # Prometheus Counter/Histogram definitions
│   ├── pricing.py            # Per-model pricing + cache math
│   ├── wrap.py               # AnthropicMeter / OpenAIMeter / wrap()
│   └── __init__.py
├── tests/                    # 17 pytest cases
├── bench/
│   ├── overhead.py           # 10K-call overhead benchmark
│   └── overhead_results.json # Last run's numbers
└── examples/
    ├── demo.py               # Synthetic traffic generator
    ├── docker-compose.yml    # Prometheus + Grafana
    ├── prometheus.yml
    └── grafana/              # Pre-provisioned datasource + dashboard
```

## Limitations

**Streaming responses.** The wrapper instruments `messages.create` and `chat.completions.create`. Streaming variants (`messages.stream`, `chat.completions.create(stream=True)`) emit usage in a final chunk — not yet wired. Tracked for v0.2.

**Async clients.** `AsyncAnthropic` and `AsyncOpenAI` follow the same shape and would slot into the same wrapper pattern — adding ~30 lines. Open to PRs.

**Cost is list-price.** Enterprise contracts, batch API discounts, and free-tier credits will shift absolute spend. Register custom pricing via `register_pricing(...)` to get your effective rate.

**No persistence.** Prometheus does the persistence. If your Prometheus retention is 7 days, that's your cost history retention.

## License

MIT — see [LICENSE](LICENSE).
