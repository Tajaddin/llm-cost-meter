"""Measure llm_cost_meter wrapper overhead per call.

Runs N iterations of a no-op mock client both unwrapped and wrapped. The
delta between the two timings is the overhead introduced by the wrapper.
Reports p50 / p95 / p99 / max / mean in microseconds and writes a JSON
artifact at ``bench/overhead_results.json``.

We use ``time.perf_counter_ns`` for sub-microsecond resolution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_cost_meter import AnthropicMeter, reset_metrics


def _make_fake_client():
    """The mock client used for the benchmark. Returns immediately with
    realistic-looking usage data so the wrapper has work to do (token counts +
    cost computation), but no network or real LLM is involved.
    """
    usage = SimpleNamespace(
        input_tokens=120,
        output_tokens=45,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    resp = SimpleNamespace(usage=usage, content=[SimpleNamespace(type="text", text="ok")])

    class _Messages:
        def create(self, **_kwargs):
            return resp

    class _Client:
        messages = _Messages()

    return _Client()


def _bench(client, n: int) -> list[int]:
    """Run ``n`` calls and return a list of per-call latencies in nanoseconds."""
    out: list[int] = []
    create = client.messages.create
    for _ in range(n):
        t0 = time.perf_counter_ns()
        create(model="claude-haiku-4-5-20251001", messages=[])
        out.append(time.perf_counter_ns() - t0)
    return out


def _quantile(samples: list[int], q: float) -> int:
    if not samples:
        return 0
    s = sorted(samples)
    k = int(q * (len(s) - 1))
    return s[k]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--iterations", type=int, default=10_000)
    p.add_argument("--warmup", type=int, default=1_000)
    p.add_argument("--out", type=str, default="bench/overhead_results.json")
    args = p.parse_args()

    n = args.iterations
    warm = args.warmup

    reset_metrics()
    bare = _make_fake_client()
    wrapped = AnthropicMeter(_make_fake_client())

    # Warm up — JIT-like effects in the Python interpreter still matter.
    _bench(bare, warm)
    _bench(wrapped, warm)
    reset_metrics()  # ignore metric churn from warmup

    bare_ns = _bench(bare, n)
    wrapped_ns = _bench(wrapped, n)

    def summary(label: str, samples: list[int]) -> dict:
        return {
            "label": label,
            "n": len(samples),
            "mean_us": round(sum(samples) / len(samples) / 1000, 3),
            "p50_us": round(_quantile(samples, 0.50) / 1000, 3),
            "p95_us": round(_quantile(samples, 0.95) / 1000, 3),
            "p99_us": round(_quantile(samples, 0.99) / 1000, 3),
            "p999_us": round(_quantile(samples, 0.999) / 1000, 3),
            "max_us": round(max(samples) / 1000, 3),
        }

    bare_summary = summary("bare", bare_ns)
    wrapped_summary = summary("wrapped", wrapped_ns)

    overhead = [w - b for w, b in zip(wrapped_ns, bare_ns)]
    overhead_summary = summary("overhead (wrapped - bare)", overhead)

    result = {
        "iterations": n,
        "bare": bare_summary,
        "wrapped": wrapped_summary,
        "overhead": overhead_summary,
        "p99_under_1ms": overhead_summary["p99_us"] < 1000.0,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nOverhead benchmark — {n:,} calls\n")
    print(f"{'Label':<28} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'p99.9':>8} {'max':>8}  (microseconds)")
    for label, row in (("bare client", bare_summary), ("wrapped client", wrapped_summary), ("overhead (delta)", overhead_summary)):
        print(
            f"{label:<28} {row['mean_us']:>8.2f} {row['p50_us']:>8.2f} "
            f"{row['p95_us']:>8.2f} {row['p99_us']:>8.2f} {row['p999_us']:>8.2f} {row['max_us']:>8.2f}"
        )
    print(f"\nTarget: overhead p99 < 1000 us")
    print(f"Result: {'PASS' if result['p99_under_1ms'] else 'FAIL'} (p99 overhead = {overhead_summary['p99_us']:.2f} us)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
