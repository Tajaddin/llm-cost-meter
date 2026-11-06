"""Metrics exposition server tests."""

from __future__ import annotations

import urllib.request

from llm_cost_meter import start_metrics_server


def test_metrics_server_binds_loopback_by_default() -> None:
    server, _thread = start_metrics_server(port=0)
    try:
        host, port = server.server_address[:2]
        assert host == "127.0.0.1"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as resp:
            assert b"llm_calls_total" in resp.read()
    finally:
        server.shutdown()
        server.server_close()
