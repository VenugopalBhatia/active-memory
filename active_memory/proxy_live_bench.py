"""Live direct-vs-proxy benchmark for very long Anthropic-format requests.

This script builds a synthetic transcript at a target raw token count,
starts an isolated active-memory proxy, and compares:

1. Direct Anthropic API request with the full transcript
2. The same request routed through the active-memory proxy

The intended use is to validate the real proxy path under 1M-token
conversation loads and capture evidence about whether the proxy can
turn an unusable request into a successful one.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .live_ab_bench import build_scenario


@dataclass
class RequestResult:
    target: str
    status: int
    elapsed_s: float
    text: str
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0
    error_type: str | None = None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_json(url: str, body: bytes, *, timeout: int = 600) -> RequestResult:
    headers = {
        "content-type": "application/json",
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
    }
    req = Request(url, data=body, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", "replace")
            elapsed = time.perf_counter() - started
            payload = json.loads(text)
            usage = payload.get("usage", {})
            result_text = _extract_response_text(payload)
            return RequestResult(
                target=url,
                status=resp.status,
                elapsed_s=elapsed,
                text=result_text,
                usage_input_tokens=int(usage.get("input_tokens", 0) or 0),
                usage_output_tokens=int(usage.get("output_tokens", 0) or 0),
            )
    except HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")
        elapsed = time.perf_counter() - started
        error_type = None
        try:
            payload = json.loads(text)
            error_type = payload.get("error", {}).get("type") or payload.get("type")
        except Exception:
            pass
        return RequestResult(
            target=url,
            status=exc.code,
            elapsed_s=elapsed,
            text=text,
            error_type=error_type,
        )
    except URLError as exc:
        elapsed = time.perf_counter() - started
        return RequestResult(
            target=url,
            status=0,
            elapsed_s=elapsed,
            text=str(exc.reason),
            error_type="url_error",
        )


def _extract_response_text(payload: dict[str, Any]) -> str:
    content = payload.get("content", [])
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(part for part in parts if part)


def _poll_health(port: int, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError(f"Proxy on port {port} did not become healthy within {timeout_s}s")


def _get_proxy_stats(port: int) -> dict[str, Any]:
    with urlopen(f"http://127.0.0.1:{port}/stats", timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _start_proxy(
    *,
    port: int,
    embedder: str,
    embed_model: str,
    budget: int,
    state_dir: str,
) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        "-m",
        "active_memory.proxy",
        "--port",
        str(port),
        "--embedder",
        embedder,
        "--embed-model",
        embed_model,
        "--budget",
        str(budget),
        "--state-dir",
        state_dir,
        "--verbose",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _read_remaining(stream: Any) -> str:
    if stream is None:
        return ""
    try:
        return stream.read()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live proxy benchmark for 1M-token Anthropic-format requests",
    )
    parser.add_argument(
        "--token-target",
        type=int,
        default=1_000_000,
        help="Target raw transcript tokens (default: 1,000,000)",
    )
    parser.add_argument(
        "--fact-id",
        default="codename",
        help="Which planted fact to probe (default: codename)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Anthropic model to use",
    )
    parser.add_argument(
        "--embedder",
        choices=["hash", "openai", "auto"],
        default="hash",
        help="Embedding provider used by the proxy",
    )
    parser.add_argument(
        "--embed-model",
        default="text-embedding-3-small",
        help="Embedding model when using OpenAI embeddings",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100_000,
        help="Proxy managed context budget",
    )
    parser.add_argument(
        "--filler-repetitions",
        type=int,
        default=10,
        help="Repetitions per filler turn",
    )
    parser.add_argument(
        "--skip-direct",
        action="store_true",
        help="Skip the direct upstream control request",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")
    if args.embedder == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set for --embedder openai")

    scenario = build_scenario(
        args.token_target,
        filler_repetitions=args.filler_repetitions,
    )
    fact = next((f for f in scenario.facts if f.fact_id == args.fact_id), None)
    if fact is None:
        available = ", ".join(f.fact_id for f in scenario.facts)
        raise SystemExit(f"Unknown fact_id '{args.fact_id}'. Available: {available}")

    messages = list(scenario.messages)
    messages.append({"role": "user", "content": fact.query})
    request_payload = {
        "model": args.model,
        "max_tokens": 128,
        "messages": messages,
    }
    body = json.dumps(request_payload).encode("utf-8")

    port = _find_free_port()
    with tempfile.TemporaryDirectory(prefix="active-memory-proxy-bench-") as state_dir:
        proxy = _start_proxy(
            port=port,
            embedder=args.embedder,
            embed_model=args.embed_model,
            budget=args.budget,
            state_dir=state_dir,
        )
        try:
            _poll_health(port)

            direct_result: RequestResult | None = None
            if not args.skip_direct:
                direct_result = _request_json(
                    "https://api.anthropic.com/v1/messages",
                    body,
                )

            proxy_result = _request_json(
                f"http://127.0.0.1:{port}/v1/messages",
                body,
            )
            stats = _get_proxy_stats(port)
        finally:
            proxy.terminate()
            try:
                proxy.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proxy.kill()
                proxy.wait(timeout=5)
            stderr = _read_remaining(proxy.stderr)
            stdout = _read_remaining(proxy.stdout)

    summary = {
        "token_target": args.token_target,
        "actual_raw_tokens": scenario.raw_tokens,
        "turns": scenario.turns,
        "fact_id": fact.fact_id,
        "query": fact.query,
        "expected_keywords": fact.expected_keywords,
        "embedder": args.embedder,
        "embed_model": args.embed_model,
        "budget": args.budget,
        "direct": asdict(direct_result) if direct_result else None,
        "proxy": asdict(proxy_result),
        "proxy_stats": stats,
        "proxy_logs": stderr.strip(),
    }

    print()
    print("=" * 88)
    print(
        f"1M Proxy Benchmark | raw={scenario.raw_tokens:,} | turns={scenario.turns} | "
        f"fact={fact.fact_id} | embedder={args.embedder}"
    )
    print("=" * 88)
    if direct_result:
        print(
            f"Direct: status={direct_result.status} elapsed={direct_result.elapsed_s:.2f}s "
            f"in={direct_result.usage_input_tokens:,} out={direct_result.usage_output_tokens:,}"
        )
        print(f"Direct text: {direct_result.text[:300]}")
    print(
        f"Proxy:  status={proxy_result.status} elapsed={proxy_result.elapsed_s:.2f}s "
        f"in={proxy_result.usage_input_tokens:,} out={proxy_result.usage_output_tokens:,}"
    )
    print(f"Proxy text: {proxy_result.text[:300]}")
    print(
        f"Proxy stats: tree_size={stats.get('tree_size')} depth={stats.get('tree_depth')} "
        f"nodes={stats.get('total_nodes')} stored_tokens={stats.get('total_tokens_stored')}"
    )

    if args.json:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

