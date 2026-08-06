from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from active_memory.context import BudgetConfig
from active_memory.proxy import MemoryEngine, ProxyConfig, ProxyHandler
from active_memory.retrieval import DeterministicTestEmbeddingProvider
from active_memory.storage.sqlite_store import SQLiteMemoryStore


class UpstreamHandler(BaseHTTPRequestHandler):
    received: ClassVar[dict] = {}
    status: ClassVar[int] = 200

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        type(self).received = json.loads(self.rfile.read(length))
        if type(self).status == 200:
            payload = {"id": "msg_http_response", "type": "message", "role": "assistant", "content": [{"type": "text", "text": "Decision recorded."}]}
        else:
            payload = {"type": "rate_limit_error", "message": "slow down"}
        body = json.dumps(payload).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def start(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def test_http_proxy_roundtrip_and_upstream_error_surface(tmp_path) -> None:
    try:
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    except PermissionError:
        pytest.skip("runner does not permit localhost socket binding")
    start(upstream)
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    ProxyHandler.engine = MemoryEngine(store, DeterministicTestEmbeddingProvider(), budget_config=BudgetConfig(model_context_limit=1000, reserved_response_tokens=100, safety_margin_tokens=50))
    ProxyHandler.config = ProxyConfig(upstream_url=f"http://127.0.0.1:{upstream.server_port}", storage_path=str(tmp_path / "memory.db"))
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
    start(proxy)
    try:
        body = json.dumps({"model": "claude-test", "max_tokens": 50, "messages": [{"role": "user", "content": "Decision: we use PostgreSQL."}]}).encode()
        request = Request(f"http://127.0.0.1:{proxy.server_port}/v1/messages", data=body, headers={"Content-Type": "application/json", "x-active-memory-session": "http-test"}, method="POST")
        response = json.loads(urlopen(request).read())
        assert response["id"] == "msg_http_response"
        assert UpstreamHandler.received["model"] == "claude-test"
        assert UpstreamHandler.received["max_tokens"] == 50
        assert UpstreamHandler.received["messages"][-1]["content"] == "Decision: we use PostgreSQL."
        assert any(message.role == "assistant" for message in store.get_recent_messages("http-test", 10))

        UpstreamHandler.status = 429
        try:
            urlopen(request)
        except HTTPError as exc:
            assert exc.code == 429
            assert json.loads(exc.read())["type"] == "rate_limit_error"
        else:
            raise AssertionError("upstream error was not surfaced")
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()
        store.close()
