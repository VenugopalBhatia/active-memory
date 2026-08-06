#!/usr/bin/env python3
"""Anthropic-compatible HTTP proxy backed by persistent semantic memory."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from active_memory.context import ApproximateTokenCounter, BudgetConfig, ContextAssembler
from active_memory.context.formatter import content_to_text
from active_memory.ingestion import MemoryIngestor
from active_memory.retrieval import EmbeddingProvider, RetrievalConfig, TwoPassRetriever, create_embedding_provider
from active_memory.storage.sqlite_store import SQLiteMemoryStore

logger = logging.getLogger("active_memory.proxy")


@dataclass(slots=True)
class ProxyConfig:
    upstream_url: str = "https://api.anthropic.com"
    api_key: str | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    storage_path: str = "~/.active-memory/memory.db"
    default_namespace: str = "global"
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    strict_memory: bool = False
    storage_enabled: bool = True
    verbose: bool = False


@dataclass(frozen=True, slots=True)
class RequestState:
    session_id: str
    namespace: str
    next_ordinal: int


class MemoryEngine:
    """Coordinates ingestion, retrieval, and context assembly for requests."""

    def __init__(self, store: SQLiteMemoryStore, embedding_provider: EmbeddingProvider, *, retrieval_config: RetrievalConfig | None = None, budget_config: BudgetConfig | None = None, storage_enabled: bool = True, store_assistant_generated: bool = True, minimum_segment_tokens: int = 12, maximum_segment_tokens: int = 500, strict: bool = False) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.counter = ApproximateTokenCounter()
        self.ingestor = MemoryIngestor(
            store, embedding_provider, self.counter, storage_enabled=storage_enabled,
            store_assistant_generated=store_assistant_generated,
            minimum_segment_tokens=minimum_segment_tokens,
            maximum_segment_tokens=maximum_segment_tokens,
        )
        self.retriever = TwoPassRetriever(store, embedding_provider, retrieval_config)
        self.assembler = ContextAssembler(store, self.counter, budget_config)
        self.strict = strict
        self.last_trace: dict[str, Any] = {}

    @staticmethod
    def _event_id(session_id: str, ordinal: int, role: str, content: str) -> str:
        digest = hashlib.sha256(f"{session_id}\0{ordinal}\0{role}\0{content}".encode("utf-8")).hexdigest()
        return f"msg_{digest}"

    @staticmethod
    def _event_role(message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, list) and content and all(isinstance(block, dict) and block.get("type") == "tool_result" for block in content):
            return "tool"
        role = str(message.get("role", "user"))
        return role if role in {"user", "assistant", "system", "tool"} else "user"

    def transform_anthropic_request(self, request: dict[str, Any], *, session_id: str, namespace: str) -> tuple[dict[str, Any], RequestState]:
        original = copy.deepcopy(request)
        messages = request.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return original, RequestState(session_id, namespace, 0)
        try:
            system = request.get("system")
            if system:
                system_text = content_to_text(system)
                self.ingestor.ingest(session_id, namespace, "system", system_text, message_id=self._event_id(session_id, -1, "system", system_text), metadata={"request_system": True})
            for ordinal, raw in enumerate(messages):
                if not isinstance(raw, dict):
                    continue
                content = content_to_text(raw.get("content", ""))
                role = self._event_role(raw)
                self.ingestor.ingest(
                    session_id, namespace, role, content,
                    message_id=self._event_id(session_id, ordinal, role, content),
                    metadata={"anthropic_role": raw.get("role"), "content_blocks": raw.get("content") if isinstance(raw.get("content"), list) else None},
                )
            latest_user = next((content_to_text(message.get("content", "")) for message in reversed(messages) if isinstance(message, dict) and message.get("role") == "user"), "")
            if not latest_user:
                return original, RequestState(session_id, namespace, len(messages))
            now = datetime.now(timezone.utc)
            ranked = self.retriever.retrieve(latest_user, namespace, now)
            assembled = self.assembler.assemble(
                session_id=session_id, namespace=namespace, system=system,
                tools=request.get("tools"), messages=messages, ranked_memories=ranked, assembled_at=now,
            )
            transformed = copy.deepcopy(request)
            transformed["system"] = assembled.system
            transformed["messages"] = assembled.messages
            self.last_trace = {
                "status": "assembled", "session_id": session_id, "namespace": namespace,
                "input_tokens": assembled.input_tokens, "available_input_tokens": assembled.available_input_tokens,
                "included": [{"id": result.memory.id, "score": result.final_score, "reasons": result.retrieval_reason} for result in assembled.included],
                "excluded": assembled.excluded, "components": assembled.component_tokens,
            }
            return transformed, RequestState(session_id, namespace, len(messages))
        except Exception as exc:
            self.last_trace = {"status": "fallback", "error": f"{type(exc).__name__}: {exc}"}
            logger.exception("memory processing failed; forwarding original request")
            if self.strict:
                raise
            return original, RequestState(session_id, namespace, len(messages))

    def ingest_anthropic_response(self, payload: dict[str, Any], state: RequestState) -> None:
        content = payload.get("content", [])
        text = content_to_text(content)
        if not text:
            return
        response_id = str(payload.get("id") or self._event_id(state.session_id, state.next_ordinal, "assistant", text))
        message_id = response_id if response_id.startswith("msg_") else f"msg_{hashlib.sha256(response_id.encode()).hexdigest()}"
        try:
            self.ingestor.ingest(state.session_id, state.namespace, "assistant", text, message_id=message_id, metadata={"upstream_response_id": payload.get("id"), "content_blocks": content})
        except Exception:
            logger.exception("assistant response persistence failed")
            if self.strict:
                raise


def build_engine(config: ProxyConfig, *, retrieval_config: RetrievalConfig | None = None, budget_config: BudgetConfig | None = None, store_assistant_generated: bool = True, minimum_segment_tokens: int = 12, maximum_segment_tokens: int = 500) -> MemoryEngine:
    store = SQLiteMemoryStore(Path(config.storage_path).expanduser())
    provider = create_embedding_provider(config.embedding_provider, model=config.embedding_model)
    return MemoryEngine(
        store, provider, retrieval_config=retrieval_config, budget_config=budget_config,
        storage_enabled=config.storage_enabled, store_assistant_generated=store_assistant_generated,
        minimum_segment_tokens=minimum_segment_tokens, maximum_segment_tokens=maximum_segment_tokens,
        strict=config.strict_memory,
    )


class ProxyHandler(BaseHTTPRequestHandler):
    engine: MemoryEngine
    config: ProxyConfig

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._json(200, {"status": "ok", "storage": "sqlite", "retrieval": "exact_cosine"})
        elif path == "/debug":
            self._json(200, self.engine.last_trace)
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        path = self.path.split("?", 1)[0]
        if path != "/v1/messages":
            self._forward(body, None)
            return
        try:
            request_data = json.loads(body)
        except json.JSONDecodeError:
            self._json(400, {"type": "invalid_request_error", "message": "invalid JSON"})
            return
        session_id = self.headers.get("x-active-memory-session", "default")
        namespace = self.headers.get("x-active-memory-namespace", self.config.default_namespace)
        transformed, state = self.engine.transform_anthropic_request(request_data, session_id=session_id, namespace=namespace)
        self._forward(json.dumps(transformed).encode("utf-8"), state)

    def _forward(self, body: bytes, state: RequestState | None) -> None:
        url = self.config.upstream_url.rstrip("/") + self.path
        headers = {"Content-Type": "application/json", "anthropic-version": self.headers.get("anthropic-version", "2023-06-01")}
        for name in ("x-api-key", "authorization", "anthropic-beta"):
            value = self.headers.get(name)
            if value:
                headers[name] = value
        if "x-api-key" not in headers and "authorization" not in headers and (self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")):
            headers["x-api-key"] = self.config.api_key or os.environ["ANTHROPIC_API_KEY"]
        try:
            with urlopen(Request(url, data=body, headers=headers, method="POST")) as response:
                payload = response.read()
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() not in {"connection", "transfer-encoding", "content-length"}:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                if state:
                    try:
                        self.engine.ingest_anthropic_response(json.loads(payload), state)
                    except json.JSONDecodeError:
                        logger.warning("upstream response was not JSON; assistant event not persisted")
        except HTTPError as exc:
            payload = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except URLError as exc:
            self._json(502, {"type": "upstream_connection_error", "message": str(exc.reason)})

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        if self.config.verbose:
            super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local semantic memory proxy for Anthropic-compatible clients")
    parser.add_argument("--config")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--upstream")
    parser.add_argument("--database")
    parser.add_argument("--namespace")
    parser.add_argument("--embedder", choices=["local", "openai", "test"])
    parser.add_argument("--embed-model")
    parser.add_argument("--strict-memory", action="store_true")
    parser.add_argument("--disable-storage", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    from active_memory.config import load_config

    application = load_config(args.config)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = ProxyConfig(
        args.upstream or application.proxy.upstream_url, None,
        args.host or application.proxy.host, args.port or application.proxy.port,
        args.database or application.storage.path, args.namespace or application.memory.default_namespace,
        args.embedder or application.embeddings.provider, args.embed_model or application.embeddings.model,
        args.strict_memory or application.proxy.strict_memory,
        application.memory.storage_enabled and not args.disable_storage, args.verbose,
    )
    engine = build_engine(
        config, retrieval_config=application.retrieval_config(), budget_config=application.budget_config(),
        store_assistant_generated=application.memory.store_assistant_generated,
        minimum_segment_tokens=application.memory.minimum_segment_tokens,
        maximum_segment_tokens=application.memory.maximum_segment_tokens,
    )
    ProxyHandler.engine = engine
    ProxyHandler.config = config
    server = ThreadingHTTPServer((config.host, config.port), ProxyHandler)
    print(f"active-memory listening on http://{config.host}:{config.port}; SQLite: {Path(config.storage_path).expanduser()}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        engine.store.close()


if __name__ == "__main__":
    main()
