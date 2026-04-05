#!/usr/bin/env python3
"""MCP server exposing active-memory as tools for Claude Code.

This lets Claude Code use the semantic B-tree as an external memory
system. Add it to your Claude Code MCP config and Claude can store,
query, and manage long-term context across sessions.

Setup in Claude Code (~/.claude/mcp.json):
{
    "mcpServers": {
        "active-memory": {
            "command": "python",
            "args": ["-m", "active_memory.mcp_server"],
            "env": {}
        }
    }
}

Exposed tools:
  - memory_store(key, value)     Store a key-value pair
  - memory_query(query, top_k)   Retrieve relevant memories
  - memory_stats()               Get tree statistics
  - memory_prune()               Prune cold entries
  - memory_compress()            Compress cold subtrees
  - memory_ingest_file(path)     Ingest a file into memory
  - memory_export()              Export all memories as JSON
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# MCP protocol uses JSON-RPC 2.0 over stdin/stdout


class ActiveMemoryMCPServer:
    """Minimal MCP server implementing the active-memory tools."""

    def __init__(self, config_path: str | None = None) -> None:
        from .config import load_config, build_scoring_config, build_btree_config
        from .embeddings import create_embedder
        from .scoring import Scorer
        from .btree import SemanticBTree

        file_overrides = load_config(config_path)

        embedder_provider = str(file_overrides.get("embedder", "hash"))
        embed_dim = int(file_overrides.get("embed_dim", 64))
        embed_model = str(file_overrides.get("embed_model", "text-embedding-3-small"))
        embedder_spec = create_embedder(
            embedder_provider,
            dim=embed_dim,
            openai_model=embed_model,
            verbose=bool(file_overrides.get("verbose", False)),
            stream=sys.stderr,
        )
        self.embedder = embedder_spec.embedder
        self.embedder_spec = embedder_spec
        self.scorer = Scorer(build_scoring_config(file_overrides))
        self.tree = SemanticBTree(
            embedder=self.embedder,
            scorer=self.scorer,
            config=build_btree_config(file_overrides),
        )

        state_dir = file_overrides.get("state_dir")
        if state_dir:
            self._state_path = Path(state_dir).expanduser() / "mcp_state.json"
        else:
            self._state_path = Path.home() / ".active-memory" / "mcp_state.json"
        self._load_state()

    # ── Tool definitions ──────────────────────────────────────────

    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "memory_store",
                "description": (
                    "Store a key-value pair in the active memory tree. "
                    "Use this to remember important facts, decisions, "
                    "code patterns, or any information that might be "
                    "useful later in the session."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Semantic label for what this memory is about (e.g. 'database choice', 'auth flow')",
                        },
                        "value": {
                            "type": "string",
                            "description": "The content to remember",
                        },
                    },
                    "required": ["key", "value"],
                },
            },
            {
                "name": "memory_query",
                "description": (
                    "Query the memory tree for relevant information. "
                    "Returns the most relevant memories scored by "
                    "recency, access frequency, and semantic similarity."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in memory",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_stats",
                "description": "Get statistics about the memory tree: size, depth, total tokens, node count.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "memory_prune",
                "description": "Remove cold/unused memories from the tree. Returns count of pruned entries.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "memory_compress",
                "description": "Compress cold subtrees into summaries. Reduces token count while preserving key information.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "memory_ingest_file",
                "description": "Read a file and ingest its contents into the memory tree as chunks.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to ingest",
                        },
                        "chunk_size": {
                            "type": "integer",
                            "description": "Words per chunk (default 300)",
                            "default": 300,
                        },
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "memory_export",
                "description": "Export all memories as a JSON list, sorted by score.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    # ── Tool handlers ─────────────────────────────────────────────

    def handle_tool(self, name: str, arguments: dict) -> dict:
        """Dispatch a tool call and return the result."""
        handlers = {
            "memory_store": self._tool_store,
            "memory_query": self._tool_query,
            "memory_stats": self._tool_stats,
            "memory_prune": self._tool_prune,
            "memory_compress": self._tool_compress,
            "memory_ingest_file": self._tool_ingest_file,
            "memory_export": self._tool_export,
        }

        handler = handlers.get(name)
        if not handler:
            return {"error": f"Unknown tool: {name}"}

        try:
            result = handler(arguments)
            self._save_state()
            return result
        except Exception as e:
            return {"error": str(e)}

    def _tool_store(self, args: dict) -> dict:
        key = args["key"]
        value = args["value"]
        t = self.tree.insert(key, value)
        return {
            "stored": True,
            "id": t.id,
            "key": key,
            "tokens": t.token_cost,
            "tree_size": self.tree.size,
        }

    def _tool_query(self, args: dict) -> dict:
        query = args["query"]
        top_k = args.get("top_k", 5)
        query_emb = self.embedder.embed([query])[0]
        results = self.tree.query(query_emb, top_k=top_k)

        return {
            "results": [
                {
                    "key": t.key_text,
                    "value": t.value_text,
                    "score": round(score, 4),
                    "hits": t.hit_count,
                    "age_seconds": round(time.time() - t.last_accessed, 1),
                }
                for score, t in results
            ],
            "total_in_tree": self.tree.size,
        }

    def _tool_stats(self, _args: dict) -> dict:
        all_tuples = self.tree.all_tuples()
        nodes = self.tree.all_nodes()
        return {
            "tree_size": self.tree.size,
            "tree_depth": self.tree.depth(),
            "total_nodes": len(nodes),
            "compressed_nodes": sum(1 for n in nodes if n.summary),
            "total_tokens": sum(t.token_cost for t in all_tuples),
            "total_hits": sum(t.hit_count for t in all_tuples),
        }

    def _tool_prune(self, _args: dict) -> dict:
        before = self.tree.size
        evicted = self.tree.prune()
        return {
            "pruned": len(evicted),
            "before": before,
            "after": self.tree.size,
            "evicted_keys": [t.key_text[:60] for t in evicted[:10]],
        }

    def _tool_compress(self, _args: dict) -> dict:
        before = self.tree.size
        count = self.tree.compress_cold_subtrees()
        return {
            "compressed_subtrees": count,
            "before": before,
            "after": self.tree.size,
        }

    def _tool_ingest_file(self, args: dict) -> dict:
        filepath = Path(args["path"]).expanduser()
        chunk_size = args.get("chunk_size", 300)

        if not filepath.exists():
            return {"error": f"File not found: {filepath}"}

        text = filepath.read_text(encoding="utf-8", errors="replace")
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            key = f"file:{filepath.name}:chunk_{i // chunk_size}"
            self.tree.insert(key, chunk)
            chunks.append(key)

        return {
            "file": str(filepath),
            "chunks_created": len(chunks),
            "tree_size": self.tree.size,
        }

    def _tool_export(self, _args: dict) -> dict:
        tuples = self.tree.all_tuples()
        export = []
        for t in tuples:
            export.append({
                "key": t.key_text,
                "value": t.value_text[:300],
                "score": round(self.scorer.score(t), 4),
                "hits": t.hit_count,
                "tokens": t.token_cost,
            })
        export.sort(key=lambda x: x["score"], reverse=True)
        return {"memories": export, "count": len(export)}

    # ── State persistence ─────────────────────────────────────────

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tuples = self.tree.all_tuples()
        state = {
            "embedding": {
                "provider": self.embedder_spec.provider,
                "model": getattr(self.embedder, "model", None),
                "dim": int(self.embedder.dim),
            },
            "tuples": [
                {
                    "key_text": t.key_text,
                    "value_text": t.value_text,
                    "key_emb": t.key_emb.tolist() if t.key_emb is not None else None,
                    "id": t.id,
                    "created_at": t.created_at,
                    "last_accessed": t.last_accessed,
                    "hit_count": t.hit_count,
                    "token_cost": t.token_cost,
                    "references": t.references,
                    "referenced_by": t.referenced_by,
                    "tags": t.tags,
                }
                for t in tuples
            ],
        }
        self._state_path.write_text(json.dumps(state))

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            import numpy as np
            from .types import KVTuple

            state = json.loads(self._state_path.read_text())
            tuple_dicts = state if isinstance(state, list) else state.get("tuples", [])
            saved_embedding = {} if isinstance(state, list) else state.get("embedding", {})
            expected = {
                "provider": self.embedder_spec.provider,
                "model": getattr(self.embedder, "model", None),
                "dim": int(self.embedder.dim),
            }
            reembed = bool(tuple_dicts) and (
                not saved_embedding
                or any(
                    saved_embedding.get(key) != expected.get(key)
                    for key in ("provider", "model", "dim")
                )
            )
            reembedded: list[np.ndarray] | None = None
            if reembed:
                keys = [td["key_text"] for td in tuple_dicts]
                reembedded = []
                for start in range(0, len(keys), 256):
                    reembedded.extend(self.embedder.embed(keys[start : start + 256]))

            for idx, td in enumerate(tuple_dicts):
                t = KVTuple(
                    key_text=td["key_text"],
                    value_text=td["value_text"],
                    key_emb=(
                        reembedded[idx]
                        if reembedded is not None
                        else (
                            np.array(td["key_emb"], dtype=np.float32)
                            if td["key_emb"] else None
                        )
                    ),
                    id=td["id"],
                    created_at=td["created_at"],
                    last_accessed=td["last_accessed"],
                    hit_count=td["hit_count"],
                    token_cost=td["token_cost"],
                    references=td.get("references", []),
                    referenced_by=td.get("referenced_by", []),
                    tags=td.get("tags", []),
                )
                self.tree.insert_tuple(t)
        except Exception:
            pass  # Start fresh if state is corrupted

    # ── MCP JSON-RPC protocol ─────────────────────────────────────

    def run(self) -> None:
        """Run the MCP server on stdin/stdout."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                continue

            response = self._handle_request(request)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

    def _handle_request(self, request: dict) -> dict | None:
        method = request.get("method", "")
        req_id = request.get("id")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "active-memory",
                        "version": "0.1.0",
                    },
                },
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self.get_tool_definitions()},
            }

        elif method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = self.handle_tool(tool_name, arguments)

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2),
                        }
                    ],
                },
            }

        elif method == "notifications/initialized":
            return None  # No response needed

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="active-memory MCP server")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON config file (default: ~/.active-memory/config.json if it exists)",
    )
    args = parser.parse_args()

    server = ActiveMemoryMCPServer(config_path=args.config)
    server.run()


if __name__ == "__main__":
    main()
