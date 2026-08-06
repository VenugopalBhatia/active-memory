"""Operational CLI for proxying, inspection, listing, and deletion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from active_memory.config import load_config
from active_memory.retrieval import TwoPassRetriever, create_embedding_provider
from active_memory.storage.sqlite_store import SQLiteMemoryStore
from active_memory.models import utc_now


def _runtime(config_path: str | None):
    config = load_config(config_path)
    store = SQLiteMemoryStore(config.storage.path)
    provider = create_embedding_provider(config.embeddings.provider, model=config.embeddings.model)
    return config, store, provider


def _confirm(prompt: str, assume_yes: bool) -> bool:
    return assume_yes or input(f"{prompt} Type 'yes' to continue: ").strip().casefold() == "yes"


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "proxy":
        sys.argv.pop(1)
        from active_memory.proxy import main as proxy_main
        proxy_main()
        return
    parser = argparse.ArgumentParser(prog="active-memory")
    parser.add_argument("--config")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--query", required=True)
    inspect_parser.add_argument("--namespace")
    inspect_parser.add_argument("--limit", type=int, default=10)
    memories = subparsers.add_parser("memories")
    memory_sub = memories.add_subparsers(dest="memory_command", required=True)
    list_parser = memory_sub.add_parser("list")
    list_parser.add_argument("--namespace")
    show_parser = memory_sub.add_parser("show")
    show_parser.add_argument("memory_id")
    delete = subparsers.add_parser("delete")
    target = delete.add_mutually_exclusive_group(required=True)
    target.add_argument("--session")
    target.add_argument("--namespace")
    target.add_argument("--memory")
    delete.add_argument("--yes", action="store_true")
    purge = subparsers.add_parser("purge")
    purge.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    config, store, provider = _runtime(args.config)
    namespace = getattr(args, "namespace", None) or config.memory.default_namespace
    try:
        if args.command == "inspect":
            retriever = TwoPassRetriever(store, provider, config.retrieval_config())
            results = retriever.retrieve(args.query, namespace, utc_now())[:args.limit]
            for result in results:
                print(json.dumps({"id": result.memory.id, "type": result.memory.memory_type, "trust": result.memory.trust_level, "tokens": result.memory.token_count, "scores": {"relevance": result.relevance, "recency": result.recency, "frequency": result.frequency, "affinity": result.affinity, "stage_one": result.stage_one_score, "final": result.final_score}, "reasons": result.retrieval_reason}, sort_keys=True))
        elif args.command == "memories" and args.memory_command == "list":
            for memory in store.get_active_memories(namespace):
                print(f"{memory.id}\t{memory.memory_type}\t{memory.trust_level}\t{memory.token_count}\t{memory.content[:100]}")
        elif args.command == "memories" and args.memory_command == "show":
            found = store.get_memories_by_ids([args.memory_id])
            if not found:
                raise SystemExit(f"memory not found: {args.memory_id}")
            memory = found[0]
            print(json.dumps({name: getattr(memory, name) for name in memory.__dataclass_fields__ if name != "embedding"}, default=str, indent=2, sort_keys=True))
        elif args.command == "delete":
            label = args.session or args.namespace or args.memory
            if not _confirm(f"Permanently delete {label}?", args.yes):
                raise SystemExit("cancelled")
            if args.session:
                count = store.delete_session(args.session)
            elif args.namespace:
                count = store.delete_namespace(args.namespace)
            else:
                count = store.delete_memory(args.memory)
            print(f"deleted {count} memory record(s)")
        elif args.command == "purge":
            path = store.path
            if not _confirm(f"Permanently delete the full database at {path}?", args.yes):
                raise SystemExit("cancelled")
            store.close()
            path.unlink(missing_ok=True)
            Path(str(path) + "-wal").unlink(missing_ok=True)
            Path(str(path) + "-shm").unlink(missing_ok=True)
            print(f"deleted {path}")
            return
    finally:
        try:
            store.close()
        except Exception:
            pass

