#!/usr/bin/env python3
"""Import legacy JSON tree/session records into the SQLite journal."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from active_memory.context.tokenizer import ApproximateTokenCounter
from active_memory.ingestion.writer import MemoryWriter
from active_memory.models import Memory, new_id
from active_memory.storage.sqlite_store import SQLiteMemoryStore


def migrate_legacy(source: Path, destination: Path, namespace: str = "legacy") -> dict[str, object]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    store = SQLiteMemoryStore(destination)
    writer = MemoryWriter(store, ApproximateTokenCounter())
    report: dict[str, object] = {"source": str(source), "destination": str(destination), "messages": 0, "memories": 0, "skipped": []}
    try:
        raw_messages = payload.get("conversation", payload.get("messages", []))
        for index, item in enumerate(raw_messages):
            try:
                role = str(item.get("role", "user"))
                content = str(item.get("content", ""))
                writer.write_raw("legacy", role if role in {"user", "assistant", "system", "tool"} else "user", content, message_id=f"legacy_msg_{index}")
                report["messages"] = int(report["messages"]) + 1
            except Exception as exc:
                report["skipped"].append({"kind": "message", "index": index, "error": str(exc)})

        tuples = payload.get("tuples", payload.get("tree", {}).get("tuples", []))
        for index, item in enumerate(tuples):
            try:
                source_id = f"legacy_tuple_source_{index}"
                message = writer.make_message("legacy", "system", str(item.get("value_text", item.get("content", ""))), message_id=source_id, metadata={"legacy_import": True})
                timestamp = datetime.fromtimestamp(float(item.get("created_at", 0)), tz=timezone.utc)
                memory = Memory(
                    id=str(item.get("id") or new_id("mem")), namespace=namespace, session_id="legacy",
                    source_message_id=source_id, memory_type="episode", content=message.content,
                    embedding=[float(v) for v in item.get("key_emb", [])], created_at=timestamp,
                    updated_at=timestamp, token_count=message.token_count, trust_level="assistant_generated",
                    inclusion_count=int(item.get("hit_count", 0)), metadata={"legacy_key": item.get("key_text", ""), "legacy_import": True},
                )
                writer.write(message, [memory])
                report["memories"] = int(report["memories"]) + 1
            except Exception as exc:
                report["skipped"].append({"kind": "tuple", "index": index, "error": str(exc)})
    finally:
        store.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--namespace", default="legacy")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = migrate_legacy(args.source, args.destination, args.namespace)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
