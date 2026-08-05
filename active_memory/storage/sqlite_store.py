"""SQLite implementation of the memory store."""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

from active_memory.models import Memory, MemoryEdge, MemoryFilters, Message
from active_memory.storage.migrations import migrate


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _embedding_blob(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values) if values else b""


def _embedding_values(blob: bytes, dimension: int) -> list[float]:
    if len(blob) != dimension * 4:
        raise ValueError("stored embedding length does not match its dimension")
    return list(struct.unpack(f"<{dimension}f", blob)) if dimension else []


class SQLiteMemoryStore:
    """Thread-safe local persistence with explicit transaction boundaries."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._lock = threading.RLock()
        migrate(self._connection)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def add_message(self, message: Message) -> None:
        with self.transaction() as connection:
            self._insert_message(connection, message)

    def _insert_message(self, connection: sqlite3.Connection, message: Message) -> None:
        connection.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (message.id, message.session_id, message.role, message.content, _dt(message.created_at),
             message.token_count, message.content_hash, json.dumps(message.metadata, sort_keys=True)),
        )

    def add_memory(self, memory: Memory) -> None:
        with self.transaction() as connection:
            self._insert_memory(connection, memory)

    def _insert_memory(self, connection: sqlite3.Connection, memory: Memory) -> None:
        connection.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (memory.id, memory.namespace, memory.session_id, memory.source_message_id,
             memory.memory_type, memory.content, _embedding_blob(memory.embedding), len(memory.embedding),
             _dt(memory.created_at), _dt(memory.updated_at), memory.token_count, memory.trust_level,
             memory.status, _dt(memory.valid_from), _dt(memory.valid_until), memory.superseded_by,
             memory.inclusion_count, _dt(memory.last_included_at), json.dumps(memory.metadata, sort_keys=True)),
        )

    def add_message_and_memories(self, message: Message, memories: Sequence[Memory], edges: Sequence[MemoryEdge] = ()) -> None:
        with self.transaction() as connection:
            self._insert_message(connection, message)
            for memory in memories:
                self._insert_memory(connection, memory)
            self._insert_edges(connection, edges)

    def add_edges(self, edges: Sequence[MemoryEdge]) -> None:
        with self.transaction() as connection:
            self._insert_edges(connection, edges)

    @staticmethod
    def _insert_edges(connection: sqlite3.Connection, edges: Sequence[MemoryEdge]) -> None:
        connection.executemany(
            "INSERT OR REPLACE INTO memory_edges VALUES (?, ?, ?, ?, ?, ?)",
            [(edge.source_id, edge.target_id, edge.edge_type, edge.weight, _dt(edge.created_at),
              json.dumps(edge.metadata, sort_keys=True)) for edge in edges],
        )

    def get_recent_messages(self, session_id: str, limit: int) -> list[Message]:
        if limit <= 0:
            return []
        rows = self._connection.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [self._message(row) for row in reversed(rows)]

    def get_message(self, message_id: str) -> Message | None:
        row = self._connection.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        return self._message(row) if row else None

    def get_active_memories(self, namespace: str, filters: MemoryFilters | None = None) -> list[Memory]:
        filters = filters or MemoryFilters()
        clauses = ["m.namespace = ?"]
        params: list[object] = [namespace]
        if filters.session_id is not None:
            clauses.append("m.session_id = ?")
            params.append(filters.session_id)
        for column, values in (("memory_type", filters.memory_types), ("trust_level", filters.trust_levels), ("status", filters.statuses)):
            if values:
                placeholders = ",".join("?" for _ in values)
                clauses.append(f"m.{column} IN ({placeholders})")
                params.extend(sorted(values))
        if filters.valid_at:
            timestamp = _dt(filters.valid_at)
            clauses.extend(["(m.valid_from IS NULL OR m.valid_from <= ?)", "(m.valid_until IS NULL OR m.valid_until > ?)"])
            params.extend([timestamp, timestamp])
        rows = self._connection.execute(
            f"SELECT m.* FROM memories m JOIN messages msg ON msg.id=m.source_message_id WHERE {' AND '.join(clauses)} ORDER BY m.created_at, m.id",
            params,
        ).fetchall()
        memories = [self._memory(row) for row in rows]
        return [memory for memory in memories if self._metadata_matches(memory, filters)]

    @staticmethod
    def _metadata_matches(memory: Memory, filters: MemoryFilters) -> bool:
        if filters.source_role and memory.metadata.get("source_role") != filters.source_role:
            return False
        if filters.file_path and memory.metadata.get("file_path") != filters.file_path:
            return False
        if filters.entity and filters.entity not in memory.metadata.get("entities", []):
            return False
        return True

    def get_memories_by_ids(self, memory_ids: Sequence[str]) -> list[Memory]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._connection.execute(f"SELECT * FROM memories WHERE id IN ({placeholders})", list(memory_ids)).fetchall()
        by_id = {row["id"]: self._memory(row) for row in rows}
        return [by_id[memory_id] for memory_id in memory_ids if memory_id in by_id]

    def get_neighbors(self, memory_ids: Sequence[str], edge_types: set[str] | None = None) -> list[MemoryEdge]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        params: list[object] = [*memory_ids, *memory_ids]
        query = f"SELECT * FROM memory_edges WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders}))"
        if edge_types:
            edge_placeholders = ",".join("?" for _ in edge_types)
            query += f" AND edge_type IN ({edge_placeholders})"
            params.extend(sorted(edge_types))
        query += " ORDER BY source_id, target_id, edge_type"
        return [self._edge(row) for row in self._connection.execute(query, params).fetchall()]

    def mark_included(self, memory_ids: Sequence[str], included_at: datetime) -> None:
        if not memory_ids:
            return
        with self.transaction() as connection:
            placeholders = ",".join("?" for _ in memory_ids)
            connection.execute(
                f"UPDATE memories SET inclusion_count=inclusion_count+1, last_included_at=?, updated_at=? WHERE id IN ({placeholders})",
                [_dt(included_at), _dt(included_at), *memory_ids],
            )

    def supersede_memory(self, old_memory_id: str, new_memory_id: str, superseded_at: datetime) -> None:
        edge = MemoryEdge(new_memory_id, old_memory_id, "supersedes", 1.0, superseded_at)
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE memories SET status='superseded', valid_until=?, superseded_by=?, updated_at=? WHERE id=? AND status='active'",
                (_dt(superseded_at), new_memory_id, _dt(superseded_at), old_memory_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"active memory not found: {old_memory_id}")
            self._insert_edges(connection, [edge])

    def record_context_assembly(self, session_id: str, namespace: str, memory_ids: Sequence[str], input_tokens: int, created_at: datetime, metadata: dict | None = None) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO context_assembly_events(session_id, namespace, included_memory_ids_json, input_tokens, created_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, namespace, json.dumps(list(memory_ids)), input_tokens, _dt(created_at), json.dumps(metadata or {}, sort_keys=True)),
            )

    def complete_context_assembly(self, session_id: str, namespace: str, memory_ids: Sequence[str], input_tokens: int, created_at: datetime, metadata: dict | None = None) -> None:
        """Atomically update inclusion frequency and record the completed assembly."""
        with self.transaction() as connection:
            if memory_ids:
                placeholders = ",".join("?" for _ in memory_ids)
                connection.execute(
                    f"UPDATE memories SET inclusion_count=inclusion_count+1, last_included_at=?, updated_at=? WHERE id IN ({placeholders})",
                    [_dt(created_at), _dt(created_at), *memory_ids],
                )
            connection.execute(
                "INSERT INTO context_assembly_events(session_id, namespace, included_memory_ids_json, input_tokens, created_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, namespace, json.dumps(list(memory_ids)), input_tokens, _dt(created_at), json.dumps(metadata or {}, sort_keys=True)),
            )

    def delete_memory(self, memory_id: str) -> int:
        with self.transaction() as connection:
            return connection.execute("UPDATE memories SET status='deleted' WHERE id=?", (memory_id,)).rowcount

    def delete_session(self, session_id: str) -> int:
        with self.transaction() as connection:
            count = connection.execute("UPDATE memories SET status='deleted' WHERE session_id=?", (session_id,)).rowcount
            connection.execute("DELETE FROM messages WHERE session_id=? AND NOT EXISTS (SELECT 1 FROM memories WHERE source_message_id=messages.id)", (session_id,))
            return count

    def delete_namespace(self, namespace: str) -> int:
        with self.transaction() as connection:
            return connection.execute("UPDATE memories SET status='deleted' WHERE namespace=?", (namespace,)).rowcount

    @staticmethod
    def _message(row: sqlite3.Row) -> Message:
        return Message(row["id"], row["session_id"], row["role"], row["content"], _parse_dt(row["created_at"]), row["token_count"], row["content_hash"], json.loads(row["metadata_json"]))  # type: ignore[arg-type]

    @staticmethod
    def _memory(row: sqlite3.Row) -> Memory:
        return Memory(row["id"], row["namespace"], row["session_id"], row["source_message_id"], row["memory_type"], row["content"], _embedding_values(row["embedding"], row["embedding_dim"]), _parse_dt(row["created_at"]), _parse_dt(row["updated_at"]), row["token_count"], row["trust_level"], row["status"], _parse_dt(row["valid_from"]), _parse_dt(row["valid_until"]), row["superseded_by"], row["inclusion_count"], _parse_dt(row["last_included_at"]), json.loads(row["metadata_json"]))  # type: ignore[arg-type]

    @staticmethod
    def _edge(row: sqlite3.Row) -> MemoryEdge:
        return MemoryEdge(row["source_id"], row["target_id"], row["edge_type"], row["weight"], _parse_dt(row["created_at"]), json.loads(row["metadata_json"]))  # type: ignore[arg-type]
