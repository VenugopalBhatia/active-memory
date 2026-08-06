"""SQLite schema migration runner."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from importlib.resources import files

SCHEMA_VERSION = 1


def migrate(connection: sqlite3.Connection) -> None:
    schema = files("active_memory.storage").joinpath("schema.sql").read_text(encoding="utf-8")
    connection.executescript(schema)
    current = connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()[0]
    if current > SCHEMA_VERSION:
        raise RuntimeError(f"database schema {current} is newer than supported version {SCHEMA_VERSION}")
    if current < SCHEMA_VERSION:
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, datetime.now(UTC).isoformat()),
        )
        connection.commit()

