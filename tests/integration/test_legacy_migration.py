from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from active_memory.models import MemoryFilters
from active_memory.storage.sqlite_store import SQLiteMemoryStore


def test_legacy_json_migration_preserves_source_ids_and_reports(tmp_path) -> None:
    source = tmp_path / "legacy.json"
    destination = tmp_path / "memory.db"
    report_path = tmp_path / "report.json"
    payload = {
        "conversation": [{"role": "user", "content": "Legacy raw event"}],
        "tuples": [{"id": "legacy-memory-id", "key_text": "database", "value_text": "Use PostgreSQL.", "key_emb": [1.0, 0.0], "created_at": 1_700_000_000, "hit_count": 3}],
    }
    original = json.dumps(payload, sort_keys=True)
    source.write_text(original, encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "scripts/migrate_legacy_store.py", str(source), str(destination), "--namespace", "legacy", "--report", str(report_path)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["messages"] == 1
    assert report["memories"] == 1
    assert report["skipped"] == []
    assert json.loads(report_path.read_text()) == report
    assert source.read_text(encoding="utf-8") == original
    store = SQLiteMemoryStore(destination)
    imported = store.get_active_memories("legacy", MemoryFilters(statuses=frozenset({"active"})))
    assert imported[0].id == "legacy-memory-id"
    assert imported[0].inclusion_count == 3
    store.close()
