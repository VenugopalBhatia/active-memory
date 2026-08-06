PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user','assistant','system','tool')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    token_count INTEGER NOT NULL CHECK (token_count >= 0),
    content_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session_time ON messages(session_id, created_at, id);
CREATE INDEX IF NOT EXISTS idx_messages_hash ON messages(content_hash);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    session_id TEXT,
    source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE RESTRICT,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL CHECK (embedding_dim >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    token_count INTEGER NOT NULL CHECK (token_count >= 0),
    trust_level TEXT NOT NULL,
    status TEXT NOT NULL,
    valid_from TEXT,
    valid_until TEXT,
    superseded_by TEXT REFERENCES memories(id) ON DELETE SET NULL,
    inclusion_count INTEGER NOT NULL DEFAULT 0 CHECK (inclusion_count >= 0),
    last_included_at TEXT,
    metadata_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_namespace_status ON memories(namespace, status);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);

CREATE TABLE IF NOT EXISTS memory_edges (
    source_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,
    weight REAL NOT NULL CHECK (weight >= 0 AND weight <= 1),
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, edge_type)
);
CREATE INDEX IF NOT EXISTS idx_edges_target ON memory_edges(target_id);

CREATE TABLE IF NOT EXISTS retrieval_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS context_assembly_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    included_memory_ids_json TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

