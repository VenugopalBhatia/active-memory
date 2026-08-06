# Migration Guide

## Architectural changes

The primary runtime moved from JSON snapshots and a custom clustered tree to SQLite, exact cosine search, explicit relationship edges, two-pass scoring, and deterministic context packing. Raw events and derived memories now have separate schemas. Querying no longer mutates access frequency.

The prior experimental tree is archived under `experiments/legacy_semantic_tree/` and is not imported by the package. Tree pruning, subtree compression, reset briefings, the legacy middleware, and tree-specific CLI/MCP commands are removed. The supported transparent proxy route is Anthropic `POST /v1/messages`; the previous branch's OpenAI/Gemini proxy adapters are not part of this migration release.

## Import legacy JSON

Keep old JSON files unchanged and run:

```bash
python scripts/migrate_legacy_store.py ~/.active-memory/proxy/proxy_state.json ~/.active-memory/memory.db --namespace legacy --report migration-report.json
```

The importer preserves tuple IDs when present, records the original key in metadata, reports every skipped record, and never modifies the source. Imported pseudo/hash vectors remain labeled legacy data; re-ingest with a semantic provider before making semantic-quality claims.

## Configuration changes

Replace `state_dir`, tree, pruning, compression, anchoring, and reset keys with typed `storage`, `embeddings`, `retrieval`, `scoring`, `budget`, `memory`, and `proxy` sections shown in the README. Unknown keys fail validation instead of being ignored.

## CLI changes

Start the proxy with `active-memory proxy`. Use `active-memory inspect`, `active-memory memories`, `active-memory delete`, and `active-memory purge` for operations. Old chat session JSON and tree export commands are not runtime APIs.

