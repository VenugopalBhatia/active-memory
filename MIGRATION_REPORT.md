# Active Memory Rebuild Report

Date: 2026-08-04

## Architectural changes

The runtime now journals immutable redacted messages and derived typed memories separately in SQLite. Normalized embeddings are stored as float32 BLOBs. Retrieval scans active namespace vectors with exact cosine similarity, scores relevance/recency/inclusion-frequency without side effects, selects semantic seeds, expands explicit one-hop edges, and reranks with relationship affinity. The context assembler pins system/tool/latest/recent content, deduplicates memory, packs dependency groups by utility per formatted token, emits provenance XML, and atomically records only completed inclusions.

The Anthropic `POST /v1/messages` proxy uses this pipeline, journals assistant responses, preserves request/tool fields, passes upstream errors through, and forwards the untouched original request when the memory subsystem fails in non-strict mode. Typed JSON/YAML configuration, inspection, physical deletion, structured traces, and a legacy JSON importer complete the operational surface.

Development was checkpointed as seven passing phase commits:

1. `d82f22b` SQLite memory foundation
2. `fb19cfc` exact cosine retrieval
3. `d4ef08f` independent retrieval scoring
4. `3a85a09` relationship affinity reranking
5. `aa01154` deterministic context budgeting
6. `2a77028` SQLite memory proxy integration
7. `fc9d4ba` evaluation and operations

## Archived and deleted components

Archived under `experiments/legacy_semantic_tree/`:

- the early custom semantic tree
- its mutable tuple/access model
- its original composite scorer

Deleted from the installed runtime:

- tree-based assembler, middleware, grounding, MCP server, and config paths
- pruning, subtree compression, and reset-briefing behavior
- legacy chat/tree persistence and release helpers
- old tree-specific evaluation, long-context, live A/B, and proxy benchmark modules
- old monolithic tests
- previous OpenAI/Gemini proxy adapters; the rebuilt compatibility target is Anthropic `/v1/messages`

The package and documentation contain no primary-runtime import or product claim for the archived structure.

## Migration guide

See `MIGRATION.md`. The supported legacy import command is:

```bash
python scripts/migrate_legacy_store.py OLD_STATE.json ~/.active-memory/memory.db \
  --namespace legacy --report migration-report.json
```

The source is read-only, original tuple IDs are retained where possible, and every rejected record appears in the report. Tree/pruning/reset configuration keys must be replaced by the typed storage, embedding, retrieval, scoring, budget, memory, and proxy sections.

## Verification

Environment: Python 3.13.7 on macOS arm64.

- Sandboxed suite: `33 passed, 1 skipped` (localhost binding unavailable).
- Full suite with localhost enabled: `34 passed in 1.02s`.
- Ruff: all checks passed across active package, scripts, and tests.
- mypy: no issues in 41 source files.
- `compileall`: passed.
- sdist and wheel: built successfully; `schema.sql` included and experiment archive excluded.

The full suite covers restart persistence, rollback, namespace isolation, supersession, dimension checks, read-only retrieval, every score, edge affinity, hard budgets, inclusion accounting, segmentation, redaction, assistant speculation, cross-session contradictions, memory fallback, response persistence, typed config, evaluation metrics, and a real localhost HTTP proxy/upstream-error roundtrip.

## Benchmark results

Command:

```bash
python -m active_memory.evaluation.end_to_end --json
```

Dataset: 12 hand-authored categories, deterministic non-semantic lexical test provider, Recall/Precision at 3. These results make CI reproducible; they do not estimate production embedding quality.

| Strategy | Recall@3 | Precision@3 | MRR | nDCG@3 | Provenance | Stale / superseded |
|---|---:|---:|---:|---:|---:|---:|
| recent only | 0.4167 | 0.1111 | 0.3500 | 0.3244 | 0.4167 | 0 / 0 |
| vector only | 0.9167 | 0.3056 | 0.7986 | 0.8750 | 0.9167 | 0 / 0 |
| three signal | 0.8750 | 0.2778 | 0.6736 | 0.7505 | 0.9167 | 0 / 0 |
| full four signal | 0.9167 | 0.3056 | 0.6875 | 0.7936 | 0.9167 | 0 / 0 |
| minus affinity | 0.8750 | 0.2778 | 0.6875 | 0.7614 | 0.9167 | 0 / 0 |
| minus frequency | 0.9167 | 0.3056 | 0.7292 | 0.8244 | 0.9167 | 0 / 0 |
| minus recency | 0.9167 | 0.3056 | 0.7569 | 0.8442 | 0.9167 | 0 / 0 |

Affinity improved full-system Recall@3 over the no-affinity and three-signal variants on this dataset. Frequency and recency did not improve Recall@3, and vector-only produced higher MRR/nDCG than the default full weighting. Therefore the repository does not claim every signal improves retrieval quality.

Context-selection metrics at top 3:

| Relevant-token ratio | Contradiction rate | Duplicate rate | Budget adherence | Coverage | Trust-weighted precision | Avg injected tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2825 | 0.0556 | 0.0000 | 1.0000 | 0.9167 | 0.3148 | 59.17 |

Exact-search timing, five trials per corpus:

| Active memories | Median | p95 | SQLite bytes | Bytes/memory |
|---:|---:|---:|---:|---:|
| 100 | 1.83 ms | 1.85 ms | 184,320 | 1,843.2 |
| 1,000 | 15.98 ms | 16.70 ms | 1,294,336 | 1,294.3 |
| 5,000 | 92.11 ms | 92.94 ms | 6,254,592 | 1,250.9 |

Embedding cost was `$0.00` because the benchmark is local. A live downstream-model evaluation was not run, so answer correctness, task completion, and hallucination-rate results are deliberately `null` rather than inferred from retrieval metrics.

## Remaining limitations

- The deterministic extractor misses implicit facts, relations, and conflicts.
- Pattern redaction cannot guarantee removal of every secret format.
- Approximate token counting can differ from provider tokenizers.
- Exact search reaches roughly 92 ms median at 5,000 records on the measured machine and loads the active namespace into memory; no universal ANN threshold is claimed.
- The default weights are policy choices and need tuning against representative production data.
- Vector-only ranking beat full scoring on MRR/nDCG in the included fixture set.
- The current transparent proxy target is Anthropic `/v1/messages`, not provider-managed hidden state.
- Persistent local storage still creates retention, backup, access-control, and deletion obligations.
- No live LLM answer-quality benchmark or fine-tuning is included.

## Supported resume wording

> Built a local semantic memory proxy for LLM agents that persistently stores prior interactions and reconstructs token-budgeted prompts using embedding relevance, recency, inclusion frequency, and relationship-based affinity—without model fine-tuning.
