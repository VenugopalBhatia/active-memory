# active-memory

`active-memory` is a local, inspectable context-management proxy for LLM agents. It persists prior conversation events, retrieves relevant memories using semantic similarity and temporal signals, expands related context through explicit memory relationships, and reconstructs each request under a fixed token budget.

The project is alpha software. It does not fine-tune models, guarantee recall, or treat every retrieved record as authoritative.

## How it works

```text
Anthropic client -> local proxy -> raw event journal -> memory ingestion
                                      |                    |
                                      v                    v
                                   SQLite <- exact cosine retrieval
                                                        |
                                         relevance + recency + frequency
                                                        |
                                              semantic seed selection
                                                        |
                                        explicit relationship expansion
                                                        |
                                                affinity reranking
                                                        |
                                       deterministic context budgeting
                                                        |
                                              Anthropic upstream API
```

SQLite stores immutable raw messages separately from derived memories, normalized embeddings, relationship edges, validity intervals, trust metadata, and completed context-assembly events. Retrieval is an exact matrix-vector cosine scan. This is simpler to test and explain than an approximate index and remains the default until local benchmarks justify another backend.

The four independent signals are:

- **Relevance:** normalized cosine similarity between the query and memory embedding.
- **Recency:** type-specific exponential decay from the memory's observation time.
- **Frequency:** the number of completed reconstructed contexts that actually included the memory.
- **Affinity:** an explicit edge from a candidate to a previously selected semantic seed, such as `same_file`, `resolves`, or `depends_on`. It is not embedding similarity under another name.

## Install

Python 3.11 or newer is required.

```bash
pip install -e '.[local,dev]'
```

The default local provider uses `sentence-transformers/all-MiniLM-L6-v2`. OpenAI embeddings are opt-in and stored memories are never sent to an external embedding provider unless configured.

```bash
active-memory proxy --database ~/.active-memory/memory.db --namespace my-project
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 claude
```

The proxy accepts Anthropic `POST /v1/messages`, forwards client authentication, preserves tool blocks, and returns upstream responses transparently. Send `x-active-memory-session` and `x-active-memory-namespace` headers when explicit isolation is needed.

## Inspect and delete

```bash
active-memory inspect --query "Why did authentication fail?" --namespace my-project
active-memory memories list --namespace my-project
active-memory memories show mem_123
active-memory delete --session session-123
active-memory delete --namespace my-project
active-memory purge
```

Inspection emits every score, relationship reason, token cost, and exclusion reason. Destructive commands require typing `yes` unless `--yes` is supplied.

## Configuration

The CLI reads `~/.active-memory/config.json` or an explicit `--config`. YAML is supported when PyYAML is installed.

```json
{
  "storage": {"backend": "sqlite", "path": "~/.active-memory/memory.db"},
  "embeddings": {"provider": "local", "model": "sentence-transformers/all-MiniLM-L6-v2", "batch_size": 32},
  "retrieval": {"candidate_limit": 80, "seed_limit": 6, "neighbor_limit": 30, "result_limit": 20, "minimum_relevance": 0.2},
  "scoring": {"relevance_weight": 0.55, "recency_weight": 0.2, "frequency_weight": 0.1, "affinity_weight": 0.15},
  "budget": {"model_context_limit": 200000, "reserved_response_tokens": 8000, "safety_margin_tokens": 2000, "recent_turn_fraction": 0.35, "memory_fraction": 0.35, "tool_context_fraction": 0.2, "recent_message_limit": 8},
  "memory": {"minimum_segment_tokens": 12, "maximum_segment_tokens": 500, "default_namespace": "global", "store_assistant_generated": true, "storage_enabled": true}
}
```

Weights are defaults selected by policy, not learned parameters.

## Evaluation

```bash
python -m active_memory.evaluation.end_to_end --json
python -m pytest -q
```

The offline harness compares recent-only, vector-only, three-signal, full four-signal, and required ablations across twelve conversation categories. It reports Recall@K, Precision@K, MRR, nDCG@K, stale/superseded retrieval, provenance accuracy, and exact-search latency. See [evaluation methodology](docs/EVALUATION.md) and [limitations](docs/LIMITATIONS.md). Results are evidence for the included fixture set, not a universal quality claim.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Scoring](docs/SCORING.md)
- [Evaluation](docs/EVALUATION.md)
- [Security and privacy](docs/SECURITY.md)
- [Limitations](docs/LIMITATIONS.md)
- [Migration guide](MIGRATION.md)

## License

MIT. See [LICENSE](LICENSE).

