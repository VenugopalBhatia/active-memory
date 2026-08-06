# Architecture

## Data flow

```text
request -> redact and journal raw events -> segment/classify/deduplicate
        -> normalize embeddings -> SQLite memories and explicit edges
        -> exact cosine candidates -> relevance/recency/frequency score
        -> select seeds -> expand one-hop neighbors -> affinity rerank
        -> pin system/tools/latest/recent -> pack memory -> upstream
        -> journal assistant response
```

## Boundaries

- `ingestion`: redaction, boundary-aware segmentation, source-aware classification, conservative duplicate/conflict handling, and deterministic edges.
- `storage`: the `MemoryStore` protocol and SQLite implementation. Database methods never call embedding providers.
- `retrieval`: provider abstraction, normalized exact cosine search, pure scoring functions, seed selection, and relationship affinity.
- `context`: provider-content normalization, hard token budgeting, provenance XML, and atomic inclusion accounting.
- `proxy`: Anthropic request adaptation, transparent forwarding, response journaling, and non-strict fallback.
- `evaluation`: fixed datasets, baselines, ablations, retrieval/context metrics, and latency measurements.

## Storage schema

`messages` is the immutable event journal. `memories` contains derived records and BLOB-encoded normalized float32 embeddings. `memory_edges` records typed weighted relationships. `context_assembly_events` records successful packing and is transactionally coupled to inclusion counters. `schema_migrations` protects version compatibility.

Foreign keys prevent derived memories without provenance. Explicit privacy deletion is the only normal path that physically removes journal records.

## Retrieval stages

1. Load active, temporally valid namespace records with optional metadata filters.
2. Compute exact cosine similarity by matrix-vector multiplication.
3. Score relevance, recency, and inclusion frequency; select semantic seeds.
4. Load one-hop edges and active neighbors.
5. Compute affinity as the maximum edge weight multiplied by a linked seed's first-pass score.
6. Rerank with all four signals and attach human-readable reasons.

## Context assembly

The assembler subtracts response reservation and safety margin first. System instructions, tools, latest user content, and recent messages are independent pinned inputs. Memories already represented by recent turns are removed. Dependency groups pack atomically, ordered by score divided by the square root of formatted token cost. The final prompt is rejected if accounting exceeds the input allowance.

## Failure handling

Embedding failure stores the redacted raw event with `embedding_failed` status. In non-strict mode, any memory pipeline failure forwards the untouched original request. Upstream HTTP errors and bodies pass through. SQLite transactions roll back partial derived-memory or context-assembly writes.

