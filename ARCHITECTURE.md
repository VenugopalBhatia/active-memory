# active-memory: Technical Documentation

## Table of Contents

1. [Motivation and Core Thesis](#1-motivation-and-core-thesis)
2. [System Architecture](#2-system-architecture)
3. [Module Reference](#3-module-reference)
   - 3.1 [types.py — Primitives](#31-typespy--primitives)
   - 3.2 [scoring.py — Composite Scorer](#32-scoringpy--composite-scorer)
   - 3.3 [btree.py — Semantic B-Tree](#33-btreepy--semantic-b-tree)
   - 3.4 [assembler.py — Context Assembler](#34-assemblerpy--context-assembler)
   - 3.5 [middleware.py — Anthropic API Middleware](#35-middlewarepy--anthropic-api-middleware)
   - 3.6 [proxy.py — HTTP Proxy Server](#36-proxypy--http-proxy-server)
   - 3.7 [cli.py — Interactive Chat CLI](#37-clipy--interactive-chat-cli)
   - 3.8 [grounding.py — Grounding Layer](#38-groundingpy--grounding-layer)
   - 3.9 [model_clients.py — Provider Abstraction](#39-model_clientspy--provider-abstraction)
   - 3.10 [code_ingest.py — AST-Aware Code Parsing](#310-code_ingestpy--ast-aware-code-parsing)
   - 3.11 [embeddings.py — Embedding Providers](#311-embeddingspy--embedding-providers)
   - 3.12 [config.py — Unified Configuration](#312-configpy--unified-configuration)
4. [Data Flow: Life of a Message](#4-data-flow-life-of-a-message)
5. [Key Algorithms](#5-key-algorithms)
   - 5.1 [Semantic Routing](#51-semantic-routing)
   - 5.2 [Node Splitting via 2-Means](#52-node-splitting-via-2-means)
   - 5.3 [Frequency-Aware Scoring](#53-frequency-aware-scoring)
   - 5.4 [Cold Subtree Compression](#54-cold-subtree-compression)
   - 5.5 [Greedy Knapsack Assembly](#55-greedy-knapsack-assembly)
6. [Design Decisions and Tradeoffs](#6-design-decisions-and-tradeoffs)
7. [Configuration Reference](#7-configuration-reference)
8. [Prior Art and Differentiation](#8-prior-art-and-differentiation)
9. [Limitations and Future Work](#9-limitations-and-future-work)

---

## 1. Motivation and Core Thesis

LLM context windows are a scarce resource. As conversations grow, models suffer from **context rot** — early information gets diluted, attention becomes unfocused, and response quality degrades. The standard industry response is to make windows bigger, but this is analogous to solving memory pressure by buying more RAM: it works until it doesn't, and it's expensive in both latency and cost (attention scales quadratically with sequence length).

active-memory proposes a different approach: **treat the context window as managed memory**. Instead of dumping the entire conversation history into the prompt, we maintain a semantic index of everything the conversation has covered and, at each turn, assemble a token-budgeted prompt containing only the most relevant and most frequently accessed information.

The core data structure is a **semantic B-tree** where each node holds a cluster of **(key, value) tuples** — an abstraction borrowed from the transformer's own KV cache. The key is a semantic identifier (what the information is *about*); the value is the content itself. Each tuple carries access metadata (hit count, recency, creation time), enabling frequency-aware eviction. When subtrees go cold, they are compressed into summary tuples stored in their parent node — the tree's depth becomes an implicit compression tier.

This is essentially **virtual memory for LLM context**: keep the working set hot, page out the cold stuff, and compress what you can.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  ActiveMemoryMiddleware                    │
│                                                          │
│  User message ──→ _ingest_message() ──→ SemanticBTree    │
│                          │                    │          │
│                          ▼                    ▼          │
│                   embedder.embed()     _find_leaf()      │
│                          │              route by         │
│                          ▼            cosine sim         │
│                   KVTuple created          │             │
│                   inserted into leaf       │             │
│                          │                 │             │
│                          ▼                 ▼             │
│                 ContextAssembler.assemble()              │
│                   │                                      │
│                   ├── 1. Pin recent N turns              │
│                   ├── 2. Query tree (score all tuples)   │
│                   ├── 3. Greedy knapsack fill budget     │
│                   └── 4. to_messages() → Anthropic API   │
│                                              │           │
│                                              ▼           │
│                                     API response         │
│                                              │           │
│                          ┌───────────────────┤           │
│                          ▼                   ▼           │
│                   _ingest_message()   Periodic maint.    │
│                   (auto-ingest        ├── prune()        │
│                    response)          └── compress()     │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

The system has a core dependency chain with additional modules for different interfaces and capabilities:

```
types.py  ←──  scoring.py  ←──  btree.py  ←──  assembler.py  ←──  middleware.py
(primitives)   (scorer)        (index)         (prompt builder)    (API wrapper)
                                                      ↑                  ↑
                                               grounding.py         proxy.py (HTTP proxy)
                                                                    cli.py   (chat CLI)

Supporting modules:
  embeddings.py    — embedding provider factory (OpenAI, hash)
  model_clients.py — LLM provider abstraction (Anthropic, OpenAI, Ollama)
  code_ingest.py   — AST-aware code parsing with call graph extraction
  config.py        — unified JSON config loader
```

Each core layer depends only on the layers to its left. You can use the B-tree and scorer without the assembler, or the assembler without the middleware, depending on how much control you want. The proxy and CLI are independent consumers of the middleware — the proxy intercepts HTTP requests, while the CLI provides a standalone interactive chat.

---

## 3. Module Reference

### 3.1 `types.py` — Primitives

This module defines the atomic building blocks that every other module depends on.

#### `Embedding`

```python
Embedding = NDArray[np.float32]  # 1-D float32 vector
```

A type alias for a one-dimensional numpy float32 array. All embedding operations throughout the system operate on this type.

#### `Embedder` (Protocol)

```python
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...
    def embed(self, texts: list[str]) -> list[Embedding]: ...
```

A structural typing protocol (PEP 544) that any embedding provider must satisfy. The system never imports a concrete embedding implementation — it depends only on this interface. This means you can swap between OpenAI embeddings, Voyage, a local sentence-transformer, or the included `HashEmbedder` without changing any other code.

The two requirements are:
- `dim`: returns the dimensionality of the embedding vectors (e.g. 1536 for OpenAI `text-embedding-3-small`).
- `embed(texts)`: takes a batch of strings, returns a list of embedding vectors of the same length.

#### `HashEmbedder`

```python
class HashEmbedder:
    def __init__(self, dim: int = 64) -> None
```

A deterministic pseudo-embedder intended **only for testing and development**. It hashes the input text to seed a numpy RandomState, then draws a random normal vector and L2-normalises it. This means the same string always produces the same embedding, but the embeddings have no semantic meaning — "database" and "cooking" are equally likely to be similar or dissimilar.

The implementation:
1. `hash(text) % 2**31` produces a deterministic seed from any string.
2. `np.random.RandomState(seed).randn(dim)` generates a reproducible random vector.
3. The vector is L2-normalised so cosine similarity is well-defined.

This is useful because it lets you run the full insertion, splitting, querying, pruning, and compression pipeline without needing an API key or a GPU. Swap in a real embedder when you're ready to test semantic behaviour.

#### `KVTuple`

```python
@dataclass
class KVTuple:
    key_text: str                     # semantic label
    value_text: str                   # actual content
    key_emb: Embedding | None         # embedding of key_text
    id: str                           # 12-char hex UUID
    created_at: float                 # time.time() at creation
    last_accessed: float              # updated on every touch()
    hit_count: int                    # incremented on every touch()
    token_cost: int                   # approximate token count of value_text
```

The **atomic unit of memory** in the system. It mirrors how attention works in a transformer — the key encodes *what to attend to*, and the value encodes *what to retrieve*. But unlike the internal KV cache (which is ephemeral and opaque), a KVTuple is persistent, inspectable, and carries access metadata.

**Why tuples instead of raw text chunks?** Raw text is unstructured and can only be matched by substring or embedding similarity over the whole blob. A KVTuple separates the *topic* (key) from the *content* (value), enabling the system to route, score, and evict at the semantic level. You can have two tuples with similar keys but different values (e.g. two facts about databases), and the system can keep the more frequently accessed one while evicting the other.

The `touch()` method updates `last_accessed` to the current time and increments `hit_count`. This is called by the B-tree's `query()` method on every tuple that gets returned in a result set — so tuples that the conversation keeps referencing naturally stay hot.

#### Utility Functions

- **`estimate_tokens(text) -> int`**: Rough token count using the ~4 characters per token heuristic. Returns `max(1, len(text) // 4)`. This is intentionally simple — for production use, you'd swap in `tiktoken` or the Anthropic tokeniser.

- **`cosine_sim(a, b) -> float`**: Standard cosine similarity between two embedding vectors. Returns 0.0 if either vector has near-zero norm (to avoid division by zero).

---

### 3.2 `scoring.py` — Composite Scorer

The scorer produces a single scalar value for each KVTuple by blending four signals. This score drives two critical decisions: what goes into the next prompt (context assembly) and what gets evicted (pruning).

#### `ScoringConfig`

```python
@dataclass
class ScoringConfig:
    recency_weight: float = 0.20      # how much recent access matters
    frequency_weight: float = 0.20    # how much repeated access matters
    relevance_weight: float = 0.45    # how much similarity to current query matters
    affinity_weight: float = 0.15     # structural relationship bonus (call graph neighbors)
    recency_half_life: float = 1800.0 # seconds (30 min) until recency drops to 0.5
    floor: float = 0.05              # minimum score (prevents total eviction)
```

The four weights must conceptually sum to 1.0 for the score to be interpretable, though this is not enforced. The `floor` ensures that even a very cold, never-accessed tuple can still be surfaced if the current query is highly relevant to it — this prevents information loss in edge cases where an old topic suddenly becomes relevant again.

#### `Scorer`

The scoring formula:

```
score(t) = w_recency × recency(t) + w_frequency × frequency(t) + w_relevance × relevance(t, q) + w_affinity × affinity(t)
```

Each component:

**Recency** uses exponential decay from the last access time:

```
recency(t) = e^(-λ × age)
where λ = ln(2) / half_life
      age = now - t.last_accessed
```

The decay constant `λ` is precomputed from the configured half-life. With the default half-life of 1800 seconds (30 minutes), a tuple that hasn't been accessed in 30 minutes scores 0.5 on recency; at 60 minutes it scores 0.25; at 90 minutes, 0.125. This creates smooth, continuous decay rather than a hard cutoff.

**Frequency** uses log-scaled hit count, normalised against a reference maximum:

```
frequency(t) = min(log(1 + hits) / log(1 + 100), 1.0)
```

The `log1p` scaling prevents high-frequency tuples from completely dominating. A tuple accessed 10 times scores about 0.5 on frequency; one accessed 100 times scores 1.0. The normalisation against `log(101)` means the frequency component saturates at ~100 hits, which is reasonable for a typical conversation session.

**Relevance** is cosine similarity between the tuple's key embedding and the current query embedding:

```
relevance(t, q) = max(0.0, cosine_sim(t.key_emb, q))
```

Negative similarities are clamped to 0. When no query embedding is available (e.g. during background pruning without a current query), relevance defaults to 0.5 (neutral), so only recency and frequency drive the score.

**Affinity** boosts tuples that are structurally related to the current context via call graph edges (e.g. a function that calls or is called by another already-selected function). This signal is computed from structural references stored on each KVTuple and rewards tuples that are topologically close to the working set, even if their embedding similarity is modest.

The final score is clamped to the `floor` value to prevent any tuple from scoring exactly zero.

---

### 3.3 `btree.py` — Semantic B-Tree

This is the heart of the system. It adapts the classical B-tree data structure for semantic memory management.

#### `BTreeNode`

```python
@dataclass
class BTreeNode:
    id: str                              # 10-char hex UUID
    tuples: list[KVTuple]                # payload (leaf nodes only)
    children: list[BTreeNode]            # child pointers (internal nodes)
    parent: BTreeNode | None             # back-pointer
    centroid: Embedding | None           # mean of tuple/child embeddings
    summary: KVTuple | None              # compressed representation
```

A node can be in one of three states:

1. **Leaf node**: `children` is empty, `tuples` contains the KV tuples. This is where data lives.
2. **Internal node**: `tuples` is empty, `children` contains child nodes. The `centroid` is the normalised mean of all children's centroids and is used for routing during insertion and query.
3. **Compressed node**: `tuples` and `children` are both empty, but `summary` contains a single KVTuple that represents the compressed content of what used to be in this subtree.

The `recompute_centroid()` method recalculates the centroid from the current contents. For leaf nodes, it averages tuple key embeddings. For internal nodes, it averages child centroids. The result is L2-normalised so cosine similarity routing works correctly.

The `total_tokens` property returns the token cost of including this node in context — either the summary's cost (if compressed) or the sum of all tuple costs.

#### `BTreeConfig`

```python
@dataclass
class BTreeConfig:
    max_tuples: int = 16              # leaf overflow threshold → triggers split
    min_tuples: int = 4               # merge threshold (not yet implemented)
    min_children: int = 2             # minimum fanout (not yet enforced)
    compress_threshold: float = 0.08  # score below which nodes are "cold"
```

`max_tuples` is the most impactful parameter. Lower values create deeper trees with more splits (better semantic separation, higher overhead). Higher values keep more tuples in flat clusters (less overhead, coarser routing). The default of 16 is a reasonable starting point — it means a leaf can hold about 16 conversation segments before splitting.

`compress_threshold` determines when a subtree is cold enough to compress. This interacts directly with the scorer's output range. With default scoring weights and the 30-minute recency half-life, a tuple with zero hits and significant age will score near or below 0.08, so this threshold catches genuinely cold content without being too aggressive.

#### `SemanticBTree`

The tree provides four key operations:

**Insert** (`insert(key_text, value_text) -> KVTuple`):
1. Embed the key text via the embedder.
2. Create a new KVTuple with the embedding, estimated token cost, and fresh access metadata.
3. Route to the most similar leaf by recursively descending through internal nodes, choosing the child with the highest cosine similarity between the key embedding and the child's centroid.
4. Append the tuple to the leaf's tuple list and recompute the leaf's centroid.
5. If the leaf now exceeds `max_tuples`, trigger a split.

**Query** (`query(query_emb, top_k) -> list[tuple[float, KVTuple]]`):
1. Recursively traverse the entire tree, scoring every tuple (and every summary) against the query embedding using the Scorer.
2. Sort all results by score descending.
3. Return the top-k results.
4. Call `touch()` on every returned tuple, updating its access metadata.

This is a full scan, not a beam search. For the expected tree sizes in a conversation context (hundreds to low thousands of tuples), this is fast enough. For larger deployments, you'd want to add branch pruning — skip subtrees whose centroid has low similarity to the query.

**Prune** (`prune(query_emb) -> list[KVTuple]`):
1. Walk every leaf node.
2. Score each tuple. If its score falls below `compress_threshold`, remove it from the leaf and add it to the evicted list.
3. Recompute centroids for affected nodes.
4. Remove any children that are now empty (no tuples, no children, no summary).
5. Return the list of evicted tuples (useful for logging/debugging).

**Compress** (`compress_cold_subtrees(summariser, query_emb) -> int`):
1. Walk the tree bottom-up (recurse into children first, then evaluate the parent).
2. For each child of an internal node, collect all raw tuples in that child's subtree and compute their average score.
3. If the average score is below `compress_threshold`, compress: generate a summary text (via the `summariser` callback or the default concatenation), embed it, wrap it in a KVTuple, clear the child's tuples and children, and set the child's `summary` field.
4. Adjust the tree's `_size` counter (subtract the removed tuples, add 1 for the summary).
5. Return the count of compressed nodes.

The `_Summariser` type is `Callable[[list[KVTuple]], str]`. The default implementation just concatenates the key and truncated value of up to 10 tuples. In production, you'd pass a function that calls a cheap LLM (like Haiku) to produce a real summary.

#### Node Splitting

When a leaf exceeds `max_tuples`, the tree splits it into two children using 2-means clustering on the tuple embeddings:

1. Stack all tuple embeddings into a matrix.
2. Initialise two cluster centres: the first tuple's embedding, and the embedding farthest from it (by L2 distance).
3. Run up to 10 iterations of k-means assignment and centroid update.
4. Partition tuples into two groups based on their cluster labels.
5. If the split is degenerate (all tuples in one group), fall back to a simple midpoint split.
6. Create two new child nodes from the groups. The original node becomes internal: its tuples are cleared, and the two children are attached.

This is where the B-tree analogy meets semantic clustering. A traditional B-tree splits on key order; this tree splits on **semantic distance**. The result is that tuples about similar topics end up in the same subtree, which makes compression more meaningful — when a subtree goes cold, the summary is coherent because the tuples were semantically related.

---

### 3.4 `assembler.py` — Context Assembler

The assembler is the bridge between the B-tree (an index of everything the conversation has covered) and the actual prompt sent to the LLM. Its job is to produce a prompt that fits within a token budget while maximising information density.

#### `AssemblerConfig`

```python
@dataclass
class AssemblerConfig:
    total_budget: int = 100_000                # max tokens for the full prompt
    pinned_reserve: int = 8_000                # tokens reserved for system prompt + pinned content
    recency_window: int = 4                    # last N conversation turns always included
    managed_top_k: int = 200                   # max tuples to score from the tree
    budget_pressure: float = 0.85              # fraction of available budget to actually fill
    anchor_relevance_threshold: float = 0.60   # cosine sim threshold for ground-truth anchoring
    anchor_budget_fraction: float = 0.25       # budget reserved for high-relevance anchors
    dependency_pull: bool = True               # pull in structurally related tuples
    dependency_budget_fraction: float = 0.10   # budget reserved for dependency pulls
```

The budget model:

```
total_budget = pinned_reserve + pinned_tokens + managed_budget
managed_budget = (total_budget - pinned_reserve - pinned_tokens) × budget_pressure
```

`pinned_reserve` accounts for the system prompt and any other fixed content. `recency_window` controls how many of the most recent turns are always included verbatim (these are the "hot pages" that never get paged out). `budget_pressure` (default 0.85) leaves headroom so the assembled prompt doesn't push right against the limit. The remaining budget is filled using a multi-phase strategy: ground-truth anchoring first, then greedy knapsack fill, then dependency pulling.

#### Assembly Process

`assemble(conversation, query_emb) -> AssembledContext`:

1. **Pin recent turns**: Take the last `recency_window × 2` messages from the conversation (×2 because each "turn" is a user message + assistant response). Calculate their total token cost.

2. **Compute managed budget**: Subtract pinned tokens and the pinned reserve from the total budget, then apply `budget_pressure` (default 0.85) to leave headroom.

3. **Query the tree**: Call `tree.query(query_emb, top_k=managed_top_k)` to get scored tuples. This returns tuples sorted by descending score, and touches each one (updating access stats).

4. **Ground-truth anchoring**: Reserve a fraction of the budget (`anchor_budget_fraction`, default 25%) for high-relevance tuples. Scan scored tuples and force-include any whose cosine similarity to the query exceeds `anchor_relevance_threshold` (default 0.60). This ensures critically relevant facts are never crowded out by merely frequent ones.

5. **Greedy knapsack fill**: Iterate through remaining scored tuples in score order. For each tuple, if adding its `token_cost` would exceed the remaining managed budget, skip it. Otherwise, wrap it in a `ManagedBlock` and add it.

6. **Dependency pull**: If `dependency_pull` is enabled (default), reserve an additional fraction (`dependency_budget_fraction`, default 10%) and pull in tuples that are structurally related (via call graph edges) to already-selected tuples. This ensures that if a function is included, its callers/callees come along for coherence.

7. **Return an `AssembledContext`** containing the pinned messages, the managed blocks, total token usage, and diagnostic counters.

#### Message Formatting

`to_messages(system_prompt, assembled) -> list[dict]`:

Converts the assembled context into Anthropic Messages API format:

1. If there are managed blocks, they are serialised into an XML-tagged `<retrieved_context>` block and prepended to the conversation as a synthetic user/assistant exchange. This pattern (user message containing context → assistant acknowledgement → actual conversation) ensures the model sees the retrieved context without it appearing in the natural conversation flow.

2. The pinned conversation messages are appended after the context injection.

The resulting message list can be passed directly to `client.messages.create()`.

---

### 3.5 `middleware.py` — Anthropic API Middleware

The middleware ties everything together into a single `send()` call that handles ingestion, assembly, API calls, and periodic maintenance automatically.

#### `MiddlewareConfig`

```python
@dataclass
class MiddlewareConfig:
    system_prompt: str = "You are a helpful assistant."
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    prune_interval: int = 5         # prune every N turns
    compress_interval: int = 10     # compress every N turns
    auto_ingest_responses: bool = True
    grounding: GroundingConfig       # grounding layer settings
    scoring: ScoringConfig
    btree: BTreeConfig
    assembler: AssemblerConfig
```

`prune_interval` and `compress_interval` control how often maintenance runs. Pruning is cheap (just scoring and filtering), so it can run frequently. Compression is more expensive (requires summarisation and re-embedding), so it runs less often. In a long conversation, the pattern is: ingest and assemble on every turn, prune every 5 turns, compress every 10 turns.

`auto_ingest_responses` controls whether assistant responses are also broken into KV tuples and added to the tree. This is useful because the model's own responses often contain important information (decisions, explanations, code) that should be retrievable in future turns. Set to `False` if you only want to index user messages and explicitly ingested content.

#### `ActiveMemoryMiddleware`

The constructor wires up all the components:

```python
def __init__(self, client, embedder, config, summariser):
    self.scorer = Scorer(config.scoring)
    self.tree = SemanticBTree(embedder, scorer, config.btree)
    self.assembler = ContextAssembler(tree, config.assembler)
```

It also maintains `_conversation` (the full raw conversation history, used for pinning recent turns) and `_turn_count` (for triggering periodic maintenance).

**`send(user_message) -> MiddlewareResponse`** is the main entry point. Its pipeline:

1. **Append** the user message to the raw conversation history.
2. **Ingest** the message: split it into sentence-ish segments via `_segment()`, then insert each segment as a KVTuple into the tree. The key is `"user:{first 60 chars}"`, the value is the full segment.
3. **Embed** the user message to get a query embedding for relevance scoring.
4. **Assemble** context from the tree + recent turns using the assembler.
5. **Format** the assembled context into Anthropic messages format.
6. **Call** the Anthropic Messages API with the formatted messages.
7. **Extract** the assistant's text response.
8. **Ingest** the assistant response (if `auto_ingest_responses` is true).
9. **Prune** if the turn count is a multiple of `prune_interval`.
10. **Compress** if the turn count is a multiple of `compress_interval`.
11. **Return** a `MiddlewareResponse` containing the response text, the raw API response, and a `ContextStats` object with diagnostic information.

**`ingest(key, value) -> KVTuple`** allows manual injection of KV tuples. Use this for tool outputs, document chunks, or any external information that should be indexed.

**`ingest_document(text, chunk_size) -> list[KVTuple]`** is a convenience method that splits a long document into word-boundary chunks and ingests each one.

**Message segmentation** (`_segment`): Splits text on sentence boundaries (`.!?` followed by whitespace), then merges very short segments (under 80 characters) with the next segment. This avoids creating tiny, low-information KV tuples while keeping segments short enough to be meaningfully scored and evicted individually.

---

### 3.6 `proxy.py` — HTTP Proxy Server

The proxy is the primary interface for use with Claude Code. It implements an HTTP server that sits between the client and the Anthropic API, transparently intercepting `/v1/messages` requests to manage context.

Key components:

- **`ProxyConfig`**: Extends the core config with proxy-specific settings — `upstream` URL, `reset_threshold` (fraction of budget triggering a full context reset, default 0.75), `reset_briefing_budget` (max tokens for the briefing after a reset), `reset_recency_turns`, and `state_dir` for persistence.
- **`ContextManager`**: The per-session state machine that wraps the B-tree, assembler, and scorer. It handles the full lifecycle: passthrough for small conversations, active management once the conversation exceeds 50% of budget, and full context resets when raw tokens exceed the reset threshold. On reset, it generates a curated topic-grouped briefing from the B-tree and continues from a fresh conversation while keeping the tree intact.
- **`ProxyHandler`**: The HTTP request handler. Intercepts `POST /v1/messages`, runs the ingest → score → assemble → forward pipeline, and returns the upstream response unchanged. Also serves `/health` and `/stats` endpoints.

The proxy forwards authentication headers from the client, so Claude Code's session credentials work without needing a separate API key.

---

### 3.7 `cli.py` — Interactive Chat CLI

A standalone terminal chat interface (`active-memory-chat`) that provides the full context management experience without the proxy. Unlike the proxy (which requires Claude Code or another client), the CLI is self-contained — it handles user input, LLM calls, and context management in a single process.

**Authentication:** The CLI requires a standard API key for the chosen provider (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). It does not support OAuth or session-based auth. For local models, use `--provider ollama` (no key needed).

Features: session save/load/resume, file ingestion (`--ingest` or `/ingest`), provider switching (Anthropic, OpenAI, Ollama), and interactive commands for inspecting tree state (`/stats`, `/tree`, `/hot`, `/cold`, `/prune`, `/compress`, `/budget`).

---

### 3.8 `grounding.py` — Grounding Layer

The grounding layer adds factual verification to the context assembly pipeline. It operates in two phases:

1. **Provenance injection** (pre-generation): Tags assembled context blocks with confidence levels based on how well they are supported by the B-tree, so the model can weigh information appropriately.
2. **Post-verification** (post-generation): After the model responds, checks claims in the response against the B-tree to detect ungrounded statements and contradictions.

```python
@dataclass
class GroundingConfig:
    enabled: bool = True
    provenance_injection: bool = True
    post_verification: bool = True
    auto_correct: bool = False              # send correction prompts automatically
    grounding_threshold: float = 0.6        # similarity needed to count as grounded
    contradiction_threshold: float = 0.75   # similarity for same-topic contradiction detection
    min_grounding_rate: float = 0.3         # below this triggers a warning
```

---

### 3.9 `model_clients.py` — Provider Abstraction

Defines the `ModelClient` protocol and concrete implementations for Anthropic, OpenAI, and Ollama. The CLI and middleware use this abstraction so the rest of the system is provider-agnostic.

- **`AnthropicModelClient`**: Wraps the Anthropic SDK. Can also read Claude Code's OAuth credentials as a fallback (for the proxy use case).
- **`OpenAIModelClient`**: Wraps the OpenAI SDK. Supports custom `base_url` for OpenAI-compatible servers (Ollama, LM Studio, vLLM).
- **`create_model_client(provider, model, base_url)`**: Factory function that returns the appropriate client.

---

### 3.10 `code_ingest.py` — AST-Aware Code Parsing

Parses source files into structured `CodeChunk` objects using Python's `ast` module (for Python files) and brace-matching heuristics (for JS/TS/Go/Rust/Java/C). Each chunk captures the function or class name, its body, start/end lines, and references to other symbols (call graph edges).

These structural references are stored on KVTuples and used by the scorer's affinity signal and the assembler's dependency pull — ensuring that when the model is working with a function, its callers and callees are available in context.

---

### 3.11 `embeddings.py` — Embedding Providers

Factory module that provides `create_embedder(mode)` returning an embedder instance:

- **`"auto"`** (default): Uses OpenAI embeddings (`text-embedding-3-small`) when `OPENAI_API_KEY` is set and the `openai` package is installed. Falls back to `HashEmbedder` otherwise.
- **`"openai"`**: Forces OpenAI embeddings; errors if not configured.
- **`"hash"`**: Forces the deterministic `HashEmbedder` (no API key needed, non-semantic).

---

### 3.12 `config.py` — Unified Configuration

Handles JSON config file loading and merging for all three interfaces (proxy, CLI, MCP server). Supports `--config` flag pointing to a specific file, or auto-discovers `~/.active-memory/config.json`. Priority: **defaults < config file < CLI flags**.

---

## 4. Data Flow: Life of a Message

Here's the complete path of a user message through the system, turn by turn:

### Turn 1: "Let's use PostgreSQL for the database"

```
1. _segment() splits into: ["Let's use PostgreSQL for the database"]
2. embed("user:Let's use PostgreSQL for the database") → vector v1
3. _find_leaf(root, v1) → root (it's empty, so root IS the leaf)
4. Insert KVTuple(key="user:Let's use PostgreSQL...", value=<full text>, emb=v1)
5. root.tuples = [tuple1], root.centroid = v1
6. assemble():
   - pinned = [{"role": "user", "content": "Let's use PostgreSQL..."}]
   - tree has 1 tuple, query returns it
   - managed_blocks = [ManagedBlock(key="user:Let's use...", ...)]
7. to_messages() → context injection block + pinned turn
8. API call → response
9. Response ingested as more KV tuples
```

### Turn 15: Tree has split several times, 40+ tuples

```
1. User: "Remind me what database we're using"
2. embed("Remind me what database we're using") → query vector
3. query(query_vec, top_k=200):
   - traverse all nodes, score all tuples
   - the PostgreSQL tuple from turn 1 scores high (high relevance
     to "database", decent recency if recent, high frequency if
     referenced often)
   - low-scoring tuples from early tangential discussion rank low
4. assemble():
   - pinned = last 8 messages (4 turns)
   - managed_blocks = top tuples that fit in budget, PostgreSQL tuple
     likely included even though it's from turn 1
5. prune() runs (turn 15 is divisible by 5):
   - tuples with score < 0.08 are evicted
   - tangential topics never referenced again get dropped
6. API receives a focused context: recent turns + the most relevant
   historical tuples, not the entire 15-turn conversation
```

### Turn 30: Compression kicks in

```
1. compress_cold_subtrees() runs (turn 30 is divisible by 10):
   - walks the tree bottom-up
   - finds a subtree about "deployment options" discussed in turns 5-7
     but never referenced again
   - avg score of that subtree's tuples: 0.08 (below threshold)
   - summariser generates: "Discussed deployment: considered K8s and
     Docker Compose, decided on Docker Compose with nginx."
   - subtree's 5 tuples (250 tokens) replaced by 1 summary (30 tokens)
   - net savings: 220 tokens, essential decision preserved
2. Tree is now leaner — hot topics have full detail, cold topics
   are compressed but still queryable
```

---

## 5. Key Algorithms

### 5.1 Semantic Routing

When inserting a new tuple, the tree must decide which leaf node to place it in. The routing algorithm:

```
_find_leaf(node, embedding):
    if node is leaf:
        return node
    for each child in node.children:
        sim = cosine_sim(embedding, child.centroid)
    return _find_leaf(child with highest sim, embedding)
```

This is a greedy descent — at each internal node, we pick the child whose centroid is most similar to the insertion key's embedding. The result is that semantically similar tuples cluster together in the same leaf, which is exactly what we want for meaningful compression later.

The time complexity is O(B × d) per level where B is the branching factor and d is the embedding dimension. With typical tree depths of 3–5, this is negligible.

### 5.2 Node Splitting via 2-Means

When a leaf overflows, we split it using k-means with k=2:

```
_split(node):
    embeddings = stack all tuple embeddings into matrix

    # Initialise: pick two farthest points
    center_a = embeddings[0]
    center_b = embeddings[argmax(||embeddings - center_a||)]

    for 10 iterations:
        assign each embedding to nearest center
        update centers as group means

    partition tuples by assignment
    create two child nodes
    node becomes internal (tuples cleared, children set)
```

**Why farthest-point initialisation?** Random initialisation can produce degenerate splits where all points end up in one cluster. Starting with the two most distant points maximises the initial separation, making convergence faster and splits more balanced.

**Degenerate split fallback**: If 2-means still produces an empty group (possible when many tuples have identical embeddings), we fall back to a simple midpoint split: first half goes to child A, second half to child B.

### 5.3 Frequency-Aware Scoring

The scoring formula is intentionally simple and interpretable:

```
score = 0.20 × e^(-λ × age) + 0.20 × log(1+hits)/log(101) + 0.45 × max(0, cos(key, query)) + 0.15 × affinity
```

**Why these weights?** Relevance gets the highest weight (0.45) because the most important signal is whether a tuple matches what's being discussed right now. Recency and frequency are tied at 0.20, reflecting the equal importance of "this was recent" and "this keeps coming up." Affinity (0.15) provides a structural bonus for call graph neighbors, ensuring code context stays coherent.

**Why exponential decay?** Linear decay doesn't model real conversation dynamics — the difference between 1 minute ago and 2 minutes ago is much more significant than the difference between 60 minutes ago and 61 minutes ago. Exponential decay captures this naturally.

**Why log-scaled frequency?** Without log scaling, a tuple accessed 100 times would dominate one accessed 10 times by 10×. Log scaling compresses this to roughly 2×, preventing hot tuples from becoming permanently sticky.

### 5.4 Cold Subtree Compression

Compression is the mechanism by which the tree stays lean over long conversations. It works bottom-up:

```
_compress_node(node, summariser, query_emb):
    if node is leaf: return 0

    for each child:
        # Recurse first (compress deeper subtrees before shallower ones)
        _compress_node(child, ...)

        # Score the child's content
        all_tuples = collect all raw tuples in child subtree
        avg_score = mean(score(t) for t in all_tuples)

        if avg_score < compress_threshold:
            summary_text = summariser(all_tuples) or default_summarise()
            summary_emb = embed(summary_text)
            child.tuples = []
            child.children = []
            child.summary = KVTuple(key="summary:{id}", value=summary_text)
```

**Bottom-up ordering matters.** By recursing into children first, we ensure that if a deep subtree is cold, it gets compressed before its parent evaluates whether the parent should also compress. This prevents compressing a subtree that contains a mix of hot and cold children — the hot children stay expanded, and only the cold ones collapse.

**The summary becomes queryable.** After compression, the summary tuple participates in normal query scoring. If the compressed topic suddenly becomes relevant again (the user asks about it), the summary will score high on relevance and appear in the assembled context. This is lossy but not catastrophic — the summary preserves the key decisions and facts.

### 5.5 Greedy Knapsack Assembly

The assembler fills the token budget using a multi-phase approach:

```
assemble(conversation, query_emb):
    pinned = last N turns (always included)
    managed_budget = (total_budget - pinned_reserve - pinned_tokens) × budget_pressure

    scored_tuples = tree.query(query_emb, top_k=200)  # pre-sorted by score

    # Phase 1: Ground-truth anchoring
    anchor_budget = managed_budget × anchor_budget_fraction
    for (score, tuple) in scored_tuples:
        if cosine_sim(tuple.key_emb, query_emb) >= anchor_relevance_threshold:
            if tuple.token_cost <= anchor_budget:
                blocks.append(tuple)
                anchor_budget -= tuple.token_cost

    # Phase 2: Greedy knapsack fill
    for (score, tuple) in scored_tuples:
        if tuple not already included:
            if used + tuple.token_cost <= managed_budget:
                blocks.append(tuple)
                used += tuple.token_cost

    # Phase 3: Dependency pull
    dep_budget = managed_budget × dependency_budget_fraction
    for block in blocks:
        for ref in block.structural_refs:
            pull related tuples within dep_budget

    return AssembledContext(pinned, blocks, used, ...)
```

**Why greedy and not optimal knapsack?** The 0/1 knapsack problem is NP-hard, but the greedy approach (sort by value, take from the top) is a good approximation when items are small relative to the budget — which they are (individual tuples are typically 20–200 tokens against a 100k budget). The greedy approach is O(n log n) for the sort and O(n) for the packing, which is fast enough to run on every turn.

**Why `continue` instead of `break`?** A high-scoring tuple might be too large to fit in the remaining budget, but a lower-scoring, smaller tuple might still fit. By continuing past oversized items, we maximise budget utilisation.

**Why anchor first?** Without anchoring, a highly relevant but infrequently accessed tuple could be crowded out by frequently accessed but less relevant ones. The anchor phase guarantees that ground-truth facts (high cosine similarity to the current query) are always present, up to the anchor budget fraction.

---

## 6. Design Decisions and Tradeoffs

**Full scan vs. beam search in query.** The current `query()` implementation scores every tuple in the tree. For conversation-scale data (hundreds to low thousands of tuples), this is fine — scoring is just a few multiplications per tuple. For larger deployments (e.g. ingesting entire codebases), you'd want beam search: skip subtrees whose centroid similarity to the query falls below a threshold. The tree structure already supports this — `_collect_scored` can be modified to check centroid similarity before recursing.

**KV tuple keys from message truncation.** The middleware generates tuple keys by truncating the message to 60 characters: `f"{role}:{text[:60]}"`. This is crude — it means two messages starting with the same 60 characters get similar keys even if they diverge later. A better approach would be to use the embedder for the full text, or to extract topic labels using an LLM. The current approach is a deliberate simplicity choice for the MVP.

**Synchronous maintenance.** Pruning and compression run synchronously after the API response, adding latency to the turn. For production use, you'd want to run them asynchronously — kick off a background task after the response is sent. The tree is not thread-safe in the current implementation, so you'd need a lock or a copy-on-write strategy.

**Persistence is partial.** The proxy and CLI support saving/loading tree state to disk via `state_dir` and session save/load, but this is not yet seamless across all interfaces.

**No branch-and-merge.** Classical B-trees support merging underfull siblings. The current implementation doesn't — pruned nodes just have fewer tuples. Merging semantically similar siblings would be valuable but adds complexity around recomputing centroids.

---

## 7. Configuration Reference

| Module | Parameter | Type | Default | Description |
|--------|-----------|------|---------|-------------|
| **ScoringConfig** | `recency_weight` | float | 0.20 | Weight of the recency component |
| | `frequency_weight` | float | 0.20 | Weight of the frequency component |
| | `relevance_weight` | float | 0.45 | Weight of the relevance component |
| | `affinity_weight` | float | 0.15 | Weight of the structural affinity component |
| | `recency_half_life` | float | 1800.0 | Seconds (30 min) until recency factor = 0.5 |
| | `floor` | float | 0.05 | Minimum score for any tuple |
| **BTreeConfig** | `max_tuples` | int | 16 | Leaf splits when exceeded |
| | `min_tuples` | int | 4 | Merge threshold (not yet used) |
| | `min_children` | int | 2 | Min fanout (not yet enforced) |
| | `compress_threshold` | float | 0.08 | Avg score below which subtree compresses |
| **AssemblerConfig** | `total_budget` | int | 100,000 | Max tokens for assembled prompt |
| | `pinned_reserve` | int | 8,000 | Tokens reserved for system prompt |
| | `recency_window` | int | 4 | Recent turns always included (×2 for messages) |
| | `managed_top_k` | int | 200 | Max tuples scored during assembly |
| | `budget_pressure` | float | 0.85 | Fraction of available budget to actually fill |
| | `anchor_relevance_threshold` | float | 0.60 | Cosine sim threshold for ground-truth anchoring |
| | `anchor_budget_fraction` | float | 0.25 | Budget fraction reserved for anchors |
| | `dependency_pull` | bool | True | Pull in structurally related tuples |
| | `dependency_budget_fraction` | float | 0.10 | Budget fraction reserved for dependency pulls |
| **MiddlewareConfig** | `model` | str | `claude-sonnet-4-20250514` | Anthropic model ID |
| | `max_tokens` | int | 4,096 | Max response tokens |
| | `prune_interval` | int | 5 | Prune every N turns |
| | `compress_interval` | int | 10 | Compress every N turns |
| | `auto_ingest_responses` | bool | True | Index assistant messages |
| **GroundingConfig** | `enabled` | bool | True | Enable grounding layer |
| | `provenance_injection` | bool | True | Tag context with confidence levels |
| | `post_verification` | bool | True | Verify responses after generation |
| | `auto_correct` | bool | False | Automatically send correction prompts |
| | `grounding_threshold` | float | 0.6 | Similarity needed to count as grounded |
| | `contradiction_threshold` | float | 0.75 | Similarity for contradiction detection |
| | `min_grounding_rate` | float | 0.3 | Below this triggers a warning |

---

## 8. Prior Art and Differentiation

| System | Approach | Key Difference from active-memory |
|--------|----------|-----------------------------------|
| **MemGPT / Letta** | Two-tier memory (context + archive), LLM-managed paging | Flat storage, no hierarchical index. LLM decides what to page — expensive and unreliable. |
| **Focus** | Agent self-compresses history into Knowledge block | No structured index, no frequency tracking. Bulk compression vs. granular management. |
| **AgentRM** | OS-inspired MLFQ scheduler + 3-tier context lifecycle | Multi-agent scheduling focus, not memory organisation. No semantic indexing. |
| **CMV** | DAG-based session history with snapshot/branch/trim | Structural deduplication, not semantic indexing. Removes bloat but doesn't score by topic. |
| **SGLang RadixAttention** | Radix tree over KV cache token prefixes | Infrastructure-level serving optimisation, not application-level memory management. |
| **LangChain SummaryBuffer** | Recent turns verbatim + running summary of older turns | Single summary loses granularity. No selective retrieval or frequency tracking. |

**Positioning**: The distinguishing choice here is the combination of (1) a B-tree index with semantic routing via embedding centroids, (2) KV tuples as the atomic storage unit with per-tuple access stats, (3) frequency-aware scoring blending recency/frequency/relevance, and (4) hierarchical compression where cold subtrees collapse into parent summaries. The value of the system should be judged by retrieval quality, token efficiency, and operational simplicity rather than by claims of category-level novelty.

---

## 9. Limitations and Future Work

**Current limitations:**

- **Full scan on query.** Scores every tuple on every query. Won't scale past ~10k tuples without beam search or index pruning.
- **No async support.** The middleware and proxy are synchronous. Needs `async send()` for production frameworks.
- **No sibling merging.** Pruned underfull leaves aren't merged with neighbours.
- **Approximate token counting.** The 4-chars-per-token heuristic is rough. Integrating `tiktoken` would improve budget accuracy.
- **Streaming is basic.** The proxy forwards streamed responses but doesn't parse SSE events for incremental ingestion.

**What's already done:**

- Real embedding provider support (OpenAI via `text-embedding-3-small`)
- Provider-agnostic model clients (Anthropic and OpenAI)
- Cross-session persistence (proxy and CLI save/load tree state to disk)
- AST-aware code ingestion with call graph analysis
- Offline and live A/B benchmarking framework
- Grounding layer with provenance injection and contradiction detection
- Transparent proxy that works with Claude Code login (no API key required)

**Planned future work:**

- LLM-based summariser for compression (Haiku for cheap, high-quality summaries)
- Async API support
- Beam search pruning in `query()` for large trees
- Sibling merging when adjacent leaves fall below `min_tuples`
- Full SSE streaming support in the proxy
