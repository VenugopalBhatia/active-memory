# active-memory

A transparent proxy that keeps long Claude Code sessions usable.

Sits between Claude Code and the Anthropic API. The proxy intercepts every request, indexes the conversation into a semantic B-tree, and rebuilds a token-budgeted prompt on each turn so earlier decisions stay retrievable without sending the full raw history every time.

```
Claude Code  -->  active-memory proxy (localhost:8080)  -->  api.anthropic.com
                  |
                  +-- Ingests new messages into B-tree
                  +-- Scores all facts by recency x frequency x relevance x affinity
                  +-- Assembles token-budgeted prompt (hot facts + recent turns)
                  +-- Prunes cold tuples, compresses cold subtrees
                  +-- Performs full context resets when conversation grows too large
```

## Install

```bash
pip install active-memory
```

Or from source:

```bash
git clone https://github.com/venugopalbhatia/active-memory.git
cd active-memory
pip install .

# With optional embedding/model providers
pip install '.[anthropic,openai]'

# With dev dependencies (tests)
pip install '.[dev]'
```

## Quick start

```bash
# Terminal 1: start the proxy
active-memory --verbose

# Terminal 2: use Claude Code as normal
ANTHROPIC_BASE_URL=http://localhost:8080 claude
```

That's it. No MCP config and no application-side code changes. The proxy starts managing context once the conversation exceeds 50% of the configured token budget.

### Authentication

The proxy forwards authentication headers from the client. If you're logged into Claude Code, your existing session credentials are forwarded automatically — no need to set `ANTHROPIC_API_KEY`. If you do have an API key set, it works as a fallback.

### Embedding providers

By default, `active-memory` uses OpenAI embeddings when `OPENAI_API_KEY` is set and the `openai` extra is installed. If not, it falls back to a deterministic hash embedder for development and testing.

```bash
# Use OpenAI embeddings (recommended for quality)
pip install 'active-memory[openai]'
export OPENAI_API_KEY=sk-...

# Or force the hash embedder (no API key needed, non-semantic)
active-memory --embedder hash
```

## How it works

### The problem

Long Claude Code sessions degrade because context windows fill up. Earlier decisions get crowded out by recent messages, and the model becomes less reliable about choices made earlier in the session.

### The solution

`active-memory` manages the context window as a constrained working set:

1. **Index everything** -- every message is segmented and inserted into a semantic B-tree as (key, value) tuples with access metadata
2. **Score by relevance** -- each tuple gets a composite score blending recency (exponential decay), frequency (log-scaled hit count), relevance (cosine similarity to current query), and structural affinity (call graph relationships)
3. **Assemble smartly** -- recent turns are pinned verbatim; remaining budget is filled greedily from the highest-scored tuples in the tree; ground-truth anchoring force-includes highly relevant facts; dependency pull brings in structurally related tuples
4. **Maintain continuously** -- cold tuples are pruned every 5 turns; cold subtrees are compressed into summaries every 10 turns
5. **Reset when needed** -- when raw tokens exceed 75% of budget, the proxy generates a curated topic-grouped briefing from the B-tree and continues from a fresh conversation while keeping the tree intact

### Proxy lifecycle

```
Turn 1-15:    Passthrough mode (small conversation, just indexing)
Turn 16-50:   Active management (assembling optimised context each turn)
Turn 50+:     Context reset triggered -- fresh conversation with B-tree briefing
Turn 51-100:  Active management continues with the same tree
Turn 100+:    Another reset if needed -- the tree keeps growing, conversations stay lean
```

## Proxy endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Intercepts and optimises Anthropic API calls |
| `/health` | GET | Health check + tree stats |
| `/stats` | GET | Detailed tree and session metrics |
| `/usage` | GET | Terminal-friendly usage graph with budget, activation, and reset markers |
| All other paths | POST | Passed through to upstream unchanged |

You can inspect usage live from a terminal:

```bash
curl http://127.0.0.1:8080/usage
```

Example output:

```text
active-memory usage
Budget:          44,000 tok
Activation:      22,000 tok
Reset:           30,800 tok

Raw history:     20,562 tok   46.7% used    23,438 left
Proxy sent:      20,562 tok   46.7% used    23,438 left
Current mode:    passthrough
Activation at:   -
Reset count:     0
Growth ( 5 turns): raw +59.2%  sent -45.2%

Trend:
   44,000 |          
   37,714 |          
   31,429 |          
   25,143 |          
   18,857 |   ◆◆H H  
   12,571 |      P P◆
    6,286 |          
        0 |          
          +----------
           PPPPPAAMMR
           ^    ^    
           Legend: H raw history  P proxy sent  ◆ overlap

Legend: P passthrough  A activation  M managed  R reset  S reset-suppressed
Recent turns:
  t121 ███████████░░░░░░░░░░░░░  19,604/44,000 (44.6%)  raw= 19,604  saved=      0  P
  t122 ███████████░░░░░░░░░░░░░  20,562/44,000 (46.7%)  raw= 20,562  saved=      0  P
  t123 █████████████░░░░░░░░░░░  23,418/44,000 (53.2%)  raw= 23,418  saved=      0  A
  t124 ████████░░░░░░░░░░░░░░░░  14,102/44,000 (32.0%)  raw= 24,887  saved= 10,785  M
  t125 ███████░░░░░░░░░░░░░░░░░  12,844/44,000 (29.2%)  raw= 31,204  saved= 18,360  R
```

## CLI options

```
active-memory [OPTIONS]

  --port, -p     Port to listen on (default: 8080)
  --host         Bind address (default: 127.0.0.1)
  --budget       Token budget for managed context (default: 100,000)
  --recency      Recent turns to always pin (default: 6)
  --embedder     auto | hash | openai (default: auto)
  --embed-model  OpenAI embedding model (default: text-embedding-3-small)
  --embed-dim    Hash embedding dimension (default: 64)
  --upstream     Upstream API URL (default: https://api.anthropic.com)
  --verbose, -v  Print context management diagnostics to stderr
  --state-dir    Directory for proxy state persistence
  --reset-threshold       Fraction of budget triggering context reset (default: 0.75)
  --reset-briefing-budget Max tokens for reset briefing (default: 8,000)
  --reset-recency-turns   Turns to keep verbatim after reset (default: 2)
  --config                Path to JSON config file
```

## Configuration file

All three interfaces (proxy, chat CLI, MCP server) support a `--config` flag pointing to a JSON file. If no `--config` is passed, `~/.active-memory/config.json` is loaded automatically when it exists.

Priority: **defaults < config file < CLI flags**.

```json
{
  "budget": 100000,
  "model": "claude-sonnet-4-20250514",
  "embedder": "auto",
  "embed_model": "text-embedding-3-small",
  "embed_dim": 64,

  "scoring": {
    "recency_weight": 0.20,
    "frequency_weight": 0.20,
    "relevance_weight": 0.45,
    "affinity_weight": 0.15,
    "recency_half_life": 1800.0,
    "floor": 0.05
  },

  "btree": {
    "max_tuples": 16,
    "min_tuples": 4,
    "min_children": 2,
    "compress_threshold": 0.08
  },

  "assembler": {
    "budget_pressure": 0.85,
    "anchor_relevance_threshold": 0.60,
    "anchor_budget_fraction": 0.25,
    "dependency_pull": true,
    "dependency_budget_fraction": 0.10,
    "managed_top_k": 200,
    "pinned_reserve": 8000,
    "recency_window": 4
  },

  "grounding": {
    "enabled": true,
    "provenance_injection": true,
    "post_verification": true,
    "auto_correct": false,
    "grounding_threshold": 0.6,
    "contradiction_threshold": 0.75,
    "min_grounding_rate": 0.3
  },

  "proxy": {
    "upstream": "https://api.anthropic.com",
    "verbose": false,
    "state_dir": null,
    "reset_threshold": 0.75,
    "reset_briefing_budget": 8000,
    "reset_recency_turns": 2
  }
}
```

All fields are optional — omit any you don't need. Top-level `budget` is a convenience alias for `assembler.total_budget`. The `proxy` section is only used by the proxy server.

```bash
# Explicit config file
active-memory --config my-config.json
active-memory-chat --config my-config.json
active-memory-mcp --config my-config.json

# Auto-discovered (no flag needed)
cp my-config.json ~/.active-memory/config.json
active-memory          # picks it up automatically
active-memory-chat     # same config, same tuning
```

## Architecture

```
types.py  <--  scoring.py  <--  btree.py  <--  assembler.py  <--  proxy.py
(primitives)   (scorer)        (index)         (prompt builder)    (HTTP proxy)
                                                       ^                ^
                                                  middleware.py    primary interface
                                                  cli.py
```

### Core data model

- **KVTuple** -- atomic unit of memory. Semantic key + content value + access stats (hit count, recency, token cost) + structural references (call graph edges).
- **SemanticBTree** -- B-tree index where nodes hold clusters of KV tuples. Routes inserts by cosine similarity to node centroids. Splits overflowing nodes via 2-means clustering. Prunes cold tuples and compresses cold subtrees into parent summaries.
- **ContextAssembler** -- greedy knapsack packer that fills a token budget from scored tuples + pinned recent turns. Includes ground-truth anchoring and dependency pulling.
- **Proxy** -- HTTP server that intercepts `/v1/messages`, runs the full ingest -> score -> assemble -> forward pipeline, and returns the upstream response unchanged.

### Scoring

```
score = 0.20 * recency + 0.20 * frequency + 0.45 * relevance + 0.15 * affinity
```

Recency decays exponentially with a 30-minute half-life. Frequency is log-scaled. Relevance is cosine similarity to the current query. Affinity boosts tuples that are structurally related (call graph neighbors) to already-selected tuples.

## Other interfaces

### Interactive chat CLI

```bash
active-memory-chat                                    # standalone chat with managed context
active-memory-chat --provider openai --model gpt-4.1  # use OpenAI instead
active-memory-chat --provider ollama --model llama3    # use local Ollama models
active-memory-chat --session myproject --resume        # resume a named session
active-memory-chat --ingest src/main.py README.md      # pre-load files
```

**Authentication:** The chat CLI requires a standard API key for your chosen provider — it does not support OAuth or session-based auth (e.g. Claude Code's login credentials). Set the appropriate environment variable before running:

```bash
# For Anthropic (default provider)
export ANTHROPIC_API_KEY=sk-ant-...

# For OpenAI
export OPENAI_API_KEY=sk-...

# For Ollama (no key needed, runs locally)
active-memory-chat --provider ollama
```

Chat commands: `/stats`, `/tree`, `/hot`, `/cold`, `/prune`, `/compress`, `/ingest <file>`, `/budget`, `/save`, `/load`, `/export`, `/quit`.

### Python library

```python
from active_memory import SemanticBTree, Scorer, create_embedder

embedder = create_embedder("auto").embedder
tree = SemanticBTree(embedder=embedder, scorer=Scorer())
tree.insert("db choice", "PostgreSQL 16 with pgvector extension")

q = tree.embedder.embed(["database"])[0]
for score, t in tree.query(q, top_k=3):
    print(f"{score:.3f}  {t.key_text}: {t.value_text[:60]}")
```

### MCP server

```bash
active-memory-mcp  # for MCP-based integrations (secondary)
```

### Evaluation and benchmarks

```bash
# Deterministic offline retrieval benchmark
active-memory-eval --mode offline

# Live A/B evaluation (full scale)
active-memory-live-bench --trials 3

# Quick sanity check (100K only)
active-memory-live-bench --quick

# Rate-safe mode (smaller scales for rate-limited accounts)
active-memory-live-bench --rate-safe

# Budget sweep (find optimal AM budget at a fixed scale)
active-memory-live-bench --sweep-budgets 10000,12000,14000,15000

# Offline long-context benchmark (no API calls, scales to 1M tokens)
active-memory-long-context-bench

# Direct-vs-proxy comparison
active-memory-proxy-live-bench
```

## Tuning

All parameters below can be set via the [config file](#configuration-file) or CLI flags where noted.

| Parameter | Config key | Default | Effect |
|---|---|---|---|
| `--budget` | `budget` | 100,000 | Total token budget for assembled prompts |
| `--recency` | `assembler.recency_window` | 6 (proxy) / 4 (chat) | Recent turns always included verbatim |
| `--reset-threshold` | `proxy.reset_threshold` | 0.75 | Fraction of budget that triggers a full context reset |
| `--reset-briefing-budget` | `proxy.reset_briefing_budget` | 8,000 | Max tokens for the briefing after a reset |
| | `btree.max_tuples` | 16 | Tuples per leaf before split |
| | `btree.compress_threshold` | 0.08 | Score below which subtrees compress |
| | `scoring.recency_half_life` | 1800 | Half-life in seconds for recency decay |
| | `assembler.budget_pressure` | 0.85 | Fraction of available budget to actually fill |
| | `assembler.anchor_relevance_threshold` | 0.60 | Cosine similarity threshold for ground-truth anchoring |

## Current status

`active-memory` is alpha software. The retrieval and assembly pipeline is implemented, but quality depends on the embedding provider and tuning choices. The included offline and live benchmarks are intended to make those tradeoffs visible rather than hide them.

## What's next

- [ ] LLM-based summariser for compression (Haiku for cheap summaries)
- [ ] Streaming support in the proxy
- [ ] Async API support
- [ ] Cross-session persistence (tree survives process restarts -- partially done via state_dir)

## License

MIT. See [LICENSE](LICENSE).
