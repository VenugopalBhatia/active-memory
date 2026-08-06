# Limitations

- Embeddings can miss relevant records or rank lexical lookalikes highly.
- Assistant-generated content may be wrong and is labeled accordingly.
- Exact search loads the active namespace into memory and does not scale indefinitely.
- No universal approximate-search threshold is asserted; use the bundled latency benchmark on the target hardware and corpus.
- Deterministic extraction can miss relationships, updates, or memory-worthy statements.
- Semantic duplicate detection can retain duplicates or suppress close paraphrases.
- Token budgeting can exclude useful context, and approximate token counts differ from provider tokenizers.
- Persistent storage creates privacy, retention, backup, and deletion obligations.
- Secret redaction is pattern-based and incomplete.
- The primary proxy currently targets Anthropic `POST /v1/messages`; provider-managed hidden conversation state is outside its view.
- Offline retrieval metrics do not prove downstream answer correctness.
- No fine-tuning occurs.

