# Security and Privacy

The default store is local SQLite at `~/.active-memory/memory.db`. Raw events and derived memories persist until explicitly deleted. File permissions and disk encryption remain the operator's responsibility.

Persistence runs configurable redaction for common API keys, passwords, bearer tokens, private keys, and connection strings. Redaction is defense in depth and can miss novel secret formats. Raw content is never written to structured logs. External embeddings are disabled unless selected in configuration.

Storage can be disabled. The CLI supports physical deletion by memory, session, namespace, or full database; destructive operations require confirmation. Namespace isolation is a filter boundary, not an OS-level access-control mechanism. Do not share one proxy among mutually untrusted users without an external authentication and authorization layer.

