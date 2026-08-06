# Archived semantic tree experiment

This directory preserves the repository's early custom semantic B-tree experiment for historical reference only. It is not packaged, imported, configured, or used by the current runtime.

The structure clustered tuple embeddings into split nodes, but queries still performed a full tuple scan. It therefore did not provide meaningful branch pruning while adding mutable access counters, JSON snapshot complexity, pruning behavior that could discard source detail, and a second structural model beside conversation history.

The current implementation uses SQLite persistence and exact cosine search. The archived files retain their original relative imports and are not maintained as a runnable package; use Git history to reconstruct the experiment if needed.
