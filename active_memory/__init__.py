"""active_memory — Active memory management for LLM context windows.

A transparent proxy that sits between Claude Code and the Anthropic API,
managing context via a semantic B-tree of KV-tuple clusters with
frequency-aware eviction and hierarchical compression.

Quick start:
    active-memory --verbose          # start the proxy
    ANTHROPIC_BASE_URL=http://localhost:8080 claude   # use it
"""

from .types import KVTuple, Embedder, HashEmbedder, cosine_sim, estimate_tokens
from .embeddings import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    EmbedderSpec,
    OpenAIEmbedder,
    create_embedder,
)
from .model_clients import (
    AnthropicModelClient,
    OpenAIModelClient,
    ModelClient,
    GeneratedResponse,
    create_model_client,
)
from .scoring import Scorer, ScoringConfig
from .btree import SemanticBTree, BTreeNode, BTreeConfig
from .assembler import ContextAssembler, AssemblerConfig, AssembledContext
from .middleware import ActiveMemoryMiddleware, MiddlewareConfig, ContextStats
from .proxy import ProxyConfig, ContextManager
from .code_ingest import CodeParser, CodeChunk, parse_code_file
from . import eval as eval

__all__ = [
    # Proxy (primary interface)
    "ProxyConfig",
    "ContextManager",
    # Core data structures
    "KVTuple",
    "Embedder",
    "HashEmbedder",
    "OpenAIEmbedder",
    "AnthropicModelClient",
    "OpenAIModelClient",
    "ModelClient",
    "GeneratedResponse",
    "create_model_client",
    "EmbedderSpec",
    "DEFAULT_OPENAI_EMBEDDING_MODEL",
    "create_embedder",
    "cosine_sim",
    "estimate_tokens",
    "Scorer",
    "ScoringConfig",
    "SemanticBTree",
    "BTreeNode",
    "BTreeConfig",
    "ContextAssembler",
    "AssemblerConfig",
    "AssembledContext",
    # Middleware (direct API wrapper)
    "ActiveMemoryMiddleware",
    "MiddlewareConfig",
    "ContextStats",
]
