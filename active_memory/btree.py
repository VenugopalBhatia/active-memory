"""Semantic B-tree with KV-tuple cluster nodes.

Design principles
-----------------
* Each leaf node holds a *cluster* of semantically related KV tuples.
* Internal nodes store centroid embeddings for routing + a compressed
  summary of their subtree.
* Insertion routes to the most similar leaf (by cosine similarity).
* Leaf nodes split when they exceed ``max_tuples`` (like a B-tree page
  overflow), using 2-means on the tuple embeddings.
* Cold subtrees can be *compressed*: all tuples are replaced by a single
  summary tuple stored in the parent, and the subtree is pruned.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from .scoring import Scorer, ScoringConfig
from .types import (
    Embedding,
    Embedder,
    KVTuple,
    cosine_sim,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class BTreeNode:
    """A node in the semantic B-tree.

    Leaf nodes hold KV tuples.  Internal nodes hold children and a
    centroid for routing.  Every node can carry a *summary* — a
    compressed representation of everything underneath it.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    # -- payload (leaf) --
    tuples: list[KVTuple] = field(default_factory=list)

    # -- structure (internal) --
    children: list[BTreeNode] = field(default_factory=list)
    parent: BTreeNode | None = field(default=None, repr=False)

    # -- semantic metadata --
    centroid: Embedding | None = field(default=None, repr=False)
    summary: KVTuple | None = field(default=None)  # compressed repr

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def total_tokens(self) -> int:
        if self.summary:
            return self.summary.token_cost
        return sum(t.token_cost for t in self.tuples)

    def recompute_centroid(self) -> None:
        """Update centroid from current tuples or children centroids."""
        vecs: list[Embedding] = []
        if self.is_leaf:
            vecs = [t.key_emb for t in self.tuples if t.key_emb is not None]
        else:
            vecs = [c.centroid for c in self.children if c.centroid is not None]
        if vecs:
            c = np.mean(vecs, axis=0).astype(np.float32)
            self.centroid = c / (np.linalg.norm(c) + 1e-9)
        else:
            self.centroid = None


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------

@dataclass
class BTreeConfig:
    max_tuples: int = 16       # max KV tuples per leaf before split
    min_tuples: int = 4        # merge threshold
    min_children: int = 2      # minimum children for internal node
    compress_threshold: float = 0.10  # score below which a node is cold


class SemanticBTree:
    """Semantic B-tree index over KV-tuple clusters."""

    def __init__(
        self,
        embedder: Embedder,
        scorer: Scorer | None = None,
        config: BTreeConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.scorer = scorer or Scorer()
        self.cfg = config or BTreeConfig()
        self.root = BTreeNode()
        self._size = 0  # total tuples stored

    @property
    def size(self) -> int:
        return self._size

    # -- insert ---------------------------------------------------------

    def insert(self, key_text: str, value_text: str) -> KVTuple:
        """Insert a new KV tuple into the tree."""
        emb = self.embedder.embed([key_text])[0]

        t = KVTuple(
            key_text=key_text,
            value_text=value_text,
            key_emb=emb,
            token_cost=estimate_tokens(value_text),
        )

        leaf = self._find_leaf(self.root, emb)
        leaf.tuples.append(t)
        leaf.recompute_centroid()
        self._size += 1

        if len(leaf.tuples) > self.cfg.max_tuples:
            self._split(leaf)

        return t

    def insert_tuple(self, t: KVTuple) -> None:
        """Insert a pre-built KVTuple (must already have key_emb)."""
        if t.key_emb is None:
            t.key_emb = self.embedder.embed([t.key_text])[0]
        if t.token_cost == 0:
            t.token_cost = estimate_tokens(t.value_text)

        leaf = self._find_leaf(self.root, t.key_emb)
        leaf.tuples.append(t)
        leaf.recompute_centroid()
        self._size += 1

        if len(leaf.tuples) > self.cfg.max_tuples:
            self._split(leaf)

    # -- query / touch --------------------------------------------------

    def query(
        self, query_emb: Embedding, top_k: int = 10
    ) -> list[tuple[float, KVTuple]]:
        """Return the top-k most relevant tuples (scored by relevance
        blended with frequency/recency).  Touching each returned tuple
        updates its access stats.
        """
        results: list[tuple[float, KVTuple]] = []
        self._collect_scored(self.root, query_emb, results)
        results.sort(key=lambda x: x[0], reverse=True)
        top = results[:top_k]
        for _, t in top:
            t.touch()
        return top

    # -- pruning / compression -----------------------------------------

    def prune(self, query_emb: Embedding | None = None) -> list[KVTuple]:
        """Prune cold tuples from the tree.  Returns evicted tuples."""
        evicted: list[KVTuple] = []
        self._prune_node(self.root, query_emb, evicted)
        return evicted

    def compress_cold_subtrees(
        self,
        summariser: _Summariser | None = None,
        query_emb: Embedding | None = None,
    ) -> int:
        """Compress cold subtrees into summary tuples in their parents.

        If *summariser* is provided it will be called to produce a
        text summary; otherwise a simple concatenation is used.

        Returns the number of nodes compressed.
        """
        return self._compress_node(self.root, summariser, query_emb)

    # -- traversal helpers ----------------------------------------------

    def all_tuples(self) -> list[KVTuple]:
        """Flat list of every tuple in the tree."""
        out: list[KVTuple] = []
        self._collect_all(self.root, out)
        return out

    def tuples_by_cluster(self) -> list[list[KVTuple]]:
        """Return tuples grouped by their leaf node (semantic cluster).

        Each inner list contains the tuples from one leaf. Leaves with
        summaries return the summary as the sole element. Empty leaves
        are skipped. The grouping reflects the tree's semantic clustering
        — tuples in the same list are about related topics.
        """
        clusters: list[list[KVTuple]] = []
        self._collect_clusters(self.root, clusters)
        return clusters

    def _collect_clusters(
        self, node: BTreeNode, out: list[list[KVTuple]]
    ) -> None:
        if node.is_leaf:
            if node.summary and not node.tuples:
                out.append([node.summary])
            elif node.tuples:
                out.append(list(node.tuples))
        else:
            if node.summary:
                out.append([node.summary])
            for child in node.children:
                self._collect_clusters(child, out)

    def all_nodes(self) -> list[BTreeNode]:
        """BFS of all nodes."""
        queue = [self.root]
        visited: list[BTreeNode] = []
        while queue:
            n = queue.pop(0)
            visited.append(n)
            queue.extend(n.children)
        return visited

    def depth(self) -> int:
        """Max depth of the tree."""
        return self._depth(self.root)

    # ===================================================================
    # INTERNAL
    # ===================================================================

    def _find_leaf(self, node: BTreeNode, emb: Embedding) -> BTreeNode:
        """Route to the most similar leaf node."""
        if node.is_leaf:
            return node
        best_child = node.children[0]
        best_sim = -2.0
        for child in node.children:
            if child.centroid is not None:
                sim = cosine_sim(emb, child.centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_child = child
        return self._find_leaf(best_child, emb)

    def _split(self, node: BTreeNode) -> None:
        """Split a leaf node into two children using 2-means."""
        if len(node.tuples) < 2:
            return

        # -- 2-means clustering on tuple embeddings --
        embeddings = [t.key_emb for t in node.tuples if t.key_emb is not None]
        if len(embeddings) < 2:
            return

        mat = np.stack(embeddings)
        labels = self._kmeans2(mat)

        group_a = [t for t, l in zip(node.tuples, labels) if l == 0]
        group_b = [t for t, l in zip(node.tuples, labels) if l == 1]

        # Avoid degenerate splits
        if not group_a or not group_b:
            mid = len(node.tuples) // 2
            group_a = node.tuples[:mid]
            group_b = node.tuples[mid:]

        child_a = BTreeNode(tuples=group_a, parent=node)
        child_b = BTreeNode(tuples=group_b, parent=node)
        child_a.recompute_centroid()
        child_b.recompute_centroid()

        # Node becomes internal
        node.tuples = []
        node.children = [child_a, child_b]
        node.recompute_centroid()

    @staticmethod
    def _kmeans2(mat: np.ndarray, max_iter: int = 10) -> list[int]:
        """Simple 2-means. Returns list of 0/1 labels."""
        n = mat.shape[0]
        # init: pick two farthest points
        idx_a = 0
        dists = np.linalg.norm(mat - mat[idx_a], axis=1)
        idx_b = int(np.argmax(dists))
        centers = np.stack([mat[idx_a], mat[idx_b]])

        labels = [0] * n
        for _ in range(max_iter):
            # assign
            for i in range(n):
                d0 = float(np.linalg.norm(mat[i] - centers[0]))
                d1 = float(np.linalg.norm(mat[i] - centers[1]))
                labels[i] = 0 if d0 <= d1 else 1
            # update
            g0 = [mat[i] for i in range(n) if labels[i] == 0]
            g1 = [mat[i] for i in range(n) if labels[i] == 1]
            if g0:
                centers[0] = np.mean(g0, axis=0)
            if g1:
                centers[1] = np.mean(g1, axis=0)
        return labels

    def _collect_scored(
        self,
        node: BTreeNode,
        query_emb: Embedding,
        out: list[tuple[float, KVTuple]],
    ) -> None:
        if node.is_leaf:
            # If node has been compressed, score the summary instead
            if node.summary and not node.tuples:
                s = self.scorer.score(node.summary, query_emb)
                out.append((s, node.summary))
            else:
                for t in node.tuples:
                    s = self.scorer.score(t, query_emb)
                    out.append((s, t))
        else:
            # Also include this node's summary if it has one
            if node.summary:
                s = self.scorer.score(node.summary, query_emb)
                out.append((s, node.summary))
            for child in node.children:
                self._collect_scored(child, query_emb, out)

    def _collect_all(self, node: BTreeNode, out: list[KVTuple]) -> None:
        if node.summary:
            out.append(node.summary)
        for t in node.tuples:
            out.append(t)
        for child in node.children:
            self._collect_all(child, out)

    def _prune_node(
        self,
        node: BTreeNode,
        query_emb: Embedding | None,
        evicted: list[KVTuple],
    ) -> None:
        """Remove tuples scoring below compress_threshold."""
        if node.is_leaf and node.tuples:
            keep: list[KVTuple] = []
            for t in node.tuples:
                score = self.scorer.score(t, query_emb)
                if score < self.cfg.compress_threshold:
                    evicted.append(t)
                    self._size -= 1
                else:
                    keep.append(t)
            node.tuples = keep
            node.recompute_centroid()
        else:
            for child in node.children:
                self._prune_node(child, query_emb, evicted)
            # Remove empty children
            node.children = [
                c for c in node.children
                if c.tuples or c.children or c.summary
            ]
            node.recompute_centroid()

    def _compress_node(
        self,
        node: BTreeNode,
        summariser: _Summariser | None,
        query_emb: Embedding | None,
    ) -> int:
        """Recursively compress cold subtrees."""
        compressed = 0

        if node.is_leaf:
            return 0

        for child in list(node.children):
            # Recurse first (bottom-up)
            compressed += self._compress_node(child, summariser, query_emb)

            # Score the entire child subtree
            child_tuples = []
            self._collect_all_raw(child, child_tuples)
            if not child_tuples:
                continue

            avg_score = sum(
                self.scorer.score(t, query_emb) for t in child_tuples
            ) / len(child_tuples)

            if avg_score < self.cfg.compress_threshold:
                # Compress this child into a summary
                if summariser:
                    summary_text = summariser(child_tuples)
                else:
                    summary_text = self._default_summarise(child_tuples)

                summary_emb = self.embedder.embed([summary_text])[0]
                summary_tuple = KVTuple(
                    key_text=f"summary:{child.id}",
                    value_text=summary_text,
                    key_emb=summary_emb,
                    token_cost=estimate_tokens(summary_text),
                )

                # Replace child contents with summary
                self._size -= len(child_tuples)
                child.tuples = []
                child.children = []
                child.summary = summary_tuple
                self._size += 1
                compressed += 1

        return compressed

    def _collect_all_raw(self, node: BTreeNode, out: list[KVTuple]) -> None:
        """Collect tuples only (not summaries) for compression input."""
        for t in node.tuples:
            out.append(t)
        for child in node.children:
            self._collect_all_raw(child, out)

    @staticmethod
    def _default_summarise(tuples: list[KVTuple]) -> str:
        """Fallback summariser: just concatenate key texts."""
        parts = [f"- {t.key_text}: {t.value_text[:80]}" for t in tuples[:10]]
        return "Compressed context:\n" + "\n".join(parts)

    def _depth(self, node: BTreeNode) -> int:
        if node.is_leaf:
            return 1
        return 1 + max(self._depth(c) for c in node.children)


# -- summariser protocol ---------------------------------------------------

from typing import Callable
_Summariser = Callable[[list[KVTuple]], str]
