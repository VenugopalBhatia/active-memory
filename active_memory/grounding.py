"""Grounding layer — reduces hallucination by verifying LLM output
against the memory tree.

Three mechanisms:
  1. Provenance injection: marks retrieved context with source metadata
     so the LLM knows what it's grounded in vs guessing.
  2. Post-generation verification: checks response claims against
     stored tuples and flags unsupported statements.
  3. Contradiction detection: compares new statements against existing
     memory to catch when the LLM contradicts earlier established facts.

This module sits between the assembler and the API call, and also
runs after the response is received.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from .btree import SemanticBTree
from .scoring import Scorer
from .types import Embedding, Embedder, KVTuple, cosine_sim, estimate_tokens


# ── Provenance Injection ──────────────────────────────────────────────

@dataclass
class GroundedBlock:
    """A context block annotated with provenance metadata."""
    key: str
    value: str
    source_turn: int | None     # conversation turn where this was established
    confidence: str             # "verbatim" | "compressed" | "inferred"
    age_seconds: float
    hit_count: int
    score: float


class ProvenanceInjector:
    """Annotates retrieved context so the LLM can distinguish between
    grounded facts and uncertain information.

    Instead of injecting raw text, we inject structured blocks like:

        [GROUNDED — established at turn 3, accessed 7 times]
        Database: ClickHouse for analytics, PostgreSQL 16 for metadata

        [COMPRESSED — summary of turns 12-18, may lack detail]
        Discussed deployment options, settled on Docker Compose

    This gives the model explicit signals about what it can state
    confidently vs what it should hedge on.
    """

    def __init__(self, tree: SemanticBTree, scorer: Scorer) -> None:
        self.tree = tree
        self.scorer = scorer

    def build_grounded_context(
        self,
        query_emb: Embedding,
        top_k: int = 50,
        conversation: list[dict] | None = None,
    ) -> list[GroundedBlock]:
        """Retrieve and annotate context blocks with provenance."""
        results = self.tree.query(query_emb, top_k=top_k)
        blocks: list[GroundedBlock] = []

        for score, t in results:
            # Determine confidence level
            if t.key_text.startswith("summary:"):
                confidence = "compressed"
            elif t.hit_count >= 3:
                confidence = "verbatim"
            else:
                confidence = "inferred"

            # Try to find the source turn
            source_turn = self._find_source_turn(t, conversation)

            blocks.append(GroundedBlock(
                key=t.key_text,
                value=t.value_text,
                source_turn=source_turn,
                confidence=confidence,
                age_seconds=time.time() - t.created_at,
                hit_count=t.hit_count,
                score=score,
            ))

        return blocks

    def format_grounded_context(self, blocks: list[GroundedBlock]) -> str:
        """Format grounded blocks into a structured context string
        that helps the LLM distinguish reliable from uncertain info."""
        parts: list[str] = []

        # Group by confidence
        verbatim = [b for b in blocks if b.confidence == "verbatim"]
        compressed = [b for b in blocks if b.confidence == "compressed"]
        inferred = [b for b in blocks if b.confidence == "inferred"]

        if verbatim:
            parts.append("<grounded_facts confidence=\"high\">")
            parts.append("The following are established facts from this conversation, "
                         "referenced multiple times. You can state these confidently.")
            for b in verbatim:
                turn_note = f" (established turn {b.source_turn})" if b.source_turn else ""
                parts.append(f"• [{b.key}]{turn_note}: {b.value}")
            parts.append("</grounded_facts>")

        if inferred:
            parts.append("")
            parts.append("<grounded_facts confidence=\"medium\">")
            parts.append("The following were mentioned but not repeatedly confirmed. "
                         "State these but be open to correction.")
            for b in inferred:
                parts.append(f"• [{b.key}]: {b.value}")
            parts.append("</grounded_facts>")

        if compressed:
            parts.append("")
            parts.append("<compressed_context confidence=\"low\">")
            parts.append("The following are compressed summaries of earlier discussion. "
                         "Details may have been lost. If asked about specifics, "
                         "acknowledge that you're working from a summary.")
            for b in compressed:
                parts.append(f"• [{b.key}]: {b.value}")
            parts.append("</compressed_context>")

        return "\n".join(parts)

    @staticmethod
    def _find_source_turn(
        t: KVTuple, conversation: list[dict] | None
    ) -> int | None:
        """Try to find which conversation turn a tuple came from."""
        if not conversation:
            return None
        # Simple heuristic: check if the tuple value appears in any message
        for i, msg in enumerate(conversation):
            content = msg.get("content", "")
            if t.value_text[:40] in content:
                return i // 2 + 1  # convert message index to turn number
        return None


# ── Post-Generation Verification ─────────────────────────────────────

@dataclass
class VerificationResult:
    """Result of verifying an LLM response against stored memories."""
    response_text: str
    claims: list[Claim]
    overall_grounding: float    # 0-1, fraction of claims that are grounded
    contradictions: list[Contradiction]
    ungrounded_claims: list[Claim]  # claims with no supporting memory


@dataclass
class Claim:
    """A factual claim extracted from the LLM response."""
    text: str
    embedding: Embedding | None = field(default=None, repr=False)
    best_match_key: str | None = None
    best_match_score: float = 0.0
    is_grounded: bool = False
    is_contradiction: bool = False


@dataclass
class Contradiction:
    """A detected contradiction between the response and stored memory."""
    response_claim: str
    stored_fact_key: str
    stored_fact_value: str
    similarity: float           # how similar the topics are
    explanation: str            # why this might be a contradiction


class ResponseVerifier:
    """Checks LLM responses against stored memories for hallucination.

    Works by:
      1. Extracting factual claims from the response (sentence-level)
      2. Embedding each claim
      3. Matching against stored tuples
      4. Flagging claims with no supporting memory (potential hallucination)
      5. Detecting claims that contradict stored facts
    """

    def __init__(
        self,
        tree: SemanticBTree,
        embedder: Embedder,
        scorer: Scorer,
        grounding_threshold: float = 0.6,
        contradiction_threshold: float = 0.75,
    ) -> None:
        self.tree = tree
        self.embedder = embedder
        self.scorer = scorer
        self.grounding_threshold = grounding_threshold
        self.contradiction_threshold = contradiction_threshold

    def verify(self, response_text: str) -> VerificationResult:
        """Verify an LLM response against stored memories."""
        # Extract claims (sentence-level)
        sentences = self._extract_claims(response_text)
        if not sentences:
            return VerificationResult(
                response_text=response_text,
                claims=[],
                overall_grounding=1.0,
                contradictions=[],
                ungrounded_claims=[],
            )

        # Embed all claims at once
        embeddings = self.embedder.embed(sentences)
        claims: list[Claim] = []

        for text, emb in zip(sentences, embeddings):
            claim = Claim(text=text, embedding=emb)

            # Find best matching memory
            results = self.tree.query(emb, top_k=3)
            if results:
                best_score, best_tuple = results[0]
                claim.best_match_key = best_tuple.key_text
                claim.best_match_score = best_score

                # Is this claim grounded?
                if best_score >= self.grounding_threshold:
                    claim.is_grounded = True

            claims.append(claim)

        # Detect contradictions
        contradictions = self._detect_contradictions(claims)
        for c in contradictions:
            # Find the claim and mark it
            for claim in claims:
                if claim.text == c.response_claim:
                    claim.is_contradiction = True

        ungrounded = [c for c in claims if not c.is_grounded and not c.is_contradiction]
        grounded_count = sum(1 for c in claims if c.is_grounded)
        grounding_rate = grounded_count / len(claims) if claims else 1.0

        return VerificationResult(
            response_text=response_text,
            claims=claims,
            overall_grounding=grounding_rate,
            contradictions=contradictions,
            ungrounded_claims=ungrounded,
        )

    def _detect_contradictions(self, claims: list[Claim]) -> list[Contradiction]:
        """Detect potential contradictions between claims and stored facts.

        A contradiction is when a claim is about the same topic as a
        stored fact (high topic similarity) but the content diverges.

        Heuristic: if the claim's KEY embedding is very similar to a
        stored tuple's key (same topic) but the claim's VALUE embedding
        has low similarity to the stored value, it might be a
        contradiction — the model is saying something different about
        the same topic.
        """
        contradictions: list[Contradiction] = []
        all_tuples = self.tree.all_tuples()

        if not all_tuples:
            return []

        for claim in claims:
            if claim.embedding is None:
                continue

            for t in all_tuples:
                if t.key_emb is None:
                    continue

                # How similar is the topic?
                topic_sim = cosine_sim(claim.embedding, t.key_emb)

                if topic_sim < self.contradiction_threshold:
                    continue  # different topics, not a contradiction

                # Same topic — now check if the content matches
                # Embed the stored value and compare
                stored_emb = self.embedder.embed([t.value_text])[0]
                content_sim = cosine_sim(claim.embedding, stored_emb)

                # High topic similarity + low content similarity = potential contradiction
                if topic_sim > self.contradiction_threshold and content_sim < 0.3:
                    contradictions.append(Contradiction(
                        response_claim=claim.text,
                        stored_fact_key=t.key_text,
                        stored_fact_value=t.value_text[:200],
                        similarity=topic_sim,
                        explanation=(
                            f"This claim is about the same topic as a stored fact "
                            f"(topic similarity: {topic_sim:.2f}) but the content "
                            f"diverges (content similarity: {content_sim:.2f}). "
                            f"The model may be contradicting an earlier decision."
                        ),
                    ))
                    break  # one contradiction per claim is enough

        return contradictions

    @staticmethod
    def _extract_claims(text: str) -> list[str]:
        """Extract factual claim sentences from a response.

        Filters out:
          - Very short sentences (likely connectors)
          - Questions
          - Hedging/meta sentences ("I think", "Let me know")
        """
        sentences = re.split(r'(?<=[.!])\s+', text)
        claims: list[str] = []

        skip_patterns = [
            r'^(I think|I believe|Maybe|Perhaps|It seems|Let me|Sure|Okay|Got it)',
            r'\?$',
            r'^(Yes|No|Right|Exactly|Absolutely)[,.]',
        ]

        for s in sentences:
            s = s.strip()
            if len(s) < 20:
                continue
            if any(re.match(p, s, re.IGNORECASE) for p in skip_patterns):
                continue
            claims.append(s)

        return claims


# ── Grounding-Aware Assembler ─────────────────────────────────────────

class GroundedAssembler:
    """Extends the base assembler with provenance and anti-hallucination
    instructions in the system prompt.

    This assembler:
      1. Injects confidence-tagged context (high/medium/low)
      2. Adds anti-hallucination instructions to the system prompt
      3. Can run post-generation verification
    """

    def __init__(
        self,
        tree: SemanticBTree,
        embedder: Embedder,
        scorer: Scorer,
        config: "GroundingConfig | None" = None,
    ) -> None:
        self.tree = tree
        self.embedder = embedder
        self.scorer = scorer
        self.cfg = config or GroundingConfig()
        self.injector = ProvenanceInjector(tree, scorer)
        self.last_blocks: list[GroundedBlock] = []
        self.verifier = ResponseVerifier(
            tree,
            embedder,
            scorer,
            grounding_threshold=self.cfg.grounding_threshold,
            contradiction_threshold=self.cfg.contradiction_threshold,
        )

    def build_grounded_prompt(
        self,
        system_prompt: str,
        conversation: list[dict],
        query_emb: Embedding,
        token_budget: int = 100_000,
        recency_window: int = 4,
    ) -> tuple[str, list[dict]]:
        """Build a prompt with grounded context and anti-hallucination
        instructions.

        Returns (enhanced_system_prompt, messages).
        """
        # Get grounded context blocks
        blocks = self.injector.build_grounded_context(
            query_emb, top_k=50, conversation=conversation
        )
        self.last_blocks = list(blocks)

        # Format with confidence tags
        grounded_context = self.injector.format_grounded_context(blocks)

        # Build enhanced system prompt
        enhanced_system = (
            f"{system_prompt}\n\n"
            "<grounding_instructions>\n"
            "You have access to retrieved context from this conversation's memory. "
            "The context is tagged by confidence level:\n"
            "- 'high' confidence: facts confirmed multiple times. State these directly.\n"
            "- 'medium' confidence: mentioned once. State these but accept correction.\n"
            "- 'low' confidence: compressed summaries. Acknowledge if details are vague.\n\n"
            "CRITICAL RULES:\n"
            "1. If you're unsure about a specific detail (number, name, date), "
            "say so rather than guessing.\n"
            "2. If the retrieved context doesn't cover a topic, explicitly state "
            "that you don't have that information from our conversation.\n"
            "3. Never invent specific details (exact numbers, names, dates) "
            "that aren't in the retrieved context.\n"
            "4. If a fact comes from compressed context, preface with "
            "'Based on our earlier discussion' rather than stating as certain.\n"
            "</grounding_instructions>"
        )

        # Build messages
        messages: list[dict] = []

        if grounded_context:
            messages.append({
                "role": "user",
                "content": (
                    "[Retrieved context from conversation memory]\n\n"
                    + grounded_context
                ),
            })
            messages.append({
                "role": "assistant",
                "content": (
                    "I have the retrieved context with confidence levels noted. "
                    "I'll ground my responses in this information and flag "
                    "anything I'm uncertain about."
                ),
            })

        # Pinned recent turns
        pinned = conversation[-recency_window * 2:]
        messages.extend(pinned)

        return enhanced_system, messages

    def verify_response(self, response_text: str) -> VerificationResult:
        """Run post-generation verification on a response."""
        return self.verifier.verify(response_text)

    def build_correction_prompt(
        self,
        verification: VerificationResult,
    ) -> str | None:
        """If verification found issues, build a follow-up prompt
        asking the model to correct itself.

        Returns None if no correction is needed.
        """
        issues: list[str] = []

        if verification.contradictions:
            for c in verification.contradictions:
                issues.append(
                    f"Your response may contradict an earlier decision. "
                    f"You said: '{c.response_claim[:100]}' "
                    f"But we previously established: '{c.stored_fact_value[:100]}' "
                    f"({c.explanation})"
                )

        if verification.overall_grounding < 0.3 and verification.ungrounded_claims:
            sample = verification.ungrounded_claims[:3]
            claims_text = "; ".join(c.text[:80] for c in sample)
            issues.append(
                f"Several claims in your response don't match our conversation "
                f"history: {claims_text}. Please verify these or indicate "
                f"you're uncertain."
            )

        if not issues:
            return None

        return (
            "I want to double-check something in your last response. "
            + " Also, ".join(issues)
            + " Can you review and correct if needed?"
        )


# ── Integration with Middleware ───────────────────────────────────────

@dataclass
class GroundingConfig:
    """Configuration for the grounding layer."""
    enabled: bool = True
    provenance_injection: bool = True   # tag context with confidence levels
    post_verification: bool = True      # verify responses after generation
    auto_correct: bool = False          # automatically send correction prompts
    grounding_threshold: float = 0.6    # similarity needed to count as grounded
    contradiction_threshold: float = 0.75  # similarity for same-topic detection
    min_grounding_rate: float = 0.3     # below this triggers a warning


@dataclass
class GroundingReport:
    """Diagnostic report from the grounding layer."""
    grounding_rate: float
    total_claims: int
    grounded_claims: int
    ungrounded_claims: int
    contradictions_found: int
    correction_sent: bool
    high_confidence_blocks: int
    medium_confidence_blocks: int
    low_confidence_blocks: int
