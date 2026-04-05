"""A/B evaluation framework for active-memory.

The key insight: to fairly compare context management strategies, the
*conversation* must be identical across all strategies. Only the *prompt
construction* for the probe should differ.

Design:
  1. Pre-script an entire conversation (both user AND assistant turns).
     This is the "frozen transcript."
  2. Feed the frozen transcript into each strategy's context manager,
     so each one ingests the same information.
  3. At probe time, each strategy assembles its own version of the
     prompt from the same underlying conversation.
  4. Send the probe to the LLM. The ONLY variable is what context
     each strategy included in the prompt.
  5. Run multiple trials per probe (temperature > 0) and measure
     recall rate with confidence intervals.

This isolates the effect of context management from conversation drift.
"""

from __future__ import annotations

import argparse
import math
import time
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from .model_clients import ModelClient, create_model_client

# ── Frozen Conversation ───────────────────────────────────────────────

@dataclass
class FrozenTurn:
    """A single turn in a pre-scripted conversation."""
    turn_number: int
    role: str           # "user" or "assistant"
    content: str
    is_planted_fact: bool = False
    fact_id: str | None = None


@dataclass
class Probe:
    """A question designed to test retrieval of a planted fact."""
    fact_id: str
    query: str
    expected_keywords: list[str]   # ALL must appear for a pass
    planted_at_turn: int


@dataclass
class FrozenScenario:
    """A complete pre-scripted conversation + probes."""
    name: str
    description: str
    transcript: list[FrozenTurn]
    probes: list[Probe]


# ── Scenario Builders ─────────────────────────────────────────────────

def build_architecture_review() -> FrozenScenario:
    """A 40-turn architecture discussion with 5 planted decisions,
    buried under realistic distractor turns.

    Both sides are scripted so every strategy sees the exact same
    conversation history.
    """
    transcript: list[FrozenTurn] = []
    t = 0

    def user(msg: str, fact_id: str | None = None) -> None:
        nonlocal t; t += 1
        transcript.append(FrozenTurn(t, "user", msg, fact_id is not None, fact_id))

    def assistant(msg: str) -> None:
        nonlocal t; t += 1
        transcript.append(FrozenTurn(t, "assistant", msg))

    # --- Turn 1-2: Opening ---
    user("Let's plan the architecture for our new analytics platform.")
    assistant("Sounds good. What's the expected data volume and query pattern? That'll drive most of our infrastructure decisions.")

    # --- Turn 3-4: PLANTED FACT 1 (database) ---
    user("We'll ingest about 50M events per day. After evaluation, let's go with ClickHouse for the analytics store and PostgreSQL 16 for the metadata catalog.", fact_id="database")
    assistant("Good split. ClickHouse handles the columnar analytics workload well at that volume, and PostgreSQL is solid for the relational metadata. I'd suggest putting them behind separate connection pools.")

    # --- Turn 5-8: Distractor (frontend discussion) ---
    user("Switching topics — what frontend framework should we use for the dashboard?")
    assistant("For an analytics dashboard, I'd lean toward Next.js with React. Server components let you stream large result sets without blocking the UI, and the built-in API routes can proxy your analytics queries.")
    user("Makes sense. Should we use a charting library or build custom visualisations?")
    assistant("Start with Recharts or Tremor for standard charts. They're React-native and handle the common cases. Build custom only for the novel visualisations that are your differentiator.")

    # --- Turn 9-10: PLANTED FACT 2 (auth) ---
    user("For authentication, we're going with Auth0 using PKCE flow for the SPA, with a 12-minute access token lifetime and 30-day refresh tokens. The refresh tokens should be rotated on every use.", fact_id="auth")
    assistant("PKCE is the right call for a public SPA. 12-minute access tokens are on the shorter side but acceptable for an analytics tool. Make sure you implement silent refresh so users don't get interrupted mid-analysis.")

    # --- Turn 11-16: Distractor (long coding discussion) ---
    user("Can you help me draft the data ingestion pipeline? We need to handle both batch and streaming.")
    assistant("For dual-mode ingestion, I'd set up a Kafka cluster as the central bus. Streaming events go directly to Kafka topics. Batch files get uploaded to S3, then a Lambda function publishes them to the same Kafka topics for unified downstream processing.")
    user("What about schema evolution? Events will change over time.")
    assistant("Use Avro with a Schema Registry. It gives you backward and forward compatibility. Each event carries a schema ID, so the consumers can decode any version. Add a compatibility check in CI so breaking changes get caught before deployment.")
    user("Good. What serialisation format for the Kafka messages themselves?")
    assistant("Avro all the way — it's compact, schema-aware, and the Schema Registry integration is native. Protobuf is an alternative if you need cross-language support, but for a Python-heavy analytics stack, Avro is simpler.")

    # --- Turn 17-18: PLANTED FACT 3 (SLA) ---
    user("We need to define our query SLA. After load testing, we're setting: p50 under 500ms, p95 under 2 seconds, p99 under 5 seconds. Any query exceeding p99 triggers an automatic alert to the on-call engineer via PagerDuty.", fact_id="sla")
    assistant("Those are reasonable for an OLAP workload on 50M daily events. The p99 at 5 seconds gives you headroom for complex aggregations. I'd add a circuit breaker that kills queries exceeding 30 seconds to protect the cluster.")

    # --- Turn 19-24: Distractor (testing, CI/CD) ---
    user("Let's talk testing strategy. What coverage targets?")
    assistant("For a data platform, I'd prioritise integration tests over unit tests — maybe 40% unit, 50% integration, 10% E2E. The data pipeline is where bugs hide, and unit tests on transformations don't catch schema mismatches or serialisation issues.")
    user("How do we test the ingestion pipeline specifically?")
    assistant("Use Testcontainers to spin up real Kafka, ClickHouse, and PostgreSQL instances in CI. Publish a known dataset through the pipeline, then assert on the output tables. This catches issues that mocking would miss — like timezone handling and null propagation.")
    user("And CI/CD setup?")
    assistant("GitHub Actions with a matrix build. Run linting and type checking first as a fast gate. Then integration tests with Testcontainers. Deploy to staging on merge to main, production on tag. Add a canary deployment step that routes 5% of traffic to the new version for 15 minutes before full rollout.")

    # --- Turn 25-26: PLANTED FACT 4 (team) ---
    user("Quick org note: the analytics platform team is 6 people. Maya is the tech lead, Raj owns the ingestion pipeline, and Sara owns the query engine. All architectural decisions need sign-off from Maya and at least one domain owner.", fact_id="team")
    assistant("Got it. That's a lean team for this scope — I'd suggest pairing Raj and Sara for the ClickHouse schema design since it bridges both their domains. Maya's sign-off requirement will keep the architecture coherent as you parallelise work.")

    # --- Turn 27-32: Distractor (monitoring, docs) ---
    user("What monitoring should we set up on day one?")
    assistant("Four things minimum: Kafka consumer lag (shows pipeline backup), ClickHouse query duration histogram (catches slow queries before users notice), ingestion throughput rate (events per second), and error rate by pipeline stage. Grafana dashboards backed by Prometheus metrics.")
    user("Should we use structured logging?")
    assistant("Absolutely. JSON logs with correlation IDs that trace an event from ingestion through storage to query. Use the OpenTelemetry SDK so you can export to whatever backend — Datadog, Elastic, Grafana Loki. Standardise the log schema across all services on day one, it's painful to retrofit.")
    user("What about documentation? I always struggle with keeping docs up to date.")
    assistant("Use architecture decision records stored in the repo alongside the code. Each ADR captures the context, decision, and consequences. They stay in sync with the code because they're in the same PR. For API docs, generate from code annotations — OpenAPI spec for the REST layer, autogenerated from the route definitions.")

    # --- Turn 33-34: PLANTED FACT 5 (budget) ---
    user("Budget constraint: our total monthly infrastructure spend cannot exceed $12,000. That includes all compute, storage, and managed services. If we project exceeding this by more than 10%, we need VP approval from David Chen before proceeding.", fact_id="budget")
    assistant("At 50M events/day with ClickHouse, you're looking at roughly $4-5K for compute, $1-2K for Kafka, $1K for PostgreSQL, and $1-2K for monitoring and misc. That puts you at $7-10K, leaving a comfortable buffer. I'd set up cost alerts at $10K and $11K in AWS Budgets.")

    # --- Turn 35-40: More distractors ---
    user("Let's discuss the data retention policy.")
    assistant("For analytics data, I'd suggest: raw events kept for 90 days in ClickHouse (hot), then archived to S3 Parquet for 2 years (cold). Aggregated tables kept indefinitely — they're small and useful for trend analysis. Metadata in PostgreSQL kept indefinitely with soft deletes.")
    user("What about GDPR compliance for the event data?")
    assistant("You'll need a deletion pipeline that can purge events by user ID across both ClickHouse and S3 archives. ClickHouse mutations are expensive on large tables, so batch deletion requests daily. For S3, tag objects with user IDs in metadata and use lifecycle policies. Log all deletion requests for audit compliance.")
    user("One more thing — what's the disaster recovery plan?")
    assistant("ClickHouse: replicated tables across 3 nodes, daily backups to S3 with 30-day retention. PostgreSQL: RDS Multi-AZ with automated backups and point-in-time recovery. Kafka: replication factor 3 across AZs. Test the restore procedure quarterly — backup that hasn't been tested isn't a backup. Target RTO of 4 hours, RPO of 1 hour.")

    probes = [
        Probe(
            fact_id="database",
            query="What databases did we choose for the analytics platform and what is each one used for?",
            expected_keywords=["ClickHouse", "PostgreSQL"],
            planted_at_turn=3,
        ),
        Probe(
            fact_id="auth",
            query="What authentication setup did we decide on? Include the token lifetimes.",
            expected_keywords=["Auth0", "PKCE", "12"],
            planted_at_turn=9,
        ),
        Probe(
            fact_id="sla",
            query="What are our query latency SLA targets at p50, p95, and p99?",
            expected_keywords=["500", "2 second", "5 second"],
            planted_at_turn=17,
        ),
        Probe(
            fact_id="team",
            query="Who is the tech lead and what is the sign-off process for architecture decisions?",
            expected_keywords=["Maya"],
            planted_at_turn=25,
        ),
        Probe(
            fact_id="budget",
            query="What is our monthly infrastructure budget cap and who approves overages?",
            expected_keywords=["12,000", "David Chen"],
            planted_at_turn=33,
        ),
    ]

    return FrozenScenario(
        name="architecture_review",
        description="40-turn architecture discussion. 5 planted decisions at turns 3, 9, 17, 25, 33. Probed after turn 40.",
        transcript=transcript,
        probes=probes,
    )


def build_long_session_decay() -> FrozenScenario:
    """A 60-turn session where facts planted early are tested at the end.
    Stresses the system's ability to retain information across longer
    distances. Uses repetitive distractor turns to fill context."""

    transcript: list[FrozenTurn] = []
    t = 0

    def user(msg: str, fact_id: str | None = None) -> None:
        nonlocal t; t += 1
        transcript.append(FrozenTurn(t, "user", msg, fact_id is not None, fact_id))

    def assistant(msg: str) -> None:
        nonlocal t; t += 1
        transcript.append(FrozenTurn(t, "assistant", msg))

    # --- PLANTED FACTS at the very start ---
    user("Project codename is AURORA. All internal references must use this codename, never the product name externally.", fact_id="codename")
    assistant("Noted — AURORA it is. I'll use the codename throughout our discussions.")

    user("The encryption key rotation schedule is every 72 hours, using AES-256-GCM. Keys are stored in AWS KMS with a separate CMK per environment.", fact_id="encryption")
    assistant("72-hour rotation with AES-256-GCM and per-environment CMKs in KMS is solid. I'd also recommend enabling automatic key rotation in KMS as a safety net.")

    user("Our vendor contact for the data partnership is Priya Sharma at Nexus Data, email priya.sharma@nexusdata.io, phone +1-555-0147.", fact_id="vendor")
    assistant("Got Priya's details saved. Is she our primary point of contact for both technical integration and contract questions?")

    # --- 54 turns of realistic but distracting work ---
    filler_exchanges = [
        ("Let's review the user story for the search feature.", "The search feature should support full-text search with fuzzy matching, filters by date range and category, and pagination. Elasticsearch would be the natural backend."),
        ("How should we handle rate limiting on the public API?", "Token bucket algorithm, 100 requests per minute per API key. Return 429 with a Retry-After header. Implement at the API gateway level so individual services don't need to handle it."),
        ("Draft a database migration for adding the tags table.", "CREATE TABLE tags (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), name VARCHAR(100) NOT NULL UNIQUE, created_at TIMESTAMPTZ DEFAULT NOW()); CREATE TABLE item_tags (item_id UUID REFERENCES items(id), tag_id UUID REFERENCES tags(id), PRIMARY KEY (item_id, tag_id));"),
        ("What's the best approach for handling file uploads over 100MB?", "Multipart upload to S3 with presigned URLs. The client requests a presigned URL from your API, uploads directly to S3, then notifies your API on completion. This keeps large files off your servers entirely."),
        ("Can you help me debug this N+1 query issue?", "The classic fix is eager loading. In SQLAlchemy, use joinedload() or subqueryload() on the relationship. Check your query count before and after — it should drop from N+1 to 2 queries."),
        ("Let's design the notification system.", "Event-driven architecture: services publish notification events to an SNS topic. A Lambda function consumes them, applies user preferences (email, push, in-app), and dispatches through the appropriate channel. Store notification state in DynamoDB for read receipts."),
        ("What caching strategy for the API responses?", "Three layers: CDN cache for public endpoints (5 min TTL), Redis for authenticated user data (2 min TTL), and application-level memoisation for expensive computations (30 sec TTL). Use cache-aside pattern — check cache first, compute on miss, write back."),
        ("Review this pull request for the payment integration.", "The payment flow looks correct but I have concerns: you're storing the full card number in the request log — that's a PCI violation. Also, the retry logic on failed charges should use exponential backoff with jitter, not fixed intervals."),
        ("How should we structure our microservices?", "Start with 4 services: API gateway, user service, core business logic, and notification service. Communicate via async events for anything that doesn't need an immediate response. Use synchronous HTTP only for user-facing request-response flows."),
        ("Let's plan the load testing approach.", "Use k6 for scripted load tests. Define three profiles: baseline (normal traffic), stress (3x normal), and spike (10x for 60 seconds). Run against a staging environment that mirrors production topology. Key metrics: response time percentiles, error rate, and resource utilisation."),
        ("Draft the API versioning strategy.", "URL-based versioning: /api/v1/, /api/v2/. Support the previous version for 6 months after a new version launches. Use feature flags within a version for gradual rollouts. Document breaking changes in a CHANGELOG."),
        ("What's the rollback procedure if a deployment fails?", "Automated rollback triggers: if error rate exceeds 5% or p99 latency exceeds 10 seconds within 5 minutes of deployment, automatically revert to the previous task definition in ECS. Manual rollback via a single CLI command that points the service back to the last known good image tag."),
        ("Help me optimise this slow database query.", "Add a composite index on (user_id, created_at DESC) — your WHERE clause filters by user_id and ORDER BY uses created_at. Also, the subquery can be rewritten as a lateral join which lets PostgreSQL use the index for both the filter and the sort."),
        ("Let's discuss the data warehouse architecture.", "Medallion architecture: bronze layer for raw ingested data, silver for cleaned and conformed, gold for business-level aggregates. Use dbt for transformations between layers. Schedule runs every 4 hours for silver, daily for gold. This gives you both fresh operational data and stable reporting tables."),
        ("What observability tools should we standardise on?", "OpenTelemetry for instrumentation — it's vendor-neutral so you can switch backends later. Grafana stack for visualisation: Loki for logs, Tempo for traces, Mimir for metrics. This avoids vendor lock-in and the Grafana stack is cost-effective at moderate scale."),
        ("Design the webhook delivery system.", "Store webhook registrations in PostgreSQL with URL, secret, and event filter. On event, enqueue a delivery job in SQS. The delivery worker signs the payload with HMAC-SHA256, sends it with a 5-second timeout, and retries up to 5 times with exponential backoff. Log every attempt for debugging."),
        ("How do we handle database connection pooling?", "PgBouncer in transaction mode, sitting between your application and PostgreSQL. Set pool_size to 2x the number of CPU cores on the database server. Each application instance connects to PgBouncer, not directly to PostgreSQL. This prevents connection exhaustion during traffic spikes."),
        ("Let's write the OpenAPI spec for the users endpoint.", "POST /api/v1/users — create user, returns 201 with user object. GET /api/v1/users/{id} — get user by ID, returns 200 or 404. PATCH /api/v1/users/{id} — partial update, returns 200. DELETE /api/v1/users/{id} — soft delete, returns 204. All endpoints require Bearer token in Authorization header."),
        ("What's our approach to secret management?", "AWS Secrets Manager for all credentials. Applications fetch secrets at startup and cache them in memory — never in config files or environment variables. Rotate database passwords every 90 days via Secrets Manager's built-in rotation Lambdas. For local development, use a .env file that's in .gitignore."),
        ("Plan the database backup and recovery strategy.", "Automated daily snapshots via RDS with 30-day retention. Enable point-in-time recovery for up to 35 days. Cross-region replication of snapshots to us-west-2 for disaster recovery. Test restore procedure monthly by spinning up a clone from the latest snapshot and running the integration test suite against it."),
        ("How should we handle feature flags?", "LaunchDarkly for user-facing features that need gradual rollout and targeting. Simple environment variables for infrastructure flags that change infrequently. Keep a registry of all flags in a YAML file in the repo with owner, creation date, and planned removal date. Review and clean up stale flags quarterly."),
        ("Design the audit logging system.", "Every mutation (create, update, delete) on sensitive resources gets an audit log entry: who, what, when, from where (IP), and the before/after state. Store in a separate PostgreSQL schema with append-only permissions. Retain for 7 years for compliance. Index on actor_id and resource_id for fast lookups."),
        ("Let's discuss the mobile API differences.", "The mobile API should be the same endpoints but with response compression enabled (gzip), smaller page sizes (20 vs 50 for desktop), and a sync endpoint that returns only changes since a timestamp. Add an API version header so you can deprecate mobile-specific workarounds independently."),
        ("What's the strategy for database schema migrations?", "Alembic for migration management, one migration file per change, never edit existing migrations. All migrations must be backward-compatible — no dropping columns or renaming tables in a single step. Use the expand-and-contract pattern: add new column, backfill, update code, then remove old column in a later release."),
        ("Help me set up the development environment.", "Docker Compose with PostgreSQL, Redis, Kafka, and the application services. A Makefile with targets: make setup (first time), make up (start services), make test (run tests), make seed (populate with sample data). Document the setup in a CONTRIBUTING.md with step-by-step instructions and troubleshooting for common issues."),
        ("Review the error handling middleware.", "The error handler should catch all unhandled exceptions, log the full traceback with the request correlation ID, and return a sanitised error response to the client. Never expose stack traces or internal details in production responses. Map known exceptions to appropriate HTTP status codes. Return a machine-readable error code alongside the human-readable message."),
        ("What's our approach to API documentation?", "Auto-generate OpenAPI specs from code annotations using FastAPI's built-in support. Host interactive docs at /docs (Swagger UI) and /redoc. Add example requests and responses for every endpoint. Maintain a separate architecture guide in Notion for high-level concepts that don't fit in API docs."),
    ]

    for q, a in filler_exchanges:
        user(q)
        assistant(a)

    probes = [
        Probe(
            fact_id="codename",
            query="What is the project codename we agreed on?",
            expected_keywords=["AURORA"],
            planted_at_turn=1,
        ),
        Probe(
            fact_id="encryption",
            query="What is our encryption key rotation schedule and what algorithm do we use?",
            expected_keywords=["72", "AES-256"],
            planted_at_turn=3,
        ),
        Probe(
            fact_id="vendor",
            query="Who is our vendor contact at Nexus Data and what is their email?",
            expected_keywords=["Priya", "nexusdata"],
            planted_at_turn=5,
        ),
    ]

    return FrozenScenario(
        name="long_session_decay",
        description="60-turn session. 3 facts planted in turns 1-5, probed after 54 distractor turns.",
        transcript=transcript,
        probes=probes,
    )


# ── Strategy Adapters ─────────────────────────────────────────────────

class StrategyAdapter:
    """Base class for context management strategies."""

    name: str

    def ingest_transcript(self, transcript: list[FrozenTurn]) -> None:
        """Feed the frozen conversation into the strategy."""
        raise NotImplementedError

    def build_probe_messages(self, probe: Probe) -> tuple[list[dict], int]:
        """Build the messages list for a probe question.
        Returns (messages, estimated_token_count)."""
        raise NotImplementedError


class FullContextAdapter(StrategyAdapter):
    """Baseline: send the entire conversation history."""
    name = "full_context"

    def __init__(self) -> None:
        self._messages: list[dict] = []

    def ingest_transcript(self, transcript: list[FrozenTurn]) -> None:
        self._messages = [
            {"role": t.role, "content": t.content} for t in transcript
        ]

    def build_probe_messages(self, probe: Probe) -> tuple[list[dict], int]:
        messages = self._messages + [{"role": "user", "content": probe.query}]
        tokens = sum(len(m["content"]) // 4 for m in messages)
        return messages, tokens


class SlidingWindowAdapter(StrategyAdapter):
    """Keep only the last N turns."""

    def __init__(self, window: int = 10) -> None:
        self._messages: list[dict] = []
        self._window = window
        self.name = f"sliding_window_{window}"

    def ingest_transcript(self, transcript: list[FrozenTurn]) -> None:
        self._messages = [
            {"role": t.role, "content": t.content} for t in transcript
        ]

    def build_probe_messages(self, probe: Probe) -> tuple[list[dict], int]:
        windowed = self._messages[-self._window * 2 :]
        messages = windowed + [{"role": "user", "content": probe.query}]
        tokens = sum(len(m["content"]) // 4 for m in messages)
        return messages, tokens


class ActiveMemoryAdapter(StrategyAdapter):
    """Uses the semantic B-tree for context assembly."""
    name = "active_memory"

    def __init__(self, embedder: Any, config: Any = None) -> None:
        from .btree import SemanticBTree, BTreeConfig
        from .scoring import Scorer, ScoringConfig
        from .assembler import ContextAssembler, AssemblerConfig
        from .types import estimate_tokens

        self._embedder = embedder
        self._scorer = Scorer(config.scoring if config else ScoringConfig())
        self._tree = SemanticBTree(
            embedder=embedder,
            scorer=self._scorer,
            config=config.btree if config else BTreeConfig(),
        )
        self._assembler = ContextAssembler(
            tree=self._tree,
            config=config.assembler if config else AssemblerConfig(),
        )
        self._messages: list[dict] = []

    def ingest_transcript(self, transcript: list[FrozenTurn]) -> None:
        import re

        self._messages = []
        for turn in transcript:
            self._messages.append({"role": turn.role, "content": turn.content})

            # Segment and insert into tree
            segments = re.split(r'(?<=[.!?])\s+', turn.content)
            buf = ""
            for seg in segments:
                buf += (" " if buf else "") + seg
                if len(buf) >= 60:
                    key = f"{turn.role}:{buf[:60]}"
                    self._tree.insert(key, buf)
                    buf = ""
            if buf and len(buf.strip()) >= 10:
                key = f"{turn.role}:{buf[:60]}"
                self._tree.insert(key, buf)

        # Run pruning to simulate a real session
        if self._tree.size > 0:
            self._tree.prune()
            self._tree.compress_cold_subtrees()

    def build_probe_messages(self, probe: Probe) -> tuple[list[dict], int]:
        query_emb = self._embedder.embed([probe.query])[0]
        assembled = self._assembler.assemble(self._messages, query_emb)
        messages = self._assembler.to_messages("You are a helpful assistant.", assembled)
        messages.append({"role": "user", "content": probe.query})
        return messages, assembled.total_tokens


# ── Evaluation Engine ─────────────────────────────────────────────────

@dataclass
class TrialResult:
    """Result of a single LLM call for one probe."""
    response_text: str
    keywords_found: dict[str, bool]    # keyword → found?
    passed: bool                       # all keywords found?
    tokens_in_prompt: int


@dataclass
class ProbeResult:
    """Aggregated results across multiple trials for one probe."""
    fact_id: str
    planted_at_turn: int
    query: str
    expected_keywords: list[str]
    trials: list[TrialResult]

    @property
    def recall_rate(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.passed) / len(self.trials)

    @property
    def avg_tokens(self) -> float:
        if not self.trials:
            return 0.0
        return sum(t.tokens_in_prompt for t in self.trials) / len(self.trials)


@dataclass
class StrategyResult:
    """Full results for one strategy on one scenario."""
    strategy: str
    scenario: str
    probe_results: list[ProbeResult]
    total_api_calls: int

    @property
    def overall_recall(self) -> float:
        if not self.probe_results:
            return 0.0
        return sum(p.recall_rate for p in self.probe_results) / len(self.probe_results)

    @property
    def avg_tokens_per_probe(self) -> float:
        if not self.probe_results:
            return 0.0
        return sum(p.avg_tokens for p in self.probe_results) / len(self.probe_results)

    def summary(self) -> dict:
        return {
            "strategy": self.strategy,
            "scenario": self.scenario,
            "overall_recall": f"{self.overall_recall:.0%}",
            "avg_tokens_per_probe": f"{self.avg_tokens_per_probe:,.0f}",
            "api_calls": self.total_api_calls,
            "per_fact": [
                {
                    "fact_id": p.fact_id,
                    "planted_turn": p.planted_at_turn,
                    "recall": f"{p.recall_rate:.0%}",
                    "avg_tokens": f"{p.avg_tokens:,.0f}",
                }
                for p in self.probe_results
            ],
        }


def _confidence_interval(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    centre = (rate + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((rate * (1 - rate) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


class ABEvalHarness:
    """Runs controlled A/B evaluations of context management strategies.

    Usage:
        from anthropic import Anthropic
        from active_memory import HashEmbedder
        from active_memory.eval import ABEvalHarness, build_architecture_review

        harness = ABEvalHarness(Anthropic())
        scenario = build_architecture_review()

        results = harness.run(
            scenario=scenario,
            strategies={
                "full_context": FullContextAdapter(),
                "sliding_window_10": SlidingWindowAdapter(10),
                "active_memory": ActiveMemoryAdapter(HashEmbedder(64)),
            },
            trials_per_probe=5,
        )

        harness.report(results)
    """

    def __init__(
        self,
        client: ModelClient,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def run(
        self,
        scenario: FrozenScenario,
        strategies: dict[str, StrategyAdapter],
        trials_per_probe: int = 5,
    ) -> list[StrategyResult]:
        """Run all strategies against the scenario.

        Each strategy:
          1. Ingests the identical frozen transcript
          2. Builds its own prompt for each probe
          3. Sends the probe to the LLM `trials_per_probe` times
          4. Checks keyword recall in each response
        """
        results: list[StrategyResult] = []

        for strat_name, adapter in strategies.items():
            print(f"\n  ▸ Strategy: {strat_name}")

            # Step 1: Ingest the same transcript
            adapter.ingest_transcript(scenario.transcript)

            probe_results: list[ProbeResult] = []
            total_calls = 0

            for probe in scenario.probes:
                # Step 2: Build the probe prompt
                messages, token_count = adapter.build_probe_messages(probe)

                # Step 3: Run N trials
                trials: list[TrialResult] = []
                for trial_idx in range(trials_per_probe):
                    response = self.client.generate(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=messages,
                    )
                    resp_text = response.text
                    total_calls += 1

                    # Step 4: Check keywords
                    kw_found = {
                        kw: kw.lower() in resp_text.lower()
                        for kw in probe.expected_keywords
                    }
                    passed = all(kw_found.values())

                    trials.append(TrialResult(
                        response_text=resp_text,
                        keywords_found=kw_found,
                        passed=passed,
                        tokens_in_prompt=token_count,
                    ))

                    status = "✓" if passed else "✗"
                    print(f"    {status} {probe.fact_id} trial {trial_idx+1}/{trials_per_probe}")

                probe_results.append(ProbeResult(
                    fact_id=probe.fact_id,
                    planted_at_turn=probe.planted_at_turn,
                    query=probe.query,
                    expected_keywords=probe.expected_keywords,
                    trials=trials,
                ))

            results.append(StrategyResult(
                strategy=strat_name,
                scenario=scenario.name,
                probe_results=probe_results,
                total_api_calls=total_calls,
            ))

        return results

    @staticmethod
    def report(results: list[StrategyResult]) -> str:
        """Print a comparative report with confidence intervals."""
        if not results:
            return ""

        scenario = results[0].scenario
        lines = [
            f"\n{'='*76}",
            f"  A/B EVALUATION REPORT: {scenario}",
            f"{'='*76}",
            "",
            f"  {'Strategy':<25} {'Recall':>8} {'95% CI':>16} {'Avg Tokens':>12} {'API Calls':>10}",
            f"  {'-'*25} {'-'*8} {'-'*16} {'-'*12} {'-'*10}",
        ]

        for r in results:
            # Compute aggregate CI
            total_trials = sum(len(p.trials) for p in r.probe_results)
            total_passed = sum(
                sum(1 for t in p.trials if t.passed) for p in r.probe_results
            )
            rate = total_passed / max(1, total_trials)
            ci_lo, ci_hi = _confidence_interval(rate, total_trials)

            lines.append(
                f"  {r.strategy:<25} {r.overall_recall:>7.0%} "
                f"[{ci_lo:>5.0%}, {ci_hi:>5.0%}]   "
                f"{r.avg_tokens_per_probe:>11,.0f} {r.total_api_calls:>10}"
            )

        # Per-fact breakdown
        lines.append(f"\n  Per-fact recall rates:")
        fact_ids = [p.fact_id for p in results[0].probe_results]

        header = f"  {'Fact':<20} {'Turn':>5}"
        for r in results:
            header += f" {r.strategy:>18}"
        lines.append(header)
        lines.append(f"  {'-'*20} {'-'*5}" + f" {'-'*18}" * len(results))

        for i, fact_id in enumerate(fact_ids):
            row = f"  {fact_id:<20} {results[0].probe_results[i].planted_at_turn:>5}"
            for r in results:
                pr = r.probe_results[i]
                row += f" {pr.recall_rate:>17.0%}"
            lines.append(row)

        # Token efficiency comparison
        if len(results) >= 2:
            baseline = results[0]  # assume first is full_context
            lines.append(f"\n  Token efficiency vs {baseline.strategy}:")
            for r in results[1:]:
                if baseline.avg_tokens_per_probe > 0:
                    ratio = r.avg_tokens_per_probe / baseline.avg_tokens_per_probe
                    savings = (1 - ratio) * 100
                    lines.append(
                        f"    {r.strategy}: {savings:+.0f}% tokens "
                        f"({r.avg_tokens_per_probe:,.0f} vs {baseline.avg_tokens_per_probe:,.0f})"
                    )

        output = "\n".join(lines)
        print(output)
        return output


# ── Quick-run helper ──────────────────────────────────────────────────

def _messages_to_text(messages: list[dict]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def run_prompt_benchmark(
    scenario: FrozenScenario,
    strategies: dict[str, StrategyAdapter],
) -> list[StrategyResult]:
    """Offline benchmark: score whether the built prompt still contains the
    expected keywords for each planted fact.

    This does not measure model generation quality, but it does provide a
    deterministic proxy for retrieval coverage and prompt size.
    """
    results: list[StrategyResult] = []

    for strat_name, adapter in strategies.items():
        adapter.ingest_transcript(scenario.transcript)
        probe_results: list[ProbeResult] = []

        for probe in scenario.probes:
            messages, token_count = adapter.build_probe_messages(probe)
            prompt_text = _messages_to_text(messages)
            kw_found = {
                kw: kw.lower() in prompt_text.lower()
                for kw in probe.expected_keywords
            }
            passed = all(kw_found.values())
            probe_results.append(ProbeResult(
                fact_id=probe.fact_id,
                planted_at_turn=probe.planted_at_turn,
                query=probe.query,
                expected_keywords=probe.expected_keywords,
                trials=[TrialResult(
                    response_text=prompt_text,
                    keywords_found=kw_found,
                    passed=passed,
                    tokens_in_prompt=token_count,
                )],
            ))

        results.append(StrategyResult(
            strategy=strat_name,
            scenario=scenario.name,
            probe_results=probe_results,
            total_api_calls=0,
        ))

    return results

def run_ab_eval(client: ModelClient, embedder: Any, trials: int = 3) -> list[StrategyResult]:
    """One-line entry point to run the full A/B evaluation.

    Example:
        from anthropic import Anthropic
        from active_memory import HashEmbedder
        from active_memory.eval import run_ab_eval

        results = run_ab_eval(Anthropic(), HashEmbedder(dim=64), trials=5)
    """
    harness = ABEvalHarness(client)
    all_results: list[StrategyResult] = []

    scenarios = [
        build_architecture_review(),
        build_long_session_decay(),
    ]

    for scenario in scenarios:
        print(f"\n{'━'*76}")
        print(f"  Scenario: {scenario.name}")
        print(f"  {scenario.description}")
        print(f"{'━'*76}")

        strategies = {
            "full_context": FullContextAdapter(),
            "sliding_window_10": SlidingWindowAdapter(10),
            "sliding_window_20": SlidingWindowAdapter(20),
            "active_memory": ActiveMemoryAdapter(embedder),
        }

        results = harness.run(scenario, strategies, trials_per_probe=trials)
        harness.report(results)
        all_results.extend(results)

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate active-memory retrieval quality and token efficiency",
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "live"],
        default="offline",
        help="offline = deterministic prompt coverage benchmark, live = Anthropic A/B eval",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="Model provider for live eval mode",
    )
    parser.add_argument(
        "--scenario",
        choices=["architecture_review", "long_session_decay", "all"],
        default="all",
        help="Which benchmark scenario to run",
    )
    parser.add_argument(
        "--embedder",
        choices=["auto", "hash", "openai"],
        default="auto",
        help="Embedding provider for active-memory strategy",
    )
    parser.add_argument(
        "--embed-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model when using auto/openai",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=64,
        help="Embedding dimension for hash embedder fallback",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Trials per probe for live eval mode",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model name for live eval mode",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary after the text report",
    )
    args = parser.parse_args()

    from .embeddings import create_embedder

    embedder_spec = create_embedder(
        args.embedder,
        dim=args.embed_dim,
        openai_model=args.embed_model,
        verbose=True,
        stream=sys.stderr,
    )

    scenario_builders = {
        "architecture_review": build_architecture_review,
        "long_session_decay": build_long_session_decay,
    }
    selected = (
        list(scenario_builders.values())
        if args.scenario == "all"
        else [scenario_builders[args.scenario]]
    )

    all_results: list[StrategyResult] = []

    if args.mode == "live":
        harness = ABEvalHarness(create_model_client(args.provider), model=args.model)
        for builder in selected:
            scenario = builder()
            strategies = {
                "full_context": FullContextAdapter(),
                "sliding_window_10": SlidingWindowAdapter(10),
                "sliding_window_20": SlidingWindowAdapter(20),
                "active_memory": ActiveMemoryAdapter(embedder_spec.embedder),
            }
            results = harness.run(scenario, strategies, trials_per_probe=args.trials)
            harness.report(results)
            all_results.extend(results)
    else:
        for builder in selected:
            scenario = builder()
            strategies = {
                "full_context": FullContextAdapter(),
                "sliding_window_10": SlidingWindowAdapter(10),
                "sliding_window_20": SlidingWindowAdapter(20),
                "active_memory": ActiveMemoryAdapter(embedder_spec.embedder),
            }
            results = run_prompt_benchmark(scenario, strategies)
            ABEvalHarness.report(results)
            all_results.extend(results)

    if args.json:
        print(json.dumps([r.summary() for r in all_results], indent=2))


if __name__ == "__main__":
    main()
