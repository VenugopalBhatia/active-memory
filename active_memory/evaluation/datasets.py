"""Hand-authored retrieval cases covering the documented failure modes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MemoryFixture:
    id: str
    content: str
    memory_type: str = "fact"
    trust: str = "user_confirmed"
    age_days: int = 0
    inclusions: int = 0
    relevant: bool = False
    status: str = "active"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    name: str
    category: str
    query: str
    memories: tuple[MemoryFixture, ...]
    edges: tuple[tuple[str, str, str, float], ...] = ()

    @property
    def relevant_ids(self) -> set[str]:
        return {memory.id for memory in self.memories if memory.relevant and memory.status == "active"}


def benchmark_cases() -> list[EvaluationCase]:
    filler = (
        MemoryFixture("noise_ui", "Review onboarding colors and typography.", age_days=1),
        MemoryFixture("noise_budget", "Finance approved the quarterly travel budget.", age_days=2),
        MemoryFixture("noise_deploy", "The staging deployment uses two replicas.", age_days=3),
    )
    return [
        EvaluationCase("long_fact", "long-distance fact recall", "Which database does the project use?", (MemoryFixture("db", "The project uses PostgreSQL for persistent storage.", "decision", age_days=120, relevant=True), *filler)),
        EvaluationCase("changed_decision", "changing decisions", "Which database is current?", (MemoryFixture("mongo", "The project uses MongoDB.", "decision", age_days=60, status="superseded"), MemoryFixture("postgres", "The project migrated to PostgreSQL.", "decision", age_days=2, relevant=True), *filler)),
        EvaluationCase("conflicting_trust", "conflicting facts", "Why did authentication fail?", (MemoryFixture("guess", "Authentication may have failed because of the database.", trust="assistant_generated", age_days=1), MemoryFixture("cause", "Authentication failed because the token fixture expired.", "resolution", "user_confirmed", age_days=2, relevant=True), *filler)),
        EvaluationCase("cross_session", "cross-session memory", "What is the API pagination limit?", (MemoryFixture("pagination", "The API pagination limit is 200 records.", age_days=45, relevant=True), *filler)),
        EvaluationCase("code_affinity", "related code entities", "How does authentication token validation work?", (MemoryFixture("auth_seed", "Authentication token validation starts in verify_token.", "code_change", age_days=5, relevant=True, metadata={"file_path": "auth.py"}), MemoryFixture("helper", "parse_claims checks issuer and expiry.", "code_change", age_days=5, relevant=True, metadata={"file_path": "auth.py"}), *filler), (("auth_seed", "helper", "same_file", 0.75),)),
        EvaluationCase("unresolved_task", "unresolved tasks", "What authentication work remains?", (MemoryFixture("task", "TODO: rotate the authentication signing keys.", "task", age_days=10, relevant=True), *filler)),
        EvaluationCase("irrelevant_repetition", "repeated irrelevant content", "What database was selected?", (MemoryFixture("db2", "PostgreSQL was selected as the database.", "decision", age_days=30, relevant=True), MemoryFixture("repeat", "Discuss onboarding colors.", inclusions=100), *filler)),
        EvaluationCase("assistant_correction", "assistant hallucination followed by correction", "Where is the service deployed?", (MemoryFixture("hallucination", "The service may be deployed to Azure.", trust="assistant_generated"), MemoryFixture("correction", "Correction: the service is deployed to AWS.", relevant=True), *filler)),
        EvaluationCase("large_tool", "large tool outputs", "Which test file failed?", (MemoryFixture("tool", "Tests failed in auth/test_tokens.py.", "tool_observation", "tool_observed", relevant=True), MemoryFixture("tool_noise", "pytest output " * 100, "summary"), *filler)),
        EvaluationCase("privacy", "privacy-sensitive content", "What credential was stored?", (*filler,)),
        EvaluationCase("expired", "facts that should expire", "Which feature flag is active?", (MemoryFixture("expired_flag", "Feature flag legacy_auth is active.", status="superseded"), MemoryFixture("current_flag", "Feature flag oauth_v2 is active.", relevant=True), *filler)),
        EvaluationCase("preference", "stable user preferences", "What response style does the user prefer?", (MemoryFixture("preference", "The user prefers concise engineering answers.", "preference", age_days=500, relevant=True), *filler)),
    ]

