"""Logical data model (PRD §5.2) and adapter I/O records (PRD §9).

Time is bi-temporal everywhere it matters:

* **valid time** (`valid_from`, `valid_to`) — when a fact is true *in the world*.
* **transaction time** (`created_at`, `expired_at`) — when the system *recorded/invalidated* it.

Superseding a fact never hard-deletes: it closes the old interval and links the
replacement, so provenance and history survive (PRD §5.2, §8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

INF = float("inf")


@dataclass
class EpisodicRecord:
    """An append-only raw observation that cleared the surprise gate (the hippocampus)."""

    id: str
    session_id: str
    seq: int
    wall_ts: float
    actor: str
    content: str
    embed_model_id: str
    surprise: float
    provenance: dict
    support_count: int = 1
    last_seen_ts: float = 0.0
    tier: str = "episodic"  # "episodic" | "semantic"
    supports: list[str] = field(default_factory=list)  # episode ids a semantic claim covers
    # Link to the (normalized) triple this episode asserts. Stored as a 3-item list — the
    # old "subject|predicate|object" string crashed consolidate() on any value containing
    # "|". Legacy string snapshots are still accepted on restore (see backend._fact_key_parts).
    fact_key: list[str] | str | None = None


@dataclass
class Fact:
    """A bi-temporal atomic fact / relation ``(subject, predicate, object)`` (PRD §5.2)."""

    id: str
    subject: str
    predicate: str
    object: str
    valid_from: float
    valid_to: float
    created_at: float
    expired_at: float
    confidence: float
    provenance: dict
    superseded_by: str | None = None

    @property
    def is_open(self) -> bool:
        """True while the fact is still the current, non-superseded record."""
        return self.expired_at == INF and self.valid_to == INF


@dataclass
class WriteReceipt:
    stored: bool
    reason: str  # "stored" | "reinforced" | "discarded"
    record_id: str | None
    surprise: float


@dataclass
class Citation:
    record_id: str
    provenance: dict
    valid_interval: tuple[float, float]


@dataclass
class Retrieval:
    context: str
    citations: list[Citation]
    tokens_in: int


@dataclass
class ConsolidationReport:
    episodes_in: int
    claims_out: int
    dupes_merged: int
    contradictions_resolved: int
    pruned: int


@dataclass
class StoreStats:
    episodic_count: int
    structured_count: int
    semantic_count: int
    bytes: int
    embed_model_id: str
