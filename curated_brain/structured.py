"""Structured tier — a bi-temporal atomic-fact store (PRD §5.2, Pillar A).

Holds ``(subject, predicate, object)`` triples, each carrying **valid time** (when the
fact is true in the world) and **transaction time** (when the system recorded/invalidated
it), plus provenance. It answers exact, relational (multi-hop) and as-of-time queries.

Superseding a fact is **non-lossy** (PRD §8, Zep/Graphiti model): the old record's
intervals are closed and it is linked to its replacement — never hard-deleted — so the
full audit trail and provenance survive.

This is the in-memory reference implementation behind the tier interface; it is trivially
byte-serializable for deterministic eval. A SQLite backend (PRD §5.3) is a drop-in swap.
"""

from __future__ import annotations

from curated_brain.models import INF, Fact
from curated_brain.util import normalize


class StructuredTier:
    def __init__(self) -> None:
        self.facts: list[Fact] = []

    # ------------------------------------------------------------------ write path ---
    def assert_fact(self, *, fact_id: str, subject: str, predicate: str, object: str,
                    valid_from: float, created_at: float, confidence: float = 1.0,
                    provenance: dict | None = None) -> tuple[Fact, Fact | None]:
        """Insert a fact, superseding any conflicting open fact for (subject, predicate).

        Returns ``(record, superseded)`` where ``record`` is the fact now considered
        current (an existing one if the value was merely reasserted) and ``superseded``
        is the fact that was closed, if any.
        """
        key = (normalize(subject), normalize(predicate))
        superseded: Fact | None = None
        for f in self.facts:
            if f.is_open and (normalize(f.subject), normalize(f.predicate)) == key:
                if normalize(f.object) == normalize(object):
                    return f, None  # same value reasserted — no new row
                # Conflict: close the old interval (valid + transaction time) and link.
                f.valid_to = valid_from
                f.expired_at = created_at
                f.superseded_by = fact_id
                superseded = f
                break
        new = Fact(
            id=fact_id, subject=subject, predicate=predicate, object=object,
            valid_from=valid_from, valid_to=INF, created_at=created_at, expired_at=INF,
            confidence=confidence, provenance=provenance or {},
        )
        self.facts.append(new)
        return new, superseded

    # ------------------------------------------------------------------ read path ----
    def current(self, subject: str, predicate: str) -> Fact | None:
        """The current (open, non-superseded) fact for (subject, predicate).

        The write path keeps a single open fact per key, but we select by latest
        ``(valid_from, created_at)`` anyway so this stays consistent with ``as_of`` even
        if a restored/corrupt store ever violates that invariant.
        """
        key = (normalize(subject), normalize(predicate))
        open_matches = [
            f for f in self.facts
            if f.is_open and (normalize(f.subject), normalize(f.predicate)) == key
        ]
        if not open_matches:
            return None
        return max(open_matches, key=lambda f: (f.valid_from, f.created_at))

    def as_of(self, subject: str, predicate: str, t: float) -> Fact | None:
        """The fact believed at time ``t`` — valid at ``t`` *and* recorded by ``t``.

        This is a true bi-temporal point query ("what did we believe on date D"):
        ``valid_from <= t < valid_to`` and ``created_at <= t < expired_at``.
        """
        key = (normalize(subject), normalize(predicate))
        cands = [
            f for f in self.facts
            if (normalize(f.subject), normalize(f.predicate)) == key
            and f.valid_from <= t < f.valid_to
            and f.created_at <= t < f.expired_at
        ]
        if not cands:
            return None
        return max(cands, key=lambda f: (f.valid_from, f.created_at))

    def resolve(self, subject: str, predicate: str, t: float | None = None) -> Fact | None:
        return self.current(subject, predicate) if t is None else self.as_of(subject, predicate, t)

    def resolve_path(self, subject: str, predicates: list[str],
                     t: float | None = None) -> Fact | None:
        """Multi-hop traversal: resolve each predicate in turn, threading the object
        forward as the next subject. e.g. ["manager", "city"] -> X's manager's city."""
        cur, last = subject, None
        for pred in predicates:
            last = self.resolve(cur, pred, t)
            if last is None:
                return None
            cur = last.object
        return last

    def history(self, subject: str, predicate: str) -> list[Fact]:
        """All facts (open + superseded) for (subject, predicate), oldest first."""
        key = (normalize(subject), normalize(predicate))
        out = [f for f in self.facts
               if (normalize(f.subject), normalize(f.predicate)) == key]
        return sorted(out, key=lambda f: (f.valid_from, f.created_at))

    @property
    def open_facts(self) -> list[Fact]:
        return [f for f in self.facts if f.is_open]

    def predicates_for(self, subject: str) -> list[str]:
        """Distinct predicates with a currently-open fact for ``subject`` (sorted, so the
        order is deterministic regardless of insertion order)."""
        key = normalize(subject)
        preds = {f.predicate for f in self.facts
                 if f.is_open and normalize(f.subject) == key}
        return sorted(preds)

    # ------------------------------------------------------------------ persistence --
    def to_dict(self) -> list[dict]:
        return [vars(f) for f in self.facts]

    def load(self, rows: list[dict]) -> None:
        self.facts = [Fact(**d) for d in rows]
