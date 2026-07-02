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
        # (norm subject, norm predicate) -> facts, holding the SAME Fact objects as `facts`
        # (so in-place supersede stays consistent). Derived, not serialized — `to_dict` emits
        # only `facts`, so snapshots stay byte-identical (AC-1). Turns per-key reads O(1).
        self._by_key: dict[tuple[str, str], list[Fact]] = {}
        # (norm predicate, norm object) -> facts: the INVERSE index, so set queries ("who
        # lives in Berlin?") don't scan every fact. Same shared-object discipline as _by_key.
        self._by_obj: dict[tuple[str, str], list[Fact]] = {}

    def _index(self, f: Fact) -> None:
        self._by_key.setdefault((normalize(f.subject), normalize(f.predicate)), []).append(f)
        self._by_obj.setdefault((normalize(f.predicate), normalize(f.object)), []).append(f)

    # ------------------------------------------------------------------ write path ---
    def assert_fact(self, *, fact_id: str, subject: str, predicate: str, object: str,
                    valid_from: float, created_at: float, confidence: float = 1.0,
                    provenance: dict | None = None) -> tuple[Fact, Fact | None]:
        """Insert a fact, superseding any conflicting open fact for (subject, predicate).

        Returns ``(record, superseded)`` where ``record`` is the fact now considered
        current (an existing one if the value was merely reasserted) and ``superseded``
        is the fact that was closed, if any.

        Out-of-order guard: a newcomer whose ``valid_from`` PRECEDES the open fact's is a
        retroactive/historical assertion, not an update — it is inserted with its valid
        interval closed at the open fact's ``valid_from`` instead of superseding forward.
        (Previously it closed the open fact with ``valid_to < valid_from`` — an inverted
        interval that silently corrupted every subsequent ``as_of`` query.)
        """
        key = (normalize(subject), normalize(predicate))
        superseded: Fact | None = None
        for f in self._by_key.get(key, []):  # only facts for this key (was: scan all facts)
            if f.is_open:
                if normalize(f.object) == normalize(object):
                    return f, None  # same value reasserted — no new row
                if valid_from < f.valid_from:
                    # Historical fact arriving late: record it as already-superseded by the
                    # open fact (closed valid interval, open transaction interval — we only
                    # learned it now, but it stopped being true when the open fact began).
                    hist = Fact(
                        id=fact_id, subject=subject, predicate=predicate, object=object,
                        valid_from=valid_from, valid_to=f.valid_from,
                        created_at=created_at, expired_at=INF,
                        confidence=confidence, provenance=provenance or {},
                        superseded_by=f.id,
                    )
                    self.facts.append(hist)
                    self._index(hist)
                    return f, hist  # the open fact stays current; the newcomer is history
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
        self._index(new)
        return new, superseded

    # ------------------------------------------------------------------ read path ----
    def current(self, subject: str, predicate: str) -> Fact | None:
        """The current (open, non-superseded) fact for (subject, predicate).

        The write path keeps a single open fact per key, but we select by latest
        ``(valid_from, created_at)`` anyway so this stays consistent with ``as_of`` even
        if a restored/corrupt store ever violates that invariant.
        """
        key = (normalize(subject), normalize(predicate))
        open_matches = [f for f in self._by_key.get(key, []) if f.is_open]
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
            f for f in self._by_key.get(key, [])
            if f.valid_from <= t < f.valid_to and f.created_at <= t < f.expired_at
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

    def resolve_path_chain(self, subject: str, predicates: list[str],
                           t: float | None = None) -> list[Fact] | None:
        """Like :meth:`resolve_path` but returns EVERY fact traversed along the chain (so each
        hop's provenance can be surfaced for attribution), or ``None`` if any hop is unresolved."""
        cur, chain = subject, []
        for pred in predicates:
            f = self.resolve(cur, pred, t)
            if f is None:
                return None
            chain.append(f)
            cur = f.object
        return chain

    def history(self, subject: str, predicate: str) -> list[Fact]:
        """All facts (open + superseded) for (subject, predicate), oldest first."""
        key = (normalize(subject), normalize(predicate))
        return sorted(self._by_key.get(key, []), key=lambda f: (f.valid_from, f.created_at))

    @property
    def open_facts(self) -> list[Fact]:
        return [f for f in self.facts if f.is_open]

    def subjects_where(self, predicate: str, object: str, t: float | None = None) -> list[str]:
        """Inverse / set query: every subject for which ``(subject, predicate, object)`` is
        currently true (or true as believed at ``t``) — "who lives in Berlin?",
        "who reports to Bob?". Sorted, distinct, O(matches) via the inverse index."""
        cands = self._by_obj.get((normalize(predicate), normalize(object)), [])
        if t is None:
            hits = [f for f in cands if f.is_open]
        else:
            hits = [f for f in cands
                    if f.valid_from <= t < f.valid_to and f.created_at <= t < f.expired_at]
        return sorted({f.subject for f in hits})

    def forget(self, subject: str, predicate: str | None = None, *,
               as_object: bool = False) -> int:
        """Hard-retract facts (the deletion path — e.g. a GDPR erasure request). Removes
        every fact (open AND superseded history) whose subject matches — and, with
        ``as_object``, every fact whose *object* names the subject (other entities' facts
        that embed this entity's data). Returns the number of facts removed.

        This is the ONE deliberate exception to the never-hard-delete invariant (PRD §8):
        supersede preserves history; ``forget`` erases it, on explicit request only."""
        ns = normalize(subject)
        np_ = normalize(predicate) if predicate is not None else None

        def doomed(f: Fact) -> bool:
            if normalize(f.subject) == ns and (np_ is None or normalize(f.predicate) == np_):
                return True
            return as_object and np_ is None and normalize(f.object) == ns

        removed = sum(1 for f in self.facts if doomed(f))
        if removed:
            self.facts = [f for f in self.facts if not doomed(f)]
            self._by_key, self._by_obj = {}, {}
            for f in self.facts:
                self._index(f)
        return removed

    def predicates_for(self, subject: str) -> list[str]:
        """Distinct predicates with a currently-open fact for ``subject`` (sorted, so the
        order is deterministic regardless of insertion order)."""
        skey = normalize(subject)
        preds = {f.predicate for (subj, _pred), facts in self._by_key.items() if subj == skey
                 for f in facts if f.is_open}
        return sorted(preds)

    # ------------------------------------------------------------------ persistence --
    def to_dict(self) -> list[dict]:
        return [vars(f) for f in self.facts]

    def load(self, rows: list[dict]) -> None:
        self.facts = [Fact(**d) for d in rows]
        self._by_key, self._by_obj = {}, {}
        for f in self.facts:
            self._index(f)
