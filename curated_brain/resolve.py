"""Conservative, deterministic entity resolution — canonicalize subject surface forms.

Resolves variants of the SAME entity ("Erin", "Erin Smith", "Ms. Smith") to one canonical
subject, WITHOUT ever merging two distinct entities — a false merge is worse than not
resolving. There is NO fuzzy / edit-distance / nickname / embedding matching: only honorific
stripping + EXACT component-token subsumption, every merge gated on store-provable uniqueness
and refused on the slightest ambiguity. Subject-only (predicates and object values are
untouched, so the bi-temporal / supersede / stale keys are unchanged). A pure function of the
surfaces seen — no wall-clock, no random — so the store stays byte-deterministic (AC-1).

It is a provable NO-OP on single-token, honorific-free subjects (e.g. the synthetic harness's
"Alice".."Heidi"): such a subject never registers a component alias and never promotes, so
``resolve_and_register(name) == normalize(name)`` and ``entities`` is unchanged — preserving AC-9.
"""

from __future__ import annotations

from curated_brain.util import normalize

_HONORIFICS = frozenset({"mr", "mrs", "ms", "miss", "dr", "prof", "mx"})
_AMBIGUOUS = object()  # sentinel: a component token claimed by >= 2 distinct full names


def strip_honorific(s: str) -> str:
    """Drop one leading honorific token ("ms.", "dr") iff present and >= 1 token remains."""
    parts = s.split()
    if len(parts) > 1 and parts[0].rstrip(".") in _HONORIFICS:
        return " ".join(parts[1:])
    return s


class EntityResolver:
    """Canonicalize a subject surface to a stable key, learning as it registers. Conservative:
    in EVERY ambiguous case it returns the input unchanged (kept as a separate entity)."""

    def __init__(self) -> None:
        self._full: set[str] = set()        # registered multi-token canonical names
        self._given: dict[str, object] = {}   # first token -> unique full name or _AMBIGUOUS
        self._surname: dict[str, object] = {}  # last token  -> unique full name or _AMBIGUOUS
        self._singletons: set[str] = set()   # standalone single-token entities

    def resolve_and_register(self, surface: str) -> str:
        s = strip_honorific(normalize(surface))
        if s in self._full:
            return s
        if " " in s:  # multi-token full name: register it + its (given, surname) components
            self._full.add(s)
            toks = s.split()
            self._register_component(self._given, toks[0], s)
            self._register_component(self._surname, toks[-1], s)
            return s
        promoted = self._promote(s)  # single token: promote to a unique full name, or stand alone
        if promoted is not None:
            return promoted
        self._singletons.add(s)
        return s

    @staticmethod
    def _register_component(index: dict, token: str, full: str) -> None:
        cur = index.get(token)
        if cur is None:
            index[token] = full
        elif cur != full:        # a second, DIFFERENT full name claims this token -> poison it
            index[token] = _AMBIGUOUS

    def _promote(self, s: str) -> str | None:
        """A single token promotes to a UNIQUE full name unless it is (or ever was) a standalone
        entity. Considers BOTH the given and surname indices together (a token that is one
        person's given name AND a different person's surname is ambiguous ACROSS indices, so it
        must refuse — not just within one index). Returns the full name, or None to refuse."""
        if s in self._singletons:
            return None
        cands: set[str] = set()
        for index in (self._given, self._surname):
            full = index.get(s)
            if full is _AMBIGUOUS:
                return None  # ambiguous in this role -> refuse (never promote via the other role)
            if isinstance(full, str):  # the only other stored value is _AMBIGUOUS (handled above)
                cands.add(full)
        return next(iter(cands)) if len(cands) == 1 else None  # exactly one distinct -> promote

    def register_canonical(self, subject: str) -> None:
        """Register an ALREADY-canonical stored subject (the restore path). A single-token
        stored subject is a standalone entity (so it must never re-promote); a multi-token one
        is a full name. This faithfully rebuilds the resolver from the facts in any order,
        keeping the snapshot schema unchanged (the resolver is derived, never serialized)."""
        s = normalize(subject)
        if " " in s:
            self._full.add(s)
            toks = s.split()
            self._register_component(self._given, toks[0], s)
            self._register_component(self._surname, toks[-1], s)
        else:
            self._singletons.add(s)

    def canonical(self, surface: str) -> str:
        """Read-only resolve (no registration) — used on the query path."""
        s = strip_honorific(normalize(surface))
        if s in self._full:
            return s
        if " " not in s:
            promoted = self._promote(s)
            if promoted is not None:
                return promoted
        return s

    def forget(self, name: str) -> None:
        """Drop an entity from the vocabulary (the erasure path). Component-index entries
        pointing at the forgotten full name are removed; entries poisoned as ambiguous stay
        poisoned — forgetting one of two homonym entities must not let the survivor silently
        claim the shared token that was refused while both existed (conservative-by-design)."""
        s = normalize(name)
        self._full.discard(s)
        self._singletons.discard(s)
        for index in (self._given, self._surname):
            for tok in [t for t, v in index.items() if v == s]:
                del index[tok]

    # ------------------------------------------------------------------ persistence --
    def to_dict(self) -> dict:
        """Full learned state, JSON-able and byte-stable (sorted). The ambiguity history
        (_AMBIGUOUS poisons, standalone singletons) is NOT derivable from the final canonical
        subjects alone — rebuilding from facts let a token refused before snapshot promote
        after restore, i.e. the same query answered differently across a restart."""
        def enc(index: dict) -> list:
            return sorted((k, None if v is _AMBIGUOUS else v) for k, v in index.items())
        return {"full": sorted(self._full), "given": enc(self._given),
                "surname": enc(self._surname), "singletons": sorted(self._singletons)}

    @classmethod
    def from_dict(cls, d: dict) -> EntityResolver:
        # Untrusted snapshot: fail with a clear error, not an opaque AttributeError/TypeError.
        if not isinstance(d, dict):
            raise ValueError(f"resolver snapshot must be an object, got {type(d).__name__}")
        r = cls()
        r._full = set(d.get("full", []))
        r._singletons = set(d.get("singletons", []))
        for attr, key in (("_given", "given"), ("_surname", "surname")):
            index = getattr(r, attr)
            entries = d.get(key, [])
            if not isinstance(entries, list):
                raise ValueError(f"resolver '{key}' must be a list")
            for pair in entries:
                try:
                    tok, val = pair
                except (TypeError, ValueError) as e:
                    raise ValueError(f"malformed resolver '{key}' entry: {e}") from e
                index[tok] = _AMBIGUOUS if val is None else val
        return r

    @property
    def entities(self) -> set[str]:
        """The vocabulary the planner matches questions against: full names + standalone single
        tokens + each component token that promotes to a UNIQUE full name (so a question saying
        'erin' or 'smith' finds the entity, which :meth:`canonical` then maps to the full key).
        A cross-index-ambiguous token is excluded — the planner is never offered a token that
        `canonical` would refuse to resolve."""
        out = set(self._full) | set(self._singletons)
        for tok in set(self._given) | set(self._surname):
            if tok not in out and self._promote(tok) is not None:
                out.add(tok)
        return out


__all__ = ["EntityResolver", "strip_honorific"]
