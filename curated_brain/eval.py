"""Shared, backend-agnostic answer reader used to score retrieval (PRD §10).

The same reader is applied to every backend's returned context, so the *only* variable is
context quality — exactly the curation thesis. The reader is deliberately naive: it scans
the returned context top-ranked line first and returns the first known candidate value
that appears and is not already in the question (so it extracts the answer, not the
subject). A well-curated context (one clean current/as-of fact) is read correctly; a noisy
or stale-laden context misleads it.
"""

from __future__ import annotations

from curated_brain.dataset import Dataset, Probe
from curated_brain.util import normalize


def candidates_for(ds: Dataset, probe: Probe) -> list[str]:
    """The closed set of plausible answer values for a probe's (final) predicate."""
    pred = probe.hops[-1] if probe.hops else probe.predicate
    vals: set[str] = set()
    for person in ds.people:
        if pred == "city":
            vals |= {person["init_city"], person["new_city"]}
        elif pred == "role":
            vals |= {person["init_role"], person["new_role"]}
        elif pred == "email":
            vals.add(person["email"])
        elif pred == "manager":
            vals.add(person["manager"])
            vals.add(person["name"])
        elif pred == "project":
            vals.add(person["project"])
    return sorted(vals)


def extract_value(context: str, candidates: list[str], question: str = "") -> str:
    """First candidate value appearing in the context (top line first), skipping any
    value already named in the question. Longer candidates win ties (email > city)."""
    q = question.lower()
    cands = sorted(candidates, key=len, reverse=True)
    for line in context.splitlines():
        low = line.lower()
        for c in cands:
            cl = c.lower()
            if cl in low and cl not in q:
                return c
    return ""


def correct(predicted: str, gold: str) -> bool:
    return normalize(predicted) == normalize(gold)


def accuracy(pairs: list[tuple[str, str]]) -> float:
    """Fraction correct over (predicted, gold) pairs."""
    if not pairs:
        return 0.0
    return sum(1 for p, g in pairs if correct(p, g)) / len(pairs)
