"""Hybrid retrieval — planner, fusion re-rank, supersede-filtering (PRD §7).

1. **Plan.** Classify the query against the *known* entity/predicate vocabulary (built
   from what was actually stored), detecting entity, predicate, multi-hop chains and
   as-of-time intent. No brittle full-sentence parsing — we match what we know.
2. **Fetch.** Exact/relational/as-of from the structured tier; top-k from the vector tier.
3. **Fuse & re-rank.** Score vector candidates by relevance × recency × importance
   (the Generative Agents weighting) and **drop superseded values** so stale facts never
   surface. The exact structured fact is surfaced first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from curated_brain.models import Fact
from curated_brain.util import tokenize

# Keyword -> predicate. Matched against the question's token set.
PRED_KEYWORDS: dict[str, list[str]] = {
    "email": ["email", "e-mail"],
    "city": ["city", "live", "lives", "living", "located", "location", "reside", "moved"],
    "role": ["role", "job", "position", "works", "promoted", "title"],
    "project": ["project"],
    "manager": ["manager", "manages", "manage", "reports", "boss"],
}
_RELATION_PREDS = {"manager"}
_SESSION_RE = re.compile(r"session\s+(\d+)")
_ASOF_RE = re.compile(r"as[-\s]of|believ|back then|at the time")

HALF_LIFE_SECONDS = 30 * 86_400.0  # recency decays with a 30-day half-life


@dataclass
class QueryPlan:
    entity: str | None
    predicate: str | None
    hops: list[str] | None
    as_of: float | None
    open_ended: bool


class Planner:
    def plan(self, question: str, *, entities: set[str],
             predicates: frozenset[str] = frozenset(),
             relation_preds: frozenset[str] = frozenset(),
             session_ts: dict[int, float]) -> QueryPlan:
        toks = set(tokenize(question, drop_stop=False))
        entity = next((e for e in sorted(entities) if e in toks), None)

        # Schema-driven FIRST: a predicate ACTUALLY STORED whose non-stop content tokens all
        # appear in the question is ground truth about what the store can answer, so it
        # outranks a keyword-mapped guess (e.g. stored "email address" beats the unstored
        # keyword predicate "email" — previously the guess won and the lookup missed).
        # Handles multi-word predicates; equivalent to single-token match for one-word ones.
        preds = [p for p in sorted(predicates)
                 if (pt := set(tokenize(p, drop_stop=True))) and pt <= toks]
        for p, kws in PRED_KEYWORDS.items():
            if p not in preds and any(k in toks for k in kws):
                preds.append(p)
        # A predicate is relational if hardwired OR if it was STORED with an entity-valued
        # object (``relation_preds``) — generalizing multi-hop beyond the "manager" relation to
        # any "X's <relation>'s <attr>" chain, without hardcoding a vocabulary.
        is_rel = _RELATION_PREDS | relation_preds
        rel = [p for p in preds if p in is_rel]
        attr = [p for p in preds if p not in is_rel]
        hops: list[str] | None = None
        predicate: str | None = None
        if rel and attr:  # "X's manager's city" -> traverse the relation then the attribute
            hops, predicate = [rel[0], attr[0]], attr[0]
        elif rel:
            predicate = rel[0]
        elif attr:
            predicate = attr[0]

        as_of: float | None = None
        ql = question.lower()
        m = _SESSION_RE.search(ql)
        if m and _ASOF_RE.search(ql):
            as_of = session_ts.get(int(m.group(1)))

        return QueryPlan(entity=entity, predicate=predicate, hops=hops, as_of=as_of,
                         open_ended=entity is None or predicate is None)


def render_fact(plan: QueryPlan, fact: Fact) -> str:
    """A compact, citation-ready statement of the resolved fact for the context payload."""
    if plan.hops:
        # render_fact is only called for a resolved (non-open-ended) plan, so entity is set.
        chain = " ".join([plan.entity or "", *plan.hops])
        return f"{chain} is {fact.object}."
    if plan.as_of is not None:
        return f"{fact.subject}'s {fact.predicate} as of that time was {fact.object}."
    return f"{fact.subject}'s current {fact.predicate} is {fact.object}."


@dataclass
class FusedItem:
    text: str
    rid: str
    provenance: dict
    valid_interval: tuple[float, float]
    score: float


def _recency(now: float, ts: float) -> float:
    return 0.5 ** (max(0.0, now - ts) / HALF_LIFE_SECONDS)


def fuse(vhits, *, now: float, stale_rids: set[str] = frozenset(),
         stale_pairs: list[tuple[frozenset[str], frozenset[str]]] = (), w_rel: float = 1.0,
         w_rec: float = 0.5, w_imp: float = 0.3, importance: float = 0.5) -> list[FusedItem]:
    """Rank vector candidates by relevance × recency × importance, dropping any record that
    states a superseded value (supersede-filtering, PRD §7 step 3).

    Two-level staleness (see ``CuratedBrain._stale_filters``): ``stale_rids`` drops records
    whose *asserted fact* is superseded (exact, provenance-linked); ``stale_pairs`` is the
    fallback for records with no fact link — dropped only when the text contains BOTH the
    subject and the full stale value (entity-scoped, so a record merely sharing words with
    some other entity's stale value is never filtered).
    The ``sim`` carried in from :meth:`VectorTier.search` is already the hybrid score."""
    items: list[FusedItem] = []
    for r, sim in vhits:
        if r.rid in stale_rids:
            continue
        rtoks = set(tokenize(r.text))
        if any(st <= rtoks and vt <= rtoks for st, vt in stale_pairs):
            continue
        score = w_rel * sim + w_rec * _recency(now, r.wall_ts) + w_imp * importance
        items.append(FusedItem(text=r.text, rid=r.rid, provenance={"session_id": r.session_id},
                               valid_interval=(r.wall_ts, float("inf")), score=score))
    items.sort(key=lambda it: (-it.score, it.rid))
    return items
