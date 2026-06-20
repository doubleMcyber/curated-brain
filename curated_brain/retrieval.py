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
             session_ts: dict[int, float]) -> QueryPlan:
        toks = set(tokenize(question, drop_stop=False))
        entity = next((e for e in sorted(entities) if e in toks), None)

        preds = [p for p, kws in PRED_KEYWORDS.items() if any(k in toks for k in kws)]
        # Schema-driven: also recognize any predicate ACTUALLY STORED when ALL of its
        # non-stop content tokens appear in the question. This lifts the planner past the 5
        # hardcoded vocab predicates AND handles multi-word predicates ("mailing address"),
        # so open-domain questions route precisely to the structured tier instead of falling
        # through to the backstop. Equivalent to a single-token match for one-word predicates.
        for p in sorted(predicates):
            ptoks = set(tokenize(p, drop_stop=True))
            if ptoks and ptoks <= toks and p not in preds:
                preds.append(p)
        rel = [p for p in preds if p in _RELATION_PREDS]
        attr = [p for p in preds if p not in _RELATION_PREDS]
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
        chain = " ".join([plan.entity, *plan.hops])
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


def fuse(vhits, *, now: float, stale_token_sets: list[frozenset[str]], w_rel: float = 1.0,
         w_rec: float = 0.5, w_imp: float = 0.3, importance: float = 0.5) -> list[FusedItem]:
    """Rank vector candidates by relevance × recency × importance, dropping any record that
    states a superseded value (supersede-filtering, PRD §7 step 3). A record is stale when it
    contains **all** tokens of some superseded value — so multi-word stale values are caught.
    The ``sim`` carried in from :meth:`VectorTier.search` is already the hybrid score."""
    items: list[FusedItem] = []
    for r, sim in vhits:
        rtoks = set(tokenize(r.text))
        if any(ts <= rtoks for ts in stale_token_sets):
            continue
        score = w_rel * sim + w_rec * _recency(now, r.wall_ts) + w_imp * importance
        items.append(FusedItem(text=r.text, rid=r.rid, provenance={"session_id": r.session_id},
                               valid_interval=(r.wall_ts, float("inf")), score=score))
    items.sort(key=lambda it: (-it.score, it.rid))
    return items
