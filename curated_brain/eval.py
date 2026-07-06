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


def extract_value(context: str, candidates: list[str], question: str = "",
                  subject: str | None = None) -> str:
    """First candidate value appearing in the context (top line first), skipping any value
    already named in the question. Longer candidates win ties (email > city).

    When ``subject`` is given the reader is *entity-aware*: a candidate only counts if it
    appears on a line that also mentions the subject. This is the fair, steelmanned reader
    — it lets a raw multi-entity dump (long-context / naive RAG) answer correctly whenever
    the right entity's line is present, instead of grabbing another person's value."""
    q = question.lower()
    subj = subject.lower() if subject else None
    cands = sorted(candidates, key=len, reverse=True)
    for line in context.splitlines():
        low = line.lower()
        if subj is not None and subj not in low:
            continue
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


# --------------------------------------------------------------------------------------
# Longitudinal harness (PRD §9.1 / AC-9). Runs the dataset through CuratedBrain and the
# three baselines, scoring every category C1–C6 with one shared, entity-aware reader so
# the only variable is each backend's curation quality.
# --------------------------------------------------------------------------------------

def _read(be, ds, probe, ts):
    r = be.query(probe.question, session_id="q", timestamp=ts, k=8)
    val = extract_value(r.context, candidates_for(ds, probe), probe.question,
                        subject=probe.subject)
    return val, r


def score_categories(be, ds, last_ts) -> dict[str, float]:
    """Per-category score in [−∞, 1]; higher is better. C3/C4 fold in cost/size so a cheap,
    bounded store is rewarded and a bloated or empty one is not."""
    def acc(cat):
        ps = ds.by_category(cat)
        return accuracy([(_read(be, ds, p, last_ts)[0], p.gold) for p in ps])

    # C1 — long-range recall: gold value present anywhere in the returned context.
    c1_ps = ds.by_category("C1")
    c1 = sum(p.gold.lower() in be.query(p.question, session_id="q", timestamp=last_ts,
                                        k=8).context.lower() for p in c1_ps) / len(c1_ps)

    # C2 — belief updating: current value returned AND no stale value present.
    c2_ps = ds.by_category("C2")
    c2_hits = 0
    for p in c2_ps:
        val, r = _read(be, ds, p, last_ts)
        stale_present = bool(p.stale) and p.stale.lower() in r.context.lower()
        c2_hits += correct(val, p.gold) and not stale_present
    c2 = c2_hits / len(c2_ps)

    # C3 — retrieval accuracy AND cost: accuracy minus a normalized token penalty.
    c3_ps = ds.by_category("C3")
    c3_acc, toks = [], []
    for p in c3_ps:
        val, r = _read(be, ds, p, last_ts)
        c3_acc.append(correct(val, p.gold))
        toks.append(r.tokens_in)
    c3 = (sum(c3_acc) / len(c3_acc), sum(toks) / len(toks))  # finalized in score_all (needs worst)

    # C5 — relational + multi-hop; C6 — temporal as-of (issued at "now", date in question).
    c5, c6 = acc("C5"), acc("C6")

    st = be.stats()
    stored = st.episodic_count + st.semantic_count
    return {"C1": c1, "C2": c2, "C3_acc": c3[0], "C3_tokens": c3[1],
            "C5": c5, "C6": c6, "stored": stored, "recall": c1}


def run_harness(seed: int = 0, *, extraction: bool = False):
    """Run the full longitudinal protocol and return ``{backend_name: {category: score}}``.

    Default mode: only CuratedBrain receives the extracted-triple metadata (it builds
    structure at write time); the baselines log raw text, exactly as in PRD §9.1. **Read
    that mode for what it is** — CB alone gets gold triples, so it validates the
    architecture wiring, not open-domain superiority (disclosed in the README).

    ``extraction=True`` is the honest configuration: CuratedBrain ingests the SAME raw
    text as every baseline (no ``metadata.fact`` spoon-feeding) and derives facts itself
    via the deterministic :class:`~curated_brain.extraction.HeuristicExtractor`.
    """
    # imported here to avoid a circular import at module load
    from curated_brain.backend import CuratedBrain
    from curated_brain.baselines import LongContext, NaiveRAG, NoMemory
    from curated_brain.dataset import generate
    from curated_brain.extraction import HeuristicExtractor

    ds = generate(seed=seed)
    last = ds.base_ts + (ds.n_sessions - 1) * ds.day
    k = len(ds.observations)
    cb = (CuratedBrain(seed=seed, extractor=HeuristicExtractor()) if extraction
          else CuratedBrain(seed=seed))
    backends = {"curated": cb, "naive": NaiveRAG(),
                "long_context": LongContext(), "no_memory": NoMemory()}

    raw: dict[str, dict] = {}
    for name, be in backends.items():
        for o in ds.observations:
            meta = ({"fact": o.fact}
                    if (name == "curated" and o.fact and not extraction) else None)
            be.write(o.content, session_id=o.session_id, timestamp=o.wall_ts, metadata=meta)
        if name == "curated":
            be.consolidate()  # the harness "sleeps" between the run and scoring
        raw[name] = score_categories(be, ds, last)

    worst_tokens = max(r["C3_tokens"] for r in raw.values()) or 1.0
    scores: dict[str, dict] = {}
    for name, r in raw.items():
        scores[name] = {
            "C1": r["C1"],
            "C2": r["C2"],
            "C3": r["C3_acc"] - 0.5 * (r["C3_tokens"] / worst_tokens),  # accuracy − cost
            "C4": r["recall"] - r["stored"] / k,                        # recall − size penalty
            "C5": r["C5"],
            "C6": r["C6"],
        }
    return scores, raw

