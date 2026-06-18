"""Stage 6 — consolidation worker. Gate: AC-6 (bounded growth), AC-7 (consolidation
quality: dedup, contradiction resolution, provenance retention)."""

from __future__ import annotations

from curated_brain.backend import CuratedBrain
from curated_brain.dataset import generate, redundant_stream
from curated_brain.eval import correct
from curated_brain.surprise import SurpriseGate


def _feed(cb, observations):
    for o in observations:
        cb.write(o.content, session_id=o.session_id, timestamp=o.wall_ts,
                 metadata={"fact": o.fact} if o.fact else None)


# --------------------------------------------------------------------------- AC-6 ----
def test_ac6_bounded_growth_and_preserved_accuracy():
    ds = generate(seed=0)
    last = ds.base_ts + (ds.n_sessions - 1) * ds.day
    cb = CuratedBrain(seed=0)
    _feed(cb, ds.observations)
    k = len(ds.observations)

    cb.consolidate()
    st = cb.stats()
    stored = st.episodic_count + st.semantic_count
    assert stored <= 0.20 * k, f"stored {stored} > 20% of {k} raw items"

    # C1–C3 accuracy must not degrade (answers are carried by the durable structured tier)
    probes = ds.by_category("C1")
    recall = sum(p.gold.lower() in cb.query(p.question, session_id="q", timestamp=last,
                                            k=8).context.lower() for p in probes) / len(probes)
    assert recall >= 0.90


def test_ac6_growth_is_sublinear_in_noise():
    # Tripling the noise volume must NOT triple the store: salient structure is bounded by
    # the entity set, so stored size grows far slower than raw input.
    def stored_for(noise):
        ds = generate(seed=0, noise_per_session=noise)
        cb = CuratedBrain(seed=0)
        _feed(cb, ds.observations)
        cb.consolidate()
        st = cb.stats()
        return len(ds.observations), st.episodic_count + st.semantic_count

    k1, s1 = stored_for(4)
    k2, s2 = stored_for(16)
    assert k2 > 1.8 * k1                      # genuinely more raw input
    assert (s2 - s1) / (k2 - k1) < 0.20       # marginal stored-per-noise stays small


# --------------------------------------------------------------------------- AC-7 ----
def test_ac7_consolidation_dedups_redundant_store():
    # Feed a duplicate-laden store (permissive gate stands in for un-curated input) and
    # show consolidate() merges near-identical records into semantic claims.
    permissive = SurpriseGate(theta0=0.0, theta_floor=0.0, theta_max=0.0, lr=0.0,
                              reinforce_sim=2.0)
    obs, facts = redundant_stream(seed=0)
    cb = CuratedBrain(seed=0, gate=permissive)
    _feed(cb, obs)

    before = cb.stats().episodic_count
    assert before > 4 * len(facts)  # the store really is duplicate-heavy
    report = cb.consolidate()

    dup_reduction = report.dupes_merged / (before - len(facts))
    assert dup_reduction >= 0.70, f"only reduced duplicates by {dup_reduction:.3f}"
    assert report.claims_out > 0

    # No salient fact lost in the merge — checked against the VECTOR tier (not just the
    # structured tier), so this genuinely exercises that consolidation preserved recall.
    for f in facts:
        hits = cb.vector.search(f["object"], k=3)
        assert any(f["object"].lower() in r.text.lower() for r, _ in hits), \
            f"merge lost vector recall of {f['object']}"
    # ...and the merged semantic claims now carry the load (episodic largely emptied).
    assert cb.stats().semantic_count >= len(facts)


def test_ac7_contradictions_resolved_and_provenance_retained():
    ds = generate(seed=0)
    cb = CuratedBrain(seed=0)
    _feed(cb, ds.observations)
    report = cb.consolidate()

    # every seeded city + role change was reconciled (6 + 6)
    assert report.contradictions_resolved == 12

    # current value is the new one; as-of returns the old one
    for p in ds.by_category("C2"):
        assert correct(cb.answer_structured(p.subject, p.predicate), p.gold)
    for p in ds.by_category("C6"):
        assert correct(cb.answer_structured(p.subject, p.predicate, at=p.as_of), p.gold)

    # provenance retained for EVERY surviving and superseded fact (PRD §8 invariant)
    assert cb.structured.facts
    assert all(f.provenance.get("session_id") for f in cb.structured.facts)
