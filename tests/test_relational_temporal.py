"""AC-8 — relational (multi-hop) + temporal (as-of) accuracy (Stage 2, C5/C6).

Pass condition: structured-tier accuracy >= 0.85 and strictly greater than naive RAG.
The win is architectural: the structured tier built triples at write time, so it answers
multi-hop and as-of queries exactly; naive RAG only has flat vector similarity.
"""

from __future__ import annotations

from curated_brain.backend import CuratedBrain
from curated_brain.baselines import NaiveRAG
from curated_brain.dataset import generate
from curated_brain.eval import accuracy, candidates_for, extract_value


def _feed(backend, ds, *, with_facts: bool):
    for obs in ds.observations:
        meta = {"fact": obs.fact} if (with_facts and obs.fact) else None
        backend.write(obs.content, session_id=obs.session_id, timestamp=obs.wall_ts,
                      metadata=meta)


def _curated_answer(cb: CuratedBrain, probe):
    if probe.hops:
        return cb.answer_path(probe.subject, probe.hops, at=probe.as_of)
    return cb.answer_structured(probe.subject, probe.predicate, at=probe.as_of)


def test_ac8_relational_and_temporal():
    ds = generate(seed=0)
    last_ts = ds.base_ts + (ds.n_sessions - 1) * ds.day

    cb = CuratedBrain(seed=0)
    naive = NaiveRAG()
    _feed(cb, ds, with_facts=True)
    _feed(naive, ds, with_facts=False)

    probes = ds.by_category("C5") + ds.by_category("C6")
    assert probes

    cb_pairs, naive_pairs = [], []
    for p in probes:
        cb_pairs.append((_curated_answer(cb, p), p.gold))
        ts = p.as_of if p.as_of is not None else last_ts
        r = naive.query(p.question, session_id="q", timestamp=ts, k=8)
        naive_pairs.append((extract_value(r.context, candidates_for(ds, p), p.question), p.gold))

    cb_acc = accuracy(cb_pairs)
    naive_acc = accuracy(naive_pairs)

    assert cb_acc >= 0.85, f"curated accuracy {cb_acc:.3f} below AC-8 threshold"
    assert cb_acc > naive_acc + 0.10, f"curated {cb_acc:.3f} not clearly > naive {naive_acc:.3f}"


def test_ac8_multi_hop_specifically_beats_naive():
    # The sharpest structural gap: 2-hop queries are impossible for flat vector search.
    ds = generate(seed=0)
    cb = CuratedBrain(seed=0)
    naive = NaiveRAG()
    _feed(cb, ds, with_facts=True)
    _feed(naive, ds, with_facts=False)
    last_ts = ds.base_ts + (ds.n_sessions - 1) * ds.day

    hop_probes = [p for p in ds.probes if p.hops]
    cb_pairs, naive_pairs = [], []
    for p in hop_probes:
        cb_pairs.append((cb.answer_path(p.subject, p.hops), p.gold))
        r = naive.query(p.question, session_id="q", timestamp=last_ts, k=8)
        naive_pairs.append((extract_value(r.context, candidates_for(ds, p), p.question), p.gold))

    assert accuracy(cb_pairs) == 1.0
    assert accuracy(naive_pairs) < accuracy(cb_pairs)
