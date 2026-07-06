"""Vector-tier tests (Stage 3): ANN index, semantic recall (subset of C1), filters."""

from __future__ import annotations

from curated_brain.backend import CuratedBrain
from curated_brain.dataset import generate
from curated_brain.fakes import DeterministicEmbedder
from curated_brain.vector import BruteForceIndex, VectorTier


def _tier() -> VectorTier:
    return VectorTier(DeterministicEmbedder())


def test_search_is_hybrid_lexical_plus_semantic():
    # With identical embeddings (cosine ties), the lexical term must decide the ranking —
    # proving search fuses token-overlap with embedding similarity (general hybrid retrieval).
    import numpy as np

    class _ConstEmb:
        model_id = "const"
        dim = 4

        def embed(self, text: str) -> np.ndarray:
            return np.ones(4, dtype=float) / 2.0  # constant unit vector -> all cosines equal

        def embed_batch(self, texts):
            return np.stack([self.embed(t) for t in texts])

    vt = VectorTier(_ConstEmb())
    # insert the NON-matching record first (lower key): without the lexical term the cosine
    # tie would break by key and rank it first, so the assertion only holds if hybrid works.
    vt.add(rid="b", text="delta epsilon zeta", wall_ts=0.0, session_id="s")
    vt.add(rid="a", text="alpha beta gamma", wall_ts=0.0, session_id="s")
    hits = vt.search("alpha beta", k=2, t=1.0)
    assert hits[0][0].rid == "a"  # lexical overlap floats the higher-key match above the tie


def test_brute_force_index_ranks_by_cosine():
    emb = DeterministicEmbedder()
    idx = BruteForceIndex(emb.dim)
    idx.add(1, emb.embed("Alice lives in Berlin"))
    idx.add(2, emb.embed("Bob works on project Falcon"))
    idx.add(3, emb.embed("Carol enjoys hiking in the mountains"))
    ranked = idx.rank(emb.embed("Where does Alice live"))
    assert ranked[0][0] == 1  # the Alice/Berlin vector is nearest


def test_semantic_recall_finds_paraphrase():
    t = _tier()
    t.add(rid="a", text="Alice's email address is alice@example.com.",
          wall_ts=1.0, session_id="s0", entities=["Alice"])
    t.add(rid="b", text="Bob is working on project Falcon.",
          wall_ts=1.0, session_id="s0", entities=["Bob"])
    hits = t.search("What is Alice's email?", k=2)
    assert hits[0][0].rid == "a"


def test_metadata_filters():
    t = _tier()
    t.add(rid="a", text="Alice lives in Berlin.", wall_ts=10.0, session_id="s0",
          tier="episodic", entities=["Alice"])
    t.add(rid="b", text="Bob lives in Lisbon.", wall_ts=20.0, session_id="s1",
          tier="semantic", entities=["Bob"])
    # entity filter
    assert [r.rid for r, _ in t.search("lives", k=5, entity="Bob")] == ["b"]
    # tier filter
    assert [r.rid for r, _ in t.search("lives", k=5, tier="semantic")] == ["b"]
    # causal filter: a query at t=15 cannot see the t=20 record
    assert all(r.wall_ts <= 15.0 for r, _ in t.search("lives", k=5, t=15.0))
    # time-window filter
    assert [r.rid for r, _ in t.search("lives", k=5, window=(0.0, 15.0))] == ["a"]


def test_search_is_deterministic():
    a, b = _tier(), _tier()
    for t in (a, b):
        t.add(rid="x", text="Alice lives in Berlin.", wall_ts=1.0, session_id="s0")
        t.add(rid="y", text="Alice lives in Berlin.", wall_ts=2.0, session_id="s0")
    ra = [(r.rid, s) for r, s in a.search("Alice Berlin", k=5)]
    rb = [(r.rid, s) for r, s in b.search("Alice Berlin", k=5)]
    assert ra == rb


def test_vector_recall_capability_subset_c1():
    # Capability check (subset of C1): the ANN recalls a specific fact among other facts.
    # (Recall over the *adversarial* full stream — where topic distractors crowd pure
    #  similarity — is a Stage-4 concern; there the structured tier carries recall.)
    ds = generate(seed=0)
    t = VectorTier(DeterministicEmbedder())
    for o in ds.observations:
        if o.fact:  # salient facts only — a clean retrieval setting
            t.add(rid=f"{o.fact['subject']}:{o.fact['predicate']}", text=o.content,
                  wall_ts=o.wall_ts, session_id=o.session_id, entities=[o.fact["subject"]])

    hit = 0
    for person in ds.people:
        q = f"What is {person['name']}'s email address?"
        hits = t.search(q, k=8)
        if any(person["email"].lower() in r.text.lower() for r, _ in hits):
            hit += 1
    recall = hit / len(ds.people)
    assert recall >= 0.90, f"vector recall@8 of email facts = {recall:.3f}"


def test_vector_state_round_trips():
    cb = CuratedBrain(seed=0)
    ds = generate(seed=0)
    for o in ds.observations[:60]:
        cb.write(o.content, session_id=o.session_id, timestamp=o.wall_ts,
                 metadata={"fact": o.fact} if o.fact else None)
    blob = cb.snapshot()
    cb.restore(blob)
    assert cb.snapshot() == blob
    assert len(cb.vector) > 0
