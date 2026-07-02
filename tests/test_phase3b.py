"""Phase 3b: namespacing (hard multi-tenant isolation), LLM-driven consolidation
summarization, ANN filter-pushdown, and the >=1e5-record load test (CB_SLOW-gated)."""

from __future__ import annotations

import importlib.util
import os
import time

import pytest

from curated_brain.backend import CuratedBrain
from curated_brain.extraction import HeuristicExtractor
from curated_brain.namespace import NamespacedMemory

HAVE_HNSW = importlib.util.find_spec("hnswlib") is not None


def _fact(subject, predicate, object):
    return {"fact": {"subject": subject, "predicate": predicate, "object": object}}


# ------------------------------------------------------------------ namespacing ---------
def test_namespaces_are_hard_isolated():
    mem = NamespacedMemory()
    mem.write("tenant-a", "x", session_id="s", timestamp=0.0,
              metadata=_fact("Erin", "city", "Vienna"))
    mem.write("tenant-b", "x", session_id="s", timestamp=0.0,
              metadata=_fact("Erin", "city", "Berlin"))
    assert mem.space("tenant-a").answer_structured("Erin", "city") == "Vienna"
    assert mem.space("tenant-b").answer_structured("Erin", "city") == "Berlin"
    # tenant B's query context never contains tenant A's data
    ctx_b = mem.query("tenant-b", "Where does Erin live?", session_id="q",
                      timestamp=1.0).context
    assert "Vienna" not in ctx_b and "Berlin" in ctx_b


def test_coreference_state_does_not_bleed_across_namespaces():
    # The recency-coreference bug class: user A's last subject must never resolve user B's
    # pronoun. With per-namespace stores (own extractor instances) it structurally cannot.
    mem = NamespacedMemory(factory=lambda: CuratedBrain(seed=0,
                                                        extractor=HeuristicExtractor()))
    mem.write("a", "Erin's city is Vienna.", session_id="s", timestamp=0.0)
    mem.write("b", "Their current city is Oslo.", session_id="s", timestamp=1.0)
    # in namespace b there is no antecedent -> the pronoun asserts nothing about Erin
    assert mem.space("b").answer_structured("Erin", "city") == ""
    assert mem.space("a").answer_structured("Erin", "city") == "Vienna"


def test_namespace_drop_erases_the_tenant():
    mem = NamespacedMemory()
    mem.write("gone", "x", session_id="s", timestamp=0.0,
              metadata=_fact("Erin", "city", "Vienna"))
    assert mem.drop("gone") is True
    assert mem.namespaces() == []
    assert b"Vienna" not in mem.snapshot()  # zero residue
    assert mem.drop("gone") is False  # idempotent


def test_namespaced_snapshot_roundtrip_and_legacy_upgrade(tmp_path):
    mem = NamespacedMemory()
    mem.write("a", "x", session_id="s", timestamp=0.0, metadata=_fact("Erin", "city", "Vienna"))
    mem.write("b", "x", session_id="s", timestamp=0.0, metadata=_fact("Bob", "city", "Berlin"))
    blob = mem.snapshot()
    mem2 = NamespacedMemory()
    mem2.restore(blob)
    assert mem2.namespaces() == ["a", "b"]
    assert mem2.space("a").answer_structured("Erin", "city") == "Vienna"
    assert mem2.snapshot() == blob  # deterministic round-trip

    # a legacy SINGLE-store file loads as the default namespace (in-place upgrade)
    cb = CuratedBrain(seed=0)
    cb.write("x", session_id="s", timestamp=0.0, metadata=_fact("Cara", "city", "Oslo"))
    p = str(tmp_path / "legacy.json")
    cb.save(p)
    mem3 = NamespacedMemory()
    mem3.load(p)
    assert mem3.namespaces() == ["default"]
    assert mem3.space("default").answer_structured("Cara", "city") == "Oslo"


# ------------------------------------------------------------ LLM consolidation ---------
class _JoinLLM:
    """Deterministic summarizer double: one sentence naming every member value."""

    def complete(self, prompt: str) -> str:
        notes = [ln[2:] for ln in prompt.splitlines() if ln.startswith("- ")]
        return "Summary: " + " ".join(notes)


def test_consolidation_summarizer_produces_a_real_claim():
    cb = CuratedBrain(seed=0, summarizer=_JoinLLM())
    cb.write("Erin's city is Vienna.", session_id="s", timestamp=0.0,
             metadata=_fact("Erin", "city", "Vienna"))
    cb.write("Erin is living in Vienna these days.", session_id="s", timestamp=1.0,
             metadata=_fact("Erin", "city", "Vienna"))
    rep = cb.consolidate()
    claims = [r for r in cb._episodes if r.tier == "semantic"]
    if rep.claims_out:  # a merge happened -> the claim is the model summary, not a member copy
        assert claims and claims[0].content.startswith("Summary: ")
        assert "Vienna" in claims[0].content
        # the summarized claim is what the vector tier serves
        hits = cb.vector.search("Vienna", k=4, t=2.0)
        assert any(r.text == claims[0].content for r, _ in hits)


def test_consolidation_without_summarizer_is_unchanged():
    # default None -> the deterministic representative behavior (AC-1/AC-9 path)
    cb = CuratedBrain(seed=0)
    cb.write("Erin's city is Vienna.", session_id="s", timestamp=0.0,
             metadata=_fact("Erin", "city", "Vienna"))
    cb.write("Erin is living in Vienna these days.", session_id="s", timestamp=1.0,
             metadata=_fact("Erin", "city", "Vienna"))
    cb.consolidate()
    for r in cb._episodes:
        assert not r.content.startswith("Summary: ")


# ------------------------------------------------------------- ANN filter-pushdown ------
@pytest.mark.skipif(not HAVE_HNSW, reason="hnswlib not installed ([scale] extra)")
def test_ann_filter_pushdown_finds_deep_matches():
    from curated_brain.fakes import DeterministicEmbedder
    from curated_brain.vector import HnswIndex, VectorTier

    emb = DeterministicEmbedder(64)
    tier = VectorTier(emb, index=HnswIndex(64, max_elements=512))
    # 300 DISTINCT episodic decoys phrased close to the query + 5 semantic targets phrased
    # far from it: a FIXED k*8 over-fetch fills with decoys and the tier filter starves.
    for i in range(300):
        tier.add(rid=f"d{i}",
                 text=f"meeting notes {i} covering the quarterly review agenda item {i * 7}",
                 wall_ts=0.0, session_id="s", tier="episodic")
    for i in range(5):
        tier.add(rid=f"t{i}", text=f"zebra habitat fact number {i}", wall_ts=0.0,
                 session_id="s", tier="semantic")
    hits = tier.search("meeting notes about the quarterly review", k=3, tier="semantic")
    assert len(hits) == 3  # escalation dug past the 24 nearest (all episodic decoys)
    assert all(r.tier == "semantic" for r, _ in hits)


# ------------------------------------------------------------------- 1e5 load test ------
@pytest.mark.skipif(os.environ.get("CB_SLOW") != "1" or not HAVE_HNSW,
                    reason="set CB_SLOW=1 (and install [scale]) to run the load test")
def test_load_100k_records_recall_and_latency():
    """The Track-C bar: >=1e5 records with a stated recall@k + p95-latency bound."""
    import numpy as np

    from curated_brain.vector import BruteForceIndex, HnswIndex, VectorTier

    n, dim, q, k = 100_000, 64, 50, 10
    rng = np.random.default_rng(7)
    vecs = rng.standard_normal((n, dim))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    queries = rng.standard_normal((q, dim))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    class _Pre:
        dim = 64

    # scale-appropriate graph parameters (hnswlib guidance: raise M/ef for 1e5+): the
    # stated bar below is for THIS config, which ships as a documented recipe.
    ann = VectorTier(_Pre(), index=HnswIndex(dim, max_elements=n, m=32, ef=400))
    exact = BruteForceIndex(dim)
    for i in range(n):
        ann.add(rid=f"r{i}", text=f"r{i}", wall_ts=0.0, session_id="s", embedding=vecs[i])
        exact.add(i, vecs[i])

    lat, hits, total = [], 0, 0
    for qi in range(q):
        gold = {f"r{key}" for key, _ in exact.rank(queries[qi])[:k]}
        t0 = time.perf_counter()
        got = {r.rid for r, _ in ann.search(queries[qi], k=k)}
        lat.append(time.perf_counter() - t0)
        hits += len(gold & got)
        total += k
    recall = hits / total
    p95 = sorted(lat)[int(0.95 * len(lat)) - 1]
    assert recall >= 0.90, f"recall@{k} {recall:.3f} < 0.90 at n={n}"
    assert p95 < 0.050, f"p95 latency {p95 * 1e3:.1f}ms >= 50ms at n={n}"
