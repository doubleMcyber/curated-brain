"""Track A — real local-model providers + the cassette reproducibility layer.

Two test classes of coverage:

* **Offline (always runs, no model):** the provider seam conforms to the protocols
  without loading any weights; a missing ``local`` extra raises an actionable error; the
  cassette records/replays embeddings + completions and can drive the whole pipeline
  deterministically off a replayed cassette (the CI-reproducibility guarantee).
* **Live (opt-in):** gated behind ``CB_LIVE=1`` so the default gate never loads a model.
  When enabled with the ``local`` extra, exercises a real ``bge`` embedder end-to-end.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

from curated_brain.backend import CuratedBrain
from curated_brain.cassette import CachedEmbedder, CachedLLM, Cassette
from curated_brain.fakes import DeterministicEmbedder, RuleBasedLLM
from curated_brain.protocols import LLM, Embedder
from curated_brain.providers import (
    OpenAICompatEmbedder,
    OpenAICompatLLM,
    SentenceTransformerEmbedder,
    TransformersLLM,
)

LIVE = os.environ.get("CB_LIVE") == "1" and importlib.util.find_spec("sentence_transformers")
LIVE_LLM = os.environ.get("CB_LIVE") == "1" and importlib.util.find_spec("transformers")


# ------------------------------------------------------------------- offline: seam ---
def test_providers_conform_to_protocols_without_loading():
    emb = SentenceTransformerEmbedder()
    llm = TransformersLLM()
    assert isinstance(emb, Embedder) and isinstance(llm, LLM)
    assert emb.model_id == "st:BAAI/bge-small-en-v1.5"
    assert emb.dim == 384  # known from the table, no model load required
    assert emb._model is None and llm._model is None  # genuinely lazy


def test_missing_extra_raises_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="local"):
        SentenceTransformerEmbedder().embed("x")
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(RuntimeError, match="local"):
        TransformersLLM().complete("x")


# ------------------------------------------------- offline: OpenAI-compatible seam ---
def test_openai_compat_embedder_offline():
    calls = []

    def fake_post(path, body):
        calls.append((path, body))
        return {"data": [{"index": 0, "embedding": [3.0, 4.0]}]}

    emb = OpenAICompatEmbedder("text-embedding-3-small", dim=2, post=fake_post)
    assert isinstance(emb, Embedder)
    assert emb.model_id == "openai:text-embedding-3-small" and emb.dim == 2
    assert np.allclose(emb.embed("hello"), [0.6, 0.8])  # 3-4-5 -> unit vector
    assert calls == [("/embeddings", {"model": "text-embedding-3-small", "input": "hello"})]


def test_openai_compat_embedder_batch_preserves_order():
    def fake_post(path, body):  # returned out of order to prove we sort by index
        return {"data": [{"index": 1, "embedding": [0.0, 2.0]},
                         {"index": 0, "embedding": [2.0, 0.0]}]}

    emb = OpenAICompatEmbedder("m", dim=2, post=fake_post)
    out = emb.embed_batch(["a", "b"])
    assert out.shape == (2, 2)
    assert np.allclose(out[0], [1.0, 0.0]) and np.allclose(out[1], [0.0, 1.0])
    assert emb.embed_batch([]).shape == (0, 2)  # empty input is safe


def test_openai_compat_llm_offline():
    seen = {}

    def fake_post(path, body):
        seen.update(path=path, body=body)
        return {"choices": [{"message": {"content": " Vienna "}}]}

    llm = OpenAICompatLLM("gpt-4o-mini", post=fake_post)
    assert isinstance(llm, LLM) and llm.model_id == "openai:gpt-4o-mini"
    assert llm.complete("Where?") == "Vienna"  # whitespace stripped
    assert seen["path"] == "/chat/completions"
    assert seen["body"]["messages"] == [{"role": "user", "content": "Where?"}]
    assert seen["body"]["temperature"] == 0.0  # greedy by default, for reproducibility


def test_openai_compat_embedder_drives_the_pipeline():
    # The remote embedder drops into CuratedBrain like any other Embedder (here via a
    # deterministic fake transport, so it stays offline + reproducible).
    def fake_post(path, body):
        h = sum(ord(c) for c in body["input"])
        return {"data": [{"index": 0, "embedding": [float(h % 7 + 1), float(h % 5 + 1)]}]}

    cb = CuratedBrain(embedder=OpenAICompatEmbedder("m", dim=2, post=fake_post), dim=2)
    cb.write("Erin lives in Vienna.", session_id="s0", timestamp=0.0,
             metadata={"fact": {"subject": "Erin", "predicate": "city", "object": "Vienna"}})
    assert cb.answer_structured("Erin", "city") == "Vienna"


# -------------------------------------------------------------- offline: cassette ----
def test_cassette_embed_roundtrip_and_replay(tmp_path):
    fake = DeterministicEmbedder(64)
    cas = Cassette()
    rec = CachedEmbedder(cas, inner=fake)  # record mode
    v = rec.embed("hello world")
    assert np.allclose(v, fake.embed("hello world"))

    path = tmp_path / "c.json"
    cas.save(str(path))
    replay = CachedEmbedder(Cassette.load(str(path)), inner=None)  # replay mode
    assert np.allclose(replay.embed("hello world"), v)
    assert replay.model_id == fake.model_id and replay.dim == fake.dim
    with pytest.raises(KeyError):
        replay.embed("never recorded")


def test_cassette_llm_record_replay():
    llm = RuleBasedLLM()
    cas = Cassette()
    out = CachedLLM(cas, inner=llm).complete("short\na much longer salient line")
    assert out == llm.complete("short\na much longer salient line")
    replay = CachedLLM(cas, inner=None)
    assert replay.complete("short\na much longer salient line") == out
    with pytest.raises(KeyError):
        replay.complete("unseen prompt")


def test_recorded_cassette_drives_pipeline_offline():
    # Record embeddings from the deterministic fake, then run CuratedBrain entirely off the
    # replayed cassette (inner=None) — proving the provider seam + cassette serve the
    # real-shaped path with zero model access, deterministically.
    obs = [("Erin works as a writer", "s1"),
           ("Erin lives in Vienna", "s1"),
           ("The quarterly report is due Friday", "s2")]
    q = "Where does Erin live?"

    cas = Cassette()
    rec = CachedEmbedder(cas, inner=DeterministicEmbedder(256))
    brain = CuratedBrain(embedder=rec, dim=256, seed=0)
    for content, sid in obs:
        brain.write(content, session_id=sid, timestamp=0.0)
    brain.query(q, session_id="q", timestamp=1.0, k=4)  # record the query embedding too

    replay = CachedEmbedder(Cassette.from_dict(cas.to_dict()), inner=None)
    brain2 = CuratedBrain(embedder=replay, dim=256, seed=0)
    for content, sid in obs:
        brain2.write(content, session_id=sid, timestamp=0.0)
    r = brain2.query(q, session_id="q", timestamp=1.0, k=4)
    assert "Vienna" in r.context


# ----------------------------------------------------- offline: re-embed migration ---
def test_reembed_on_upgrade_migrates_all_vectors():
    # Upgrading the embedding model must re-embed every record (new dimensionality),
    # stamp the new model id, preserve recall, and stay byte-deterministic.
    stream = [("Erin lives in Vienna", "s1"), ("Bob lives in Paris", "s1"),
              ("Cara writes Rust", "s2")]

    def fresh():
        cb = CuratedBrain(embedder=DeterministicEmbedder(64), dim=64, seed=0)
        for content, sid in stream:
            cb.write(content, session_id=sid, timestamp=0.0)
        return cb

    cb, cb2 = fresh(), fresh()
    n = len(cb.vector)
    report = cb.reembed(DeterministicEmbedder(128))
    cb2.reembed(DeterministicEmbedder(128))

    assert report == {"reembedded": n, "from": "det-hash-64-v1", "to": "det-hash-128-v1"}
    assert cb.vector.index.dim == 128
    assert all(v.shape[0] == 128 for v in cb.vector.index._vecs.values())
    assert all(r.embed_model_id == "det-hash-128-v1" for r in cb._episodes)
    assert cb.stats().embed_model_id == "det-hash-128-v1"
    # recall survives AND is served from the NEW model's vector space: a fresh 128-dim
    # query must locate Erin's record via the migrated index (a no-op reembed would leave
    # 64-dim vectors and raise on the dim-mismatched dot product — so this is load-bearing).
    top = cb.vector.nearest(DeterministicEmbedder(128).embed("Where does Erin live?"))
    assert top is not None and top[0].text == "Erin lives in Vienna"
    assert "Vienna" in cb.query("Where does Erin live?", session_id="q",
                                timestamp=1.0, k=4).context
    # deterministic: identical inputs + same upgrade => byte-identical snapshot
    assert cb.snapshot() == cb2.snapshot()


# -------------------------------------------------------------------- live (opt-in) --
@pytest.mark.skipif(not LIVE_LLM, reason="set CB_LIVE=1 with the 'local' extra + a cached model")
def test_live_llm_extracts_a_triple():
    # Genuine non-faked LLM run: a real cached chat model extracts a (subject|predicate|
    # object) triple from raw text — the feasibility proof for Track B. CPU-forced because
    # MPS mis-handles some models' grouped-query attention in this environment.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model = os.environ.get("CB_LLM_MODEL", "Qwen/Qwen3.5-0.8B")
    llm = TransformersLLM(model_name=model, device="cpu", max_new_tokens=48)
    try:
        out = llm.complete(
            "Extract the fact as 'subject | predicate | object', output only that line.\n"
            "Sentence: Erin moved to Vienna last spring.")
    except Exception as e:  # cached model absent / unloadable in this env
        pytest.skip(f"cached model {model} unavailable: {e}")
    low = out.lower()
    assert out.strip() and "erin" in low and "vienna" in low


@pytest.mark.skipif(not LIVE, reason="set CB_LIVE=1 with the 'local' extra to run real-model tests")
def test_live_bge_embedder_semantics():
    emb = SentenceTransformerEmbedder()
    a, b, c = (emb.embed(t) for t in (
        "Erin lives in Vienna", "Erin is based in Vienna",
        "The quarterly report is due Friday"))
    assert emb.dim == len(a) == 384
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5      # unit norm per the contract
    assert float(a @ b) > float(a @ c)                     # related closer than unrelated


@pytest.mark.skipif(not LIVE, reason="set CB_LIVE=1 with the 'local' extra to run real-model tests")
def test_live_real_embedder_end_to_end():
    emb = SentenceTransformerEmbedder()
    brain = CuratedBrain(embedder=emb, dim=emb.dim, seed=0)
    for content, sid in [("Erin lives in Vienna", "s1"),
                         ("Erin works as a writer", "s1"),
                         ("The cafeteria serves lunch at noon", "s2")]:
        brain.write(content, session_id=sid, timestamp=0.0)
    r = brain.query("Where does Erin live?", session_id="q", timestamp=1.0, k=4)
    assert "Vienna" in r.context
