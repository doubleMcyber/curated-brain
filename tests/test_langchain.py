"""LangChain integration (Track E). The retrieval logic lives in the plain `memories()`
helper, tested here without LangChain; the BaseRetriever wrapper is smoke-tested when
`langchain-core` is installed."""

from __future__ import annotations

import importlib.util

import pytest

from curated_brain.backend import CuratedBrain
from curated_brain.extraction import HeuristicExtractor
from curated_brain.langchain import build_retriever, memories


def _brain() -> CuratedBrain:
    cb = CuratedBrain(seed=0, extractor=HeuristicExtractor())
    cb.write("Erin lives in Vienna.", session_id="s", timestamp=0.0)
    return cb


def test_memories_returns_curated_context_and_provenance():
    cb = _brain()
    docs = memories(cb, "Where does Erin live?", k=4)
    assert len(docs) == 1
    assert "Vienna" in docs[0]["page_content"]
    assert docs[0]["metadata"]["record_ids"]  # provenance citations carried through
    assert docs[0]["metadata"]["tokens_in"] > 0


def test_memories_empty_when_nothing_relevant():
    cb = CuratedBrain(seed=0, extractor=HeuristicExtractor())  # nothing written
    assert memories(cb, "Where does Erin live?") == []


@pytest.mark.skipif(importlib.util.find_spec("langchain_core") is None,
                    reason="langchain extra not installed")
def test_retriever_returns_documents():
    retriever = build_retriever(_brain(), k=4)
    docs = retriever.invoke("Where does Erin live?")  # standard LangChain Runnable API
    assert docs and "Vienna" in docs[0].page_content
    # the wrapped brain is reachable for writes
    retriever.cb.write("Bob lives in Berlin.", session_id="s", timestamp=1.0)
    assert "Berlin" in retriever.invoke("Where does Bob live?")[0].page_content
