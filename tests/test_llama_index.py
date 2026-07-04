"""LlamaIndex integration (Track E). The retrieval logic reuses the framework-free
`memories()` core (tested in test_langchain.py); the BaseRetriever wrapper is smoke-tested
when `llama-index-core` is installed."""

from __future__ import annotations

import importlib.util

import pytest

from curated_brain.backend import CuratedBrain
from curated_brain.extraction import HeuristicExtractor


def _brain() -> CuratedBrain:
    cb = CuratedBrain(seed=0, extractor=HeuristicExtractor())
    cb.write("Erin lives in Vienna.", session_id="s", timestamp=0.0)
    return cb


@pytest.mark.skipif(importlib.util.find_spec("llama_index.core") is None,
                    reason="llama-index extra not installed")
def test_retriever_returns_scored_nodes_with_provenance():
    from curated_brain.llama_index import build_retriever
    retriever = build_retriever(_brain(), k=4)
    nodes = retriever.retrieve("Where does Erin live?")  # standard LlamaIndex API
    assert nodes and "Vienna" in nodes[0].text
    assert nodes[0].node.metadata["record_ids"]  # provenance carried through
    # the wrapped brain is reachable for writes
    retriever.cb.write("Bob lives in Berlin.", session_id="s", timestamp=1.0)
    assert "Berlin" in retriever.retrieve("Where does Bob live?")[0].text


@pytest.mark.skipif(importlib.util.find_spec("llama_index.core") is None,
                    reason="llama-index extra not installed")
def test_missing_query_returns_no_nodes():
    from curated_brain.llama_index import build_retriever
    retriever = build_retriever(k=4)  # empty brain
    assert retriever.retrieve("Where does Erin live?") == []
