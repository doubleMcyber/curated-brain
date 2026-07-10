"""LlamaIndex integration (Track E). The retrieval logic reuses the framework-free
`memories()` core (tested in test_langchain.py); the BaseRetriever wrapper is smoke-tested
when `llama-index-core` is installed."""

from __future__ import annotations

import pytest

from curated_brain.backend import CuratedBrain
from curated_brain.extraction import HeuristicExtractor

# Skip the whole module cleanly when the extra is absent. NOTE: find_spec("llama_index.core")
# RAISES ModuleNotFoundError when the parent `llama_index` is missing (it is a submodule), so
# it can't guard collection — importorskip catches the import failure and skips instead.
pytest.importorskip("llama_index.core", reason="llama-index extra not installed")


def _brain() -> CuratedBrain:
    cb = CuratedBrain(seed=0, extractor=HeuristicExtractor())
    cb.write("Erin lives in Vienna.", session_id="s", timestamp=0.0)
    return cb


def test_retriever_returns_scored_nodes_with_provenance():
    from curated_brain.llama_index import build_retriever
    retriever = build_retriever(_brain(), k=4)
    nodes = retriever.retrieve("Where does Erin live?")  # standard LlamaIndex API
    assert nodes and "Vienna" in nodes[0].text
    assert nodes[0].node.metadata["record_ids"]  # provenance carried through
    # the wrapped brain is reachable for writes
    retriever.cb.write("Bob lives in Berlin.", session_id="s", timestamp=1.0)
    assert "Berlin" in retriever.retrieve("Where does Bob live?")[0].text


def test_missing_query_returns_no_nodes():
    from curated_brain.llama_index import build_retriever
    retriever = build_retriever(k=4)  # empty brain
    assert retriever.retrieve("Where does Erin live?") == []
