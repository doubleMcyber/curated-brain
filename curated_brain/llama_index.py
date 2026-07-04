"""LlamaIndex integration — a Retriever over The Curated Brain (Track E adoption surface).

The sibling to :mod:`curated_brain.langchain`: CB's curated context for a query is exposed as
a LlamaIndex ``NodeWithScore``, so the memory layer drops into any LlamaIndex query engine or
agent. The framework-free core is :func:`curated_brain.langchain.memories` (a plain dict — no
LlamaIndex needed to unit-test it); :func:`build_retriever` wraps it as a
``llama_index.core.retrievers.BaseRetriever`` (imported lazily — the ``[llama-index]`` extra).

    from curated_brain.llama_index import build_retriever
    retriever = build_retriever()              # or build_retriever(my_curated_brain)
    retriever.cb.write("Erin lives in Vienna.", session_id="s", timestamp=0.0)
    nodes = retriever.retrieve("Where does Erin live?")
"""

from __future__ import annotations

from curated_brain.backend import CuratedBrain
from curated_brain.extraction import HeuristicExtractor
from curated_brain.langchain import memories  # framework-neutral core (returns plain dicts)

__all__ = ["build_retriever"]


def build_retriever(cb: CuratedBrain | None = None, *, k: int = 8):
    """A LlamaIndex ``BaseRetriever`` over a ``CuratedBrain`` (defaults to one with the
    heuristic extractor, so raw text in). The wrapped brain is exposed as ``.cb`` for writes.
    Requires the ``[llama-index]`` extra."""
    try:
        from llama_index.core.retrievers import BaseRetriever
        from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "LlamaIndex retriever requires: pip install 'curated-brain[llama-index]'") from e

    brain = cb or CuratedBrain(seed=0, extractor=HeuristicExtractor())

    class CuratedBrainRetriever(BaseRetriever):
        def __init__(self, cb: CuratedBrain, k: int) -> None:
            self._cb = cb
            self._k = k
            super().__init__()

        @property
        def cb(self) -> CuratedBrain:
            return self._cb

        def _retrieve(self, query_bundle: QueryBundle) -> list:
            # CB returns one curated payload; wrap it as a single scored node carrying the
            # provenance record-ids so downstream nodes stay attributable.
            out: list = []
            for d in memories(self._cb, query_bundle.query_str, k=self._k):
                node = TextNode(text=d["page_content"], metadata=d["metadata"])
                out.append(NodeWithScore(node=node, score=1.0))
            return out

    return CuratedBrainRetriever(brain, k)
