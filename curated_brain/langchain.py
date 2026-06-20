"""LangChain integration — a Retriever over The Curated Brain (Track E adoption surface).

The curated context for a query is exposed as a LangChain ``Document``, so the memory layer
drops into any retrieval chain or agent. The plain :func:`memories` helper is unit-testable
without LangChain; :func:`build_retriever` wraps it as a ``BaseRetriever`` (``langchain-core``
imported lazily — the ``[langchain]`` extra). Write raw text via the wrapped ``CuratedBrain``
(heuristic extractor by default), exactly as elsewhere.

    from curated_brain.langchain import build_retriever
    retriever = build_retriever()              # or build_retriever(my_curated_brain)
    retriever.cb.write("Erin lives in Vienna.", session_id="s", timestamp=0.0)
    docs = retriever.invoke("Where does Erin live?")
"""

from __future__ import annotations

from curated_brain.backend import CuratedBrain
from curated_brain.extraction import HeuristicExtractor

# Far-future default query time: with no explicit clock, "now" sees everything written.
_NOW = 1e12


def memories(cb: CuratedBrain, query: str, *, k: int = 8, session_id: str = "langchain",
             timestamp: float = _NOW) -> list[dict]:
    """Curated context for ``query`` as a list of 0 or 1 ``(page_content, metadata)`` dicts —
    the LangChain-free core. The content is CB's small curated payload (stale values already
    supersede-filtered); the metadata carries the provenance citation record-ids."""
    r = cb.query(query, session_id=session_id, timestamp=timestamp, k=k)
    if not r.context:
        return []
    return [{"page_content": r.context,
             "metadata": {"record_ids": [c.record_id for c in r.citations],
                          "tokens_in": r.tokens_in}}]


def build_retriever(cb: CuratedBrain | None = None, *, k: int = 8):
    """A LangChain ``BaseRetriever`` over a ``CuratedBrain`` (defaults to one with the
    heuristic extractor, so raw text in). The wrapped brain is exposed as ``.cb`` for writes.
    Requires the ``[langchain]`` extra."""
    try:
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
        from pydantic import PrivateAttr
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "LangChain retriever requires: pip install 'curated-brain[langchain]'") from e

    brain = cb or CuratedBrain(seed=0, extractor=HeuristicExtractor())

    class CuratedBrainRetriever(BaseRetriever):
        k: int = 8
        _cb: CuratedBrain = PrivateAttr()

        def __init__(self, cb: CuratedBrain, k: int, **kw) -> None:
            super().__init__(k=k, **kw)
            self._cb = cb

        @property
        def cb(self) -> CuratedBrain:
            return self._cb

        def _get_relevant_documents(self, query: str, *, run_manager=None):
            return [Document(page_content=d["page_content"], metadata=d["metadata"])
                    for d in memories(self._cb, query, k=self.k)]

    return CuratedBrainRetriever(brain, k)


__all__ = ["memories", "build_retriever"]
