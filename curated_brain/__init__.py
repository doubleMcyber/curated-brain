"""The Curated Brain — a persistent, self-organizing memory layer for LLM agents.

Memory is treated as a *curation* problem, not a search problem: we invest at write
time (selective, surprise-gated) and run background consolidation so the store grows
sublinearly and gets more useful the longer it runs.

Everything is built on frozen-model protocols (`Embedder`, `LLM`) — no training — and
ships with deterministic local implementations so the whole layer is reproducible and
scorable without any network access.
"""

from curated_brain.backend import CuratedBrain, MemoryBackend
from curated_brain.extraction import LLMExtractor
from curated_brain.fakes import DeterministicEmbedder, RuleBasedLLM
from curated_brain.models import (
    Citation,
    ConsolidationReport,
    EpisodicRecord,
    Fact,
    Retrieval,
    StoreStats,
    WriteReceipt,
)
from curated_brain.protocols import LLM, Embedder
from curated_brain.providers import (
    OpenAICompatEmbedder,
    OpenAICompatLLM,
    SentenceTransformerEmbedder,
    TransformersLLM,
)

__all__ = [
    # core
    "CuratedBrain",
    "MemoryBackend",
    # frozen-model seams: protocols, deterministic fakes (test doubles), real providers
    "Embedder",
    "LLM",
    "DeterministicEmbedder",
    "RuleBasedLLM",
    "SentenceTransformerEmbedder",
    "TransformersLLM",
    "OpenAICompatEmbedder",
    "OpenAICompatLLM",
    "LLMExtractor",
    # data types
    "EpisodicRecord",
    "Fact",
    "WriteReceipt",
    "Retrieval",
    "Citation",
    "ConsolidationReport",
    "StoreStats",
]
