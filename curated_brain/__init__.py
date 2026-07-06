"""The Curated Brain — a persistent, self-organizing memory layer for LLM agents.

Memory is treated as a *curation* problem, not a search problem: we invest at write
time (selective, surprise-gated) and run background consolidation so the store grows
sublinearly and gets more useful the longer it runs.

Everything is built on frozen-model protocols (`Embedder`, `LLM`) — no training — and
ships with deterministic local implementations so the whole layer is reproducible and
scorable without any network access.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from curated_brain.backend import CuratedBrain, MemoryBackend
from curated_brain.extraction import HeuristicExtractor, LLMExtractor, resolve_first_person
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
from curated_brain.namespace import NamespacedMemory
from curated_brain.pricing import Pricing
from curated_brain.protocols import LLM, Embedder
from curated_brain.providers import (
    OpenAICompatEmbedder,
    OpenAICompatLLM,
    SentenceTransformerEmbedder,
    TransformersLLM,
)

try:
    __version__ = _pkg_version("curated-brain")
except PackageNotFoundError:  # running from a source checkout without an install
    __version__ = "0.1.0"

__all__ = [
    # core
    "CuratedBrain",
    "MemoryBackend",
    "NamespacedMemory",
    "Pricing",
    "resolve_first_person",
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
    "HeuristicExtractor",
    # data types
    "EpisodicRecord",
    "Fact",
    "WriteReceipt",
    "Retrieval",
    "Citation",
    "ConsolidationReport",
    "StoreStats",
    # package metadata
    "__version__",
]
