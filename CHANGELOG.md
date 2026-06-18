# Changelog

Notable changes. Format loosely follows [Keep a Changelog](https://keepachangelog.com);
the project is pre-1.0, so the API may still change.

## [Unreleased]

### Added
- Real local-model providers — `SentenceTransformerEmbedder` (bge/e5) and `TransformersLLM`
  (local chat model) — behind the frozen-model protocols; the deterministic fakes remain the
  default test doubles so the offline gate needs no model stack.
- Record/replay **cassette** layer for reproducible real-model runs in CI.
- **Raw-text fact extraction** (`LLMExtractor`) with a groundedness anti-hallucination guard,
  wired into the write path (optional `extractor=`, off by default).
- Re-embed-on-model-upgrade migration (`CuratedBrain.reembed`).
- Observability metrics (`CuratedBrain.metrics`): write-decision breakdown, discard rate, size.
- Durable persistence (`CuratedBrain.save` / `load`).
- Robustness / property test suite; runnable `examples/`; Apache-2.0 LICENSE, README, CI.
- Complete top-level public API in `curated_brain/__init__` (imports stay lazy).

### Fixed
- **Silent bi-temporal corruption:** a non-finite timestamp created an "open" fact invisible
  to as-of queries — now rejected with a clear error at the write boundary.
- **Restore fidelity:** the session→timestamp map driving as-of-by-session (C6) queries was not
  persisted and was rebuilt lossily on restore, silently shifting answers — now persisted.

## [0.1.0]

- Initial two-tier memory layer: a bi-temporal structured tier (entities/relations, valid +
  transaction time, non-lossy supersede, multi-hop and as-of queries) and a vector tier, with
  surprise-gated selective writes, hybrid retrieval, and self-organizing consolidation.
- Acceptance tests AC-1…AC-9 on a seeded synthetic longitudinal dataset (beats naive-RAG,
  long-context, and no-memory baselines on every category C1–C6).
