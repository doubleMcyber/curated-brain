# Changelog

Notable changes. Format loosely follows [Keep a Changelog](https://keepachangelog.com);
the project is pre-1.0, so the API may still change.

## [Unreleased]

### Added
- Real local-model providers — `SentenceTransformerEmbedder` (bge/e5) and `TransformersLLM`
  (local chat model) — behind the frozen-model protocols; the deterministic fakes remain the
  default test doubles so the offline gate needs no model stack. **bge-small verified offline**.
- **OpenAI-compatible providers** (`OpenAICompatEmbedder` / `OpenAICompatLLM`, stdlib HTTP) so
  the layer (and rival systems) can share one hosted/vLLM endpoint.
- **Heuristic (no-LLM) extractor** (`HeuristicExtractor`): deterministic possessive/verb triple
  extraction, predicate canonicalization, and recency-based pronoun coreference.
- **Entity resolution** (`curated_brain.resolve.EntityResolver`): conservative subject
  canonicalization — "Erin" / "Erin Smith" / "Ms. Smith" resolve to one entity via honorific
  stripping + exact component subsumption, gated on store-provable uniqueness, with NO fuzzy
  matching (a false merge is structurally impossible).
- Record/replay **cassette** layer for reproducible real-model runs in CI.
- **Raw-text fact extraction** (`LLMExtractor`) with a groundedness anti-hallucination guard,
  wired into the write path (optional `extractor=`, off by default).
- **Schema-driven + multi-entity retrieval:** the planner recognizes any stored predicate
  (incl. multi-word), and `query()` surfaces facts for every named entity when a plan
  open-domains or mis-routes; multi-hop now cites the whole support chain.
- **Real ANN backend** (`HnswIndex`, `[scale]` extra) behind `VectorIndex` — ~20× faster top-k
  at scale; brute force stays the deterministic default. Now drops into `VectorTier` via
  `VectorTier(embedder, index=HnswIndex(...))`: `search`/`nearest` use the index's `topk`
  fast path (over-fetch + filter + hybrid re-rank), keeping the exact default byte-identical.
- **MCP server** (`curated_brain.mcp_server`, `[mcp]` extra) + `curated-brain-mcp` console
  script — mount the memory layer on any agent host.
- **LangChain Retriever** (`curated_brain.langchain.build_retriever`, `[langchain]` extra) —
  drop the curated context into any retrieval chain/agent via the standard `.invoke()` API.
- Re-embed-on-model-upgrade migration (`CuratedBrain.reembed`).
- Observability metrics (`CuratedBrain.metrics`): write-decision breakdown, discard rate, size,
  and **cost accounting** (embed/extract calls + tokens, `avg_context_tokens`).
- Durable persistence (`CuratedBrain.save` / `load`).
- Robustness / property test suite; runnable `examples/`; Apache-2.0 LICENSE, README, CI.
- **Soak/scale test** — 5000 redundant observations → ~174 episodic records + all 500 distinct
  facts (the store grows with distinct facts, not observations), with full recall after
  consolidation and a byte-identical snapshot round-trip at scale.
- Complete top-level public API in `curated_brain/__init__` (imports stay lazy).
- **Benchmarks** on an independent offline harness (see README): CB wins precision +
  contradiction-resolution vs strong RAG references and (preliminary, small local model) vs
  **Mem0**; competitive on recall. Not yet the full named-rival claim — see README scope.

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
