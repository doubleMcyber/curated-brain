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
- **Complete PyPI package metadata** (`readme`, `license`, `authors`, `keywords`, `classifiers`,
  `project.urls`) — the README now renders as the project long-description; `twine check` passes.
- **Typed public API** — the package is now `mypy`-clean and ships a `py.typed` marker, so
  downstream type checkers see the annotations. `mypy` runs in the CI gate to keep it honest.
- Robustness / property test suite; runnable `examples/`; Apache-2.0 LICENSE, README, CI.
- **Soak/scale test** — 5000 redundant observations → ~174 episodic records + all 500 distinct
  facts (the store grows with distinct facts, not observations), with full recall after
  consolidation and a byte-identical snapshot round-trip at scale.
- Complete top-level public API in `curated_brain/__init__` (imports stay lazy).
- **Benchmarks** on our companion offline diagnostic harness (same-author — not an independent
  benchmark; see `benchmark/README.md` for the full provenance and tuning disclosures), now
  **reproducible from a clean checkout** via `benchmark/run_offline.sh`: CB leads precision +
  contradiction-resolution + staleness vs the harness's RAG references; the offline Mem0
  comparison is n=3 and mixed (answer ties on plain recall; provenance-metric caveats apply).
  Not yet the full named-rival claim — the doc states the exact endpoint/throughput needed.

### Changed (Phase-1 extraction-default pass, 2026-07-02)
- **One predicate vocabulary.** The heuristic extractor's verb patterns now emit the same
  canonical predicates as the possessive path / planner / dataset (`location`→`city` via
  `PREDICATE_ALIASES`), so "Erin moved to Vienna" supersedes "Erin's city is Berlin" instead
  of living in a parallel schema. (Breaking for callers who queried the old `location` key.)
- **First-person extraction** (`resolve_first_person`): "My email is X" / "I moved to Berlin"
  extract as facts about a declared speaker (`metadata={"speaker": ...}`; the MCP server
  defaults to "User"). Opt-in — without a declared speaker, first-person text asserts nothing
  (and a bare capitalized pronoun is no longer mistaken for a name).
- **Echo suppression (Pillar B).** A verbatim restatement of an already-asserted statement
  reinforces instead of re-asserting — a late echo of "Alice lives in Riga" no longer flips
  her city back after "Alice has moved to Tallinn". A genuine revert needs fresh phrasing.
- **AC-9 without spoon-feeding.** `run_harness(extraction=True)` runs Curated Brain on the
  same raw text as every baseline (no `metadata.fact`); it scores identically to the spoon-fed
  wiring (C1–C6 = 1.0/1.0/0.99/0.91/1.0/1.0) and strictly beats all three baselines on every
  category — locked in CI (`tests/test_extraction_default.py`). External harness results are
  bit-identical (same determinism hash).
- PRD §6: the logprob/predictive-surprise estimator is explicitly **deferred post-v1** (cost
  multiplier on the write path; unvalidatable without a capable endpoint; no claim depends on
  it). Planner: predicates actually stored now outrank keyword-mapped guesses.

### Fixed
- **Silent bi-temporal corruption:** a non-finite timestamp created an "open" fact invisible
  to as-of queries — now rejected with a clear error at the write boundary.
- **Restore fidelity:** the session→timestamp map driving as-of-by-session (C6) queries was not
  persisted and was rebuilt lossily on restore, silently shifting answers — now persisted.
- **(Phase-2 correctness pass)** `consolidate()` no longer crashes on fact values containing
  `"|"` (fact links stored structured; legacy snapshots still load). Bi-temporality is real:
  `metadata.fact["valid_from"]` records retroactive facts, and an out-of-order older assertion
  becomes closed history instead of inverting the open fact's interval. Non-Latin text is
  storable/retrievable (unicode tokenizer + NFKC/casefold; previously zero-vector — ASCII
  unchanged). Supersede-filtering is provenance-linked + entity-scoped per (subject, predicate)
  — superseding one entity's value no longer filters store-wide records sharing its words
  (tradeoff: a stale free-text record with neither fact link nor subject name is no longer
  caught). Snapshots persist the resolver's ambiguity history; `restore()` rejects
  dim-mismatched snapshots. MCP: lock-serialized ops, wall-clock default timestamps (was 0.0 —
  silently disabled temporal semantics), atomic batched persistence. Consolidated claims keep
  entity tags + fact link (visible to entity-filtered search and later supersedes). New
  `CuratedBrain(max_context_items=)` knob so callers asking for more context get it (default 4
  unchanged).

## [0.1.0]

- Initial two-tier memory layer: a bi-temporal structured tier (entities/relations, valid +
  transaction time, non-lossy supersede, multi-hop and as-of queries) and a vector tier, with
  surprise-gated selective writes, hybrid retrieval, and self-organizing consolidation.
- Acceptance tests AC-1…AC-9 on a seeded synthetic longitudinal dataset (beats naive-RAG,
  long-context, and no-memory baselines on every category C1–C6).
