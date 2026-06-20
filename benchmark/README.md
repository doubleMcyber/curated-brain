# Benchmark — Curated Brain on an independent longitudinal-memory harness

Curated Brain is evaluated on a **third-party, fully-offline, deterministic** harness it never
saw during development: [longitudinal-memory-eval-harness](https://github.com/doubleMcyber/longitudinal-memory-eval-harness).
Scoring is provenance-based (each retrieved item is resolved to a gold fact by its
`(source_session, source_turn)`), so a backend cannot game the metric by phrasing.

## Reproduce (offline, ~1–2 min, no GPU / network / keys)

```bash
benchmark/run_offline.sh                 # clones the harness next to this repo if absent
# or point at an existing checkout:
LMEH_PATH=/path/to/harness benchmark/run_offline.sh
```

It installs this library editable, runs Curated Brain + the shipped references on the standard
suite (seed 42, k=10), and prints the `compare` scoreboard. The token-cosine embedder and the
no-model normalizing judge ship in the harness, so the run is byte-deterministic (every backend
emits a `determinism_hash`). The harness is pinned to a reviewed commit. Network is used only on
first run (to clone the harness + pip-install numpy); the evaluation itself needs none.

## Result — vs the shipped references (standard suite, seed 42, k=10)

| metric | **curated_brain** | temporal_rag | semantic_rag | naive_rag | long_context | no_memory |
|---|---|---|---|---|---|---|
| recall@k | 0.88 | **0.92** | **0.92** | 0.80 | 0.60 | 0.00 |
| precision@k | **0.79** | 0.54 | 0.40 | 0.39 | 0.19 | 0.00 |
| contradiction_acc | **1.00** | 0.80 | 0.10 | 0.10 | 0.10 | 0.00 |
| staleness *(lower=better)* | **0.00** | 0.06 | 0.28 | 0.28 | 0.28 | 0.00 |
| answer_acc | 0.76 | 0.76 | 0.76 | 0.68 | 0.68 | 0.00 |
| cost_per_query_usd *(modeled)* | **~0.0000** | 0.0001 | 0.0001 | 0.0001 | 0.0004 | 0.00 |

Curated Brain wins or ties **every quality axis except aggregate recall**, where it trails
`temporal_rag` by 0.04 — entirely the `recency_relevance` category (0.50 vs 1.00). That category's
update turn (*"… briefly noted the project changed to July"*) needs definite-NP + ellipsis
coreference; closing it generally also requires original-vs-current intent detection, and a
phrasing-specific regex would be benchmark-tuning — so we **deliberately did not special-case it**.
Not tuning to the benchmark is itself a credibility property. Full per-category breakdown, the
`bge` real-embedder ablation, and the provenance audit live in the harness repo's
[`RESULTS_curated_brain.md`](https://github.com/doubleMcyber/longitudinal-memory-eval-harness/blob/claude/curated-brain-adapter/RESULTS_curated_brain.md).

## Result — vs a named system (Mem0), preliminary

A real head-to-head vs **Mem0** (`mem0ai` 2.0.7), run fully offline (Mem0 driven by a small local
model + the same offline embedder, in-memory qdrant). Mem0 is CPU-bound (~2.7 min/add), so this is
a small **n=3** subset, not the full suite:

| metric | curated_brain | temporal_rag | mem0 (Qwen-2B) |
|---|---|---|---|
| answer accuracy | **1.00** | 0.67 | 0.67 |
| recall@k | 1.00 | 1.00 | 1.00 |
| precision@k | **1.00** | 0.67 | 0.37 |
| contradiction-resolution | **1.00** | 0.00 | 0.00 |

## Scope & the path to the full named-rival claim

This is **preliminary** evidence: Curated Brain is competitive-to-better vs strong RAG references
(fully reproducible above) and beats Mem0 on a small offline subset. It is **not yet** the full
headline claim — *"LongMemEval ≥ each of Mem0 / Letta / Zep at ≤ cost"* — which needs:

1. A **capable shared inference endpoint** (one OpenAI-compatible model that CB's extractor and
   every rival call, so the model is held constant). The providers for this already ship in CB
   (`OpenAICompatLLM` / `OpenAICompatEmbedder`); set `OPENAI_BASE_URL` + `OPENAI_API_KEY`.
2. **Letta** and **Zep** adapters in the harness (Mem0 + a Letta stub exist; Zep needs its Docker
   server) — each pointed at the same endpoint.
3. Enough throughput to run the **full suite** for every system (the Mem0-on-CPU run above is ~5 h
   for one system; a hosted endpoint removes that wall).

When (1)–(3) are available the run is one command per backend (`mem_eval.runner.cli run --backend
<name> …`) followed by `compare`. Until then, the result here is the strongest reproducible
evidence the architecture wins on the quality axes that distinguish a memory layer from plain RAG:
**precision, contradiction-resolution, and staleness.**

### Honest caveats

- **Embedder parity:** CB runs on its deterministic embedder; the references use the harness's
  token-cosine embedder. A fully-fair run gives every backend the same embedder (`CB_EMBEDDER=bge`
  wires a real one into CB; the references would get `bge` too).
- **Storage slope** (517 vs 46) is a serialization artifact — `len(snapshot())` is a verbose JSON
  dump vs the references' raw chunks — not claimed as a win or a loss.
- The Mem0 number is **n=3**; treat it as directional, not conclusive.
