# The Curated Brain

A persistent, self-organizing **memory layer for LLM agents** — **the memory you can replay
and audit**. Built entirely on **frozen-model APIs** (no training, no fine-tuning): all the
intelligence is in how a frozen embedder + chat model are *orchestrated*.

> **Thesis:** memory is a **curation** problem, not a search problem. RAG fails at *write*
> time, not query time — log everything and retrieval drowns in noise. The Curated Brain
> invests at write time: it decides *what to keep*, imposes *structure*, and *continuously
> reconciles* old and new facts, so it gets **more useful the longer it runs**, not more bloated.

**What it uniquely owns** (properties you can verify today, not benchmark claims):

- **Deterministic & replayable** — identical input + seed ⇒ byte-identical store and answers;
  `snapshot()`/`restore()` round-trip exactly. Debug a memory bug by replaying it.
- **Bi-temporal with full provenance** — every fact carries valid time *and* transaction time
  plus its source; updates supersede without destroying history, so "what did we believe on
  date D?" is a first-class query and every answer is auditable back to its source turn.
- **Offline-capable** — the whole layer (and its test gate) runs with zero network, zero keys,
  zero GPUs; real models plug in behind the same protocols when you have them.
- **Hard tenant isolation & real erasure** — one store per namespace (`NamespacedMemory`;
  cross-tenant bleed is structurally impossible) and `forget()` erases an entity with zero
  residue in the snapshot bytes (the GDPR path), verified by test.

If you need a managed cloud service, a mature ecosystem, or LLM-native extraction quality at
scale today, Mem0 / Letta / Zep are more established; the head-to-head against them has **not**
been run (see Status).

## Status

Honest snapshot (see [`PROGRESS.md`](PROGRESS.md) for the live, detailed state):

**Working and tested** (`pytest -q` green, `ruff` clean, fully deterministic):
- Two-tier store: a **bi-temporal structured tier** (valid + transaction time, retroactive
  facts, non-lossy supersede, multi-hop / as-of-time / **inverse set queries** — "who lives
  in Berlin?") and a **vector tier** (cosine ANN, metadata filters with pushdown).
- **Surprise-gated writes** (semantic novelty + contradiction override + adaptive threshold),
  with **echo suppression** — a verbatim restatement reinforces instead of resurrecting a
  superseded value.
- **Hybrid retrieval** (planner → fuse by relevance × recency × importance → supersede-filter,
  provenance-linked and entity-scoped).
- **Self-organizing consolidation** (dedupe, prune, resolve contradictions, retain provenance;
  plug in an LLM `summarizer=` for real cluster summaries).
- **Raw-text fact extraction** — third-person *and* first-person ("My email is …" with a
  declared speaker), unicode-safe, with an anti-hallucination groundedness guard. The
  structured tier needs no spoon-fed facts.
- **Multi-tenant namespacing** (`NamespacedMemory`) and **hard erasure** (`forget()`, GDPR path).
- **Real local models** behind the protocols: `bge`/`e5` embeddings and a 🤗 Transformers chat
  model — with the deterministic fakes retained as test doubles so CI needs no model stack.
- Byte-deterministic `snapshot`/`restore`; re-embed-on-model-upgrade migration; **1e5-record
  load bar met** on the `[scale]` ANN backend (recall@10 1.00, p95 2.4 ms; `CB_SLOW=1`).

**Verified by tests:** AC-1…AC-9 on a seeded synthetic longitudinal dataset — **in both
configurations**: the original wiring check (gold triples supplied via `metadata.fact`) *and*
the honest one, where Curated Brain ingests the **same raw text as every baseline** and derives
facts itself with the deterministic extractor (`run_harness(extraction=True)`,
`tests/test_extraction_default.py`). Extraction-ON scores match spoon-fed
(C1–C6 = 1.0/1.0/0.99/0.91/1.0/1.0) and strictly beat naive-RAG, long-context, and no-memory on
every category. Caveats that remain: the scorer is a closed-set reader over the dataset's own
vocabulary, and the corpus phrasing is template-generated — this validates the architecture on
in-distribution text, not open-domain superiority.

**External benchmark — LongMemEval vs Mem0 / Letta / Zep** (2026-07-03; see Benchmark below).
Every system ran on the same local model (`qwen2.5:7b`) and embedder. The result is
**regime-split**: when the conversation history *fits* the model's context (oracle variant),
**Letta wins** (0.471 vs CB 0.261) by reading transcripts directly; when history *overflows*
context (`_s` variant — what a memory layer is for), **CB ties the best system (0.167, tied
with Mem0) at 8–24× lower cost per question**, Letta drops to ~0.083 (a disclosed partial that
ties CB on the 12 questions it finished), and Zep can't finish a question. **CB ≥ Mem0 and
CB ≥ Zep hold on both variants; CB never posts an accuracy win over Letta** (loses oracle, ties
the shared `_s` subset) — so the unconditional "beats all three" claim is **not** met, but CB
is the accuracy co-leader and cost leader in the regime that matters. Local 7B judge (not
GPT-4o); numbers internally comparable, not leaderboard-comparable.

**Not done yet** (tracked in `PROGRESS.md`): closing CB's temporal-reasoning and preference
gaps (both ~0 at `_s`) to actually overtake Letta there; a hosted-model rerun with the official
GPT-4o judge at larger n (and Letta/Zep run to completion). The strongest honest claim today is
"ties or beats Mem0/Zep everywhere and leads all three on cost; competitive with Letta —
behind it when history fits context, level on the questions Letta could finish once it doesn't."

## Benchmark (preliminary — our own diagnostic suite, not an independent benchmark)

On our companion **fully-offline & deterministic** longitudinal-memory harness
([longitudinal-memory-eval-harness](https://github.com/doubleMcyber/longitudinal-memory-eval-harness)
— built by the same author; corpus, scoring, and reference backends all share that lineage),
Curated Brain vs the harness's contradiction-aware reference `temporal_rag` (standard suite, seed 42,
~25 scored queries — treat per-metric deltas of a few hundredths as within noise):

| metric | Curated Brain | temporal_rag |
|---|---|---|
| precision@k | **0.79** | 0.54 |
| contradiction-resolution | **1.00** | 0.80 |
| staleness *(lower better)* | **0.00** | 0.06 |
| answer accuracy | 0.76 | 0.76 *(tie)* |
| recall@k | 0.88 | **0.92** |
| cost / query | **lowest of all backends** | — |

Curated Brain wins or ties **every quality metric except overall recall**, where it trails by
0.04 (= one query) — entirely one category (`recency_relevance`) needing definite-NP/ellipsis
coreference we did not add. It also wins long-range recall (0.83 vs 0.67) and is the cheapest
backend. **Tuning disclosure:** the first run lost on recall; we then added three general-purpose
capabilities (multi-entity routing, relational extraction patterns, recency pronoun coreference)
and re-ran. Each is defensible as general, but they were accepted under benchmark selection
pressure — judge accordingly.

**Reproduce it yourself** (offline, ~1–2 min, no GPU/keys): `benchmark/run_offline.sh`.

**Scope:** this is vs RAG *references* written by the same author. The offline **Mem0** comparison
is **n=3 and mixed** — CB led on the initial favorable subset, but a broader partial run showed
**answer-accuracy ties** on plain recall, and the harness's provenance-based recall/precision
structurally penalize Mem0's consolidating design (details in
[`benchmark/README.md`](benchmark/README.md)). The full named-rival claim (Mem0 / Letta / Zep on
the whole suite) has **not** been run — it needs a capable shared LLM endpoint.

## Install

```bash
pip install -e ".[dev]"        # core + test/lint
pip install -e ".[local]"      # + real local models (sentence-transformers, transformers, torch)
```

## Quickstart

Default (deterministic fakes — no model download, byte-reproducible):

```python
from curated_brain.backend import CuratedBrain

cb = CuratedBrain(seed=0)
cb.write(
    "Erin lives in Vienna.", session_id="s1", timestamp=0.0,
    metadata={"fact": {"subject": "Erin", "predicate": "city", "object": "Vienna"}},
)
print(cb.query("Where does Erin live?", session_id="q", timestamp=1.0, k=8).context)
print(cb.answer_structured("Erin", "city"))   # -> "Vienna"
```

With real local models (raw text in, facts extracted automatically):

```python
from curated_brain.backend import CuratedBrain
from curated_brain.providers import SentenceTransformerEmbedder, TransformersLLM
from curated_brain.extraction import LLMExtractor

emb = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
llm = TransformersLLM("Qwen/Qwen2.5-1.5B-Instruct", device="cpu")  # CPU avoids an MPS attn bug
cb = CuratedBrain(embedder=emb, dim=emb.dim, extractor=LLMExtractor(llm))

cb.write("Erin moved to Vienna last spring.", session_id="s1", timestamp=0.0)  # facts extracted
print(cb.answer_structured("Erin", "city"))   # -> "Vienna"
```

## Develop

```bash
pytest -q          # the gate: fast, offline, deterministic (real-model tests skip)
ruff check .
mypy curated_brain # type gate (also enforced in CI)
CB_LIVE=1 pytest -q -k live   # opt in to the real-model tests (needs the [local] extra + a model)
```

Real-model behaviour that CI must reproduce is captured with the record/replay
[`cassette`](curated_brain/cassette.py) layer, so the gate stays deterministic without weights.

## Mount it on an agent (MCP)

Expose the memory layer to any MCP host (Claude, agents) — `write` / `query` / `answer` /
`consolidate` / `stats` tools, raw text in:

```bash
pip install -e ".[mcp]"
curated-brain-mcp            # stdio server; set CB_MCP_PATH=store.json to persist across runs
```

## Use it from LangChain

```python
from curated_brain.langchain import build_retriever   # pip install -e ".[langchain]"

retriever = build_retriever()                          # heuristic extractor: raw text in
retriever.cb.write("Erin lives in Vienna.", session_id="s", timestamp=0.0)
docs = retriever.invoke("Where does Erin live?")       # standard LangChain Runnable API
```

## Use it from LlamaIndex

```python
from curated_brain.llama_index import build_retriever  # pip install -e ".[llama-index]"

retriever = build_retriever()                          # heuristic extractor: raw text in
retriever.cb.write("Erin lives in Vienna.", session_id="s", timestamp=0.0)
nodes = retriever.retrieve("Where does Erin live?")    # standard LlamaIndex Retriever API
```

## How it works

| Layer | Module |
|---|---|
| Adapter + orchestration (`write`/`query`/`consolidate`/`snapshot`) | `curated_brain/backend.py` |
| Bi-temporal structured tier | `curated_brain/structured.py` |
| Vector tier (ANN + metadata filters) | `curated_brain/vector.py` |
| Surprise gate | `curated_brain/surprise.py` |
| Hybrid retrieval (plan, fuse, supersede-filter) | `curated_brain/retrieval.py` |
| Consolidation worker | `curated_brain/consolidation.py` |
| Frozen-model protocols + real providers | `curated_brain/protocols.py`, `providers.py` |
| Raw-text fact extraction | `curated_brain/extraction.py` |

Design and full roadmap: the PRD is in [`PRD.md`](PRD.md); progress and next steps in
[`PROGRESS.md`](PROGRESS.md).

## License

[Apache-2.0](LICENSE).
