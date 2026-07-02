# Benchmark — Curated Brain on our companion longitudinal-memory harness

Curated Brain is evaluated on a **fully-offline, deterministic** harness:
[longitudinal-memory-eval-harness](https://github.com/doubleMcyber/longitudinal-memory-eval-harness).

**Provenance disclosure (read first):** the harness is **same-author** — its corpus generators,
scoring, and reference backends were written in the same ecosystem as this library, and the
capability work below was iterated against it. It is a *diagnostic suite*, **not** an independent
third-party benchmark; nothing here substitutes for an externally-authored eval (LongMemEval /
LoCoMo) run on a frozen configuration.

Scoring is provenance-based (each retrieved item is resolved to a gold fact by its
`(source_session, source_turn)`). That makes it robust to phrasing tricks, but it **structurally
favors designs that preserve raw turns** (like Curated Brain) and penalizes designs that rewrite
facts and lose turn provenance (like Mem0) — see the caveats on the Mem0 table below.

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

## Result — vs the shipped references (standard suite, seed 42, k=10; ~25 scored queries)

| metric | **curated_brain** | temporal_rag | semantic_rag | naive_rag | long_context | no_memory |
|---|---|---|---|---|---|---|
| recall@k | 0.88 | **0.92** | **0.92** | 0.80 | 0.60 | 0.00 |
| precision@k | **0.79** | 0.54 | 0.40 | 0.39 | 0.19 | 0.00 |
| contradiction_acc | **1.00** | 0.80 | 0.10 | 0.10 | 0.10 | 0.00 |
| staleness *(lower=better)* | **0.00** | 0.06 | 0.28 | 0.28 | 0.28 | 0.00 |
| answer_acc | 0.76 | 0.76 | 0.76 | 0.68 | 0.68 | 0.00 |
| cost_per_query_usd *(modeled)* | **~0.0000** | 0.0001 | 0.0001 | 0.0001 | 0.0004 | 0.00 |

Curated Brain wins or ties **every quality axis except aggregate recall**, where it trails
`temporal_rag` by 0.04 — at this suite size, **one query** — entirely the `recency_relevance`
category (0.50 vs 1.00), whose update turn (*"… briefly noted the project changed to July"*)
needs definite-NP + ellipsis coreference we did not add.

**Statistical weight:** the suite is ~25 queries; per-category cells rest on 2–10 scenarios and
carry no confidence intervals. Treat the table as directional. **Tuning disclosure:** the first
scored run *lost* on recall and answer accuracy; three general-purpose capabilities (multi-entity
routing, relational extraction patterns, recency pronoun coreference) were then added and the
suite re-run until CB led the quality axes. Each lever is general-purpose code (no gold peeking,
no phrasing-specific regexes), but they were accepted under benchmark selection pressure by the
same author who wrote the corpus generators — a stronger claim requires a frozen configuration on
an externally-authored benchmark. Full per-category breakdown, the `bge` real-embedder ablation,
and the provenance audit live in the harness repo's
[`RESULTS_curated_brain.md`](https://github.com/doubleMcyber/longitudinal-memory-eval-harness/blob/claude/curated-brain-adapter/RESULTS_curated_brain.md).

## Result — LongMemEval vs ALL THREE named rivals (2026-07-02, first full run)

The named-rival run finally executed locally (ollama + qwen2.5:7b for every system's LLM
calls, answer generation, and judging; same nomic embedder for all — the memory layer is the
only variable). Dataset: **LongMemEval (externally authored)**, oracle variant, stratified
n=138 (seed 42). Full protocol, per-type table, disclosures:
[`RESULTS_longmemeval.md`](https://github.com/doubleMcyber/longitudinal-memory-eval-harness/blob/claude/curated-brain-adapter/RESULTS_longmemeval.md).

| system | accuracy | wall time |
|---|---|---|
| Letta 0.16.8 | **0.471** | 159 min |
| **Curated Brain** | 0.261 | **20 min** |
| Mem0 2.0.7 | 0.203 | 77 min |
| Zep (Graphiti+Kuzu) | 0.065 | 254 min |

**Stated plainly:** Curated Brain **decisively beats Zep on accuracy** (p<0.0001) and beats
both Mem0 and Zep on cost; vs **Mem0** its accuracy edge (0.261 vs 0.203) is **within noise
at n=138** (paired McNemar p=0.24 — a statistical tie on accuracy, a clear cost win).
**Letta beats Curated Brain on accuracy** (0.471 vs 0.261, p=0.0002) at ~8× the wall time —
so the full "≥ each of Mem0/Letta/Zep" claim is **NOT met on this variant**. Key context: with only
~2 evidence sessions per oracle question, Letta's agent answers with the transcripts
effectively still in its context window (agentic full-context reading, the accuracy ceiling
memory systems trade against); the `_s` variant (~115k-token haystacks that overflow any
context) is the follow-up setting where retrieval quality, not context capacity, decides.
CB is the strongest of all four on **knowledge-update** (belief revision — the bi-temporal
supersede design working as intended); its measured weak spots are temporal reasoning and
preference summarization. Judge is the shared local model, not the official GPT-4o —
numbers are internally comparable, not leaderboard-comparable.

## Result — vs a named system (Mem0), preliminary and **mixed** (older, offline)

A head-to-head vs **Mem0** (`mem0ai` 2.0.7), run fully offline (Mem0 driven by a small local
model + the same offline embedder, in-memory qdrant). Mem0 is CPU-bound (~2.7 min/add), so this is
a small **n=3** subset — and, we later determined, a **CB-favorable** subset (smallest scenario
per category). The initial table:

| metric | curated_brain | temporal_rag | mem0 (Qwen-2B) |
|---|---|---|---|
| answer accuracy | **1.00** | 0.67 | 0.67 |
| recall@k | 1.00 | 1.00 | 1.00 |
| precision@k | **1.00** | 0.67 | 0.37 |
| contradiction-resolution | **1.00** | 0.00 | 0.00 |

**Corrections that must be read with that table:**

- **Broader partial run (long-0/1, lexgap-0):** on plain longitudinal recall all three systems
  **tie on answer accuracy (1.00)**; paraphrase (lexical-gap) fails for *all* three under the
  shared offline embedder. The honest synthesis is "CB leads precision and
  contradiction-resolution; answer accuracy ties on plain recall" — **not** a sweep.
- **Provenance artifact:** Mem0 rewrites facts, losing `(source_session, source_turn)` provenance,
  so this harness's recall/precision **cannot credit Mem0's design**. In a later run where Mem0
  was made fully functional (grammar-constrained JSON decoding, same Qwen3-1.7B for both systems),
  Mem0 scored **answer accuracy 1.00 — an exact tie with CB** — while its recall/precision still
  read 0.00 purely as a metric artifact. Mem0's precision/contradiction cells above are therefore
  **not claimed as CB wins**.
- **Rival handicap:** Mem0 ran on a small local model through an OpenAI-shaped shim with frequent
  JSON-parse failures in the first run — not Mem0 at its recommended cloud configuration.

## Scope & the path to the full named-rival claim

This is **preliminary** evidence: Curated Brain is competitive-to-better vs the harness's RAG
references (fully reproducible above), and the Mem0 comparison is mixed (see corrections above).
It is **not yet** the full headline claim — *"LongMemEval ≥ each of Mem0 / Letta / Zep at ≤
cost"* — which needs:

1. A **capable shared inference endpoint** (one OpenAI-compatible model that CB's extractor and
   every rival call, so the model is held constant). The providers for this already ship in CB
   (`OpenAICompatLLM` / `OpenAICompatEmbedder`); set `OPENAI_BASE_URL` + `OPENAI_API_KEY`.
2. **Letta** and **Zep** adapters in the harness pointed at that endpoint. Mem0's adapter is built
   and proven functional; a Docker-free **Zep** adapter exists (Graphiti over embedded Kuzu,
   `adapters/zep_graphiti.py`); Letta is still a stub.
3. Enough throughput to run the **full suite** for every system (the Mem0-on-CPU run above is ~5 h
   for one system; a hosted endpoint removes that wall).
4. **An externally-authored benchmark and a frozen CB configuration** — the run must use
   LongMemEval/LoCoMo as published, with CB's config tagged *before* the first scored run, rivals
   at their recommended configs, an LLM-judged answer metric, and bootstrap CIs. A win on our own
   diagnostic suite, however reproducible, is not the claim.

When (1)–(4) are available the run is one command per backend (`mem_eval.runner.cli run --backend
<name> …`) followed by `compare`. Until then, the result here is reproducible evidence that the
architecture leads **on our own diagnostic suite** on the quality axes that distinguish a memory
layer from plain RAG — precision, contradiction-resolution, and staleness — with the authorship
and metric caveats above.

**Why not just run it locally now?** Measured, not assumed (2026-06-20): the rivals all run
Docker-free (Zep via Graphiti + embedded Kuzu), but every local-inference avenue was measured to
be throughput-infeasible on this hardware — Mem0 makes many LLM calls per add; a tiny fast model
projects to **~11.5 h** for the small suite *and* is too weak to be a fair rival (0.00), a fair
model (≥2B–8B) runs at 0.03–1.2 tok/s here, and the one working forced-JSON configuration costs
~28 min per 2-turn scenario. The bottleneck is a capable shared **endpoint**, which is exactly
requirement (1).

### Honest caveats

- **Embedder parity:** CB runs on its deterministic embedder; the references use the harness's
  token-cosine embedder. A fully-fair run gives every backend the same embedder (`CB_EMBEDDER=bge`
  wires a real one into CB; the references would get `bge` too).
- **Storage slope** (517 vs 46) is a serialization artifact — `len(snapshot())` is a verbose JSON
  dump vs the references' raw chunks — not claimed as a win or a loss.
- The Mem0 numbers are **n=3 on a CB-favorable subset with a handicapped rival** — see the
  corrections block above; the only claim we stand behind is the mixed synthesis stated there.
