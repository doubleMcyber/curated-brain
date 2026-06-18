# The Curated Brain

A persistent, self-organizing **memory layer for LLM agents**, built entirely on
**frozen-model APIs** — no training, no fine-tuning. All the intelligence is in how a frozen
embedder + chat model are *orchestrated*.

> **Thesis:** memory is a **curation** problem, not a search problem. RAG fails at *write*
> time, not query time — log everything and retrieval drowns in noise. The Curated Brain
> invests at write time: it decides *what to keep*, imposes *structure*, and *continuously
> reconciles* old and new facts, so it gets **more useful the longer it runs**, not more bloated.

## Status

Honest snapshot (see [`PROGRESS.md`](PROGRESS.md) for the live, detailed state):

**Working and tested** (`pytest -q` green, `ruff` clean, fully deterministic):
- Two-tier store: a **bi-temporal structured tier** (valid + transaction time, non-lossy
  supersede, multi-hop and as-of-time queries) and a **vector tier** (cosine ANN, metadata filters).
- **Surprise-gated writes** (semantic novelty + contradiction override + adaptive threshold).
- **Hybrid retrieval** (planner → fuse by relevance × recency × importance → supersede-filter).
- **Self-organizing consolidation** (dedupe, prune, resolve contradictions, retain provenance).
- **Real local models** behind the protocols: `bge`/`e5` embeddings and a 🤗 Transformers chat
  model — with the deterministic fakes retained as test doubles so CI needs no model stack.
- **Raw-text fact extraction** with an anti-hallucination groundedness guard (the structured
  tier no longer needs spoon-fed facts).
- Byte-deterministic `snapshot`/`restore`; re-embed-on-model-upgrade migration.

**Verified by tests:** AC-1…AC-9 on a seeded synthetic longitudinal dataset (beats naive-RAG,
long-context, and no-memory baselines on every category C1–C6), plus a scoped extraction-ON
vs spoon-fed capability eval.

**Not done yet** (tracked in `PROGRESS.md`): the head-to-head **LongMemEval** benchmark against
Mem0 / Letta / Zep (the headline "contender" proof — *not yet run*); production scale (real ANN
at 10⁵–10⁶ records, durable persistence); packaging to PyPI; framework/MCP integrations. The
goal is to compete with the best open-source memory systems — **the benchmark numbers do not
exist yet**, and nothing here claims to beat them until they do.

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
CB_LIVE=1 pytest -q -k live   # opt in to the real-model tests (needs the [local] extra + a model)
```

Real-model behaviour that CI must reproduce is captured with the record/replay
[`cassette`](curated_brain/cassette.py) layer, so the gate stays deterministic without weights.

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
