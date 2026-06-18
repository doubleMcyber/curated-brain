# Contributing

Thanks for your interest in The Curated Brain — a persistent, self-organizing memory layer
for LLM agents, built on frozen-model APIs.

## Setup

```bash
pip install -e ".[dev]"     # core + test/lint tooling
pip install -e ".[local]"   # optional: real local models (sentence-transformers, transformers, torch)
```

## The gate (must stay green)

```bash
pytest -q
ruff check .
```

Real-model tests are gated behind `CB_LIVE=1` and skip by default, so the gate runs offline
and deterministically with no model weights or network.

## Conventions

- **Keep the deterministic fakes as the default test doubles.** Real providers plug in behind
  the existing `Embedder` / `LLM` protocols — no concrete vendor leaks into core logic.
- **Preserve byte-deterministic `snapshot` → `restore`** (AC-1). If you add state that drives a
  query but can't be faithfully rebuilt from the stores, persist it in the snapshot.
- **New behavior needs a test.** New *real-model* behavior should be captured as a cassette
  (`curated_brain/cassette.py`) so CI replays genuine output deterministically — no mocks that
  fake away the model's real quality.
- One logical change per PR; conventional commit messages.

## Where things live

`backend.py` (adapter + orchestration), `structured.py` (bi-temporal tier), `vector.py`
(ANN + filters), `surprise.py` (gate), `retrieval.py` (plan/fuse/supersede-filter),
`consolidation.py`, `providers.py` / `extraction.py` (real models). Design: `PRD.md`;
roadmap and current state: `PROGRESS.md`.
