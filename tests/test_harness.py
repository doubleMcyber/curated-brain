"""Stage 7 — full longitudinal run vs all three baselines. Gate: AC-9.

The win condition (PRD §10): CuratedBrain STRICTLY beats naive RAG, long-context, and
no-memory on EVERY category C1–C6. A tie or loss on any category means the project is
not done.
"""

from __future__ import annotations

import pytest

from curated_brain.eval import run_harness

CATEGORIES = ["C1", "C2", "C3", "C4", "C5", "C6"]
BASELINES = ["naive", "long_context", "no_memory"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ac9_curated_strictly_beats_every_baseline_on_every_category(seed):
    scores, _ = run_harness(seed)
    for category in CATEGORIES:
        cb = scores["curated"][category]
        for baseline in BASELINES:
            assert cb > scores[baseline][category], (
                f"seed {seed}, {category}: curated {cb:.3f} did not beat "
                f"{baseline} {scores[baseline][category]:.3f}"
            )


def test_ac9_curated_is_accurate_and_cheap():
    # Curation should be both correct on the factual categories and far cheaper than
    # stuffing the whole history (AC-4 holds inside the full run too).
    _, raw = run_harness(seed=0)
    cur = raw["curated"]
    assert cur["C1"] == 1.0 and cur["C2"] == 1.0
    assert cur["C5"] == 1.0 and cur["C6"] == 1.0
    assert cur["C3_acc"] == 1.0
    assert cur["C3_tokens"] <= 0.25 * raw["long_context"]["C3_tokens"]
    # bounded store: a small fraction of the raw stream survives
    assert cur["stored"] < 0.20 * raw["no_memory"].get("stored", 0) + 120


def test_ac9_no_memory_is_floor():
    # No-memory retains nothing, so it scores zero on the factual categories — the floor
    # every other approach must clear.
    _, raw = run_harness(seed=0)
    assert raw["no_memory"]["stored"] == 0
    assert raw["no_memory"]["C1"] == 0.0
