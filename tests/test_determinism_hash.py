"""Gate A determinism anchor — in-repo behavior-lock regression (PRD §10, AC-1).

This is the in-repo *Gate A anchor*. It runs a fixed scripted workload on the
deterministic test doubles and pins a sha256 over the final ``snapshot()`` bytes and
over the concatenated query outputs. Both hashes are computed from genuine current
behavior of the CURRENT tree, verified stable across two in-process builds, and hard-coded
below.

Any INTENTIONAL behavior change (write path, retrieval, consolidation, snapshot schema, the
dataset generator, or the deterministic fakes) will move one or both hashes and fail this
test. When that happens the pinned constant must be updated *in the same commit* as the
change, with a one-line justification in the commit message explaining why the new bytes are
correct. Never adjust the workload to make the hash "look nice" — recompute once from real
behavior, confirm it is stable, and re-pin.

The workload deliberately covers the whole pipeline: write every observation, run every
probe as a query (reusing the dataset's own timestamps — no wall-clock), consolidate, then
snapshot. It runs in well under a second on the fakes, so it stays inside the fast/offline
gate.
"""

from __future__ import annotations

import hashlib

from curated_brain.backend import CuratedBrain
from curated_brain.dataset import generate
from curated_brain.fakes import DeterministicEmbedder, RuleBasedLLM

SEED = 42

# Pinned anchors — computed from the current tree, verified stable across a double build.
EXPECTED_SNAPSHOT_SHA256 = "374fff1c6a587e8c5a1f6525674c0118cddba6a69e9f43c1245c7de9942b7304"
EXPECTED_QUERY_SHA256 = "39e99110f007ea0c05a500e575cbb2d295a18c37caa9f11261f38f55925e7d0a"


def _run() -> tuple[str, str]:
    """One full build of the scripted Gate A workload; returns (snapshot_sha, query_sha)."""
    ds = generate(seed=SEED)
    cb = CuratedBrain(embedder=DeterministicEmbedder(), seed=SEED)
    last_ts = ds.base_ts + (ds.n_sessions - 1) * ds.day

    for obs in ds.observations:
        cb.write(obs.content, session_id=obs.session_id, timestamp=obs.wall_ts,
                 metadata={"fact": obs.fact} if obs.fact else None)

    outputs: list[str] = []
    for p in ds.probes:
        t = p.as_of if p.as_of is not None else last_ts
        r = cb.query(p.question, session_id="sQ", timestamp=t, k=8)
        outputs.append(r.context)

    cb.consolidate()

    snapshot_sha = hashlib.sha256(cb.snapshot()).hexdigest()
    query_sha = hashlib.sha256("\n".join(outputs).encode("utf-8")).hexdigest()
    return snapshot_sha, query_sha


def test_rule_based_llm_double_is_deterministic():
    # The RuleBasedLLM test double is one of the frozen fakes the Gate A workload rests on;
    # pin its determinism so a change there surfaces here rather than silently.
    llm = RuleBasedLLM()
    prompt = "short line\na noticeably longer salient line\nmid line"
    assert llm.complete(prompt) == llm.complete(prompt) == "a noticeably longer salient line"


def test_workload_is_stable_in_process():
    # Two independent builds in the same process must agree before we trust the pin.
    assert _run() == _run()


def test_gate_a_hash_matches_pin():
    snapshot_sha, query_sha = _run()
    assert snapshot_sha == EXPECTED_SNAPSHOT_SHA256, (
        "snapshot() bytes changed — if intentional, update EXPECTED_SNAPSHOT_SHA256 in this "
        "same commit with justification"
    )
    assert query_sha == EXPECTED_QUERY_SHA256, (
        "query outputs changed — if intentional, update EXPECTED_QUERY_SHA256 in this same "
        "commit with justification"
    )
