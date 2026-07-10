"""Predictive (log-prob) surprise, pinned to GENUINE qwen2.5:7b output via a recorded cassette.

The opt-in ``PredictiveSurprise`` estimator (curated_brain/surprise.py) turns a model's
per-token prediction error into a 0..1 surprise score, fused into the write gate by ``max``.
Its promise is the *paraphrased-update dead zone*: a line that is lexically a near-duplicate
of a known memory but buries a NEW value, which the lexical gate reinforces/discards while a
predictive gate could store. The synthetic ``FakeLogprobLLM`` in test_selectivity proves the
mechanism *can* fire; this file measures whether the REAL model actually does, on a fixed
scenario, and pins the honest outcome so it cannot silently drift.

Cassette provenance: recorded 2026-07-10 from Ollama-served ``qwen2.5:7b`` at
``http://localhost:11434/v1`` (temperature 0.0, greedy, max_tokens 32, logprobs on) via
``CachedLLM(inner=OpenAICompatLLM(...))`` over the scenario below. 22 distinct prompts,
~15 KiB. The tests here replay offline (``inner=None``); a replay miss RAISES, so no
assertion can be satisfied by fabricated logprobs. Do NOT hand-edit the cassette; re-record
by re-running the recorder against the same model if the scenario changes.

HONEST MEASURED RESULT (qwen2.5:7b, 2026-07-10). On this scenario the predictive gate rescued
ZERO of the five dead-zone updates — the same zero the default gate managed. The mean log-prob
over a 32-token continuation is dominated by predictable generic filler ("Based on the provided
text, ..."), so the surprising update token is diluted: measured predictive scores for the
buried updates land at 0.24-0.30, well under the gate's threshold theta (>=0.5, and driven up
to ~0.88 by the seed writes). The predictive signal did NOT degrade discard selectivity (it
rose slightly, 0.333 -> 0.375). This test pins that negative: with the shipped
mean-logprob mapping and this real model, predictive surprise does not close the dead zone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from curated_brain.backend import CuratedBrain
from curated_brain.cassette import CachedLLM, Cassette
from curated_brain.fakes import DeterministicEmbedder

FIXTURE = Path(__file__).parent / "fixtures" / "logprob_cassette.json"

# Fixed measurement scenario. Each entry: (content, fact_or_None, kind[, new_value]).
#   seed   - a distinct fact, stated once (fact supplied -> stored).
#   dup    - an exact duplicate of a seed (both gates reinforce/discard).
#   para   - a pure paraphrase, no new information (neither gate should store).
#   update - a paraphrase that BURIES a new value (the dead zone). NO fact is supplied: if it
#            were, the structured tier's contradiction detector would force BOTH gates to store
#            it and erase the difference the predictive gate is meant to show. As raw text an
#            update is only LEXICALLY near-duplicate, so the default gate (lexical novelty
#            only) reinforces/discards it; a predictive gate could score the unpredictable
#            update token high. ``new_value`` is the token whose presence in the episodic store
#            proves the update was rescued.
SCENARIO: list[tuple] = [
    ("Alice works at Acme as an engineer.",
     {"subject": "Alice", "predicate": "role", "object": "engineer"}, "seed"),
    ("Bob lives in Berlin.",
     {"subject": "Bob", "predicate": "city", "object": "Berlin"}, "seed"),
    ("Carol's email is carol@example.com.",
     {"subject": "Carol", "predicate": "email", "object": "carol@example.com"}, "seed"),
    ("Dan is working on project Falcon.",
     {"subject": "Dan", "predicate": "project", "object": "Falcon"}, "seed"),
    ("Erin leads the platform team.",
     {"subject": "Erin", "predicate": "role", "object": "platform lead"}, "seed"),

    ("Alice works at Acme as an engineer.", None, "dup"),
    ("Bob lives in Berlin.", None, "dup"),
    ("Carol's email is carol@example.com.", None, "dup"),
    ("Dan is working on project Falcon.", None, "dup"),
    ("Erin leads the platform team.", None, "dup"),
    ("Alice works at Acme as an engineer.", None, "dup"),
    ("Bob lives in Berlin.", None, "dup"),

    ("Alice is employed at Acme in an engineering role.", None, "para"),
    ("Bob's home is in the city of Berlin.", None, "para"),
    ("You can reach Carol at carol@example.com.", None, "para"),
    ("Dan has been assigned to the Falcon project.", None, "para"),
    ("Erin is the lead of the platform team.", None, "para"),
    ("Alice does engineering work over at Acme.", None, "para"),
    ("Bob makes his home in Berlin.", None, "para"),

    ("Alice works at Acme as a senior principal engineer.", None, "update", "principal"),
    ("Bob lives in Munich now.", None, "update", "Munich"),
    ("Carol's new email is carol.smith@example.org.", None, "update", "example.org"),
    ("Dan is now working on project Mercury.", None, "update", "Mercury"),
    ("Erin leads the security team now.", None, "update", "security"),
]

_UPDATES = [e for e in SCENARIO if e[2] == "update"]


def _replayer() -> CachedLLM:
    return CachedLLM(Cassette.load(str(FIXTURE)), inner=None)


def _run(*, predictive: bool) -> CuratedBrain:
    """Run the full scenario through a brain, default gate vs cassette-replay predictive gate.
    The embedder is the deterministic fake, so the ONLY live-model influence is the replayed
    logprob surprise — the measurement isolates the predictive signal."""
    kw = {"surprise_llm": _replayer()} if predictive else {}
    cb = CuratedBrain(embedder=DeterministicEmbedder(), seed=0, **kw)
    for i, entry in enumerate(SCENARIO):
        content, fact = entry[0], entry[1]
        cb.write(content, session_id="s0", timestamp=float(i),
                 metadata={"fact": fact} if fact else None)
    return cb


def _updates_rescued(cb: CuratedBrain) -> int:
    """How many dead-zone updates' new value reached the episodic store."""
    store = "\n".join(r.content for r in cb._episodes).lower()
    return sum(1 for e in _UPDATES if e[3].lower() in store)


def test_dead_zone_rescue_measured_on_real_qwen():
    # The measurement. Default gate vs predictive gate over the identical scenario.
    default = _run(predictive=False)
    predictive = _run(predictive=True)

    default_rescued = _updates_rescued(default)
    predictive_rescued = _updates_rescued(predictive)

    # Baseline sanity: the DEFAULT lexical gate misses the entire dead zone (this is the gap
    # the predictive estimator exists to close). If this ever becomes non-zero the scenario
    # stopped exercising the dead zone and the comparison below is meaningless.
    assert default_rescued == 0, (
        f"default gate unexpectedly rescued {default_rescued} updates — scenario no longer "
        f"exercises the paraphrased-update dead zone")

    # HONEST PINNED OUTCOME (qwen2.5:7b, 2026-07-10): the predictive gate rescues NONE of the
    # five buried updates either. The mean-logprob-over-continuation mapping dilutes the
    # surprising token below the gate threshold. The predictive gate stores AT LEAST as many
    # as the default (never fewer), but on this real model that floor is zero.
    assert predictive_rescued >= default_rescued
    assert predictive_rescued == 0, (
        f"predictive gate rescued {predictive_rescued}/{len(_UPDATES)} dead-zone updates on "
        f"qwen2.5:7b — if the model/scenario changed and this is now a genuine rescue, update "
        f"this pin and the module docstring's measured-result section together")


def test_predictive_does_not_blow_the_discard_rate():
    # Selectivity guardrail: turning the predictive signal on must not wreck the gate's
    # discard rate on redundant input. Measured: default 0.333, predictive 0.375 (it discards
    # one update the default reinforced). Assert predictive stays within tolerance of default
    # and, in particular, is not WORSE (lower) than default beyond that tolerance.
    default_rate = _run(predictive=False).metrics()["discard_rate"]
    predictive_rate = _run(predictive=True).metrics()["discard_rate"]
    tolerance = 0.10
    assert predictive_rate >= default_rate - tolerance, (
        f"predictive gate degraded discard rate from {default_rate:.3f} to "
        f"{predictive_rate:.3f} (tolerance {tolerance})")


def test_double_replay_is_byte_identical():
    # Determinism: two independent replay runs of the predictive brain produce byte-identical
    # snapshots (the cassette + deterministic embedder pin every input).
    a = _run(predictive=True)
    b = _run(predictive=True)
    assert a.snapshot() == b.snapshot()


def test_replay_miss_raises_so_the_pin_rests_on_real_logprobs():
    # A prompt that was never recorded must RAISE on replay, so no assertion above can be
    # met by an unrecorded (fabricated) logprob payload.
    with pytest.raises(KeyError):
        _replayer().complete_with_logprobs(
            "Given these known memories: nothing. Text: an unrecorded probe line")


# ---- cassette extension unit tests (offline, injected fake inner) --------------------
class _FakeInner:
    """Deterministic logprob source keyed on the prompt, for the cassette unit tests."""

    model_id = "fake-inner"

    def __init__(self) -> None:
        self.calls = 0

    def complete_with_logprobs(self, prompt: str) -> tuple[str, list[float]]:
        self.calls += 1
        return f"echo:{prompt}", [-0.5, -1.25, -2.0]


def test_cassette_records_then_replays_logprobs_without_calling_inner():
    inner = _FakeInner()
    cass = Cassette()
    rec = CachedLLM(cass, inner=inner)
    text, lps = rec.complete_with_logprobs("p1")
    assert text == "echo:p1" and lps == [-0.5, -1.25, -2.0]
    assert inner.calls == 1
    # A second call for the same prompt is served from the recording (inner not hit again).
    text2, lps2 = rec.complete_with_logprobs("p1")
    assert (text2, lps2) == (text, lps)
    assert inner.calls == 1

    # Replay (inner=None) from the same cassette returns the recorded pair exactly.
    replay = CachedLLM(cass, inner=None)
    assert replay.complete_with_logprobs("p1") == ("echo:p1", [-0.5, -1.25, -2.0])


def test_cassette_logprob_namespace_is_distinct_from_complete():
    # complete() and complete_with_logprobs() of the SAME prompt use separate namespaces
    # and never collide.
    class _Inner(_FakeInner):
        def complete(self, prompt: str) -> str:
            return f"plain:{prompt}"

    cass = Cassette()
    rec = CachedLLM(cass, inner=_Inner())
    assert rec.complete("p") == "plain:p"
    text, _lps = rec.complete_with_logprobs("p")
    assert text == "echo:p"
    # Both namespaces populated, keyed by the same prompt hash, holding different payloads.
    assert set(cass.complete) and set(cass.complete_lp)


def test_cassette_logprob_survives_dict_roundtrip():
    cass = Cassette()
    CachedLLM(cass, inner=_FakeInner()).complete_with_logprobs("p1")
    restored = Cassette.from_dict(cass.to_dict())
    assert CachedLLM(restored, inner=None).complete_with_logprobs("p1") == (
        "echo:p1", [-0.5, -1.25, -2.0])


def test_cassette_logprob_replay_miss_raises():
    with pytest.raises(KeyError):
        CachedLLM(Cassette(), inner=None).complete_with_logprobs("never recorded")
