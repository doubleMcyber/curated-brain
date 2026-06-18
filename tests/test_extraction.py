"""Track B — LLM fact extraction from raw text (no `metadata.fact` spoon-feeding).

The offline tests replay a cassette of **genuine** completions recorded from a real local
model (`tests/fixtures/extract_cassette.json`), so they exercise the real extractor on real
model output — deterministically, with no model present and no hand-written regex stand-in.
A replay *miss* raises, which guarantees the assertions can't be satisfied by fabricated
text. The live test (opt-in) re-derives the same extraction from the model directly.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from curated_brain.cassette import CachedLLM, Cassette
from curated_brain.extraction import LLMExtractor
from curated_brain.providers import TransformersLLM

FIXTURE = Path(__file__).parent / "fixtures" / "extract_cassette.json"
LIVE_LLM = os.environ.get("CB_LIVE") == "1" and importlib.util.find_spec("transformers")


def _replayer() -> LLMExtractor:
    return LLMExtractor(CachedLLM(Cassette.load(str(FIXTURE)), inner=None))


def test_extracts_grounded_triples_from_recorded_llm_output():
    ext = _replayer()
    assert ext.extract("Erin moved to Vienna last spring.") == [
        {"subject": "Erin", "predicate": "city", "object": "Vienna"}]
    assert ext.extract("Bob was promoted to engineering manager.") == [
        {"subject": "Bob", "predicate": "role", "object": "engineering manager"}]
    assert ext.extract("Cara's email address is cara@example.com.") == [
        {"subject": "Cara", "predicate": "email", "object": "cara@example.com"}]


def test_grounding_filters_leaked_examples_and_chitchat():
    ext = _replayer()
    # On the Cara sentence the weak model LEAKS the few-shot exemplars (Alice/Frank) as if
    # they were facts; grounding must drop them since they aren't in the source text.
    cara = "Cara's email address is cara@example.com."
    leaked = {"Alice", "Frank"}
    raw_subjects = {f["subject"] for f in ext.extract(cara, ground=False)}
    grounded_subjects = {f["subject"] for f in ext.extract(cara, ground=True)}
    assert leaked & raw_subjects            # the leakage really is present in raw output...
    assert not (leaked & grounded_subjects)  # ...and grounding removes it
    # chit-chat yields no grounded facts — without grounding the model would inject triples.
    assert ext.extract("The weather was pleasant and nothing else happened.") == []


def test_replay_miss_raises_so_assertions_use_real_recorded_output():
    with pytest.raises(KeyError):
        _replayer().extract("An unrecorded sentence about nobody in particular.")


@pytest.mark.skipif(not LIVE_LLM, reason="set CB_LIVE=1 with the 'local' extra + a cached model")
def test_live_extraction_is_grounded():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model = os.environ.get("CB_LLM_MODEL", "Qwen/Qwen3.5-0.8B")
    try:
        ext = LLMExtractor(TransformersLLM(model_name=model, device="cpu", max_new_tokens=48))
        facts = ext.extract("Erin moved to Vienna last spring.")
    except Exception as e:  # cached model absent / unloadable
        pytest.skip(f"cached model {model} unavailable: {e}")
    assert any(f["subject"] == "Erin" and f["object"] == "Vienna" for f in facts)
    hay = "erin moved to vienna last spring."
    assert all(f["subject"].lower() in hay and f["object"].lower() in hay for f in facts)
