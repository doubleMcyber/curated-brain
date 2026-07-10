"""Stage 5 — surprise-gated selective write path. Gate: AC-5 (selectivity), with the
AC-2 long-range recall guarantee preserved."""

from __future__ import annotations

from curated_brain.backend import CuratedBrain
from curated_brain.dataset import generate, redundant_stream
from curated_brain.eval import correct
from curated_brain.surprise import DISCARD, REINFORCE, STORE, PredictiveSurprise, SurpriseGate


def test_ac5_discards_redundant_stream_without_losing_facts():
    obs, facts = redundant_stream(seed=0)
    cb = CuratedBrain(seed=0)
    stored = total = 0
    for o in obs:
        r = cb.write(o.content, session_id=o.session_id, timestamp=o.wall_ts,
                     metadata={"fact": o.fact})
        total += 1
        stored += r.stored
    discard_rate = 1 - stored / total
    assert discard_rate >= 0.80, f"gate only discarded {discard_rate:.3f} of a redundant stream"

    # No loss of salient facts: every distinct fact is still answerable.
    recalled = sum(correct(cb.answer_structured(f["subject"], f["predicate"]), f["object"])
                   for f in facts) / len(facts)
    assert recalled == 1.0, f"selectivity lost salient facts (recall {recalled:.3f})"


def test_ac5_preserves_long_range_recall():
    # AC-2 must still hold once the gate is active on the main longitudinal stream.
    ds = generate(seed=0)
    last = ds.base_ts + (ds.n_sessions - 1) * ds.day
    cb = CuratedBrain(seed=0)
    stored = total = 0
    for o in ds.observations:
        r = cb.write(o.content, session_id=o.session_id, timestamp=o.wall_ts,
                     metadata={"fact": o.fact} if o.fact else None)
        total += 1
        stored += r.stored

    probes = ds.by_category("C1")
    recall = sum(p.gold.lower() in cb.query(p.question, session_id="q", timestamp=last,
                                            k=8).context.lower() for p in probes) / len(probes)
    assert recall >= 0.90, f"gate degraded long-range recall to {recall:.3f}"
    assert 1 - stored / total >= 0.50  # the redundancy-heavy main stream is mostly dropped


def test_gate_retains_novel_salient_non_fact():
    # The gate's own no-loss property, on an item that lives ONLY in the episodic/vector
    # store (no structured fact to fall back on): a genuinely novel salient line must
    # survive even after a flood of redundant distractor noise.
    cb = CuratedBrain(seed=0)
    for i in range(40):
        cb.write("Alice email address", session_id="s000", timestamp=float(i))
    r = cb.write("The quarterly security audit found a critical vulnerability in the gateway.",
                 session_id="s001", timestamp=100.0)
    assert r.stored and r.reason == STORE
    hit = cb.query("security audit critical vulnerability", session_id="q", timestamp=200.0, k=5)
    assert "vulnerability" in hit.context.lower()


def test_gate_decision_policy():
    # Freeze theta (lr=0) to test the decision branches in isolation.
    g = SurpriseGate(theta0=0.5, reinforce_sim=0.7, theta_floor=0.0, lr=0.0)
    assert g.decide(0.6, contradiction=False) == STORE        # novel enough
    assert g.decide(0.05, contradiction=True) == STORE        # contradiction overrides novelty
    assert g.decide(0.2, contradiction=False) == REINFORCE    # near-duplicate (sim 0.8)
    assert g.decide(0.4, contradiction=False) == DISCARD      # low value, not a duplicate


def test_gate_theta_adapts_to_write_budget():
    g = SurpriseGate(budget=0.2, theta0=0.5)
    for _ in range(60):  # everything looks novel -> store-rate >> budget -> tighten
        g.decide(0.99, contradiction=False)
    assert g.theta > 0.5


# ------------------------------------------- predictive (log-prob) surprise (opt-in) --
class FakeLogprobLLM:
    """Deterministic logprob LLM keyed purely on the prompt. It parses the estimator's
    ``"Given these known memories: <ctx>. Text: <obs>"`` prompt, then assigns each generated
    token a logprob by predictability: a token from the observation that does NOT appear in
    the context is unpredictable (very negative logprob); everything already in context is
    predictable (near-zero). So a paraphrase whose tokens are all known scores ~0 surprise,
    while a paraphrase carrying a buried NEW value scores high — the dead-zone case."""

    model_id = "fake-logprob"

    def complete_with_logprobs(self, prompt: str):
        ctx, _, obs = prompt.partition(". Text: ")
        ctx = ctx.removeprefix("Given these known memories: ")
        known = set(ctx.lower().replace(",", " ").replace(".", " ").split())
        lps = []
        for tok in obs.lower().replace(",", " ").replace(".", " ").split():
            lps.append(-0.02 if tok in known else -6.0)
        return obs, lps or [-0.02]


def test_predictive_surprise_stores_paraphrased_update_a_default_brain_drops():
    # The motivating dead zone: after a known statement, a lexical paraphrase carrying a
    # buried update. Lexically it is a near-duplicate (the default gate reinforces/discards);
    # predictively the update token is unpredictable, so the predictive-on brain STORES it.
    stream = [
        ("Alice works at Acme as an engineer", "s0", 0.0),
        # same subject/words, but "senior" and "principal" are the buried update — not in the
        # prior memory, so predictively surprising while lexically near-identical.
        ("Alice works at Acme as a senior principal engineer", "s0", 1.0),
    ]

    def run(predictive: bool):
        kw = {"surprise_llm": FakeLogprobLLM()} if predictive else {}
        cb = CuratedBrain(seed=0, **kw)
        receipts = [cb.write(c, session_id=s, timestamp=t) for c, s, t in stream]
        return cb, receipts

    default_cb, default_r = run(predictive=False)
    pred_cb, pred_r = run(predictive=True)

    # The second write's decision must differ: default does not STORE it; predictive does.
    assert default_r[1].reason != STORE
    assert pred_r[1].reason == STORE
    # And the update reached the store only on the predictive brain.
    assert pred_cb.metrics()["store_size"] > default_cb.metrics()["store_size"]


def test_predictive_default_off_is_byte_identical():
    # surprise_llm=None must be byte-identical to a plain brain over the same stream.
    obs, _facts = redundant_stream(seed=0)
    a = CuratedBrain(seed=0)
    b = CuratedBrain(seed=0, surprise_llm=None)
    for o in obs:
        for cb in (a, b):
            cb.write(o.content, session_id=o.session_id, timestamp=o.wall_ts,
                     metadata={"fact": o.fact})
    assert a.snapshot() == b.snapshot()


def test_predictive_cost_counter_increments_only_when_path_runs():
    off = CuratedBrain(seed=0)
    on = CuratedBrain(seed=0, surprise_llm=FakeLogprobLLM())
    for cb in (off, on):
        cb.write("Alice works at Acme", session_id="s0", timestamp=0.0)
        cb.write("Bob writes Rust", session_id="s0", timestamp=1.0)
    assert off.metrics()["cost"]["surprise_calls"] == 0
    assert on.metrics()["cost"]["surprise_calls"] == 2  # one per gated write when enabled


def test_predictive_surprise_end_to_end_uses_provider_estimator():
    # Exercise the real PredictiveSurprise + a canned logprob provider through the brain,
    # confirming the estimator object is what the brain wires (not just the fake shape).
    ps = PredictiveSurprise(FakeLogprobLLM(), k_context=4)
    assert ps.score("a brand new unseen fact", ["Alice works at Acme"]) > 0.5
