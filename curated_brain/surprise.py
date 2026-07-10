"""Surprise-gated selective write path (PRD §6, Pillar B).

The default action on input is **discard**. An item is persisted to the episodic/vector
store only when its **surprise** clears an adaptive threshold:

* **Semantic novelty (primary):** ``1 − max cosine`` to the nearest existing memory.
* **Contradiction (override):** a candidate that conflicts with a stored fact is forced
  high — it must be stored so the conflict can be reconciled (PRD §8).

Redundant input that closely matches an existing memory **reinforces** it (a support/recency
bump, no new row) rather than duplicating it; everything else is dropped.

θ is **adaptive, calibrated to a write budget** (persist ≤ ``budget`` of the stream): when
the running store-rate runs above budget θ rises (stricter); it relaxes toward a floor when
below, so a redundant stream stays selective regardless of volume. All updates are pure
functions of the observed sequence, so the gate is fully deterministic.
"""

from __future__ import annotations

import math
from typing import Protocol

STORE = "stored"
REINFORCE = "reinforced"
DISCARD = "discarded"


class _LogprobLLM(Protocol):
    """The narrow seam PredictiveSurprise needs: a completion that returns per-token
    logprobs (``OpenAICompatLLM.complete_with_logprobs`` / any deterministic fake)."""

    def complete_with_logprobs(self, prompt: str) -> tuple[str, list[float]]: ...


class PredictiveSurprise:
    """Predictive (log-prob) surprise estimator — PRD §6 #2, an OPT-IN second surprise signal.

    Surprise is prediction error: text the model finds UNPREDICTABLE given what memory already
    holds is surprising, even when it is lexically near-duplicate. This closes the
    paraphrased-update dead zone — a paraphrase of a known memory carrying a buried update has
    low cosine novelty but the update tokens are unpredictable, so this scores high where the
    lexical gate would reinforce/discard.

    ``score`` builds a deterministic prompt (a pure function of ``observation`` and
    ``context_texts`` — no timestamps, no randomness) and asks the model for the per-token
    logprobs of a short continuation. We map the MEAN token logprob to 0..1 via
    ``1 - exp(mean_logprob)``. mean_logprob is the log of the geometric-mean per-token
    probability (equivalently ``-log(perplexity)``), so ``exp(mean_logprob)`` is that mean
    probability in (0, 1]; ``1 - it`` is monotone decreasing in predictability — a confidently
    predicted continuation (prob→1) scores →0, an unpredictable one (prob→0) scores →1. Chosen
    over raw perplexity because it is already bounded to 0..1, needs no squashing constant, and
    fuses directly with lexical novelty (also 0..1) by ``max``.
    """

    def __init__(self, llm: _LogprobLLM, k_context: int = 4) -> None:
        self.llm = llm
        self.k_context = k_context

    def prompt(self, observation: str, context_texts: list[str]) -> str:
        """The scored prompt. Pure function of its inputs (nearest-k context, in the order
        given) so the score is reproducible for a given LLM."""
        ctx = " ".join(context_texts[: self.k_context])
        return f"Given these known memories: {ctx}. Text: {observation}"

    def score(self, observation: str, context_texts: list[str]) -> float:
        _text, logprobs = self.llm.complete_with_logprobs(
            self.prompt(observation, context_texts))
        if not logprobs:
            # No logprobs returned -> no predictive signal; contribute nothing (the caller
            # fuses by max, so 0.0 leaves the lexical novelty untouched).
            return 0.0
        mean_lp = sum(logprobs) / len(logprobs)
        return 1.0 - math.exp(mean_lp)


class SurpriseGate:
    def __init__(self, *, budget: float = 0.2, theta0: float = 0.5, theta_floor: float = 0.35,
                 theta_max: float = 0.95, lr: float = 0.05, reinforce_sim: float = 0.7) -> None:
        self.budget = budget
        self.theta = theta0
        self.theta_floor = theta_floor
        self.theta_max = theta_max
        self.lr = lr
        self.reinforce_sim = reinforce_sim
        self.seen = 0
        self.stored = 0

    def decide(self, novelty: float, *, contradiction: bool) -> str:
        if contradiction or novelty >= self.theta:
            decision = STORE
        elif (1.0 - novelty) >= self.reinforce_sim:
            decision = REINFORCE
        else:
            decision = DISCARD

        self.seen += 1
        if decision == STORE:
            self.stored += 1
        rate = self.stored / self.seen
        self.theta = min(self.theta_max, max(self.theta_floor,
                                             self.theta + self.lr * (rate - self.budget)))
        return decision

    def to_dict(self) -> dict:
        return {"budget": self.budget, "theta": self.theta, "theta_floor": self.theta_floor,
                "theta_max": self.theta_max, "lr": self.lr, "reinforce_sim": self.reinforce_sim,
                "seen": self.seen, "stored": self.stored}

    _KEYS = ("budget", "theta", "theta_floor", "theta_max", "lr", "reinforce_sim",
             "seen", "stored")

    @classmethod
    def from_dict(cls, d: dict) -> SurpriseGate:
        # Untrusted snapshot: fail with a clear error, not an opaque KeyError/TypeError.
        if not isinstance(d, dict):
            raise ValueError(f"gate snapshot must be an object, got {type(d).__name__}")
        missing = [k for k in cls._KEYS if k not in d]
        if missing:
            raise ValueError(f"gate snapshot missing keys {missing}")
        if not all(isinstance(d[k], (int, float)) and not isinstance(d[k], bool)
                   for k in cls._KEYS):
            raise ValueError("gate snapshot values must all be numbers")
        g = cls(budget=d["budget"], theta0=d["theta"], theta_floor=d["theta_floor"],
                theta_max=d["theta_max"], lr=d["lr"], reinforce_sim=d["reinforce_sim"])
        g.seen = int(d["seen"])
        g.stored = int(d["stored"])
        return g
