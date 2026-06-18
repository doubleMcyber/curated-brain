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

STORE = "stored"
REINFORCE = "reinforced"
DISCARD = "discarded"


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

    @classmethod
    def from_dict(cls, d: dict) -> SurpriseGate:
        g = cls(budget=d["budget"], theta0=d["theta"], theta_floor=d["theta_floor"],
                theta_max=d["theta_max"], lr=d["lr"], reinforce_sim=d["reinforce_sim"])
        g.seen = d["seen"]
        g.stored = d["stored"]
        return g
