"""Belief updating: when a fact changes, only the *current* value surfaces — yet you can
still ask what was true earlier (bi-temporal), and stale values never reappear as current.

Uses the deterministic local fakes, so it runs with no model download and no network.

    python examples/02_belief_update.py
"""

from curated_brain import CuratedBrain


def main() -> None:
    cb = CuratedBrain(seed=0)

    cb.write("Erin lives in Vienna.", session_id="s1", timestamp=0.0,
             metadata={"fact": {"subject": "Erin", "predicate": "city", "object": "Vienna"}})
    cb.write("Erin moved to Berlin.", session_id="s2", timestamp=100.0,
             metadata={"fact": {"subject": "Erin", "predicate": "city", "object": "Berlin"}})

    print("Current city        :", cb.answer_structured("Erin", "city"))            # Berlin
    print("As-of t=50 (earlier):", cb.answer_structured("Erin", "city", at=50.0))   # Vienna
    print()
    print("Retrieved context (the superseded 'Vienna' is not surfaced as current):")
    print(cb.query("Where does Erin live?", session_id="q", timestamp=200.0).context)


if __name__ == "__main__":
    main()
