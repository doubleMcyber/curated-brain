"""Basic memory: store facts, retrieve a compact context, answer exact + multi-hop queries.

Uses the deterministic local fakes, so it runs with no model download and no network.

    python examples/01_basic_memory.py
"""

from curated_brain import CuratedBrain


def main() -> None:
    cb = CuratedBrain(seed=0)

    facts = [
        ("Erin lives in Vienna.",
         {"subject": "Erin", "predicate": "city", "object": "Vienna"}),
        ("Bob reports to Erin.",
         {"subject": "Bob", "predicate": "manager", "object": "Erin"}),
        ("Erin's email is erin@example.com.",
         {"subject": "Erin", "predicate": "email", "object": "erin@example.com"}),
    ]
    for i, (text, fact) in enumerate(facts):
        cb.write(text, session_id="s1", timestamp=float(i), metadata={"fact": fact})

    print("Q: Where does Erin live?")
    print(cb.query("Where does Erin live?", session_id="q", timestamp=10.0).context)
    print()
    print("Exact lookup     — Erin's city:", cb.answer_structured("Erin", "city"))
    print("Multi-hop        — Bob's manager's city:",
          cb.answer_path("Bob", ["manager", "city"]))
    print("Observability    — metrics:", cb.metrics())


if __name__ == "__main__":
    main()
