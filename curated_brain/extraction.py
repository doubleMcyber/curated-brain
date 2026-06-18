"""Track B — extract atomic facts from raw text with a frozen LLM.

This is the *general* path that replaces the eval dataset's spoon-fed ``metadata.fact``:
given an unstructured observation, prompt the LLM for ``subject | predicate | object``
triples and parse them into the fact dicts the structured tier already consumes
(``{"subject", "predicate", "object"}`` — see ``CuratedBrain.write``). Subjects/objects are
lightly canonicalized so they resolve against existing entities (PRD §5.1).

The LLM is the only real dependency and it stays behind the ``LLM`` protocol, so the same
extractor runs on a real local model, on a hosted model, or — for deterministic CI — on a
:class:`~curated_brain.cassette.CachedLLM` replaying genuine recorded completions.
"""

from __future__ import annotations

from curated_brain.util import normalize, tokenize

# Schema-constrained, few-shot prompt (the roadmap's mitigation for weak small-model
# extraction): it pins the allowed predicates, shows the exact triple format, and
# demonstrates NONE for non-facts so chit-chat doesn't hallucinate triples.
_PROMPT = (
    "Extract facts about people as lines of 'subject | predicate | object'.\n"
    "Allowed predicates: city, role, email, manager, project.\n"
    "Use the person's name as the subject. If there is no such fact, output NONE.\n"
    "One fact per line, no commentary.\n\n"
    "Text: Alice relocated to Berlin in March.\nFacts:\nAlice | city | Berlin\n\n"
    "Text: Frank was promoted to senior analyst.\nFacts:\nFrank | role | senior analyst\n\n"
    "Text: It was a sunny afternoon and nothing happened.\nFacts:\nNONE\n\n"
    "Text: {text}\nFacts:"
)


def _supported(value: str, text_tokens: set[str]) -> bool:
    """True iff every content token of ``value`` occurs in the source text, matched at word
    level so ``man`` is not satisfied by ``management``. Token-empty values are unsupported."""
    toks = tokenize(value, drop_stop=False)
    return bool(toks) and set(toks) <= text_tokens


class LLMExtractor:
    """Turn raw text into atomic ``(subject, predicate, object)`` facts via a frozen LLM."""

    def __init__(self, llm, *, max_facts: int = 8, prompt: str = _PROMPT) -> None:
        self.llm = llm
        self.max_facts = max_facts
        self.prompt = prompt

    def extract(self, text: str, *, ground: bool = True) -> list[dict]:
        """Return parsed facts (possibly empty). Robust to bullets, blank lines, and the
        model echoing prose around the triples — only well-formed ``a | b | c`` lines with
        three non-empty fields are kept, deduplicated, and capped at ``max_facts``.

        When ``ground`` is set (default), a fact is kept only if its subject *and* object
        are supported by the source text. This is the anti-hallucination guard (PRD §12):
        it discards few-shot exemplars the model leaks and triples invented from chit-chat,
        without which a weak local model would poison the store with facts that were never
        stated. The predicate is inferred (mapped to the schema) so it is not grounded.
        """
        raw = self.llm.complete(self.prompt.format(text=text))
        text_tokens = set(tokenize(text, drop_stop=False))
        facts: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for line in raw.splitlines():
            line = line.strip().lstrip("-*•").strip()
            if not line or line.upper() == "NONE":
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 3 or not all(parts):
                continue
            subject, predicate, obj = parts
            if ground and not (_supported(subject, text_tokens)
                               and _supported(obj, text_tokens)):
                continue
            key = (normalize(subject), normalize(predicate), normalize(obj))
            if key in seen:
                continue
            seen.add(key)
            facts.append({"subject": subject, "predicate": predicate, "object": obj})
            if len(facts) >= self.max_facts:
                break
        return facts
