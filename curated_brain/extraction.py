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

import re

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


# --------------------------------------------------------------------------------------
# Heuristic (no-LLM) extractor — a deterministic, general pattern-based fallback.
# --------------------------------------------------------------------------------------

# Temporal markers stripped when canonicalizing a predicate phrase, so "current mailing
# address", "previous mailing address" and "mailing address" collapse to ONE predicate and
# the structured tier's supersede logic fires (mirrors the topic-key idea in curation refs).
_TEMPORAL_MARKERS = frozenset(
    "current previous old new latest former currently now recent earlier originally "
    "first prior updated nowadays presently initially".split()
)

# Possessive-attribute copula: "Alice's mailing address is X", "After that, Bob's role was Y".
# Non-greedy attribute, greedy object (which only needs to *contain* the value).
_POSSESSIVE_RE = re.compile(r"\b([A-Z][a-zA-Z]*)'s\s+(.+?)\s+(?:is|was|are|were|will be)\s+(.+)")
# Verb/copula forms mapped to a canonical predicate: (regex over a clause, predicate).
_VERB_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b([A-Z][a-zA-Z]*)\s+(?:moved|relocated)\s+to\s+(.+)"), "location"),
    (re.compile(r"\b([A-Z][a-zA-Z]*)\s+(?:lives|resides)\s+in\s+(.+)"), "location"),
    (re.compile(r"\b([A-Z][a-zA-Z]*)\s+is\s+(?:based|located)\s+in\s+(.+)"), "location"),
    (re.compile(r"\b([A-Z][a-zA-Z]*)\s+is\s+headquartered\s+in\s+(.+)"), "headquarters"),
    (re.compile(r"\b([A-Z][a-zA-Z]*)\s+(?:works|worked)\s+(?:at|for)\s+(.+)"), "employer"),
    (re.compile(r"\b([A-Z][a-zA-Z]*)\s+(?:works|worked)\s+as\s+(?:an?\s+)?(.+)"), "role"),
    (re.compile(r"\b([A-Z][a-zA-Z]*)\s+was\s+promoted\s+to\s+(?:an?\s+)?(.+)"), "role"),
    (re.compile(r"\b([A-Z][a-zA-Z]*)\s+reports\s+to\s+(.+)"), "manager"),
    (re.compile(r"\b([A-Z][a-zA-Z]*)\s+is\s+an?\s+(.+)"), "role"),  # "Bob is a designer"
]
# Split into clauses only at terminal punctuation FOLLOWED BY whitespace, so emails/decimals
# ("a@b.com", "3.5") stay intact within a clause.
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")


def _canon_predicate(attr: str) -> str:
    """Canonical predicate key: content tokens minus temporal markers, space-joined.
    'current mailing address' -> 'mailing address'; 'role' -> 'role' (empty -> '')."""
    return " ".join(t for t in tokenize(attr, drop_stop=True) if t not in _TEMPORAL_MARKERS)


def _clean_object(value: str) -> str:
    """Trim surrounding whitespace and trailing sentence punctuation from an object value."""
    return value.strip().rstrip(".!?,;: ").strip()


class HeuristicExtractor:
    """Deterministic, no-LLM ``(subject, predicate, object)`` extractor.

    Parses naturalistic entity-attribute statements with a few general patterns (the same
    family of surface forms a contradiction-aware RAG reference keys on) — no model, no
    network, fully deterministic. Subjects/objects are substrings of the source by
    construction, so it never hallucinates a fact that was not stated (no grounding pass
    needed). Predicates are canonicalized (temporal markers stripped) so repeated/updated
    assertions about the same (subject, attribute) supersede rather than duplicate.

    Same ``extract(text) -> list[{"subject","predicate","object"}]`` shape as
    :class:`LLMExtractor`, so it drops into ``CuratedBrain(extractor=...)`` unchanged.
    """

    def __init__(self, *, max_facts: int = 8) -> None:
        self.max_facts = max_facts

    def extract(self, text: str) -> list[dict]:
        facts: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for clause in _CLAUSE_SPLIT_RE.split(text):
            fact = self._parse_clause(clause)
            if fact is None:
                continue
            key = (normalize(fact["subject"]), normalize(fact["predicate"]),
                   normalize(fact["object"]))
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)
            if len(facts) >= self.max_facts:
                break
        return facts

    def _parse_clause(self, clause: str) -> dict | None:
        """First matching pattern wins; possessive form (most specific) is tried first."""
        m = _POSSESSIVE_RE.search(clause)
        if m:
            subject, predicate, obj = m.group(1), _canon_predicate(m.group(2)), m.group(3)
            if predicate and _clean_object(obj):
                return {"subject": subject, "predicate": predicate, "object": _clean_object(obj)}
        for pat, predicate in _VERB_PATTERNS:
            m = pat.search(clause)
            if m:
                obj = _clean_object(m.group(2))
                if obj:
                    return {"subject": m.group(1), "predicate": predicate, "object": obj}
        return None
