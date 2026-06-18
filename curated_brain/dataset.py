"""Seeded synthetic longitudinal dataset generator (PRD §10, Stage 1 deliverable).

Produces a deterministic, multi-session stream that exercises every eval category:

* **C1 long-range recall** — stable facts (e.g. email) injected in the first sessions,
  probed at the very end (>= 50 sessions later).
* **C2 belief updating** — attributes (city, role) that change mid-run; the old value
  must never resurface.
* **C3 retrieval cost** — a probe set whose answers live in a few facts, not the whole log.
* **C4 bounded growth / C5 selectivity** — the stream is redundancy-heavy: most lines are
  near-duplicate restatements or chit-chat that should be discarded/reinforced, not stored.
* **C5 relational / multi-hop** — manager relations support 1- and 2-hop queries.
* **C6 temporal reasoning** — as-of-time probes land inside a *past* validity interval.

Ground truth (which observation is salient, its (subject, predicate, object) triple, the
valid-from session, and the gold/stale answers) is attached so tests can score directly.
Determinism comes from a single seeded ``random.Random`` and a fixed base timestamp.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

DAY = 86_400.0
BASE_TS = 1_700_000_000.0  # fixed epoch — never the real wall-clock

_NAMES = ["Alice", "Bob", "Carol", "Dan", "Erin", "Frank", "Grace", "Heidi"]
_CITIES = ["Berlin", "Munich", "Lisbon", "Oslo", "Vienna", "Dublin", "Prague", "Madrid",
           "Helsinki", "Zurich", "Porto", "Tallinn", "Riga", "Sofia", "Athens", "Naples"]
_ROLES = ["engineer", "designer", "analyst", "manager", "researcher", "writer",
          "architect", "scientist", "recruiter", "marketer", "lawyer", "accountant"]
_PROJECTS = ["Falcon", "Mercury", "Atlas", "Nimbus", "Cobalt", "Vesper", "Onyx", "Quartz"]

_CHITCHAT = [
    "{name} mentioned the weather was pleasant today.",
    "We had a brief chat about lunch options with {name}.",
    "{name} said good morning during the standup.",
    "Someone joked about the coffee machine again near {name}.",
    "{name} shared a meme in the team channel.",
]

# Terse "topic distractors": they echo a probe's vocabulary (entity + attribute) but carry
# NO answer value. To a pure-similarity store they look maximally relevant and crowd the
# top-k, so naive RAG's recall/precision degrades — the classic RAG failure (PRD §1.1).
# A structured tier is immune: it answers by exact (subject, predicate) lookup, not topic.
_TERSE = {
    "email": ["{name} email address", "What {name} email address",
              "{name} email address what", "Email address {name}"],
    "city": ["{name} city", "What city {name}", "{name} lives city", "City {name} lives"],
    "role": ["{name} role", "What role {name}", "{name} role title", "Role {name}"],
    "manager": ["{name} manager", "Who manages {name}", "Manager {name}", "{name} manager who"],
}


@dataclass
class Observation:
    """One line in the stream, with ground-truth annotations for scoring."""

    session_id: str
    seq: int
    wall_ts: float
    actor: str
    content: str
    salient: bool
    redundant: bool  # a near-duplicate restatement (should reinforce, not store anew)
    fact: dict | None = None  # {"subject","predicate","object"} when this asserts a triple


@dataclass
class Probe:
    question: str
    category: str  # "C1".."C6"
    subject: str
    predicate: str
    gold: str
    stale: str | None = None
    as_of: float | None = None
    hops: list[str] | None = None  # predicate chain for multi-hop, e.g. ["manager", "city"]


@dataclass
class Dataset:
    seed: int
    observations: list[Observation]
    probes: list[Probe]
    base_ts: float
    day: float
    n_sessions: int
    people: list[dict] = field(default_factory=list)

    def by_category(self, cat: str) -> list[Probe]:
        return [p for p in self.probes if p.category == cat]


def _email(name: str) -> str:
    return f"{name.lower()}@example.com"


def generate(
    seed: int = 0,
    *,
    n_people: int = 6,
    n_sessions: int = 64,
    noise_per_session: int = 4,
    distractors_min: int = 3,
    distractors_max: int = 15,
) -> Dataset:
    """Build the longitudinal dataset. Same seed + params => byte-identical output."""
    rng = random.Random(seed)
    n_people = min(n_people, len(_NAMES))

    names = _NAMES[:n_people]
    init_cities = rng.sample(_CITIES, n_people)
    # New cities drawn disjoint from initial ones so stale != gold.
    new_cities = rng.sample([c for c in _CITIES if c not in init_cities], n_people)
    init_roles = rng.sample(_ROLES, n_people)
    new_roles = rng.sample([r for r in _ROLES if r not in init_roles], n_people)
    projects = rng.sample(_PROJECTS, n_people)

    people: list[dict] = []
    for p, name in enumerate(names):
        people.append(
            {
                "name": name,
                "email": _email(name),
                "init_city": init_cities[p],
                "new_city": new_cities[p],
                "init_role": init_roles[p],
                "new_role": new_roles[p],
                "project": projects[p],
                "manager": names[(p + 1) % n_people],
                "city_update_session": 30 + p,
                "role_update_session": 45 + p,
            }
        )

    # ---- build the chronological stream, session by session -------------------------
    sessions: list[list[Observation]] = [[] for _ in range(n_sessions)]

    def emit(s: int, content: str, *, salient: bool, redundant: bool, fact: dict | None,
             actor: str = "user") -> None:
        seq = len(sessions[s])
        sessions[s].append(
            Observation(
                session_id=f"s{s:03d}",
                seq=seq,
                wall_ts=BASE_TS + s * DAY + seq * 60.0,
                actor=actor,
                content=content,
                salient=salient,
                redundant=redundant,
                fact=fact,
            )
        )

    def fact(subj: str, pred: str, obj: str) -> dict:
        return {"subject": subj, "predicate": pred, "object": obj}

    # Introductions: person p's facts arrive in session p (all within the first sessions).
    for p, person in enumerate(people):
        name = person["name"]
        emit(p, f"{name}'s email address is {person['email']}.",
             salient=True, redundant=False, fact=fact(name, "email", person["email"]))
        emit(p, f"{name} lives in {person['init_city']}.",
             salient=True, redundant=False, fact=fact(name, "city", person["init_city"]))
        emit(p, f"{name} works as a {person['init_role']}.",
             salient=True, redundant=False, fact=fact(name, "role", person["init_role"]))
        emit(p, f"{name} is working on project {person['project']}.",
             salient=True, redundant=False, fact=fact(name, "project", person["project"]))
        emit(p, f"{name}'s manager is {person['manager']}.",
             salient=True, redundant=False, fact=fact(name, "manager", person["manager"]))

    # Updates: city and role change mid-run (contradictions / temporal history).
    for person in people:
        name = person["name"]
        cu = person["city_update_session"]
        emit(cu, f"{name} has moved to {person['new_city']}.",
             salient=True, redundant=False, fact=fact(name, "city", person["new_city"]))
        ru = person["role_update_session"]
        emit(ru, f"{name} was promoted to {person['new_role']}.",
             salient=True, redundant=False, fact=fact(name, "role", person["new_role"]))

    # Topic distractors: valueless, maximally-on-topic lines per (person, attribute),
    # scattered across sessions. A randomized count per pair gives natural variance, so a
    # pure-similarity store recalls some buried facts and loses others (not all-or-nothing).
    for person in people:
        for templates in _TERSE.values():
            for _ in range(rng.randint(distractors_min, distractors_max)):
                s = rng.randrange(n_sessions)
                emit(s, rng.choice(templates).format(name=person["name"]),
                     salient=False, redundant=False, fact=None)

    # Noise: dominate the stream with near-duplicate restatements + chit-chat so the
    # gate has an >=80%-discardable redundancy-heavy stream to prove selectivity against.
    known_facts: list[str] = []  # salient lines said so far, available to restate as noise
    for s in range(n_sessions):
        for _ in range(noise_per_session):
            r = rng.random()
            if r < 0.6 and known_facts:
                # near-duplicate restatement of an already-stored fact -> reinforce
                content = rng.choice(known_facts)
                emit(s, content, salient=False, redundant=True, fact=None)
            else:
                name = rng.choice(names)
                content = rng.choice(_CHITCHAT).format(name=name)
                emit(s, content, salient=False, redundant=False, fact=None)
        # after a session, anything salient said in it becomes restatable noise
        for obs in sessions[s]:
            if obs.salient:
                known_facts.append(obs.content)

    observations = [obs for sess in sessions for obs in sess]

    # ---- probes ---------------------------------------------------------------------
    probes: list[Probe] = []
    last_ts = BASE_TS + (n_sessions - 1) * DAY
    for person in people:
        name = person["name"]
        manager = person["manager"]
        mgr_person = next(q for q in people if q["name"] == manager)

        # C1 — stable long-range fact (email), injected early, probed now.
        probes.append(Probe(f"What is {name}'s email address?", "C1",
                             name, "email", person["email"]))
        # C2 — belief updating: current city/role; old value must not resurface.
        probes.append(Probe(f"Where does {name} live now?", "C2",
                             name, "city", person["new_city"], stale=person["init_city"]))
        probes.append(Probe(f"What is {name}'s current role?", "C2",
                             name, "role", person["new_role"], stale=person["init_role"]))
        # C5 — relational (1-hop) and multi-hop (2-hop).
        probes.append(Probe(f"Who is {name}'s manager?", "C5",
                             name, "manager", manager))
        probes.append(Probe(f"What city does {name}'s manager live in?", "C5",
                             name, "manager", mgr_person["new_city"], hops=["manager", "city"]))
        # C6 — temporal as-of: a date inside the *pre-move* interval.
        as_of_ts = BASE_TS + (person["city_update_session"] - 1) * DAY
        probes.append(Probe(
            f"Where did {name} live as of session "
            f"{person['city_update_session'] - 1}?",
            "C6", name, "city", person["init_city"], as_of=as_of_ts))

    # C3 — retrieval-cost probe set reuses C1/C2/C5 questions (answerable from few facts).
    for base in list(probes):
        if base.category in ("C1", "C2") and base.hops is None:
            probes.append(Probe(base.question, "C3", base.subject, base.predicate,
                                base.gold, stale=base.stale, as_of=last_ts))

    return Dataset(
        seed=seed,
        observations=observations,
        probes=probes,
        base_ts=BASE_TS,
        day=DAY,
        n_sessions=n_sessions,
        people=people,
    )
