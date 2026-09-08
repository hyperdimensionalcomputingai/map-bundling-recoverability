"""Canonical role-value facts remain visible beside their bound vectors."""

import unicodedata

import torch

from .readout import bind, bundle
from .settings import PREFIXES, ROLES


def canonical(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Expected text")
    text = " ".join(unicodedata.normalize("NFKC", text).split()).lower()
    if not text:
        raise ValueError("Empty text is not a semantic value")
    return text


def prefixed(text: str, mode: str) -> str:
    if mode not in PREFIXES:
        raise ValueError(f"Unknown embedding mode: {mode}")
    return PREFIXES[mode] + canonical(text)


def person_facts(person: dict) -> list[dict]:
    pairs = []
    if person.get("age") is not None:
        age = person["age"]
        if not isinstance(age, int) or isinstance(age, bool) or age < 0:
            raise ValueError("age must be a nonnegative integer or missing")
        pairs.append(("age", f"{age} years old"))
    if person.get("eye_color") is not None:
        pairs.append(("eye_color", canonical(person["eye_color"]) + " eyes"))
    interests = [canonical(text) for text in (person.get("interests") or [])]
    if len(set(interests)) != len(interests):
        raise ValueError(f"Duplicate canonical interests for {person['id']}")
    pairs.extend(("interest", text) for text in sorted(interests))
    return [
        {
            "fact_id": f"{person['id']}:{role}:{text}",
            "person_id": person["id"],
            "role": role,
            "text": text,
            "input": prefixed(text, "document"),
        }
        for role, text in pairs
    ]


def query_input(query: dict) -> str:
    if query["role"] not in ROLES:
        raise ValueError(f"Unknown query role: {query['role']}")
    return prefixed(query["text"], "query")


def required_inputs(people: list[dict], queries: list[dict]) -> list[str]:
    if len({p["id"] for p in people}) != len(people):
        raise ValueError("Duplicate person IDs")
    if len({q["id"] for q in queries}) != len(queries):
        raise ValueError("Duplicate query IDs")
    documents = [fact["input"] for person in people for fact in person_facts(person)]
    return sorted(set(documents + [query_input(query) for query in queries]))


def encode_person(person: dict, values: dict[str, torch.Tensor], roles: dict) -> tuple:
    facts = person_facts(person)
    bound = [bind(roles[fact["role"]], values[fact["input"]]) for fact in facts]
    return bundle(bound), facts, bound
