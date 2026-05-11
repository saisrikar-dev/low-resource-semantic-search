from __future__ import annotations

import re
from dataclasses import dataclass

NORMALIZE_PATTERN = re.compile(r"[^a-z0-9\s-]+")
WHITESPACE_PATTERN = re.compile(r"\s+")

SEMANTIC_GROUPS = (
    ("agriculture", "farming", "kilimo", "gonar", "oko"),
    ("water", "maji", "omi", "irrigation", "umwagiliaji"),
    ("health", "clinic", "hospital", "afya", "asibiti", "lafiya"),
    ("education", "school", "eko", "makaranta"),
    ("market", "trade", "business", "kasuwa", "ciniki"),
    ("climate", "drought", "ukame", "adaptation"),
)

LANGUAGE_HINTS = {
    "kilimo": "sw",
    "maji": "sw",
    "afya": "sw",
    "umwagiliaji": "sw",
    "gonar": "ha",
    "asibiti": "ha",
    "lafiya": "ha",
    "kasuwa": "ha",
    "ciniki": "ha",
    "oko": "yo",
    "omi": "yo",
    "eko": "yo",
}


def normalize_text(text: str) -> str:
    lowered = text.lower().replace("-", " ")
    cleaned = NORMALIZE_PATTERN.sub(" ", lowered)
    return WHITESPACE_PATTERN.sub(" ", cleaned).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")


def _build_expansion_table() -> dict[str, tuple[str, ...]]:
    table: dict[str, tuple[str, ...]] = {}
    for group in SEMANTIC_GROUPS:
        for term in group:
            table[term] = tuple(other for other in group if other != term)
    return table


TERM_EXPANSIONS = _build_expansion_table()


def infer_language(tokens: list[str]) -> str:
    scores: dict[str, int] = {}
    for token in tokens:
        language = LANGUAGE_HINTS.get(token)
        if language is None:
            continue
        scores[language] = scores.get(language, 0) + 1

    if not scores:
        return "unknown"

    return max(scores.items(), key=lambda item: item[1])[0]


def dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


@dataclass(slots=True)
class QueryAnalysis:
    raw_query: str
    normalized_query: str
    inferred_language: str
    tokens: list[str]
    expansion_terms: list[str]

    def expanded_query(self) -> str:
        values = dedupe_preserving_order(self.tokens + self.expansion_terms)
        return " ".join(values)

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "inferred_language": self.inferred_language,
            "tokens": list(self.tokens),
            "expansion_terms": list(self.expansion_terms),
            "expanded_query": self.expanded_query(),
        }


class QueryUnderstandingPipeline:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def analyze(self, query: str) -> QueryAnalysis:
        tokens = tokenize(query)
        normalized = " ".join(tokens)
        inferred_language = infer_language(tokens)

        if not self.enabled:
            return QueryAnalysis(
                raw_query=query,
                normalized_query=normalized,
                inferred_language=inferred_language,
                tokens=tokens,
                expansion_terms=[],
            )

        expansion_terms: list[str] = []
        for token in tokens:
            expansion_terms.extend(TERM_EXPANSIONS.get(token, ()))

        return QueryAnalysis(
            raw_query=query,
            normalized_query=normalized,
            inferred_language=inferred_language,
            tokens=tokens,
            expansion_terms=dedupe_preserving_order(expansion_terms),
        )

