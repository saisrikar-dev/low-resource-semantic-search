from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pipeline import SemanticSearchEngine


@dataclass(slots=True)
class LabeledQuery:
    query: str
    relevant_doc_id: str
    language: str = "unknown"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabeledQuery":
        return cls(
            query=str(payload["query"]),
            relevant_doc_id=str(payload["relevant_doc_id"]),
            language=str(payload.get("language", "unknown")),
        )


def load_labeled_queries(path: str | Path) -> list[LabeledQuery]:
    query_path = Path(path)
    queries: list[LabeledQuery] = []

    with query_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            queries.append(LabeledQuery.from_dict(payload))

    if not queries:
        raise ValueError(f"No labeled queries found in {query_path}")

    return queries


def evaluate_queries(
    engine: SemanticSearchEngine,
    labeled_queries: list[LabeledQuery],
    top_k: int = 5,
) -> dict[str, Any]:
    reciprocal_rank_sum = 0.0
    hits = 0
    breakdown: list[dict[str, Any]] = []

    for labeled_query in labeled_queries:
        response = engine.search(labeled_query.query, top_k=top_k)
        ranked_ids = [result.document.doc_id for result in response.results]

        try:
            rank = ranked_ids.index(labeled_query.relevant_doc_id) + 1
        except ValueError:
            rank = None

        if rank is not None:
            hits += 1
            reciprocal_rank_sum += 1.0 / rank

        breakdown.append(
            {
                "query": labeled_query.query,
                "language": labeled_query.language,
                "relevant_doc_id": labeled_query.relevant_doc_id,
                "rank": rank,
                "returned_ids": ranked_ids,
            }
        )

    total = len(labeled_queries)
    return {
        "queries": total,
        "top_k": top_k,
        "recall_at_k": round(hits / total, 6),
        "mrr": round(reciprocal_rank_sum / total, 6),
        "breakdown": breakdown,
    }

