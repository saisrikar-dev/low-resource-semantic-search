from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .encoders import DenseEncoder
from .index import SearchIndex, SearchResult
from .query_understanding import QueryAnalysis, QueryUnderstandingPipeline


@dataclass(slots=True)
class SearchResponse:
    analysis: QueryAnalysis
    results: list[SearchResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis.to_dict(),
            "results": [result.to_dict() for result in self.results],
        }


class SemanticSearchEngine:
    def __init__(
        self,
        index: SearchIndex,
        encoder: DenseEncoder,
        query_understanding: QueryUnderstandingPipeline | None = None,
    ) -> None:
        self.index = index
        self.encoder = encoder
        self.query_understanding = query_understanding or QueryUnderstandingPipeline(enabled=True)

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        analysis = self.query_understanding.analyze(query)
        encoded_query = analysis.expanded_query() or analysis.normalized_query or query
        vector = self.encoder.encode([encoded_query])[0]
        results = self.index.search(vector, top_k=top_k)
        return SearchResponse(analysis=analysis, results=results)

