from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .corpus import Document
from .encoders import DenseEncoder
from .vectors import dot_product


@dataclass(slots=True)
class IndexedDocument:
    document: Document
    vector: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "vector": list(self.vector),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IndexedDocument":
        return cls(
            document=Document.from_dict(dict(payload["document"])),
            vector=[float(value) for value in payload["vector"]],
        )


@dataclass(slots=True)
class SearchResult:
    document: Document
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.document.doc_id,
            "language": self.document.language,
            "title": self.document.title,
            "text": self.document.text,
            "english_gloss": self.document.english_gloss,
            "keywords": list(self.document.keywords),
            "metadata": dict(self.document.metadata),
            "score": round(self.score, 6),
        }


@dataclass(slots=True)
class SearchIndex:
    metadata: dict[str, Any]
    items: list[IndexedDocument]

    @classmethod
    def build(cls, documents: list[Document], encoder: DenseEncoder) -> "SearchIndex":
        searchable_texts = [document.searchable_text() for document in documents]
        vectors = encoder.encode(searchable_texts)
        items = [
            IndexedDocument(document=document, vector=vector)
            for document, vector in zip(documents, vectors, strict=True)
        ]
        metadata = {
            "created_at": datetime.now(UTC).isoformat(),
            "documents": len(documents),
            "encoder": encoder.metadata(),
        }
        return cls(metadata=metadata, items=items)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "items": [item.to_dict() for item in self.items],
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SearchIndex":
        index_path = Path(path)
        with index_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        items = [IndexedDocument.from_dict(item) for item in payload["items"]]
        return cls(metadata=dict(payload["metadata"]), items=items)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        scored = [
            SearchResult(document=item.document, score=dot_product(query_vector, item.vector))
            for item in self.items
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

