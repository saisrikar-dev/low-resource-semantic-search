from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("id", "language", "title", "text")


@dataclass(slots=True)
class Document:
    doc_id: str
    language: str
    title: str
    text: str
    english_gloss: str = ""
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Document":
        missing = [field_name for field_name in REQUIRED_FIELDS if not payload.get(field_name)]
        if missing:
            raise ValueError(f"Document is missing required fields: {', '.join(missing)}")

        return cls(
            doc_id=str(payload["id"]),
            language=str(payload["language"]),
            title=str(payload["title"]),
            text=str(payload["text"]),
            english_gloss=str(payload.get("english_gloss", "")),
            keywords=[str(keyword) for keyword in payload.get("keywords", [])],
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.doc_id,
            "language": self.language,
            "title": self.title,
            "text": self.text,
            "english_gloss": self.english_gloss,
            "keywords": list(self.keywords),
            "metadata": dict(self.metadata),
        }

    def searchable_text(self) -> str:
        parts = [self.title, self.text, self.english_gloss, " ".join(self.keywords)]
        return " ".join(part.strip() for part in parts if part and part.strip())


def load_jsonl_corpus(path: str | Path) -> list[Document]:
    corpus_path = Path(path)
    documents: list[Document] = []

    with corpus_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {corpus_path}") from exc

            documents.append(Document.from_dict(payload))

    if not documents:
        raise ValueError(f"No documents found in {corpus_path}")

    return documents

