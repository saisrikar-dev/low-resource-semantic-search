from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from .query_understanding import tokenize
from .vectors import l2_normalize


class DenseEncoder:
    kind = "base"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError

    def metadata(self) -> dict[str, object]:
        return {"kind": self.kind, "dimension": self.dimension}


@dataclass(slots=True)
class HashingDenseEncoder(DenseEncoder):
    embedding_dimension: int = 256
    trigram_weight: float = 0.4

    kind = "hashing"

    @property
    def dimension(self) -> int:
        return self.embedding_dimension

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_single(text) for text in texts]

    def _encode_single(self, text: str) -> list[float]:
        vector = [0.0] * self.embedding_dimension
        tokens = tokenize(text)

        for token in tokens:
            self._accumulate(vector, token, 1.0)
            padded = f"__{token}__"
            for index in range(max(0, len(padded) - 2)):
                trigram = padded[index : index + 3]
                self._accumulate(vector, trigram, self.trigram_weight)

        return l2_normalize(vector)

    def _accumulate(self, vector: list[float], token: str, weight: float) -> None:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for offset in range(0, 16, 4):
            bucket = int.from_bytes(digest[offset : offset + 2], "big") % self.embedding_dimension
            sign = 1.0 if digest[offset + 2] % 2 == 0 else -1.0
            magnitude = 1.0 + (digest[offset + 3] / 255.0)
            vector[bucket] += sign * magnitude * weight


class SentenceTransformerDenseEncoder(DenseEncoder):
    kind = "sentence-transformer"

    def __init__(self, model_name: str) -> None:
        if not model_name:
            raise ValueError("A model name is required for the sentence-transformer encoder.")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it with: python -m pip install -e .[transformer]"
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def metadata(self) -> dict[str, object]:
        metadata = super().metadata()
        metadata["model_name"] = self.model_name
        return metadata

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self._model.encode(list(texts), normalize_embeddings=False)
        if hasattr(embeddings, "tolist"):
            values = embeddings.tolist()
        else:
            values = [list(row) for row in embeddings]
        return [l2_normalize([float(value) for value in row]) for row in values]


def build_encoder(kind: str, dimension: int = 256, model_name: str | None = None) -> DenseEncoder:
    if kind == "hashing":
        return HashingDenseEncoder(embedding_dimension=dimension)
    if kind == "sentence-transformer":
        return SentenceTransformerDenseEncoder(model_name=model_name or "")
    raise ValueError(f"Unsupported encoder kind: {kind}")

