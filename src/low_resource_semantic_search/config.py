from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EncoderConfig:
    kind: str = "hashing"
    dimension: int = 256
    model_name: str | None = None


@dataclass(slots=True)
class QueryUnderstandingConfig:
    enabled: bool = True


@dataclass(slots=True)
class BuildConfig:
    corpus_path: str
    output_path: str
    encoder: EncoderConfig
    query_understanding: QueryUnderstandingConfig


def load_build_config(path: str | Path) -> BuildConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)

    encoder_payload = dict(payload.get("encoder", {}))
    query_payload = dict(payload.get("query_understanding", {}))

    return BuildConfig(
        corpus_path=str(payload["corpus_path"]),
        output_path=str(payload["output_path"]),
        encoder=EncoderConfig(
            kind=str(encoder_payload.get("kind", "hashing")),
            dimension=int(encoder_payload.get("dimension", 256)),
            model_name=(
                str(encoder_payload["model_name"])
                if encoder_payload.get("model_name") is not None
                else None
            ),
        ),
        query_understanding=QueryUnderstandingConfig(
            enabled=bool(query_payload.get("enabled", True))
        ),
    )

