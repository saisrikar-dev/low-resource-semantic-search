from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .api import run_api_server
from .config import BuildConfig, EncoderConfig, QueryUnderstandingConfig, load_build_config
from .corpus import load_jsonl_corpus
from .encoders import build_encoder
from .evaluation import evaluate_queries, load_labeled_queries
from .index import SearchIndex
from .pipeline import SemanticSearchEngine
from .query_understanding import QueryUnderstandingPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lsrs",
        description="Low-resource multilingual semantic search",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_index = subparsers.add_parser("build-index", help="Build a search index from a JSONL corpus")
    build_index.add_argument("--config", type=Path, help="Optional JSON config file")
    build_index.add_argument("--corpus", type=Path, help="Path to the JSONL corpus")
    build_index.add_argument("--output", type=Path, help="Path to the output index JSON")
    build_index.add_argument(
        "--encoder",
        choices=("hashing", "sentence-transformer"),
        help="Encoder backend",
    )
    build_index.add_argument("--model-name", help="Model name for sentence-transformer mode")
    build_index.add_argument("--dimension", type=int, help="Embedding size for hashing mode")

    search = subparsers.add_parser("search", help="Search an existing index")
    search.add_argument("--index", type=Path, required=True, help="Path to a saved index JSON")
    search.add_argument("--query", required=True, help="Query string")
    search.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    search.add_argument("--model-name", help="Override transformer model name if needed")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate labeled search queries")
    evaluate.add_argument("--index", type=Path, required=True, help="Path to a saved index JSON")
    evaluate.add_argument("--queries", type=Path, required=True, help="Path to labeled queries JSONL")
    evaluate.add_argument("--top-k", type=int, default=5, help="Cutoff for retrieval metrics")
    evaluate.add_argument("--model-name", help="Override transformer model name if needed")

    serve = subparsers.add_parser("serve", help="Start the HTTP search API")
    serve.add_argument("--index", type=Path, required=True, help="Path to a saved index JSON")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve.add_argument("--port", type=int, default=8080, help="Bind port")
    serve.add_argument("--model-name", help="Override transformer model name if needed")

    return parser


def load_runtime_build_config(args: argparse.Namespace) -> BuildConfig:
    if args.config:
        config = load_build_config(args.config)
    else:
        corpus_path = str(args.corpus or "data/demo_corpus.jsonl")
        output_path = str(args.output or "artifacts/demo_index.json")
        encoder = EncoderConfig(
            kind=args.encoder or "hashing",
            dimension=args.dimension or 256,
            model_name=args.model_name,
        )
        return BuildConfig(
            corpus_path=corpus_path,
            output_path=output_path,
            encoder=encoder,
            query_understanding=QueryUnderstandingConfig(enabled=True),
        )

    if args.corpus:
        config.corpus_path = str(args.corpus)
    if args.output:
        config.output_path = str(args.output)
    if args.encoder:
        config.encoder.kind = args.encoder
    if args.dimension:
        config.encoder.dimension = args.dimension
    if args.model_name:
        config.encoder.model_name = args.model_name
    return config


def build_engine_from_index(index_path: Path, model_name_override: str | None = None) -> SemanticSearchEngine:
    index = SearchIndex.load(index_path)
    encoder_metadata = dict(index.metadata.get("encoder", {}))
    encoder_kind = str(encoder_metadata.get("kind", "hashing"))
    dimension = int(encoder_metadata.get("dimension", 256))
    model_name = model_name_override or encoder_metadata.get("model_name")
    encoder = build_encoder(encoder_kind, dimension=dimension, model_name=model_name)
    query_understanding = QueryUnderstandingPipeline(enabled=True)
    return SemanticSearchEngine(index=index, encoder=encoder, query_understanding=query_understanding)


def handle_build_index(args: argparse.Namespace) -> int:
    config = load_runtime_build_config(args)
    documents = load_jsonl_corpus(config.corpus_path)
    encoder = build_encoder(
        kind=config.encoder.kind,
        dimension=config.encoder.dimension,
        model_name=config.encoder.model_name,
    )
    index = SearchIndex.build(documents, encoder)
    index.save(config.output_path)
    print(
        json.dumps(
            {
                "status": "ok",
                "documents": len(documents),
                "output_path": str(Path(config.output_path).resolve()),
                "encoder": index.metadata["encoder"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


def handle_search(args: argparse.Namespace) -> int:
    engine = build_engine_from_index(args.index, model_name_override=args.model_name)
    response = engine.search(args.query, top_k=args.top_k)
    print(json.dumps(response.to_dict(), ensure_ascii=True, indent=2))
    return 0


def handle_evaluate(args: argparse.Namespace) -> int:
    engine = build_engine_from_index(args.index, model_name_override=args.model_name)
    labeled_queries = load_labeled_queries(args.queries)
    metrics = evaluate_queries(engine, labeled_queries, top_k=args.top_k)
    print(json.dumps(metrics, ensure_ascii=True, indent=2))
    return 0


def handle_serve(args: argparse.Namespace) -> int:
    engine = build_engine_from_index(args.index, model_name_override=args.model_name)
    run_api_server(engine, host=args.host, port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handlers: dict[str, Any] = {
        "build-index": handle_build_index,
        "search": handle_search,
        "evaluate": handle_evaluate,
        "serve": handle_serve,
    }
    return handlers[args.command](args)

