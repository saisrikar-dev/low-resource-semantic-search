from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TEMP_ROOT = PROJECT_ROOT / ".tmp-tests"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from low_resource_semantic_search.cli import main
from low_resource_semantic_search.corpus import load_jsonl_corpus
from low_resource_semantic_search.encoders import HashingDenseEncoder
from low_resource_semantic_search.evaluation import evaluate_queries, load_labeled_queries
from low_resource_semantic_search.index import SearchIndex
from low_resource_semantic_search.pipeline import SemanticSearchEngine
from low_resource_semantic_search.query_understanding import QueryUnderstandingPipeline


def make_scratch_dir() -> Path:
    TEMP_ROOT.mkdir(exist_ok=True)
    scratch_dir = TEMP_ROOT / uuid.uuid4().hex
    scratch_dir.mkdir()
    return scratch_dir


class SemanticSearchEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TEMP_ROOT.mkdir(exist_ok=True)
        cls.documents = load_jsonl_corpus(PROJECT_ROOT / "data" / "demo_corpus.jsonl")
        cls.encoder = HashingDenseEncoder(embedding_dimension=192)
        cls.index = SearchIndex.build(cls.documents, cls.encoder)
        cls.engine = SemanticSearchEngine(
            index=cls.index,
            encoder=cls.encoder,
            query_understanding=QueryUnderstandingPipeline(enabled=True),
        )

    def test_build_and_search_returns_expected_document(self) -> None:
        response = self.engine.search("kilimo na maji wakati wa ukame", top_k=3)
        self.assertEqual(response.results[0].document.doc_id, "doc-sw-001")

    def test_query_understanding_detects_language_and_expands_terms(self) -> None:
        analysis = self.engine.query_understanding.analyze("kasuwa farashi")
        self.assertEqual(analysis.inferred_language, "ha")
        self.assertIn("market", analysis.expansion_terms)
        self.assertIn("trade", analysis.expansion_terms)

    def test_index_round_trip(self) -> None:
        scratch_dir = make_scratch_dir()
        try:
            index_path = scratch_dir / "demo_index.json"
            self.index.save(index_path)
            reloaded = SearchIndex.load(index_path)
            response = SemanticSearchEngine(reloaded, self.encoder).search(
                "rural clinic nurse training", top_k=1
            )
            self.assertEqual(response.results[0].document.doc_id, "doc-ha-001")
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def test_evaluation_metrics(self) -> None:
        labeled_queries = load_labeled_queries(PROJECT_ROOT / "data" / "demo_queries.jsonl")
        metrics = evaluate_queries(self.engine, labeled_queries, top_k=3)
        self.assertGreaterEqual(metrics["recall_at_k"], 0.8)
        self.assertGreaterEqual(metrics["mrr"], 0.7)


class CommandLineTests(unittest.TestCase):
    def test_build_index_cli_creates_file(self) -> None:
        scratch_dir = make_scratch_dir()
        try:
            index_path = scratch_dir / "cli_index.json"
            exit_code = main(
                [
                    "build-index",
                    "--corpus",
                    str(PROJECT_ROOT / "data" / "demo_corpus.jsonl"),
                    "--output",
                    str(index_path),
                    "--encoder",
                    "hashing",
                    "--dimension",
                    "128",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(index_path.exists())

            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["encoder"]["kind"], "hashing")
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
