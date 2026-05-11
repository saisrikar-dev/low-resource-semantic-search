# Multilingual Semantic Search and Query Understanding for Low-Resource Languages Using Transformer-Based Dense Retrieval

Short repo name: `low-resource-semantic-search`

This repository is a working reference implementation for multilingual semantic search over low-resource language content. It includes:

- an installable Python package
- an offline-safe dense retrieval baseline that works without external downloads
- optional `sentence-transformers` integration for true transformer-based dense retrieval
- query understanding with lightweight language hints and semantic expansion
- JSONL corpus and evaluation data
- a CLI for indexing, search, evaluation, and serving an HTTP API
- unit tests

## Quickstart

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install the project

```powershell
python -m pip install -e .
```

Optional transformer support:

```powershell
python -m pip install -e .[transformer]
```

### 3. Build the demo index

```powershell
python main.py build-index --config configs/demo_build.json
```

### 4. Run a search

```powershell
python main.py search --index artifacts/demo_index.json --query "kilimo na maji" --top-k 3
```

### 5. Evaluate the demo set

```powershell
python main.py evaluate --index artifacts/demo_index.json --queries data/demo_queries.jsonl --top-k 3
```

### 6. Start the API

```powershell
python main.py serve --index artifacts/demo_index.json --host 127.0.0.1 --port 8080
```

Then query it:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/search -ContentType 'application/json' -Body '{"query":"rural clinic support","top_k":2}'
```

## Transformer Mode

The default hashing encoder keeps the repo fully functional offline. If you have a locally available or cached multilingual sentence-transformer model, build a transformer index with:

```powershell
python main.py build-index --corpus data/demo_corpus.jsonl --output artifacts/transformer_index.json --encoder sentence-transformer --model-name your-model-name
```

The search and serve commands automatically reuse encoder metadata stored in the index.

## Project Layout

```text
configs/                 Demo build configuration
data/                    Demo corpus and labeled evaluation queries
src/low_resource_semantic_search/
  api.py                 HTTP API server
  cli.py                 Command-line interface
  config.py              Config loading helpers
  corpus.py              JSONL corpus loader
  encoders.py            Hashing and transformer encoders
  evaluation.py          Retrieval metrics
  index.py               Index build, save, load, search
  pipeline.py            Query understanding + retrieval orchestration
  query_understanding.py Query normalization and expansion
  vectors.py             Pure-Python vector math
tests/                   Unit tests
main.py                  Repo-root entry point
```

## Demo Data

The included corpus covers agriculture, health, education, commerce, climate adaptation, and water access across English plus romanized Swahili, Hausa, and Yoruba examples. The query-understanding layer expands domain and language terms so the offline baseline can still behave like a multilingual retrieval system.

## Test

```powershell
python -m unittest discover -s tests -v
```

