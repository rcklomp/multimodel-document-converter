# Qdrant Search Tool

Semantic search across all ingested document collections.

## Usage

```bash
python3 scripts/search_qdrant.py "your question here"
```

That's it. The tool searches all collections, filters by keyword, reranks by semantic relevance, and shows the best results.

## Examples

```bash
# Simple keyword
python3 scripts/search_qdrant.py "Mauser"

# Natural language question
python3 scripts/search_qdrant.py "Why was Mr. Dursley enraged?"

# Technical query
python3 scripts/search_qdrant.py "How to disassemble a Mauser 1898?"

# Specific model name
python3 scripts/search_qdrant.py "British SMLE No. 1, MKIII"

# Cross-domain
python3 scripts/search_qdrant.py "how to build an energy efficient home"
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-n 10` | 5 | Number of results per collection |
| `-c firearms_pdf` | all | Search a specific collection |
| `-m image` | all | Filter by modality: `text`, `image`, `table` |
| `--no-rerank` | rerank on | Skip cloud reranking, use raw vector scores |
| `--list` | | Show all collections and their chunk counts |

## How It Works

1. **Keyword extraction** — picks the most distinctive word from your query (proper nouns first, then longest word). Filters Qdrant results to only chunks containing that keyword.
2. **Vector search** — embeds the query with `nomic-embed-text` (local, via Ollama) and retrieves the top candidates from each collection.
3. **Reranking** — sends candidates to `qwen3-rerank` (Alibaba cloud API) which reads the actual text and reorders by semantic relevance. This is what turns a 40% vector match into a 96% precise answer.

## Reading the Output

```
  ███████████████████░ 96%          ← relevance bar (green = strong)
  TEXT  page 15  (CHAPTER ONE)      ← modality, page, section heading
    Mr. Dursley was enraged to      ← content preview
    see that a couple of them...
```

- **Green bars (75%+)** — strong match, directly relevant
- **Yellow bars (65-75%)** — good match, contextually relevant
- **Dim bars (<65%)** — weak match, may not be relevant

Image results also show the asset path:
```
  IMAGE  page 51  (MAUSER)
    Four-panel instructional layout showing sequential disassembly...
    Asset: assets/29f7c8bb7680_051_figure_10.png
```

## Requirements

- **Ollama** running with `nomic-embed-text` model (`ollama pull nomic-embed-text`)
- **Qdrant** running on `localhost:6333`
- **Internet** for reranking (Alibaba API). Use `--no-rerank` for offline mode.

## Collections

List what's ingested:

```bash
python3 scripts/search_qdrant.py --list
```

To add a new document, convert it and ingest:

```bash
mmrag-v2 process document.pdf -o output/document_name
python3 scripts/ingest_to_qdrant.py output/document_name/ingestion.jsonl
```
