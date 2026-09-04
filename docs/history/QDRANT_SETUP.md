# Phase 3: Qdrant Vector Database Setup

## Overview
This guide sets up a local Qdrant vector database for the multimodal RAG ingestion system. The setup includes:
- **Docker Container**: Qdrant running in isolation
- **Persistent Storage**: Local `./qdrant_storage` directory
- **Dashboard**: Web UI at `http://localhost:6333/dashboard`
- **Configuration**: Environment-driven via `.env`

## Quick Start (3 Steps)

### 1. Start Qdrant Docker Container
```bash
docker-compose up -d
```

**Verify it's running:**
```bash
curl http://localhost:6333/health
# Response: {"status":"ok"}
```

### 2. Initialize the Collection
```bash
python scripts/init_db.py
```

**Expected output:**
```
================================================================================
QDRANT COLLECTION INITIALIZATION
ENGINE_USE: Claude 4.5 Opus (Architect)
================================================================================

[1/3] Loading configuration from .env...
  ✓ Collection: multi_modal_knowledge
  ✓ Vector Size: 384
  ✓ Embedding Model: minilm
  ✓ Qdrant URL: http://localhost:6333

[2/3] Connecting to Qdrant...
  ✓ Connected successfully
  ✓ Collections in database: 1

[3/3] Creating collection...
  ✓ Collection created: multi_modal_knowledge

  Collection Info:
    - Points: 0
    - Vectors: 0
    - Status: green

================================================================================
✓ INITIALIZATION COMPLETE
================================================================================
```

### 3. Access the Dashboard
Open your browser: `http://localhost:6333/dashboard`

---

## Configuration Files

### `.env` - Environment Variables
```
QDRANT_URL=http://localhost:6333
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=multi_modal_knowledge
VECTOR_SIZE=384
EMBEDDING_MODEL=minilm
```

**Available embedding models:**
- `minilm` (384-dim) - Default, fast local embedding
- `bge-m3` (1024-dim) - Multilingual, requires local inference
- `openai` (1536-dim) - Requires OPENAI_API_KEY environment variable
- `openai-large` (3072-dim) - High quality, requires API key

### `docker-compose.yml` - Container Configuration
```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"       # REST API & Dashboard
      - "6334:6334"       # gRPC API
    volumes:
      - ./qdrant_storage:/qdrant/storage
      - ./qdrant_snapshots:/qdrant/snapshots
```

---

## Usage Workflows

### Ingest Documents
```bash
# 1. Generate ingestion.jsonl from Phase 2
python -m mmrag_converter.v2.cli process input.pdf --output output/ingestion.jsonl

# 2. Ingest into Qdrant
python -m src.database.ingestor output/ingestion.jsonl --batch-size 50
```

### Search Knowledge Base
```bash
# Simple search
python -m src.database.retriever "machine learning basics"

# Search with filters
python -m src.database.retriever "neural networks" --modality image --top-k 10

# JSON output
python -m src.database.retriever "clustering" --json
```

### Python API
```python
from src.database.ingestor import ingest_jsonl_file
from src.database.retriever import search_knowledge_base

# Ingest
result = ingest_jsonl_file("output/ingestion.jsonl")
print(f"Ingested {result.inserted} chunks")

# Search
results = search_knowledge_base("How to install?", top_k=5)
for r in results:
    print(f"[{r.score:.2f}] {r.citation}")
    print(f"  {r.text[:100]}...")
```

---

## Troubleshooting

### Connection Failed
```
ERROR: Cannot connect to http://localhost:6333
```

**Solution:**
```bash
# Check if Docker is running
docker ps | grep qdrant

# Start if not running
docker-compose up -d

# Check logs
docker-compose logs qdrant
```

### Port Already in Use
```
ERROR: Bind for 0.0.0.0:6333 failed: port is already allocated
```

**Solution:**
```bash
# Option 1: Stop the existing container
docker stop multimodal-doc-converter-qdrant

# Option 2: Change port in docker-compose.yml
# Modify: ports: ["6333:6333"] to ["6334:6333"]
```

### Collection Not Found
```
ValueError: Collection multi_modal_knowledge not found
```

**Solution:**
```bash
# Run initialization
python scripts/init_db.py

# Or recreate it
python scripts/init_db.py --recreate
```

---

## Collection Schema

### Vector Configuration
- **Size**: 384 dimensions (optimized for all-MiniLM-L6-v2)
- **Distance**: Cosine similarity
- **Type**: Dense vectors (float32)

### Payload Fields (Auto-Indexed)
| Field | Type | Purpose |
|-------|------|---------|
| `chunk_id` | UUID | Unique chunk identifier |
| `chunk_hash` | String | Deduplication (SHA256) |
| `doc_id` | String | Document identifier |
| `modality` | Keyword | text, image, or table |
| `content` | Text | Actual chunk content |
| `source_file` | Keyword | Original filename |
| `file_type` | Keyword | pdf, epub, docx, etc. |
| `page_number` | Integer | Page in source |
| `breadcrumb_path` | List | Hierarchical section path |
| `parent_heading` | String | Immediate parent section |
| `asset_path` | String | Path to extracted image |
| `visual_description` | String | AI description of visual |
| `prev_text_snippet` | Text | Context before |
| `next_text_snippet` | Text | Context after |

---

## Docker Commands Reference

### Start Container
```bash
docker-compose up -d
```

### View Logs
```bash
docker-compose logs -f qdrant
```

### Stop Container
```bash
docker-compose down
```

### Stop Container & Clear Data
```bash
docker-compose down -v
```

### Access Container Shell
```bash
docker exec -it multimodal-doc-converter-qdrant bash
```

### View Storage
```bash
ls -lh qdrant_storage/
```

---

## Production Deployment

### For Qdrant Cloud
```bash
# Update .env
QDRANT_URL=https://your-project.qdrant.io
QDRANT_API_KEY=your_api_key_here
```

### For Kubernetes
Use the Qdrant Helm chart:
```bash
helm repo add qdrant https://qdrant.github.io/qdrant-helm
helm install qdrant qdrant/qdrant
```

---

## Performance Tuning

### Increase Batch Size (for fast ingestion)
```bash
python -m src.database.ingestor ingestion.jsonl --batch-size 200
```

### Enable Vector Quantization
Edit `src/database/config.py`:
```python
quantization_config=qdrant_models.ScalarQuantization(
    scalar=qdrant_models.ScalarQuantizationConfig(
        type=qdrant_models.ScalarType.INT8,  # Lower from 32-bit
        quantile=0.99,
        always_ram=True,
    ),
)
```

---

## Monitoring

### Collection Size
```bash
curl http://localhost:6333/collections/multi_modal_knowledge
```

### Disk Usage
```bash
du -sh qdrant_storage/
```

### Check Health
```bash
curl http://localhost:6333/health
```

---

## Next Steps

1. **Generate Ingestion JSONL**: Process documents with Phase 2
2. **Ingest Chunks**: Load into Qdrant
3. **Search & Retrieve**: Build RAG applications
4. **Fine-tune**: Adjust batch sizes, embedding models, vector dimensions

---

## Support

For issues or questions:
- Check Docker logs: `docker-compose logs qdrant`
- Verify configuration: `cat .env`
- Test connection: `curl http://localhost:6333/health`
- Review Qdrant docs: https://qdrant.tech/documentation/
