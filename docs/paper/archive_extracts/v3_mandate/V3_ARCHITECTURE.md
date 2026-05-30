# V3 Architecture: UIR-in, Chunk-out Pipeline

**Version:** 3.0.0-alpha
**Contract:** Every engine returns `UniversalDocument`. Every chunker
accepts only `UniversalDocument`. Violation is a build failure (enforced
by `tests/test_v3_security.py`).

---

## 1. Pipeline Topology

```
SOURCE FILE (PDF, EPUB, HTML, DOCX, ...)
        │
        ▼
┌──────────────────────┐
│   Format Engine       │  ← One engine per format.
│   (engines/pdf.py,    │    Returns UniversalDocument.
│    engines/epub.py,   │    May use Docling, PyMuPDF, etc.
│    ...)               │    Implementation detail —
└────────┬─────────────┘    not exposed to rest of pipeline.
         │
         ▼  UniversalDocument
┌──────────────────────┐
│   Quality Router      │  ← Reads page-level classification
│   (routing/)          │    (DIGITAL/SCANNED/HYBRID).
│                       │    Defers OCR/VLM decisions.
└────────┬─────────────┘
         │
         ▼  UniversalDocument (possibly enriched)
┌──────────────────────┐
│   Chunker             │  ← Accepts UniversalDocument ONLY.
│   (chunking/)         │    Emits IngestionChunk list.
│                       │    Format-agnostic; no engine
│                       │    knowledge.
└────────┬─────────────┘
         │
         ▼  List[IngestionChunk]
┌──────────────────────┐
│   Processor (CLI)    │  ← Orchestrates the pipeline.
│   (processor.py)      │    Docling-agnostic.
│                       │    Calls engines via adapter
│                       │    interface only.
└──────────────────────┘
```

---

## 2. Core Types (Source of Truth)

All types live in `src/mmrag_v3/universal/`. They are NOT re-exported
from legacy locations. There is no backward compatibility layer.

### 2.1 UniversalDocument

```python
@dataclass
class UniversalDocument:
    doc_id: str
    source_file: str
    file_type: str                     # pdf, epub, html, docx, ...
    pages: List[UniversalPage]
    metadata: DocumentMetadata
    total_pages: int
    created_at: datetime
```

### 2.2 UniversalPage

```python
@dataclass
class UniversalPage:
    page_number: int                   # 1-indexed
    elements: List[Element]
    classification: PageClassification # DIGITAL | SCANNED | HYBRID
    dimensions: Tuple[int, int]        # (width, height) in pixels
    raw_image: Optional[np.ndarray]
    text_density: float
    avg_confidence: float
```

### 2.3 Element

```python
@dataclass
class Element:
    type: ElementType                  # TEXT | IMAGE | TABLE
    content: str
    bbox: Optional[BoundingBox]        # Normalized [0,1000]
    confidence: float
    raw_image: Optional[np.ndarray]
    extraction_method: ExtractionMethod
    element_index: int
    source_label: str
    metadata: Dict[str, Any]
```

### 2.4 BoundingBox

Integer coordinates in [0, 1000] range. Validation enforces:
- All four coordinates are `int`
- `x_min < x_max`, `y_min < y_max`
- All values in [0, 1000]

### 2.5 IngestionChunk (Output Schema)

The pipeline output is a JSONL file where every line validates against
the `IngestionChunk` Pydantic model. The canonical definition lives in
`src/mmrag_v3/schema/ingestion_schema.py`. Every chunk emitted by the
chunker must serialize through this model without exception.

---

## 3. Engine Contract

Every format engine in `src/mmrag_v3/engines/` MUST:

1. Expose a function with signature:
   ```python
   def extract(file_path: str | Path) -> UniversalDocument: ...
   ```
   The return type annotation MUST reference `UniversalDocument`.
   Enforced by `tests/test_v3_security.py`.

2. Be the **only** module in the codebase that imports format-specific
   libraries (Docling, PyMuPDF, EbookLib, etc.).

3. Never be imported by the processor, chunker, or routing modules
   except through a defined adapter interface.

---

## 4. Processor Contract

The processor at `src/mmrag_v3/processor.py` MUST:

1. Accept an engine (by name/import string) and invoke it to produce
   a `UniversalDocument`.

2. Pass the `UniversalDocument` through quality routing (optional) and
   then to the chunker.

3. **Import zero Docling symbols.** Enforced by AST audit in
   `tests/test_v3_security.py`.

4. Not inspect, modify, or depend on engine-internal data structures.
   The `UniversalDocument` is the only communication channel between
   engine and processor.

---

## 5. Chunker Contract

The chunker in `src/mmrag_v3/chunking/` MUST:

1. Accept `UniversalDocument` as its sole input type. Enforced by
   signature inspection in `tests/test_v3_security.py`.

2. Emit `List[IngestionChunk]` as its output.

3. Be format-agnostic. It must not branch on `file_type` or inspect
   engine-specific metadata fields beyond what `UniversalDocument`
   exposes.

---

## 6. What Is Forbidden

The following patterns are architectural violations and will cause
`test_v3_security.py` to fail:

- Any `import docling` or `from docling ...` in `processor.py`
- Any engine that does not return `UniversalDocument`
- Any chunker that accepts anything other than `UniversalDocument`
- Any `v2x_to_v3_mapper` or bridge module
- Any re-export of V3 types from legacy paths
- Any module that imports from `.legacy_archive/`

---

## 7. Directory Layout

```
src/mmrag_v3/
    __init__.py
    processor.py          # Orchestrator — Docling-agnostic
    universal/
        __init__.py
        document.py       # UniversalDocument, UniversalPage, Element,
                          #   BoundingBox, ElementType, PageClassification,
                          #   ExtractionMethod, DocumentMetadata
    engines/
        __init__.py
        pdf.py            # PDF engine (may use Docling internally)
        epub.py           # EPUB engine
        html.py           # HTML engine
    chunking/
        __init__.py
        chunker.py        # UniversalDocument → List[IngestionChunk]
    schema/
        __init__.py
        ingestion_schema.py  # IngestionChunk Pydantic model
    routing/
        __init__.py
        quality_router.py # Optional quality-based routing
tests/
    test_v3_security.py   # Architecture enforcement
    ...