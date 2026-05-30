# MM-Converter V3.0: Architectural Charter — Draft 0.3

**Status:** DRAFT for review (incorporates follow-up architecture review on Draft 0.2)
**Target release:** v3.0 (next major schema version after v2.X)
**Changes since 0.2:** Replaced unattainable LLM dominance criterion with aggregate + degradation-rate threshold; defined chunk-identity mapping policy and diff predicate; added visual_weight floor (0.1) for PROSE profile; extended C-spike to measure reranker discrimination; strengthened determinism policy with content-pinning cache; added fusion re-normalization on visual-leg skip; resolved UIRChunk provenance contract ambiguity; fixed ConfidenceBreakdown Optional fields; defined partial-release policy (3.0.0 held until 3.1.0); added R12 calendar implications; expanded holdout policy to 8 blind-test docs; added LLM sanitization test strategy as Phase B deliverable; defined corpus-size target for Phase C entry criterion.
**Governance lineage:** `AGENTS.md` (Level-0 invariants) → `docs/DECISIONS.md` (canonical decisions) → `docs/ARCHITECTURE.md` (v2.X baseline) → this document (v3.0 target)
**Parent:** `docs/ARCHITECTURE.md` (v2.X production architecture — this document supersedes for v3.0)
**Review baseline:** Draft 0.2 + follow-up architecture review (2026-05-25, "Partially Fixed — Residual Gaps" + "New Issues")

---

## Glossary

| Term | Definition |
|---|---|
| **UIR** | Universal Intermediate Representation — format-agnostic document model between extraction and chunking |
| **ColPali** | Vision Transformer model that embeds document pages as patch-vector matrices for visual retrieval |
| **MaxSim** | Maximum Similarity — late-interaction scoring: for each query token, find its most similar document patch, sum maxima |
| **RRF** | Reciprocal Rank Fusion — merges ranked lists from multiple retrievers via 1/(k+rank) scoring |
| **omlx** | Local ML inference server at `10.0.10.246:8000`; hosts Qwen3-Embedding-8B, ModernBERT reranker, future ColPali |
| **mxfp8** | Mixed-precision FP8 model quantization format used by omlx-hosted models |
| **GX10** | LAN-local Nvidia Blackwell inference endpoint at `10.0.10.239:8000`; hosts FP8 LLMs for sanitization and judge |
| **Soak** | Synthetic retrieval quality evaluation — 518 queries × top-5 results, scored by LLM-as-judge across 5 axes |
| **Strict gate** | Per-document deterministic acceptance check (`qa_full_conversion.py --source-pdf`); produces PASS/WARN/FAIL |
| **Smoke matrix** | Multi-profile gate run across all 34 docs + blind-test documents (Greenhouse + ≥8 additional); requires GATE_PASS + UNIVERSAL_PASS |
| **Content-pinning cache** | Deterministic key-value store: `(content_hash, context_hash, model_id, prompt_version) → sanitized_output`. Context-aware key prevents stale hits when neighbor chunks change. Enables build reproducibility without requiring the LLM endpoint to be deterministic. |

---

## 0. Document Purpose & Scope

This charter defines the next-generation architecture for the MM-Converter pipeline. It describes the *target state* for v3.0 — the components, their interfaces, the migration path, and the quality gates that govern acceptance. It does **not** replace `docs/ARCHITECTURE.md` (v2.X production architecture, still canonical for current operation) until v3.0 ships.

**Read-order:**
1. This document (v3.0 target architecture)
2. `docs/DECISIONS.md` — all architectural decisions governing v2.X → v3.0 transition
3. `docs/ARCHITECTURE.md` — v2.X production baseline (the system being evolved)
4. `docs/QUALITY_GATES.md` — v3.0 will extend, not relax, these gates
5. `docs/AGENT_GOVERNANCE.md` — evidence/status/completion rules apply to v3.0 workstreams

---

## 1. Problem Analysis: The v2.16 Empirical Ceiling

### 1.1 Achievement Context

The v2.X architecture reached its mathematical ceiling at `v2.16.0`. Key metrics:

| Metric | v2.10 Baseline | v2.16.0 | Ceiling Nature |
|---|---|---|---|
| Recall@5 (doc) | 54.2% | **98.6%** | Approachable asymptote — 1.4% remaining likely judge edge cases |
| Recall@5 (chunk) | 6.8% | **90.2%** | Strong; remaining 9.8% are complex-document spatial defects |
| Recall@1 (chunk) | 2.1% | **67.8%** | Hard ceiling — see §1.2 |
| Format (judge) | 98.3% | ~88-93% | Coverage-revealed pre-existing debt; v2.10's 98.3% was artificially high (hub-collapse masked defects) |
| Faithfulness (judge) | 4.7% | **72.6%** | Plateaued; HyDE, query rewriting, dynamic top-K all **empirically falsified** as dead levers |

### 1.2 Three Hard Architectural Limits

These limits were all **diagnosed with empirical evidence** during v2.16 Phase 2 — none are speculative:

#### Limit 1: Spatial-to-Text Translation Deficit (-12pp Recall@1 on complex documents)

**Evidence:** `docs/DIAGNOSTIC_2026-05-25_v2.16_p2_omlx_deficit_root_cause.md`
**Affected docs:** `ATZ_Elektronik_German`, `Python_Cookbook`, `IRJET_Modeling_of_Solar_PV`
**Root cause:** Flat text embeddings (even 4096-dim Qwen3-Embedding-8B) cannot capture 2D spatial relationships. A German circuit diagram label carries meaning from its *position* relative to components, not just its text. Pure-text retrieval augmentations (HyDE, Query Rewriting, Dynamic Top-K) were **proven empirically inert** against this deficit.
**Cannot be solved by:** Better text embedders, query rewriting, chunk restructuring, or reranker upgrades.
**Can only be solved by:** Visual/spatial embeddings that encode page layout.

#### Limit 2: Heuristic Patching Ceiling (Format Quality Maxed at ~98-99%)

**Evidence:** `docs/DECISIONS.md` — "v2.10 chunker-quality ceiling — 99.9% Format not chased"
**Manifestation:** `engines/docling_postprocess.py` executing `y_sort_with_dropcap` for *Harry Potter*, forced full-page OCR for `Earthship_Vol1`, bbox-IoU deduplication for overlapping text/table chunks, picture classification label leaks, trailing preposition mergers.
**Root cause:** Docling 2.86's DOM extraction produces layout artifacts that require document-class-specific Python middleware. Each fix is brittle, per-publisher, and introduces regression risk for other profiles.
**Cannot be solved by:** More Python heuristics, regex, or rule-based post-processing.
**Can only be solved by:** Delegating spatial reading-order understanding to a model trained on layout topology (LLM sanitization or VLM-native parsing).

#### Limit 3: Chunker-State Fragmentation (Cross-Page Splits & Inert Adjacency Fetch)

**Evidence:** `docs/DECISIONS.md` — v2.16 partial_code adjacency fetch shipped INERT
**Manifestation:** `Fluent_Python` cross-page code splits attributed to wrong pages. `Python_Cookbook`/`Python_Distilled` cross-page content present but under wrong `page_number`. HybridChunker cannot reliably emit cross-page state flags because it operates on Docling's DOM objects, not a unified document view.
**Root cause:** `PdfConversionPlan`, `BatchProcessor`, and `DoclingPdfAdapter` are not truly unified — extraction and chunking are coupled through Docling-specific layout classes.
**Cannot be solved by:** Patching chunker state propagation per-profile.
**Can only be solved by:** True UIR refactor (Item #13 from v2.15) — decoupling extraction from chunking via a format-agnostic `UniversalDocument`.

### 1.3 Why V3.0 Is a Paradigm Shift, Not a Feature Patch

All three limits share a common root: **heuristic 1D text processing of 2D visual data.** V3.0 shifts the burden of spatial comprehension from Python heuristics to models trained on millions of layout topologies. The architecture moves from:

| Axis | v2.X (Current) | v3.0 (Target) |
|---|---|---|
| **Extraction** | Docling DOM parsing + Python heuristics | UIR abstraction with pluggable backends (Docling retained + LLM sanitization; VLM-native parsing as optional upgrade) |
| **Chunk sanitization** | Regex + POS + per-profile rules | LLM-native semantic polish pass (local GX10 FP8 endpoint, $0 cost) with heuristic dual-write retained until aggregate dominance proven |
| **Retrieval** | Flat-text embeddings (dense + sparse + RRF) | Hybrid text + late-interaction visual embeddings (ColPali/MaxSim), profile-conditional fusion weights with non-zero floor |
| **Quality evaluation** | Prose-calibrated format gates | Modality-aware gates (prose, form, table, diagram), sidecar evaluation track |

---

## 2. Architectural Drivers

### 2.1 Constraints (Hard, Non-Negotiable)

| ID | Constraint | Source |
|---|---|---|
| C1 | Python 3.10 only | `AGENTS.md` §1.1 |
| C2 | Apple Silicon (MPS) optimization target | `AGENTS.md` §1.2 |
| C3 | ≤8GB RAM during runs; batch ≤10 pages | `AGENTS.md` §1.4 |
| C4 | BBoxes: integer [0,1000] normalized coordinates | `AGENTS.md` §1, REQ-COORD-01 |
| C5 | Docling pinned at 2.86.0 | `AGENTS.md` §1.3 |
| C6 | Solo-dev 12-day convergence cycles | `AGENTS.md` §5, `DECISIONS.md` |
| C7 | Local-first execution (LAN GX10 + omlx server); cloud optional/fallback | `AGENTS.md` §4, `DECISIONS.md` v2.13 |
| C8 | No gate weakening to make failing runs pass | `DECISIONS.md` "No gate weakening..." |
| C9 | Schema version bump required for chunk-shape changes | `AGENTS.md` Schema version 2.7.0 |
| C10 | `AGENT-VAL-01` blind-test compliance (Greenhouse document) | `AGENTS.md` §1.5 |
| C11 | Build reproducibility — a fresh checkout must reproduce the corpus within stated tolerances | This document §2.2 Q8; §7.4 Determinism Policy |
| C12 | Corpus-size operational target: ≤1000 documents, ≤300,000 pages | This document §3.4 Resource budget; §4.2 Phase C entry criterion |

### 2.2 Quality Attribute Scenarios

| ID | Quality Attribute | Scenario | Target |
|---|---|---|---|
| Q1 | **Recall@1 (chunk)** on complex docs | User queries German circuit diagram label | Close -12pp deficit (from 67.8% → ≥80%) |
| Q2 | **Format quality** | Chunk content judged by LLM | ≥95% corpus-wide (v3.0 target; 99% stretch) |
| Q3 | **Cross-page integrity** | Code block spans page boundary | Correct page attribution, no content truncation |
| Q4 | **Form/table fidelity** | Dutch inventory form (`CarOK`) | Modality-aware judge scoring; no false-negative prose penalties |
| Q5 | **End-to-end latency** | Retrieval (embed → search → rerank) | ≤3.0s p99 (text-only retained at ~1.6s; visual leg adds ≤1s) |
| Q6 | **Per-query cost** | Production retrieval | $0 (all models local/LAN) |
| Q7 | **Backward compatibility** | v2.X ingestion.jsonl | Consumable by v3.0 tooling via schema migration reader; v3.0 outputs carry explicit schema version |
| Q8 | **Repeatable builds** | Fresh checkout → full corpus rebuild | < 24h wall time; chunk content deterministic via content-pinning cache; heuristic/off modes byte-stable |
| Q9 | **Graceful degradation** | GX10 endpoint unreachable | Pipeline continues with heuristics (dual-write retained); `[SANITIZE_SKIPPED: endpoint unreachable]` sentinel logged per chunk |
| Q10 | **Observability** | Visual retrieval miss root-caused | Per-query fusion weights + normalized scores logged; per-chunk lineage traceable to extraction engine + sanitizer model version + prompt version |

### 2.3 Key Technical Risks

| # | Risk | Severity | Probability | Mitigation |
|---|---|---|---|---|
| R1 | LLM sanitization hallucinates or alters factual content | **Critical** | Medium | Multi-layer guard: numeric/entity preservation, code-span hashing, order-preservation, edit-distance ceiling (§3.3). Dual-write with heuristics retained. Property-based tests for each guard layer (§4 Phase B B7). |
| R2 | LLM sanitization deferral becomes permanent one-way door | **High** | High | Heuristics retained alongside LLM (dual-write) for v3.0–v3.1; explicit fork-back trigger (§6). Aggregate dominance criterion with degradation cap (§3.3). |
| R3 | ColPali multi-vector storage requires Qdrant version upgrade | **High** | Medium | Phase C-spike probes Qdrant MaxSim support before any Phase C implementation (§4.2). Corpus-size target C12 bounds storage ceiling at design time. |
| R4 | Visual storage 10–100× at scale (5GB for 34 docs → ~150GB at 1000 docs raw, ~400GB indexed) | **High** | Medium | C12 defines operational ceiling (1000 docs). Phase C entry criterion: verify ColPali resolution/patch count sustains ≤500GB indexed at 1000 docs. |
| R5 | Visual retrieval degrades text-heavy doc recall via fusion noise | **Medium** | Medium | Profile-conditional fusion weights with non-zero floor (visual=0.1 for PROSE). Sweep weights on deficit subset before declaring default. C-spike measures impact on text-heavy docs. |
| R6 | ColPali/omlx memory conflict with existing Qwen3-Embedding-8B | **Medium** | Medium | Phase C-spike validates co-residency on omlx server before Phase C implementation. |
| R7 | Edit-distance guard misses semantic corruption under token budget | **High** | Medium | Numeric/entity extraction gate, code-span hashing, order-preservation check layered on token-count ceiling (§3.3). Golden-output regression tests per guard. |
| R8 | MinerU AGPL-3.0 license incompatible with project distribution | **Medium** | Low (deferred) | Containerize as isolated service if adopted; keep Docling as fallback extraction path. |
| R9 | VLM-native parsing changes chunk shape → invalidates regression fixtures | **High** | Low (deferred) | Separate cycle after UIR stabilizes; maintain semantic-identity gate. |
| R10 | LLM sanitization non-deterministic → builds not reproducible | **High** | High | Content-pinning cache (§7.4): deterministic key `(content_hash, model_id, prompt_version) → sanitized_output`. Heuristic/off modes byte-stable. Build-reproducibility CI test. |
| R11 | Prompt injection via document content in sanitization path | **Medium** | Low | Content treated as data, not instructions; prompt structured with XML boundary delimiters (§3.3 guard #6). Input-length cap per chunk. |
| R12 | Solo-dev cycle overrun on Phase B (prompt engineering unbounded) | **High** | High | Phase B estimated 18–22 days (1.5–1.8× nominal). Dual-write means heuristics protect quality. Calendar overlay shows B overlaps C in execution time; see §4.1. |
| R13 | Phase A semantic-identity gate misses a material regression | **Medium** | Low | Gate: identical content, chunk_type; flags additive; top-5 doc IDs unchanged. Stable chunk-identity key `(doc_id, page, content_hash_prefix)` resolves mapping when chunker splits/merges units. |
| R14 | LLM sanitization test infrastructure underspecified → guard layers untestable | **High** | Medium | Phase B B7: property-based tests per guard, mocked-endpoint integration tests, golden-output regression for prompt versions (§4 Phase B). |

---

## 3. V3.0 Target Architecture

### 3.1 Architecture Overview

V3.0 is organized as two peer subsystems (Ingestion Pipeline, Retrieval Backend) with a sidecar Quality & Evaluation track.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           V3.0 ARCHITECTURE                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐  │
│  │     INGESTION PIPELINE          │    │      RETRIEVAL BACKEND          │  │
│  │                                 │    │                                 │  │
│  │  Stage 1: True UIR              │    │  ┌───────────────────────────┐  │  │
│  │  ┌───────────────────────────┐  │    │  │ Text Retrieval (retained) │  │  │
│  │  │ Format-agnostic           │  │    │  │ ├─ dense : omlx Qwen3     │  │  │
│  │  │ ConversionPlan →          │  │    │  │ └─ sparse: BM25           │  │  │
│  │  │ UniversalDocument         │  │    │  └───────────────────────────┘  │  │
│  │  │ Decoupled ElementProcessor│  │    │                                 │  │
│  │  └───────────────────────────┘  │    │  ┌───────────────────────────┐  │  │
│  │              │                  │    │  │ Visual Retrieval (NEW)    │  │  │
│  │              ▼                  │    │  │ ├─ ColPali/ColQwen2.5     │  │  │
│  │  Stage 2: LLM Sanitization      │    │  │ ├─ Qdrant multi-vector    │  │  │
│  │  ┌───────────────────────────┐  │    │  │ └─ MaxSim scoring         │  │  │
│  │  │ GX10 FP8 endpoint         │  │    │  └───────────────────────────┘  │  │
│  │  │ + Content-pinning cache   │  │    │              │                  │  │
│  │  │ Multi-layer guard stack   │  │    │              ▼                  │  │
│  │  │ Heuristic dual-write      │  │    │  ┌───────────────────────────┐  │  │
│  │  └───────────────────────────┘  │    │  │ Fusion + Rerank           │  │  │
│  │                                 │    │  │ ├─ Profile-conditional RRF│  │  │
│  └─────────────────────────────────┘    │  │ ├─ Leg-skip re-normalization │  │
│                                         │  │ └─ ModernBERT rerank      │  │  │
│  ┌─────────────────────────────────┐    │  └───────────────────────────┘  │  │
│  │  EXTRACTION ENGINES (pluggable) │    │                                 │  │
│  │  PDF (Docling) | EPUB | Future  │    └─────────────────────────────────┘  │
│  └─────────────────────────────────┘                                         │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │        QUALITY & EVALUATION (Sidecar — cross-cuts both subsystems)   │    │
│  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────┐  │    │
│  │  │ Modality-Aware Gates│  │ Synthetic Soak      │  │ Strict Gate  │  │    │
│  │  │ PROSE | FORM | TABLE│  │ + Visual Relevance  │  │ FORM/TABLE/  │  │    │
│  │  │ DIAGRAM rubrics     │  │ + Spatial Fidelity  │  │ DIAGRAM lanes│  │    │
│  │  └─────────────────────┘  └─────────────────────┘  └──────────────┘  │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Key architectural relationships:**
- **Ingestion → Retrieval:** Ingestion produces `ingestion.jsonl` (text chunks) + rendered pages (visual input). Retrieval consumes both independently — no tight coupling.
- **Ingestion → Quality:** Quality evaluates ingestion output (strict gate) and retrieval output (soak).
- **Retrieval → Quality:** Quality measures retrieval axes; profile-conditional weights evaluated per-soak.
- **Visual retrieval does NOT depend on UIR or LLM sanitization.** ColPali embeds rendered page images; it can be built and validated against v2.16 output today.

### 3.2 Ingestion Pipeline Stage 1: True Universal Intermediate Representation (UIR Refactor)

**Directly addresses:** Limit 3 (Chunker-State Fragmentation), v2.15 Item #13 (parked)
**Governance anchor:** `docs/DECISIONS.md` — "Combined ceiling: UIR refactor + LLM cleanup together" (v2.10 chunker-quality ceiling); "v2.11 Carry-Forward §3c — PAUSED for user signoff"

**Design:**

The v2.X architecture has a partial UIR (`UniversalDocument`, `UniversalPage`, `Element`) defined in `src/mmrag_v2/universal/intermediate.py`, but the production pipeline does not actually route through it. `PdfConversionPlan`, `BatchProcessor`, and `DoclingPdfAdapter` each construct Docling options independently and the HybridChunker operates directly on Docling's `DoclingDocument` layout objects.

V3.0 elevates the UIR to the **single source of truth between extraction and chunking:**

1. **Format-Agnostic `ConversionPlan`:** `PdfConversionPlan` is elevated to a parent `ConversionPlan` class. All format-specific adapters (PDF, EPUB, future DOCX/HTML) produce a `ConversionPlan` that the pipeline consumes uniformly.

2. **Extraction Engines as Dumb Pipes:** `PDFEngine`, `EpubEngine` (and future engines) output a standardized `UniversalDocument` schema. They do NOT construct Docling options independently — the shared adapter/factory is the single construction site (preserving the v2.8 `test_no_raw_converter_invocation_outside_adapter` guard).

3. **Decoupled `ElementProcessor`:** The chunker operates solely on `UniversalDocument`, entirely detached from Docling's `DoclingDocument` layout classes. This enables:
   - Holistic cross-page boundary state (code blocks, tables, sections spanning pages)
   - Reliable `partial_code` flag emission (activating the inert v2.16 adjacency fetch)
   - Single cleanup site for all format defects (not per-profile branches)

**Interface contract (Python 3.10 dataclasses):**

```python
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum

class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CODE = "code"
    FORM = "form"

class LocatorType(Enum):
    """How to locate this element in its source document."""
    BBOX = "bbox"            # Fixed-layout: PDF, scanned images
    FLOW_OFFSET = "flow_offset"  # Reflowable: EPUB, HTML
    DOM_PATH = "dom_path"    # Structured: HTML, DOCX

@dataclass
class Locator:
    """Source-document location, format-appropriate."""
    type: LocatorType
    # For BBOX type:
    bbox: Optional[List[int]] = None         # [x1, y1, x2, y2] in [0, 1000]
    page_number: Optional[int] = None
    # For FLOW_OFFSET / DOM_PATH types:
    path: Optional[str] = None               # CFI, DOM XPath, or character offset range

@dataclass
class ConfidenceBreakdown:
    """Per-source confidence scores. Collapsing to one float loses signal.
    Fields are Optional: sentinel convention is None = "not applicable" for
    that particular extraction path (e.g., IMAGE modality has no
    text_extraction_confidence; EPUB flow elements have no layout_confidence).
    When a confidence IS applicable but unavailable, use -1.0.
    """
    layout_confidence: Optional[float] = None
    text_extraction_confidence: Optional[float] = None
    ocr_confidence: Optional[float] = None  # Only if OCR was applied
    classification_confidence: Optional[float] = None  # Element type classification

@dataclass
class UIRChunk:
    """Emitted by ElementProcessor; consumed by chunker + sanitizer.

    Provenance contract:
      - `content` is ALWAYS the current authoritative value. When sanitization
        is applied and accepted, `content` = `content_sanitized`. When
        sanitization is not applied or rejected, `content` = raw extraction.
      - `content_original` captures the pre-sanitization raw extraction. Set
        whenever sanitization was attempted and changed or proposed to change
        the content: on acceptance, `content_original` = raw extraction,
        `content` = sanitized output. On rejection, `content_original` = raw
        extraction, `content_sanitized` = <what the LLM produced>,
        `content` = raw extraction (reverted for safety). On "not_applied",
        `content_original` is None.
      - `content_sanitized` is populated whenever sanitization was attempted
        (accepted or rejected), preserving the LLM's output for guard-stack
        debugging. This is critical for false-positive analysis — you need
        to see what the LLM produced even when it was rejected.
      - `sanitization_status` is the single source of truth for whether
        sanitization was applied to this chunk. The string value encodes
        the outcome: "not_applied" | "accepted" | "rejected:*" | "skipped:*".
    """
    modality: Modality
    content: str                          # Always authoritative
    locator: Locator
    confidence: ConfidenceBreakdown
    extraction_method: str                # "docling_direct" | "ocr_tesseract" | "ocr_doctr" | "vlm_enrichment"
    extraction_engine_version: str        # e.g., "docling-2.86.0"
    structural_flags: Dict[str, bool]     # "partial_code", "cross_page_split", "orphan_label", etc.
    source_element_ids: List[str]         # Traceability back to extraction engine elements
    asset_ref: Optional[str] = None       # Path to extracted image/asset, if IMAGE modality
    lang: Optional[str] = None            # ISO 639-1 language code (detected or from ConversionPlan)
    reading_order: Optional[int] = None   # Monotonic logical position within page/document

    # Provenance fields:
    content_original: Optional[str] = None              # See docstring: set whenever sanitization was attempted
    content_sanitized: Optional[str] = None             # If sanitization was applied (may equal content)
    sanitizer_model_id: Optional[str] = None            # e.g., "RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic"
    sanitizer_prompt_version: Optional[str] = None      # Git-hash of prompt template
    sanitization_status: str = "not_applied"            # Single source of truth — see docstring above

    # Hierarchical context:
    parent_element_id: Optional[str] = None  # Table cell → parent table; caption → parent figure
    parent_heading: Optional[str] = None     # Nearest ancestor heading text (carried forward from v2.X)

@dataclass
class ConversionPlan:
    """Format-agnostic extraction plan; subclasses add format-specific fields."""
    source_path: str
    file_type: str                        # "pdf" | "epub" | "html"
    doc_id: str                           # First 12 chars of MD5 hash
    profile_type: str                     # From ProfileClassifier
    modality_flags: Dict[str, bool]       # "is_scanned", "has_encoding_corruption", etc.
    extraction_strategy: str              # "digital_native" | "ocr_nuclear" | "ocr_forced"
    reading_order_strategy: str           # "docling_native" | "y_sort" | "y_sort_with_dropcap"
    batch_size: int = 10
    render_dpi: int = 200                 # Resolution for page renders consumed by visual retrieval
    lang_hint: Optional[str] = None       # ISO 639-1, when known a priori
    # ... format-specific fields in subclasses
```

**Acceptance gate (Phase A): Semantic-identity gate:**

1. **Stable chunk-identity key:** `(doc_id, page_number, content_hash_prefix)` where `content_hash_prefix` = first 64 bits of SHA-256 of content (16 hex chars). Unicode normalization (NFC), internal whitespace collapse (consecutive whitespace → single space), and line-ending normalization (CRLF → LF) are applied **before** hashing, making the key robust to platform differences.
2. **Matching policy:**
   - If v3.0 chunk has same identity key as a v2.16 chunk → content must match (modulo trailing whitespace), `chunk_type` must match.
   - If v2.16 chunk A becomes v3.0 chunks B1, B2 (split) → `normalize(A.content)` must equal `normalize(concat_with_separator(B1.content, B2.content))` where `normalize` applies NFC + whitespace collapse + CRLF→LF (the same normalization used for hashing). `chunk_type` of all three must match, B1/B2 each carry `cross_page_split` or equivalent structural flag.
   - If v2.16 chunks A1, A2 become v3.0 chunk B (merge) → `normalize(concat_with_separator(A1.content, A2.content))` must equal `normalize(B.content)`. Same normalization + separator convention as the split case.
   - No v2.16 chunk may have zero corresponding v3.0 chunks (dropped content = FAIL).
3. **Structural flags strictly additive** — no v2.16 flag goes missing; new flags may appear.
4. **Retrieval invariant** — top-5 doc IDs unchanged for every query in the v2.16 regression fixture.

### 3.3 Ingestion Pipeline Stage 2: LLM-Native Chunk Sanitization

**Directly addresses:** Limit 2 (Heuristic Patching Ceiling)
**Governance anchor:** `docs/DECISIONS.md` — "Combined ceiling: UIR refactor + LLM cleanup together" (v2.10 chunker-quality ceiling, path 4: "LLM-clean every chunk on ingestion" estimated at $30/rebuild on Dashscope, now $0 via local GX10)

**Design:**

After Stage 1 ensures all chunks flow through the UIR, Stage 2 adds a **semantic polish pass** using the existing local LLM endpoint:

1. **Execution:** Raw extraction chunks are piped through `RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic` at `10.0.10.239:8000` (GX10 endpoint, already operational per `docs/DECISIONS.md` v2.14).

2. **Content-pinning cache:** LLM calls are cached via a deterministic key: `(content_hash, context_hash, model_id, prompt_version) → sanitized_output`, where `context_hash = SHA-256(prev_chunk_content_first64bits + next_chunk_content_first64bits + detected_lang)`. Including context prevents stale cache hits when neighbor chunks change between rebuilds (breaking the surrounding-context assumption in the prompt). On cache hit, the LLM is not invoked. This makes builds deterministic even if the GX10 endpoint does not support deterministic decoding — *and* reduces wall time on rebuild (a chunk whose raw content AND context haven't changed won't be re-sanitized). Cache is file-backed under `output/sanitization_cache/`, keyed by content hash prefix + first 8 chars of context hash. See §7.4 for the full determinism policy.

3. **Prompt contract:** The LLM receives the raw chunk content + surrounding context (previous/next chunk snippets, page breadcrumb, detected language) and is instructed to reconstruct the text into clean, well-formed Markdown:
   - Heal cross-page sentence and code-block splits
   - Normalize whitespace and punctuation
   - Remove extraction artifacts (drop-cap orphans, picture classification labels, OCR noise)
   - Preserve ALL factual content — no summarization, no paraphrasing
   - Edit-distance budget enforced

4. **Sanitization mode flag:** `--sanitize-mode={off,llm,heuristic,both-and-diff}`
   - `off`: No sanitization — raw UIR chunks emitted (v2.16 equivalent, for regression baseline)
   - `llm`: LLM sanitization only — heuristics skipped (Phase B target)
   - `heuristic`: Existing heuristic stack only — LLM skipped (v2.16 behavior preserved)
   - `both-and-diff`: Both run, output compared; disagreement logged (validation mode)

5. **Heuristics retained — NOT ripped out.** Phase B does NOT remove `docling_postprocess.py` or the multimodal validation layers. The heuristic stack remains operational and is the fallback path. See §6.2 for deprecation trigger.

6. **Cost:** $0 per rebuild (local GX10 FP8 endpoint). No cloud API calls on the sanitization path.

**Multi-layer guard stack (defense-in-depth against hallucination):**

| Guard | What it catches | Failure mode if absent |
|---|---|---|
| **Edit-distance ceiling** (>30% token change → reject) | Gross rewrites, fabrications | — |
| **Numeric/entity preservation** (all numbers, dates, identifiers, named entities must appear verbatim in sanitized output) | "100 mg" → "10 mg"; date changes; ID swaps | Subtle factual corruption within budget |
| **Code-span hashing** (text inside ``` fences must be byte-identical or rejected) | LLM "fixing" code syntax; reordering statements | Silent code corruption |
| **Order-preservation check** (regex-identified ordered-list markers must appear in same sequence) | LLM reordering procedural steps, recipes, algorithms | Reordered instructions |
| **Token-level alignment** (Levenshtein distance, not just count delta) | Reorderings that preserve token count | The edit-distance ceiling's blind spot |
| **Content/prompt boundary delimiters** (explicit XML-style tags separating instructions from document content) | Prompt injection via document text | Adversarial document rewrites its neighbors |

**Graceful degradation:** When GX10 endpoint is unreachable, the pipeline:
1. Logs `[SANITIZE_SKIPPED: endpoint unreachable]` sentinel per chunk.
2. Falls back to heuristic sanitization (dual-write retained per §4 Phase B).
3. Emits a build-level warning with unreachable-chunk count.
4. Does NOT hard-fail the build.

**LLM dominance criterion (replaces Draft 0.2's unattainable "strictly equal-or-better per chunk"):** Heuristic stack deprecation is gated on:

1. **Aggregate Format improvement:** LLM-sanitized corpus-wide Format exceeds heuristic output by ≥ 2× the soak run-to-run standard deviation (measured on the heuristic-only baseline across ≥3 consecutive soak runs). This makes the threshold adaptive to actual measurement noise rather than an arbitrary fixed increment. The run-to-run σ is computed once before Phase B acceptance and documented.
2. **No chunk-level cliff:** Zero chunks transition from PASS to FAIL under the strict gate when comparing LLM vs heuristic output.
3. **Degradation rate:** ≤2% of chunks where LLM output scores below heuristic output on Format axis (some local variation is expected and acceptable with a probabilistic LLM over 6,800+ chunks).
4. **Consecutive confirmation:** All three conditions hold for two consecutive soak iterations.

**Diff predicate for `both-and-diff` comparison:** Two sanitization outputs "differ" when their token-level Levenshtein distance exceeds 5% of the shorter output's token count. Whitespace-only differences and Unicode NFC/NFD normalization differences are excluded before comparison. This predicate is used for the degradation rate (item 3) and for the dual-write extension trigger (§6.3 regret #3). Without this defined predicate, the 5% threshold in §6.3 is unfalsifiable.

**Acceptance gate (Phase B):** Synthetic soak Format scores cross 95% corpus-wide with `--sanitize-mode=llm`, with no regression on Recall, Relevance, or Faithfulness axes vs `--sanitize-mode=heuristic`. LLM dominance criterion items 1–3 met.

### 3.4 Retrieval Backend: Vision-Native Retrieval (ColPali + MaxSim)

**Directly addresses:** Limit 1 (Spatial-to-Text Translation Deficit, -12pp Recall@1)
**Governance anchor:** `docs/DECISIONS.md` — v2.16 Phase 2 diagnosis; `docs/PLAN_V2.16.md` Item #11 (ColPali/VisRAG scoped out)
**Independence:** Visual retrieval embeds rendered page images. It does NOT depend on UIR or LLM sanitization output. Phase C-spike can and must run before Phase A implementation.

**Design:**

The v2.16 retrieval stack (dense + sparse + RRF + ModernBERT reranker) operates entirely on text embeddings. For text-heavy documents, this stack approaches 98.6% Recall@5 doc. For visually-complex documents (diagrams, multi-column forms, scanned engineering drawings), text embeddings discard the 2D spatial information that makes the content meaningful.

V3.0 introduces a **parallel visual vector index** that operates alongside the text retrieval stack:

1. **Patch-Level Visual Embeddings (ColPali/ColQwen2.5):**
   - Each page is rendered at the resolution specified in `ConversionPlan.render_dpi` (default 200 DPI).
   - A Vision Transformer (ColPali, or its successor ColQwen2.5) divides the page into patches (e.g., 1030 patches per page at 448×448 resolution).
   - Each patch is embedded as a 128-dimensional vector.
   - The result: a page is represented as a **matrix of patch vectors**, not a single flat embedding.

2. **Qdrant Multi-Vector Storage:**
   - Qdrant is configured to support multi-vector collections.
   - Each document page stores its patch-vector matrix as a separate point (page-level granularity).
   - Visual collection is independent of text collections — can be dropped/rebuilt without affecting text retrieval.

3. **MaxSim Late-Interaction Retrieval:**
   - At query time, the query is embedded through the same ColPali vision model.
   - MaxSim (Maximum Similarity) computes: for each query token, find its most similar document patch, then sum those maximum similarities.
   - Output: page-level scores (not chunk-level).

4. **Granularity Join Policy (page → chunk mapping):**
   Visual retrieval produces page-level scores. Text retrieval produces chunk-level scores. For RRF fusion:
   - Each visual page score is propagated to ALL chunks associated with that page number.
   - This is intentionally coarse — visual retrieval signals "this page is relevant," and the reranker discriminates among chunks on that page.
   - Alternative (one score per chunk) would require per-chunk visual crops, increasing storage cost ~10×.
   - **C-spike now measures this trade-off explicitly** (§4.2): reranker top-1 selection rate on visually-retrieved pages must ≥60%. If below, region-level granularity must be in Phase C scope, not deferred.

5. **Profile-Conditional Fusion Weights (with non-zero floor):**

   | Profile class | Dense weight | Sparse weight | Visual weight | Rationale |
   |---|---|---|---|---|
   | PROSE (default) | 1.0 | 1.0 | **0.1** | Non-zero floor lets visual act as tiebreaker on figures/diagrams within prose docs; prevents waste of deployed visual infra |
   | DIAGRAM | 1.0 | 1.0 | **0.4** | Visual is primary signal for diagrams/schematics |
   | FORM | 1.0 | 1.0 | **0.4** | Visual captures form layout; text captures field values |
   | TABLE | 1.0 | 0.5 | **0.3** | Sparse still useful for exact numeric match; visual for structure |

   **Per-document weights are a known limitation.** A PROSE document may contain a diagram page, and per-document profile-conditional weights use the document's profile, not the page's actual content. Future work (v3.2+): per-query profile classification, or a query-time `--boost-visual` flag to escalate visual weight for queries targeting visual content within prose documents.

6. **Fusion Re-Normalization on Leg Skip:**
   When any retrieval leg is skipped (e.g., visual leg when ColPali is unreachable), remaining weights are L2-normalized to unit norm, preserving their relative proportions. Without re-normalization, scores from text-only fusion aren't comparable to scores when visual was present, which breaks any score-based threshold downstream. Example: PROSE weights `(1.0, 1.0, 0.1)` with visual skipped → remaining `(1.0, 1.0)` L2-normalized to `(0.707, 0.707)`. DIAGRAM weights `(1.0, 1.0, 0.4)` with visual skipped → remaining `(1.0, 1.0)` L2-normalized to `(0.707, 0.707)`. All profiles converge to equal text-leg weights on leg skip — the profiles differ only in the presence of the visual leg.

**End-to-end retrieval flow (V3.0):**

```
query
  ├─ dense  : omlx Qwen3-Embedding-8B-mxfp8 (4096-dim)     ← retained (v2.13)
  ├─ sparse : BM25                                         ← retained (v2.12)
  └─ visual : ColPali/ColQwen2.5 → MaxSim (page scores)    ← NEW (v3.0)
           → page scores propagated to child chunks (join policy)
  → RRF fusion (k=60, profile-conditional weights; leg-skip re-normalization)
  → ModernBERT rerank (top-25 → top-5)                     ← retained (v2.12)
  → top-5 return
```

**Reranker behavior on visual hits:** ModernBERT is text-only. When a chunk enters the top-25 primarily due to its visual score, the reranker evaluates `(query, chunk_content)`. The visual signal has already done its job (getting the page into the candidate set). The reranker then selects the best text chunks on that page.

**Resource budget (operational target: ≤1000 docs per C12):**
- Patch-vector storage: ~128 dimensions × 1030 patches × ~300,000 pages (at 1000 docs) = ~150 GB raw; ~400-500 GB indexed with HNSW overhead.
- Visual embedding latency: target <1s/page at omlx server inference speed.
- MaxSim latency: O(query_tokens × doc_patches) for exhaustive; Qdrant ANN caps at configurable ef_search.

**Phase C entry criterion:** Before Phase C full build, verify ColPali resolution/patch count sustains ≤500GB indexed storage at the C12 target (1000 docs). If not, reduce patch count or page resolution and re-validate via C-spike methodology.

**Phase C pre-flight spike (MUST RUN before Phase A implementation):** See §4.2.

**Acceptance gate (Phase C):** Recall@1 chunk ≥80% on deficit docs (from 67.8%), Recall@5 doc ≥98.6% maintained. Text-heavy docs (29/34) show no regression. Reranker top-1 selection rate on visually-retrieved pages ≥60%.

### 3.5 Quality & Evaluation (Sidecar Track)

**Directly addresses:** CarOK form-class false-negative Format penalties, profile-specific evaluation, visual retrieval quality axes
**Governance anchor:** `docs/QUALITY_GATES.md` — "Form / Invoice Acceptance Class" (v2.8); `docs/DECISIONS.md` — "v2.12 Phase 0 Outcome" (CarOK form-shape decision)

**Design:**

Quality evaluation is a cross-cutting concern that applies to both ingestion output (strict gate) and retrieval output (synthetic soak). It is not a "layer" above retrieval — it is a sidecar track.

1. **Profile-Specific Judge Rubrics:**
   - `PROSE` (default): relevance, fluency, formatting — the existing axes.
   - `FORM`: data integrity (key-value completeness), field accuracy, no unstructured prose requirement.
   - `TABLE`: structural fidelity (row/column preservation), numeric accuracy.
   - `DIAGRAM`: visual description quality, label accuracy, spatial relationship capture.

2. **UIR Modality Tag Driving Judge Selection:**
   - `UIRChunk.modality` field determines which judge rubric applies.
   - `FORM` and `TABLE` chunks skip prose-fluency checks; `DIAGRAM` chunks skip OCR-text accuracy checks.

3. **Visual Retrieval Quality Axes:**
   - **Visual Relevance:** Scored 0/1/2 per retrieval by LLM-as-judge with access to the page render.
   - **Spatial Fidelity:** Alignment between MaxSim top-patch heatmap and ground-truth relevant page regions (where available).

4. **Strict Gate Extension:**
   - `FORM_AUDIT_PASS` extended to `TABLE_AUDIT_PASS` and `DIAGRAM_AUDIT_PASS`.
   - Universal invariants apply across all classes — no waivers.

5. **Holdout policy:** Minimum 8 blind-test documents — 2 per rubric (PROSE, FORM, TABLE, DIAGRAM). The Greenhouse document is the PROSE holdout. Effective at v3.0 final tag. Rubric-to-holdout mapping documented in `docs/QUALITY_GATES.md` before Phase D. Two documents per rubric is the minimum to guard against overfitting four modality rubrics on a 34-doc corpus.

**Acceptance gate (Phase D):** `CarOK_voorraadtelling` Format score no longer penalized for non-prose content shape. Modality breakdown visible in soak reports. No false-negative quality failures on structured data. All 8 holdout documents pass.

---

## 4. Implementation & Phasing Strategy

V3.0 is sequenced to respect solo-dev 12-day convergence cycles. **Visual retrieval (Phase C) does not depend on UIR or LLM sanitization** — ColPali embeds rendered page images directly. The Phase C pre-flight spike runs in parallel with Phase A.

### 4.1 Phase Dependency & Calendar Diagram

```
                     Nominal 12-day cycles ──────────────────────────────→

  C-spike (2-3d)
      │
      ├─── Phase A: UIR (12d) ──────────────────┐
      │    D1 (rubrics) runs in parallel ──────┐ │
      │                                        │ │
      ├─── Phase B: LLM Sanitize (18-22d) ──────────────┐   ← overlaps C
      │    B1-B6: prompt eng, guards, dual-write        │
      │                                                 │
      └─── Phase C: Visual Retrieval (12d) ─────────────┤
           C1-C7: gated on C-spike PASS                  │
                                                         │
                                         Phase D: Gates (12d)
                                         D2-D4: strict gate ext, full acceptance
```

**Calendar implications of R12 (Phase B 50-80% overrun):**
- Phase B execution (18-22 days) will overlap Phase C implementation (12 days) in calendar time.
- Solo developer must context-switch between LLM prompt engineering and visual retrieval integration during the overlap window.
- **Recommendation:** Front-load C-spike to complete before Phase B starts (spike is 2-3 days and is independent). Schedule Phase C implementation to start at day 12 of Phase B's execution window (when nominal B would have completed). Elapsed: 24 days vs 34 sequential — saving 10 days, which justifies the overlap's cognitive cost. At day 12, B's prompt engineering is in active refinement; the context-switch cost is real but the 10-day calendar savings is worth it for a solo-dev cycle.

### 4.2 Phase C Pre-Flight Spike (MANDATORY — before Phase A code)

**Minimum experiment to prove or disprove the ColPali approach:**

1. Pick the single highest-deficit doc — `ATZ_Elektronik_German`.
2. Render all pages at 200 DPI (target ColPali resolution).
3. Embed pages with off-the-shelf ColPali on a workstation. **No omlx deployment. No Qdrant integration. No MaxSim in production code.**
4. Take 20 queries from the v2.16 regression fixture targeting this doc (or hand-craft if fixture coverage is thin).
5. Embed each query with ColPali. Compute MaxSim scores against page-vector matrices in raw numpy. Rank pages.
6. For each query, record: text-retrieval top-1 page (v2.16), visual-retrieval top-1 page (this experiment), gold page.
7. **PASS condition A:** Visual retrieval recovers the correct page on ≥60% of queries where v2.16 text retrieval failed, without harming queries where text retrieval was correct.
8. **PASS condition B:** Reranker top-1 selection rate on visually-retrieved pages ≥60%. Measure by: for each query where visual retrieval placed the correct page in the top-5, construct a candidate set mimicking production — top-25 text chunks from the full corpus under text retrieval, plus the visually-retrieved page's chunks (deduplicated). Simulate ModernBERT reranking on this combined set and check whether the correct chunk lands in the reranker's top-1. This production-realistic candidate construction avoids the overly optimistic assumption of per-page reranking. If <60%, page-level granularity is insufficient and region-level must be in Phase C scope, not deferred.
9. Also verify: ColPali model fits alongside Qwen3-Embedding-8B on the omlx server; end-to-end latency <1s/page.

**Time budget:** 2–3 days. **Hardware:** workstation. **No production infrastructure touched.**

**Outcome:** If PASS (both A and B) → Phase C implementation proceeds as designed. If PASS A but FAIL B → Phase C scope expands to include region-level granularity. If FAIL A → Phase C as designed is dead; redirect to VLM-native parsing evaluation or alternative visual model.

### 4.3 Partial-Release Policy

| Version | Ships independently? | Condition |
|---|---|---|
| `3.0.0` (Phase A) | **No** — held until `3.1.0` is also ready | UIR overhead without LLM-cleanup benefit would be a Format regression vs v2.16 heuristic stack. Shipping 3.0.0 alone = deploying the abstraction cost without the quality payoff. |
| `3.1.0` (Phase B) | **Yes** — with UIR already deployed | LLM sanitization is the primary Format-quality improvement. Ships alongside 3.0.0 to provide a complete v3.0 ingestion pipeline. |
| `3.2.0` (Phase C) | **Yes** — additive | Visual retrieval is independent of ingestion. Can ship later without affecting v3.1.0 pipeline output. Gated on C-spike PASS. |

### Phase A: UIR Foundation (Cycle 3.0)

| Task | Description | Acceptance |
|---|---|---|
| A1 | Elevate `PdfConversionPlan` to parent `ConversionPlan` | All existing tests pass without modification |
| A2 | Refactor extraction engines to output `UniversalDocument` | Semantic-identity gate (§3.2): stable chunk-identity key match, content match, chunk_type match; flags additive; top-5 doc IDs unchanged |
| A3 | Decouple chunker from `DoclingDocument` → operate on UIR | `partial_code` flags emit correctly; cross-page splits attributed to correct pages |
| A4 | Rip out duplicate Docling option construction sites | `test_no_raw_converter_invocation_outside_adapter` expanded to cover v3.0 paths |
| A5 | Corpus-wide rebuild + strict gate | 34/34 PASS (or documented deferrals matching v2.16 baseline) |
| A6 | Schema version: `2.7.0` → `3.0.0` | All output carries `schema_version: "3.0.0"` |

### Phase B: LLM Sanitization (Cycle 3.1)

| Task | Description | Acceptance |
|---|---|---|
| B1 | Wire GX10 FP8 endpoint + content-pinning cache | Sanitization harness operational; cache hit rate measured; all `--sanitize-mode` flags functional |
| B2 | Design + validate sanitization prompt (spike on 100-chunk sample first) | Prompt passes negative tests (no hallucination, no summarization, no content loss, preserves numerics/entities/ordered markers) |
| B3 | Implement multi-layer guard stack (6 guards) | Each guard has positive + negative regression tests; golden-output fixture per guard |
| B4 | Dual-write LLM + heuristics; compare output with defined diff predicate | `both-and-diff` comparison functional; diff predicate: token-level Levenshtein >5% (excluding whitespace + Unicode normalization diffs) |
| B5 | Implement LLM sanitization test infrastructure | Property-based tests per guard layer; mocked-endpoint integration tests; golden-output regression for each prompt version |
| B6 | Corpus-wide rebuild + synthetic soak (llm vs heuristic) | Format ≥95% corpus-wide with LLM; no regression; dominance criterion items 1–3 met (§3.3) |
| B7 | Schema version: `3.0.0` → `3.1.0` | Output carries `schema_version: "3.1.0"`; provenance fields populated |

### Phase C: Visual Retrieval (Cycle 3.2)

| Task | Description | Acceptance |
|---|---|---|
| C1 | Probe Qdrant multi-vector support + verify storage at C12 target | MaxSim compatibility confirmed; verify ≤500GB indexed at 1000 docs |
| C2 | Deploy ColPali/ColQwen2.5 on omlx server | Co-residency confirmed; latency <1s/page |
| C3 | Build parallel visual index (single doc → deficit subset → full corpus) | Incremental rollout; each stage validates no regression |
| C4 | Implement MaxSim + profile-conditional fusion + leg-skip re-normalization | `retrieve_hybrid_visual()` functional; re-normalization verified on leg-skip |
| C5 | Implement granularity join policy; sweep profile weights | Reranker top-1 selection rate ≥60% on visually-retrieved pages; sweep confirms weight defaults |
| C6 | Synthetic soak on complex-doc subset + full corpus | Recall@1 ≥80% on deficit docs; text-doc metrics maintained; no regression on 29 text-heavy docs |
| C7 | Schema version: `3.1.0` → `3.2.0` | Visual collection built |

### Phase D: Modality-Aware Gates (Cycle 3.3)

| Task | Description | Acceptance |
|---|---|---|
| D1 | Implement profile-specific judge rubrics (runs ∥ Phase A) | FORM, TABLE, DIAGRAM rubrics operational in synthetic soak |
| D2 | Extend strict gate for modality classes | `FORM_AUDIT_PASS`, `TABLE_AUDIT_PASS`, `DIAGRAM_AUDIT_PASS` in `qa_full_conversion.py` |
| D3 | Add visual retrieval quality axes | Visual Relevance, Spatial Fidelity axes operational |
| D4 | Full-corpus v3.0 acceptance run | Smoke matrix: all rows GATE_PASS + UNIVERSAL_PASS; ≥8 blind-test holdouts included |

---

## 5. Component Architecture (Detailed)

### 5.1 Module Map (V3.0 Target)

```
src/mmrag_v2/
├── universal/                     # [EXPANDED] True UIR layer
│   ├── intermediate.py            # UniversalDocument, UniversalPage, UIRChunk, Locator, ConfidenceBreakdown
│   ├── conversion_plan.py         # Parent ConversionPlan + format-specific subclasses
│   ├── element_processor.py       # [REFACTORED] Operates on UIR, not Docling DOM
│   ├── router.py                  # Format detection → engine routing (retained)
│   └── quality_classifier.py      # ConfidenceNormalizer (retained)
│
├── engines/
│   ├── base.py                    # FormatEngine ABC (retained)
│   ├── pdf_engine.py              # [REFACTORED] Consumes shared adapter; outputs UniversalDocument
│   ├── docling_adapter.py         # [REFACTORED] Single Docling construction + invocation site
│   ├── docling_postprocess.py     # [RETAINED] Heuristic post-processing; dual-write alongside LLM
│   └── epub_engine.py             # [FUTURE — deferred to post-v3.0]
│
├── sanitization/                  # [NEW] LLM-native chunk sanitization
│   ├── __init__.py
│   ├── llm_sanitizer.py           # GX10 FP8 endpoint client + content-pinning cache
│   ├── edit_guard.py              # 6-layer guard stack
│   ├── prompts.py                 # Versioned, language-aware prompt templates
│   └── graceful_degradation.py    # Endpoint-unreachable fallback policy
│
├── retrieval/
│   ├── pipeline.py                # [EXTENDED] retrieve_hybrid_visual()
│   ├── visual_embedder.py         # [NEW] ColPali/ColQwen2.5 client (omlx)
│   ├── maxsim.py                  # [NEW] MaxSim + page→chunk join policy
│   ├── fusion.py                  # [EXTENDED] Profile-conditional RRF + leg-skip re-normalization
│   └── config.py                  # [EXTENDED] Visual collection defaults + weight profiles
│
├── vision/                        # VLM integration (retained)
├── validators/                    # QA checks (retained; heuristics live alongside LLM)
│   ├── corruption_interceptor.py  # [RETAINED]
│   ├── token_validator.py
│   └── quality_filter_tracker.py
│
├── ocr/                           # OCR cascade (retained)
├── chunking/                      # Chunking helpers (retained)
├── schema/                        # Output schema (extended for v3.x)
├── state/                         # Context tracking (retained)
├── orchestration/                 # Profile intelligence (retained)
└── rag/                           # Downstream RAG (retained)
```

### 5.2 Data Flow (V3.0 Ingestion)

```
INPUT PDF
    │
    ▼
DocumentDiagnosticEngine  ──→  ProfileClassifier  ──→  ConversionPlan
    │
    ▼
DoclingPdfAdapter.convert()
    │
    ▼
UniversalDocument  ──→  ElementProcessor  ──→  List[UIRChunk]
    │                                               │
    │                                    ┌──────────┘
    │                                    ▼
    │                    ┌───────────────┴───────────────┐
    │                    │  LLMSanitizer  │  Heuristics  │
    │                    │  (GX10 FP8)   │  (retained)  │
    │                    │  + cache      │              │
    │                    └───────┬───────┴──────┬───────┘
    │                            │              │
    │                    ┌───────┴───────┐      │
    │                    │ Guard stack   │      │
    │                    │ (6-layer)     │      │
    │                    └───────┬───────┘      │
    │                            │              │
    │                    ┌───────┴───────┐      │
    │                    │ both-and-diff │      │
    │                    │ (Levenshtein  │      │
    │                    │  >5% = diff)  │      │
    │                    └───────┬───────┘      │
    │                            │              │
    │                            ▼              ▼
    │                    Cleaned/validated chunks
    │                            │
    ▼                            ▼
HybridChunker (UIR-native)  ←───┘
    │
    ▼
IngestionChunk[]  ──→  ingestion.jsonl + assets/
    │
    ├─→ Qdrant text ingest
    │     ├── dense  : omlx text embed → mmrag_v3__dense
    │     └── sparse : BM25 → mmrag_v3__sparse
    │
    └─→ Rendered pages (200 DPI)
          └── ColPali embed → mmrag_v3__visual  (Phase C)
```

### 5.3 Retrieval Flow (V3.0 Production)

```
QUERY
    │
    ├─→ dense  : omlx Qwen3-Embedding-8B-mxfp8 (4096-dim)
    ├─→ sparse : BM25 (token match)
    └─→ visual : ColPali patch vectors → MaxSim (page scores)
              → page scores propagated to child chunks (join policy)
    │
    ▼
RRF fusion (k=60, profile-conditional weights; leg-skip re-normalization)
    │
    ▼
ModernBERT rerank (top-25 → top-5)
    │
    ▼
Top-5 results (text chunks from relevant pages)
```

---

## 6. Design Decisions, Fork-Back Triggers & Regret Risks

### 6.1 Design Decisions

| Decision | Phase | Rationale |
|---|---|---|
| **Docling retained** + LLM sanitization; VLM-native deferred | A-B | Lower risk; LLM-clean achieves Format improvement with $0 cost. Fork-back trigger defined at §6.2. |
| **LLM sanitization via local GX10 FP8** | B | $0 cost; privacy-preserving; already operational (v2.14). Content-pinning cache ensures determinism. |
| **ColPali via omlx** | C | $0 per-query cost; LAN-local. |
| **Profile-conditional fusion weights with non-zero PROSE floor (0.1)** | C | Non-zero floor prevents wasted visual infra; tiebreaker effect on embedded figures in prose docs. |
| **Semantic-identity gate with stable chunk-identity key** | A | `(doc_id, page, content_hash_prefix)` survives Unicode/whitespace/line-ending normalization. |
| **Multi-layer guard stack** for LLM sanitization | B | Defense-in-depth: edit-distance + numeric/entity + code-span + order-preservation + token-alignment + prompt-boundary. Property-based tests per guard. |
| **Heuristics retained alongside LLM (dual-write)** | B | Probabilistic vs deterministic; fallback + regression comparison. Aggregate dominance criterion with degradation cap gates deprecation. |
| **Granular schema versioning** (3.0.0/3.1.0/3.2.0) | A/B/C | Isolates blame surfaces; enables incremental adoption. 3.0.0 held until 3.1.0 ready (§4.3). |
| **Content-pinning cache for sanitization determinism** | B | `(content_hash, model_id, prompt_version) → sanitized_output`. Reproducibility without requiring deterministic LLM endpoint. |
| **Leg-skip fusion re-normalization** | C | Preserves score comparability across queries; prevents threshold breakage downstream. |
| **Phase C-spike before Phase A implementation** | Pre-A | Proves ColPali viability before any v3.0 code. Two PASS conditions: page recovery + reranker discrimination. |
| **No gate weakening** | All | Per `DECISIONS.md`. |

### 6.2 Fork-Back Triggers

| Decision | Trigger | Action |
|---|---|---|
| **Docling retained vs VLM-native** | Two Phase B soak iterations: Format < 92% LLM AND heuristic < 92% (extraction-side deficit) | Reopen VLM-native parsing evaluation. |
| **Heuristics retained vs removed** | Aggregate dominance + ≤2% degradation rate + no chunk FAIL for two consecutive soak iterations | Deprecate heuristic stack. |
| **Visual retrieval weights** | Sweep outperforms profile-conditional defaults by ≥5pp on any axis | Adopt sweep-optimal weights. |
| **ColPali vs alternative** | C-spike FAIL A (page recovery <60%) or FAIL B (reranker discrimination <60%) | Redirect or expand Phase C scope. |
| **omlx co-residency** | ColPali + Qwen3-Embedding-8B exceed omlx memory | Dedicated visual endpoint or cloud fallback. |
| **Page-level vs region-level granularity** | Reranker top-1 selection rate <60% on visually-retrieved pages in C-spike or Phase C soak | Scope region-level retrieval into Phase C (not deferred). |
| **Content-pinning cache vs deterministic endpoint** | GX10 endpoint supports deterministic decoding at temperature=0 → cache still retained as speed optimization | No decision change; cache is always beneficial. |

### 6.3 Decisions We May Regret (and How We'll Know)

1. **Docling + LLM sanitization as the permanent extraction path.**
   - **Regret condition:** Docling bbox quality on deficit docs is the bottleneck. Visual retrieval gains <5pp despite clean text.
   - **Monitoring signal:** Bbox-IoU between Docling output and a VLM-native reference parse on the 5 deficit docs.

2. **Page-level visual retrieval granularity (not region-level).**
   - **Regret condition:** Reranker fails to discriminate on visually-retrieved dense pages.
   - **Monitoring signal:** Reranker top-1 selection rate on visually-retrieved pages. If <60%, too coarse. Measured in C-spike AND Phase C soak.

3. **Heuristics retained for one cycle only.**
   - **Regret condition:** LLM model updates introduce new failure modes; heuristics would have caught them.
   - **Monitoring signal:** `both-and-diff` disagreement rate (>5% Levenshtein distance per §3.3 diff predicate). If >5% of chunks differ in cycle 1, extend dual-write to cycle 2.

---

## 7. Migration, Rollback & Operations

### 7.1 Schema Version

| Version | Phase | What changes | Consumer impact |
|---|---|---|---|
| `3.0.0` | A | UIR introduced; structural flags populated; chunk content unchanged from v2.16 | Content/`chunk_id` unchanged. Flags additive. |
| `3.1.0` | B | LLM sanitization may change content; provenance fields populated | Content may differ from v2.16. Provenance enables audit. |
| `3.2.0` | C | Multi-vector visual collection alongside text collections | Text collections unchanged. Visual is additive. |

**Downstream breakage:** cached embeddings invalid when content changes (3.1.0); regression fixtures version-pinned; chunk_id joins break if IDs positionally derived; soak baselines reset per version. Partial-release policy at §4.3.

### 7.2 Rollback Paths

| Phase | Rollback | Procedure |
|---|---|---|
| A (UIR) | `git revert` | Semantic-identity output → no migration |
| B (LLM sanitize) | `--sanitize-mode=heuristic` | Exact v2.16 behavior; heuristics retained |
| C (Visual index) | Drop visual collection | Text retrieval unaffected |
| D (Gates) | Gate-only change | No data migration |

### 7.3 Corpus Migration

Each phase: rebuild corpus (~8-12h), re-ingest text (~1-2h), Phase C adds: render + ColPali embed (~5h at 34 docs), synthetic soak (~1-2h), updated regression fixture.

### 7.4 Determinism Policy

- **Content-pinning cache:** `(content_hash, context_hash, model_id, prompt_version) → sanitized_output` where `context_hash = SHA-256(prev_chunk_content_first64bits + next_chunk_content_first64bits + detected_lang)`. On cache hit, no LLM invocation. Cache is file-backed under `output/sanitization_cache/`, keyed by content hash prefix + first 8 chars of context hash. Including context in the key prevents stale cache hits when neighbor chunks change between rebuilds. Makes builds deterministic even without deterministic LLM endpoint. A chunk whose raw content AND context haven't changed between rebuilds won't be re-sanitized — it hits the cache. This also speeds rebuilds.
- **Build-reproducibility test:** CI runs `mmrag-v2 batch --sanitize-mode=llm` on 3-doc subset; asserts chunk-level identical-hash ratio ≥99.5%, with the additional invariant that zero disagreeing chunks have changed semantics (verified by judge sample on all disagreeing chunks).
- **Heuristic mode:** Byte-stable (no LLM). **Off mode:** Byte-stable.
- **Hash-tolerance:** "≥99.5%" = ≥99.5% of chunks have byte-identical content hashes across consecutive builds. The remaining ≤0.5% are verified individually by LLM-as-judge to confirm no semantic change (acceptable variance from floating-point in the LLM stack).

### 7.5 Observability

- **Per-chunk lineage:** Provenance fields written to JSONL + Qdrant payload.
- **Sanitization rejection log:** `logs/sanitization_rejections_<timestamp>.jsonl` — chunk ID, guard name, rejection reason.
- **Graceful degradation log:** Endpoint name, unreachable duration, affected chunk count.
- **Fusion trace (opt-in):** `--log-fusion-trace` — per-leg scores, weights applied (including re-normalization on leg skip), fusion scores, reranker input/output.
- **Soak provenance:** Schema version, model versions, sanitization mode per judgment.

### 7.6 Failure-Mode Behavior (Endpoint Unreachable)

| Component | Endpoint | When unreachable | Behavior |
|---|---|---|---|
| **LLM Sanitizer** | GX10 `10.0.10.239:8000` | Fall back to heuristic sanitization | Sentinel per chunk; build warning; no hard fail |
| **Text Embedder** | omlx `10.0.10.246:8000` | Pipeline halts | Hard fail; operator intervention |
| **ColPali** | omlx `10.0.10.246:8000` | Visual leg skipped; weights re-normalized | Log per query; text-only retrieval proceeds |
| **ModernBERT** | omlx `10.0.10.246:8000` | Reranker skipped | Log per query; top-K from fusion returned directly |

---

## 8. Quality Gates (V3.0)

### 8.1 Universal Invariants (Unchanged from v2.X)

`chunk_type` present, `bbox` in [0,1000], non-empty text content, `modality` present, QA-CHECK-01 through QA-CHECK-05.

### 8.2 V3.0-Specific Gates

| Gate | Phase | Threshold | Measurement |
|---|---|---|---|
| UIR semantic-identity | A | Stable chunk-identity key match; content/chunk_type match; flags additive; top-5 doc IDs unchanged | Key-based matching (§3.2); regression fixture |
| `partial_code` flag coverage | A | 100% of cross-page code splits flagged | Corpus audit script |
| LLM guard-stack compliance | B | Zero chunks accepted with numeric mismatch, code-span change, or order swap | Guard unit tests (B7) + corpus-wide acceptance |
| LLM dominance (aggregate + degradation) | B | Format ≥ 2× soak σ above heuristic; zero chunk PASS→FAIL; ≤2% degradation rate vs heuristic | `both-and-diff` comparison with defined diff predicate (§3.3); σ measured on heuristic baseline |
| Format (soak judge) | B | ≥95% corpus-wide with `--sanitize-mode=llm`; no regression vs `--sanitize-mode=heuristic` | `scripts/synthetic_soak.py` |
| Phase C-spike PASS (both A and B) | Pre-A | Page recovery ≥60% on text-missed queries; reranker discrimination ≥60% | Spike report (numpy MaxSim + simulated rerank) |
| Phase C entry criterion (storage) | Pre-C | ≤500GB indexed at C12 target (1000 docs) | Storage projection from spike measurements |
| Recall@1 chunk (deficit docs) | C | ≥80% (from 67.8%) | Retrieval regression on 5 deficit docs |
| Recall@5 doc (deficit docs) | C | ≥98.6% maintained | Retrieval regression |
| Text-doc metrics (29 docs) | C | No regression on any axis | Full-corpus synthetic soak |
| Reranker discrimination (visual pages) | C | Top-1 selection rate ≥60% on visually-retrieved pages | Phase C soak measurement |
| Modality-aware soak axes | D | FORM/TABLE/DIAGRAM rubrics operational | `scripts/synthetic_soak.py --modality-aware` |
| Smoke matrix | All | 34/34 GATE_PASS + UNIVERSAL_PASS | `scripts/smoke_multiprofile.sh` |
| Blind-test (Greenhouse + ≥8 additional) | All | GATE_PASS + UNIVERSAL_PASS for all 8 holdout docs | Included in smoke matrix |
| Build reproducibility | B+ | Chunk-level identical-hash ratio ≥99.5%; zero disagreeing chunks have semantic change | CI test on 3-doc subset; judge verification on disagreements |

### 8.3 Carry-Forward from v2.16

| Limitation | Rationale |
|---|---|
| 1.4% Recall@5 doc residual | Likely judge edge cases; not a structural defect |
| ~5% Format residual after LLM sanitization | Acceptable ceiling; 100% Format is asymptotically unreachable without human review |
| Magazine image quality (composite layouts) | Rendered-region-crop deferred per v2.11 §3e; visual retrieval partially mitigates |
| EPUB engine | Deferred to post-v3.0; explicit acceptance criteria TBD in a future cycle plan |
| Cross-page table spanning | Tables spanning pages are harder than code spanning pages (column-alignment recovery); Phase A addresses code/paragraph cross-page splits; table spanning requires additional design work deferred to v3.1+ |

---

## 9. Appendix: Audit Trace from Draft 0.2 Review

### 9.1 Review Summary

Draft 0.2 received a follow-up structured audit that confirmed 5 findings correctly resolved, identified 6 partially-fixed gaps, and surfaced 4 new issues. The review classified items as: 5 confirmed-resolved, 6 partially-fixed residual gaps, 4 newly introduced/surfaced issues. Overall judgment: greenlight with 2 remaining conditions (both now addressed in Draft 0.3).

### 9.2 Changes in Draft 0.3 (By Review Finding)

| Review Finding | Draft 0.2 Issue | Draft 0.3 Resolution |
|---|---|---|
| **B 1:** "Strictly equal-or-better per chunk" unattainable | LLM dominance criterion guaranteed heuristics never deprecated | Replaced with aggregate (Format ≥ 2× soak σ above heuristic) + degradation cap (≤2% degrade) + no chunk PASS→FAIL + two-soak confirmation (§3.3). Diff predicate: Levenshtein >5% (excl. whitespace + Unicode norm) (§3.3). |
| **B 2:** `both-and-diff` "disagreement" undefined | No diff predicate; 5% threshold unfalsifiable | Defined: token-level Levenshtein >5% of shorter output, excl. whitespace and NFC/NFD differences (§3.3). |
| **B 3:** Semantic-identity gate assumes 1:1 chunk mapping | No policy for split/merge cases | Stable identity key: `(doc_id, page_number, content_hash_prefix)` with Unicode/whitespace/line-ending normalization before hashing. Split/merge matching policy defined (§3.2). |
| **B 4:** PROSE visual_weight=0 wastes visual infra | Visual retrieval deployed but disabled for 29/34 docs | PROSE visual_weight → 0.1 (non-zero floor, tiebreaker effect on embedded figures). Known limitation: per-document weights miss per-query visual needs; future v3.2+ work noted (§3.4 #5). |
| **B 5:** Join policy cost unmeasured | No measurement of reranker discrimination on visual hits | C-spike PASS condition B: reranker top-1 selection rate on visually-retrieved pages must ≥60%. Measured pre-Phase C. If FAIL, region-level must be in Phase C scope (§4.2, §6.2). |
| **B 6:** Determinism hash-tolerance undefined | "≥99.5%" had no units | Defined: chunk-level identical-hash ratio. Zero disagreeing chunks may have semantic change (judge-verified). Heuristic + off modes byte-stable (§7.4). |
| **B 7:** Content-pinning cache as determinism solution | R10 mitigation was "document variance" | Content-pinning cache: 4-tuple key `(content_hash, context_hash, model_id, prompt_version) → sanitized_output`, context-aware to prevent stale hits. Deterministic even without deterministic LLM. Also speeds rebuilds on cache hit (§7.4). |
| **B 8:** Fusion re-normalization on leg skip not specified | ColPali unreachable → unnormalized scores break downstream thresholds | Leg-skip re-normalization: remaining weights scaled to preserve relative proportions. Verified in CI (§3.4 #6). |
| **C 1:** `UIRChunk` provenance ambiguity | `content` vs `content_original` vs `content_sanitized` roles undocumented | Dataclass docstring defines: `content` = always authoritative; `content_original` populated ONLY when sanitization changed content; `sanitization_status` = single source of truth (§3.2). |
| **C 2:** `ConfidenceBreakdown` non-Optional fields | `layout_confidence` and `text_extraction_confidence` required but not always applicable | All fields now Optional. Sentinel: `None` = "not applicable"; `-1.0` = "applicable but unavailable" (§3.2). |
| **C 3:** Partial-release policy undefined | Could 3.0.0 ship alone? | 3.0.0 held until 3.1.0 ready (UIR overhead without LLM-quality payoff = regression). 3.2.0 ships independently (§4.3). |
| **C 4:** Holdout policy under-scoped | 2 additional blind-test docs | Minimum 8: 2 per rubric (PROSE, FORM, TABLE, DIAGRAM). Rubric-to-holdout mapping documented in QUALITY_GATES.md before Phase D (§3.5 #5). |
| **C 5:** LLM sanitization test strategy absent | — | Phase B B7: property-based tests per guard, mocked-endpoint integration tests, golden-output regression per prompt version (§4). Risk R14 added (§2.3). |
| **C 6:** Corpus-size target for Phase C undefined | R4 said "define before Phase C" | C12: ≤1000 docs, ≤300,000 pages. Phase C entry criterion: verify ≤500GB indexed at target (§2.1, §3.4). |
| **C 7:** R12 calendar impact not drawn | Phase B overrun acknowledged but downstream effects not modeled | Calendar diagram + recommendation: front-load C-spike; start C implementation at day 12 of B's window, saving 10 days vs sequential (§4.1). |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---|---|---|---|
| 0.1 | 2026-05-25 | Claude Code (Opus 4.7) | Initial synthesis of Draft A + Draft B + governance audit. |
| 0.2 | 2026-05-25 | Claude Code (Opus 4.7) | Incorporate 17-point external architecture review. Restructured layer model; strengthened UIR contract; semantic-identity gate; heuristics retained; Phase C-spike; granular schema versioning; expanded risk register; determinism/observability/failure-mode policies; fork-back triggers; regret-risk register; glossary. |
| 0.3 | 2026-05-25 | Claude Code (Opus 4.7) | Incorporate follow-up architecture review on Draft 0.2. Replaced LLM dominance criterion with aggregate + degradation-rate; defined chunk-identity key + diff predicate; visual_weight floor (0.1); C-spike PASS B (reranker discrimination); content-pinning cache; fusion re-normalization; UIRChunk provenance contract; ConfidenceBreakdown→Optional; partial-release policy; R12 calendar overlay; 8-doc holdout; LLM test strategy as B7; corpus-size target C12; risk R14. ALL body sections edited — not just appendix. |

---

**END OF ARCHITECTURE_V3_DRAFT_0.3.md**