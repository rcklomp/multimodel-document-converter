# MM-Converter V3.0: Architectural Charter — Draft 0.2

**Status:** DRAFT for review (incorporates external architecture review feedback on Draft 0.1)
**Target release:** v3.0 (next major schema version after v2.X)
**Changes since 0.1:** Restructured layer model; strengthened UIR contract with provenance, flow-aware locators, and per-source confidence; replaced byte-identical gate with semantic-identity gate; Phase B retains heuristic stack alongside LLM (dual-write); added Phase C pre-flight spike; granular schema versioning (3.0.0/3.1.0/3.2.0); expanded risk register (5→13 entries); added determinism, observability, and failure-mode policies; added glossary; profile-conditional fusion weights; fork-back trigger for Docling vs VLM-native decision.
**Governance lineage:** `AGENTS.md` (Level-0 invariants) → `docs/DECISIONS.md` (canonical decisions) → `docs/ARCHITECTURE.md` (v2.X baseline) → this document (v3.0 target)
**Parent:** `docs/ARCHITECTURE.md` (v2.X production architecture — this document supersedes for v3.0)
**Review baseline:** Draft 0.1 + external architecture review (2026-05-25, 17-point structured audit)

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
| **GX10** | LAN-local Apple Silicon inference endpoint at `10.0.10.239:8000`; hosts FP8 LLMs for sanitization and judge |
| **Soak** | Synthetic retrieval quality evaluation — 518 queries × top-5 results, scored by LLM-as-judge across 5 axes |
| **Strict gate** | Per-document deterministic acceptance check (`qa_full_conversion.py --source-pdf`); produces PASS/WARN/FAIL |
| **Smoke matrix** | Multi-profile gate run across all 34 docs + blind-test document (Greenhouse); requires GATE_PASS + UNIVERSAL_PASS |

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
| **Chunk sanitization** | Regex + POS + per-profile rules | LLM-native semantic polish pass (local GX10 FP8 endpoint, $0 cost) with heuristic dual-write for one cycle |
| **Retrieval** | Flat-text embeddings (dense + sparse + RRF) | Hybrid text + late-interaction visual embeddings (ColPali/MaxSim), profile-conditional fusion weights |
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
| Q8 | **Repeatable builds** | Fresh checkout → full corpus rebuild | < 24h wall time; chunk content deterministic when `--sanitize-mode=heuristic` or `off`; LLM-sanitized content matches within hash-tolerance when seed/temperature=0 |
| Q9 | **Graceful degradation** | GX10 endpoint unreachable | Pipeline continues with heuristics (dual-write retained); `[SANITIZE_SKIPPED: endpoint unreachable]` sentinel logged per chunk |
| Q10 | **Observability** | Visual retrieval miss root-caused | Per-query fusion weights logged; per-chunk lineage traceable to extraction engine + sanitizer model version + prompt version |

### 2.3 Key Technical Risks

| # | Risk | Severity | Probability | Mitigation |
|---|---|---|---|---|
| R1 | LLM sanitization hallucinates or alters factual content | **Critical** | Medium | Multi-layer guard: numeric/entity preservation check, code-span hashing, order-preservation check, edit-distance ceiling (§3.3). Dual-write with heuristics retained for one full cycle (§4 Phase B). |
| R2 | LLM sanitization deferral becomes permanent one-way door | **High** | High | Heuristics retained alongside LLM (dual-write) for v3.0–v3.1; explicit fork-back trigger (§6). VLM-native parsing evaluated post-v3.0 if Format < 92% after two soak iterations. |
| R3 | ColPali multi-vector storage requires Qdrant version upgrade | **High** | Medium | Phase C-spike probes Qdrant MaxSim support before any Phase C implementation (§4.2). Budget side-collection rebuild as fallback. |
| R4 | Visual storage 10–100× at scale (5GB for 34 docs → 150GB at 1000 docs raw, ~400GB with HNSW overhead) | **High** | Medium | Storage budget must scale linearly with corpus; define corpus size target and memory ceiling before Phase C full build. MaxSim latency scales O(query_tokens × doc_patches) — measure and cap. |
| R5 | Visual retrieval degrades text-heavy doc recall via fusion noise | **High** | Medium | Profile-conditional fusion weights (§3.4): visual_weight=0 for PROSE, 0.4 for DIAGRAM/FORM. Sweep weights on deficit subset before declaring default. |
| R6 | ColPali/omlx memory conflict with existing Qwen3-Embedding-8B | **Medium** | Medium | Phase C-spike validates co-residency on omlx server before Phase C implementation. |
| R7 | Edit-distance guard misses semantic corruption under token budget | **High** | Medium | Numeric/entity extraction gate, code-span hashing, order-preservation check layered on top of token-count ceiling (§3.3 guard stack). |
| R8 | MinerU AGPL-3.0 license incompatible with project distribution | **Medium** | Low (deferred) | Containerize as isolated service if adopted; keep Docling as fallback extraction path. |
| R9 | VLM-native parsing changes chunk shape → invalidates regression fixtures | **High** | Low (deferred) | Separate cycle after UIR stabilizes; maintain semantic-identity gate. |
| R10 | LLM sanitization non-deterministic → builds not reproducible | **High** | High | Determinism policy (§7.4): temperature=0, fixed seed, deterministic sampling for sanitization. Build-reproducibility test hashes corpus output. |
| R11 | Prompt injection via document content in sanitization path | **Medium** | Low | Content treated as data, not instructions; prompt structured with clear content/prompt boundary delimiters. Input-length cap per chunk. |
| R12 | Solo-dev cycle overrun on Phase B (prompt engineering unbounded) | **High** | High | Phase B estimated 18-22 days; accept as likely overrun. Spike prompt on 100-chunk sample early. Dual-write means heuristics catch LLM gaps. |
| R13 | Phase A semantic-identity gate misses a material regression | **Medium** | Low | Gate checks: identical chunk content (modulo whitespace), identical chunk_type, structural flags strictly additive, top-5 doc IDs unchanged in regression fixture. |

---

## 3. V3.0 Target Architecture

### 3.1 Architecture Overview

V3.0 is organized as two peer subsystems (Ingestion Pipeline, Retrieval Backend) with a sidecar Quality & Evaluation track. This replaces Draft 0.1's four-layer stack, which incorrectly implied strict bottom-up dependencies between ingestion and retrieval.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           V3.0 ARCHITECTURE                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐  │
│  │     INGESTION PIPELINE          │    │      RETRIEVAL BACKEND           │  │
│  │                                 │    │                                 │  │
│  │  Stage 1: True UIR              │    │  ┌───────────────────────────┐  │  │
│  │  ┌───────────────────────────┐  │    │  │ Text Retrieval (retained) │  │  │
│  │  │ Format-agnostic           │  │    │  │ ├─ dense : omlx Qwen3     │  │  │
│  │  │ ConversionPlan →          │  │    │  │ └─ sparse: BM25           │  │  │
│  │  │ UniversalDocument         │  │    │  └───────────────────────────┘  │  │
│  │  │ Decoupled ElementProcessor│  │    │                                 │  │
│  │  └───────────────────────────┘  │    │  ┌───────────────────────────┐  │  │
│  │              │                  │    │  │ Visual Retrieval (NEW)     │  │  │
│  │              ▼                  │    │  │ ├─ ColPali/ColQwen2.5     │  │  │
│  │  Stage 2: LLM Sanitization      │    │  │ ├─ Qdrant multi-vector    │  │  │
│  │  ┌───────────────────────────┐  │    │  │ └─ MaxSim scoring          │  │  │
│  │  │ GX10 FP8 endpoint         │  │    │  └───────────────────────────┘  │  │
│  │  │ Multi-layer guard stack   │  │    │              │                  │  │
│  │  │ Heuristic dual-write      │  │    │              ▼                  │  │
│  │  └───────────────────────────┘  │    │  ┌───────────────────────────┐  │  │
│  │                                 │    │  │ Fusion + Rerank            │  │  │
│  └─────────────────────────────────┘    │  │ ├─ Profile-conditional RRF │  │  │
│                                         │  │ └─ ModernBERT rerank       │  │  │
│  ┌─────────────────────────────────┐    │  └───────────────────────────┘  │  │
│  │  EXTRACTION ENGINES (pluggable) │    │                                 │  │
│  │  PDF (Docling) | EPUB | Future  │    └─────────────────────────────────┘  │
│  └─────────────────────────────────┘                                         │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │           QUALITY & EVALUATION (Sidecar — cross-cuts both subsystems) │   │
│  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────┐  │   │
│  │  │ Modality-Aware Gates│  │ Synthetic Soak      │  │ Strict Gate   │  │   │
│  │  │ PROSE | FORM | TABLE│  │ + Visual Relevance  │  │ FORM/TABLE/   │  │   │
│  │  │ DIAGRAM rubrics     │  │ + Spatial Fidelity  │  │ DIAGRAM lanes │  │   │
│  │  └─────────────────────┘  └─────────────────────┘  └──────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
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
    """Per-source confidence scores. Collapsing to one float loses signal."""
    layout_confidence: float       # Docling layout model confidence
    text_extraction_confidence: float  # Native text extraction confidence
    ocr_confidence: Optional[float] = None  # Only if OCR was applied
    classification_confidence: Optional[float] = None  # Element type classification

@dataclass
class UIRChunk:
    """Emitted by ElementProcessor; consumed by chunker + sanitizer."""
    modality: Modality
    content: str
    locator: Locator
    confidence: ConfidenceBreakdown
    extraction_method: str                # "docling_direct" | "ocr_tesseract" | "ocr_doctr" | "vlm_enrichment"
    extraction_engine_version: str        # e.g., "docling-2.86.0"
    structural_flags: Dict[str, bool]     # "partial_code", "cross_page_split", "orphan_label", etc.
    source_element_ids: List[str]         # Traceability back to extraction engine elements
    asset_ref: Optional[str] = None       # Path to extracted image/asset, if IMAGE modality
    lang: Optional[str] = None            # ISO 639-1 language code (detected or from ConversionPlan)
    reading_order: Optional[int] = None   # Monotonic logical position within page/document

    # Provenance fields (populated after sanitization, if applied):
    content_original: Optional[str] = None              # Pre-sanitization content
    content_sanitized: Optional[str] = None             # Post-sanitization content (if different)
    sanitizer_model_id: Optional[str] = None            # e.g., "RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic"
    sanitizer_prompt_version: Optional[str] = None      # Git-hash of prompt template
    sanitization_status: Optional[str] = None           # "accepted" | "rejected:edit_ratio" | "rejected:numeric_mismatch" | "skipped:endpoint_unreachable" | "not_applied"

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

**Key contract improvements from Draft 0.1:**
- **`Locator`** replaces bare `bbox` — supports reflowable formats (EPUB, HTML) via `flow_offset`/`dom_path`
- **`ConfidenceBreakdown`** prevents lossy collapse of multi-source confidence into one scalar
- **`reading_order`** — explicit monotonic sequence position, not inferred from emission order
- **`lang`** — language routing for sanitization prompt selection
- **`parent_element_id` / `parent_heading`** — hierarchical context for table cells, captions, list items
- **Provenance block** (`content_original`, `sanitizer_model_id`, `sanitizer_prompt_version`, `sanitization_status`) — enables reproducible builds and regression debugging
- **`extraction_engine_version`** — which engine version produced this element
- **`render_dpi`** in `ConversionPlan` — Phase C needs deterministic render resolution for ColPali

**Acceptance gate (Phase A): Semantic-identity gate** — replaces Draft 0.1's "byte-for-byte identical" (unachievable once structural flags change):

1. **Identical chunk content** (modulo trailing whitespace) for all chunks that existed in v2.16.
2. **Identical `chunk_type`** for all chunks.
3. **Structural flags strictly additive** — no v2.16 flag goes missing; new flags (`partial_code`, etc.) may appear.
4. **Retrieval invariant** — top-5 doc IDs unchanged for every query in the v2.16 regression fixture.

### 3.3 Ingestion Pipeline Stage 2: LLM-Native Chunk Sanitization

**Directly addresses:** Limit 2 (Heuristic Patching Ceiling)
**Governance anchor:** `docs/DECISIONS.md` — "Combined ceiling: UIR refactor + LLM cleanup together" (v2.10 chunker-quality ceiling, path 4: "LLM-clean every chunk on ingestion" estimated at $30/rebuild on Dashscope, now $0 via local GX10)

**Design:**

After Stage 1 ensures all chunks flow through the UIR, Stage 2 adds a **semantic polish pass** using the existing local LLM endpoint:

1. **Execution:** Raw extraction chunks are piped through `RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic` at `10.0.10.239:8000` (GX10 endpoint, already operational per `docs/DECISIONS.md` v2.14).

2. **Prompt contract:** The LLM receives the raw chunk content + surrounding context (previous/next chunk snippets, page breadcrumb, detected language) and is instructed to reconstruct the text into clean, well-formed Markdown:
   - Heal cross-page sentence and code-block splits
   - Normalize whitespace and punctuation
   - Remove extraction artifacts (drop-cap orphans, picture classification labels, OCR noise)
   - Preserve ALL factual content — no summarization, no paraphrasing
   - Edit-distance budget enforced

3. **Sanitization mode flag:** `--sanitize-mode={off,llm,heuristic,both-and-diff}`
   - `off`: No sanitization — raw UIR chunks emitted (v2.16 equivalent, for regression baseline)
   - `llm`: LLM sanitization only — heuristics skipped (Phase B target)
   - `heuristic`: Existing heuristic stack only — LLM skipped (v2.16 behavior preserved)
   - `both-and-diff`: Both run, output compared; disagreement logged (validation mode for one full cycle)

4. **Heuristics retained — NOT ripped out.** Phase B does NOT remove `docling_postprocess.py` or the multimodal validation layers. The heuristic stack remains operational and is the fallback path. Deprecation requires a measured cycle where `both-and-diff` shows LLM strictly dominates on the soak set, not just aggregate Format scores. See §6 fork-back trigger.

5. **Cost:** $0 per rebuild (local GX10 FP8 endpoint). No cloud API calls on the sanitization path.

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

**Acceptance gate (Phase B):** Synthetic soak Format scores cross 95% corpus-wide with `--sanitize-mode=llm`, with no regression on Recall, Relevance, or Faithfulness axes vs `--sanitize-mode=heuristic`. `both-and-diff` comparison shows LLM output is strictly equal-or-better per chunk (no chunk where LLM degrades and heuristic is correct).

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
   - Alternative (one score per chunk) would require per-chunk visual crops, increasing storage cost ~10×. Documented as a deliberate trade-off; reevaluate if reranker discrimination proves insufficient.

5. **Profile-Conditional Fusion Weights:**
   Draft 0.1 proposed equal 3-way RRF weights (dense=1, sparse=1, visual=1). This implicitly weights text at 2/3 globally, including on text-heavy docs where visual is noise. Replaced with:

   | Profile class | Dense weight | Sparse weight | Visual weight | Rationale |
   |---|---|---|---|---|
   | PROSE (default) | 1.0 | 1.0 | **0.0** | Visual adds noise on text-heavy docs; 29/34 docs |
   | DIAGRAM | 1.0 | 1.0 | **0.4** | Visual is primary signal for diagrams/schematics |
   | FORM | 1.0 | 1.0 | **0.4** | Visual captures form layout; text captures field values |
   | TABLE | 1.0 | 0.5 | **0.3** | Sparse still useful for exact numeric match; visual for structure |

   Profile is determined by `ConversionPlan.profile_type` → modality classifier. Defaults are empirical starting points; final weights determined by sweep on the deficit-doc subset during Phase C.

**End-to-end retrieval flow (V3.0):**

```
query
  ├─ dense  : omlx Qwen3-Embedding-8B-mxfp8 (4096-dim)     ← retained (v2.13)
  ├─ sparse : BM25                                         ← retained (v2.12)
  └─ visual : ColPali/ColQwen2.5 → MaxSim (page scores)    ← NEW (v3.0)
           → page scores propagated to child chunks (join policy)
  → RRF fusion (k=60, profile-conditional weights)
  → ModernBERT rerank (top-25 → top-5)                     ← retained (v2.12)
  → top-5 return
```

**Reranker behavior on visual hits:** ModernBERT is text-only. When a chunk enters the top-25 primarily due to its visual score, the reranker evaluates `(query, chunk_content)`. The visual signal has already done its job (getting the page into the candidate set). The reranker then selects the best text chunks on that page. This is intentional — the visual leg handles page discovery; the text leg + reranker handle chunk selection.

**Resource budget (34-doc corpus):**
- Patch-vector storage: ~128 dimensions × 1030 patches × ~10,000 pages = ~5 GB raw; ~15-20 GB with HNSW indexing overhead.
- Visual embedding latency: target <1s/page at omlx server inference speed.
- Storage scaling at 1000 docs (~300,000 pages): ~150 GB raw, ~400 GB indexed. Must define corpus size target and memory ceiling before Phase C full build.
- MaxSim latency: O(query_tokens × doc_patches) for exhaustive search; Qdrant approximate nearest neighbor search caps this at a configurable ef_search.

**Phase C pre-flight spike (MUST RUN before Phase A implementation):** See §4.2 — minimum experiment proving ColPali viability on the single highest-deficit doc.

**Acceptance gate (Phase C):** The -12pp Recall@1 deficit on the 5 complex engineering documents is closed. Target: Recall@1 chunk ≥80% (from 67.8%), Recall@5 doc ≥98.6% maintained. Text-heavy docs (29/34) show no regression.

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
   - **Visual Relevance:** Did the retrieved visual patch match the query intent? Scored 0/1/2 per retrieval by LLM-as-judge with access to the page render.
   - **Spatial Fidelity:** Did the patch encoding preserve layout information? Measured as alignment between MaxSim top-patch heatmap and ground-truth relevant page regions (where available).

4. **Strict Gate Extension:**
   - `FORM_AUDIT_PASS` (existing v2.8) extended to `TABLE_AUDIT_PASS` and `DIAGRAM_AUDIT_PASS`.
   - Universal invariants (bbox range, modality present, non-empty content) apply across all classes — no waivers.

5. **Holdout policy:** 34-doc corpus is small. Rubric calibration on all 34 risks overfitting. Policy: the Greenhouse blind-test document is the minimum holdout; before v3.0 final tag, add ≥2 additional blind-test documents from categories not yet represented in the corpus.

**Acceptance gate (Phase D):** `CarOK_voorraadtelling` Format score no longer penalized for non-prose content shape. Modality breakdown visible in soak reports. No false-negative quality failures on structured data.

---

## 4. Implementation & Phasing Strategy

V3.0 is sequenced to respect solo-dev 12-day convergence cycles. Critically, **visual retrieval (Phase C) does not depend on UIR or LLM sanitization** — ColPali embeds rendered page images directly. The Phase C pre-flight spike runs in parallel with Phase A. The critical path is A → (B ∥ C-implementation) → D.

### 4.1 Phase Dependency Diagram

```
Phase C-spike (pre-light)
    │
    ▼
Phase A: UIR Foundation ──────────────────────┐
    │                                          │
    ├──→ Phase B: LLM Sanitization ────────────┤
    │    (heuristics retained as dual-write)    │
    │                                          │
    └──→ Phase C: Visual Retrieval ────────────┤
         (gated on C-spike pass)               │
                                               ▼
                                    Phase D: Modality Gates
```

- **C-spike** runs before Phase A implementation — it validates the v3.0 visual retrieval thesis independently.
- **Phase D1** (rubric design) runs in parallel with Phase A — modality tagging exists in v2.16 already.
- **B and C** can overlap — they share infrastructure (GX10, omlx) but have independent data dependencies.

### 4.2 Phase C Pre-Flight Spike (MANDATORY — before Phase A code)

**Minimum experiment to prove or disprove the ColPali approach:**

1. Pick the single highest-deficit doc — `ATZ_Elektronik_German`.
2. Render all pages at 200 DPI (target ColPali resolution).
3. Embed pages with off-the-shelf ColPali on a workstation. **No omlx deployment. No Qdrant integration. No MaxSim in production code.**
4. Take 20 queries from the v2.16 retrieval regression fixture targeting this doc (or hand-craft if fixture coverage is thin).
5. Embed each query with ColPali. Compute MaxSim scores against page-vector matrices in raw numpy. Rank pages.
6. For each query, record: text-retrieval top-1 page (v2.16), visual-retrieval top-1 page (this experiment), gold page.
7. **PASS condition:** Visual retrieval recovers the correct page on ≥60% of queries where v2.16 text retrieval failed, without harming queries where text retrieval was correct.
8. Also verify: ColPali model fits alongside Qwen3-Embedding-8B on the omlx server; end-to-end latency <1s/page.

**Time budget:** 2–3 days. **Hardware:** workstation. **No production infrastructure touched.**

**Outcome:** If PASS → Phase C implementation proceeds as designed. If FAIL → Phase C as designed is dead; redirect to VLM-native parsing evaluation or a different visual model. This decision is documented as a fork-back trigger (§6).

### Phase A: UIR Foundation (Cycle 3.0)

**Scope:** Ingestion Pipeline Stage 1 only. No ML changes. No behavioral changes beyond structural flag emission.

| Task | Description | Acceptance |
|---|---|---|
| A1 | Elevate `PdfConversionPlan` to parent `ConversionPlan` | All existing tests pass without modification |
| A2 | Refactor extraction engines to output `UniversalDocument` | Semantic-identity gate: identical chunk content, chunk_type; structural flags additive; top-5 doc IDs unchanged in regression fixture |
| A3 | Decouple chunker from `DoclingDocument` → operate on UIR | `partial_code` flags emit correctly; cross-page splits attributed to correct pages |
| A4 | Rip out duplicate Docling option construction sites | `test_no_raw_converter_invocation_outside_adapter` expanded to cover v3.0 paths |
| A5 | Corpus-wide rebuild + strict gate | 34/34 PASS (or documented deferrals matching v2.16 baseline) |
| A6 | Schema version: `2.7.0` → `3.0.0` | All output carries `schema_version: "3.0.0"`; consumer migration doc added to `docs/` |

**Risk:** Lowest-risk phase. Pure refactoring. Rollback is `git revert`.

**Schema version — granular approach (replaces Draft 0.1's single 3.0.0 bump):**
- `3.0.0` — UIR introduction. Chunk shape unchanged from v2.16 (semantic-identity gate). Structural flags now populated. This is the Phase A artifact.
- `3.1.0` — Sanitization provenance. Chunk content may differ from v2.16 (LLM-sanitized). `UIRChunk` provenance fields populated. Phase B artifact.
- `3.2.0` — Visual vectors. Multi-vector Qdrant collection alongside text collections. Phase C artifact.

This lets downstream consumers adopt incrementally and isolates the blame surface when something breaks.

### Phase B: LLM Sanitization (Cycle 3.1)

**Scope:** Ingestion Pipeline Stage 2. Requires Phase A complete.

| Task | Description | Acceptance |
|---|---|---|
| B1 | Wire GX10 FP8 endpoint into UIR output phase | Sanitization harness operational; all `--sanitize-mode` flags functional |
| B2 | Design + validate sanitization prompt | Prompt passes negative tests (no hallucination, no summarization, no content loss, preserves all numerics/entities/ordered markers) |
| B3 | Implement multi-layer guard stack | All 6 guards operational (§3.3); each guard has positive + negative regression tests |
| B4 | Dual-write LLM + heuristics; compare output | `--sanitize-mode=both-and-diff` functional; disagreement log emitted per chunk |
| B5 | Corpus-wide rebuild + synthetic soak (llm vs heuristic) | Format ≥95% corpus-wide with LLM; no regression on other axes; llm output strictly equal-or-better per chunk vs heuristic |
| B6 | Schema version: `3.0.0` → `3.1.0` | Output carries `schema_version: "3.1.0"`; `UIRChunk` provenance fields populated |

**Risk:** Medium-High. LLM may introduce subtle content alterations. Multi-layer guard stack mitigates. Heuristics retained (dual-write) means fallback path is live. Phase B is the most-likely-to-overrun phase — estimate 18–22 days against 12-day cycle due to prompt engineering iteration and guard tuning.

**⚠ Heuristics are NOT removed in Phase B.** `docling_postprocess.py`, `corruption_interceptor.py`, and the four multimodal validation layers remain operational. Deprecation requires a full cycle where `both-and-diff` shows LLM strictly dominates on the soak set, not just on aggregate Format scores. The `--sanitize-mode=heuristic` flag preserves full v2.16 behavior for regression comparison.

### Phase C: Visual Retrieval (Cycle 3.2)

**Scope:** Retrieval Backend visual leg. Gated on C-spike PASS (§4.2). Does NOT require Phase B complete (visual retrieval embeds rendered pages, not text chunks).

| Task | Description | Acceptance |
|---|---|---|
| C1 | Probe Qdrant multi-vector support | Confirm MaxSim compatibility on current Qdrant version; if upgrade needed, scope migration |
| C2 | Deploy ColPali/ColQwen2.5 on omlx server | Visual embedding endpoint operational; co-residency with Qwen3-Embedding-8B confirmed; latency <1s/page |
| C3 | Build parallel visual index (single doc → deficit subset → full corpus) | Incremental rollout; each stage validates no regression before expanding |
| C4 | Implement MaxSim retrieval + profile-conditional fusion | `retrieve_hybrid_visual()` functional; weights per profile class; sweep confirms weights on deficit subset |
| C5 | Implement granularity join policy (page → chunk score propagation) | Verified: reranker correctly discriminates among chunks on visually-retrieved pages |
| C6 | Synthetic soak on complex-doc subset + full corpus | Recall@1 chunk ≥80% on deficit docs; text-doc metrics maintained; no regression on 29 text-heavy docs |
| C7 | Schema version: `3.1.0` → `3.2.0` | Visual collection built; Qdrant metadata carries `schema_version: "3.2.0"` |

**Risk:** Highest-risk phase. Mitigated by: (a) C-spike validates approach before any code; (b) incremental rollout (single doc → subset → full); (c) visual collection is independent of text collections — can be dropped without affecting text retrieval; (d) profile-conditional weights prevent visual noise on text-heavy docs.

### Phase D: Modality-Aware Gates (Cycle 3.3)

**Scope:** Quality & Evaluation sidecar. D1 can start in parallel with Phase A. D2–D4 require upstream phases.

| Task | Description | Acceptance |
|---|---|---|
| D1 | Implement profile-specific judge rubrics | FORM, TABLE, DIAGRAM rubrics operational in synthetic soak; D1 runs in parallel with Phase A |
| D2 | Extend strict gate for modality classes | `FORM_AUDIT_PASS`, `TABLE_AUDIT_PASS`, `DIAGRAM_AUDIT_PASS` in `qa_full_conversion.py` |
| D3 | Add visual retrieval quality axes | Visual Relevance, Spatial Fidelity axes in soak reports |
| D4 | Full-corpus v3.0 acceptance run | Smoke matrix: all rows GATE_PASS + UNIVERSAL_PASS; blind-test document (Greenhouse) + ≥2 additional holdout docs included |

**Risk:** Low. Extends existing v2.8 form-audit precedent. No pipeline changes.

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
├── engines/                       # Format-specific extraction
│   ├── base.py                    # FormatEngine ABC (retained)
│   ├── pdf_engine.py              # [REFACTORED] Consumes shared adapter; outputs UniversalDocument
│   ├── docling_adapter.py         # [REFACTORED] Single Docling construction + invocation site
│   ├── docling_postprocess.py     # [RETAINED] Heuristic post-processing; dual-write alongside LLM in Phase B
│   └── epub_engine.py             # [FUTURE — deferred to post-v3.0, with explicit acceptance criteria TBD]
│
├── sanitization/                  # [NEW] LLM-native chunk sanitization
│   ├── __init__.py
│   ├── llm_sanitizer.py           # GX10 FP8 endpoint client + prompt management
│   ├── edit_guard.py              # Multi-layer guard stack (edit-distance, numeric/entity, code-span, order-preservation)
│   ├── prompts.py                 # Sanitization prompt templates (versioned, language-aware)
│   └── graceful_degradation.py    # Endpoint-unreachable fallback policy
│
├── retrieval/                     # [EXPANDED] Visual retrieval
│   ├── pipeline.py                # [EXTENDED] retrieve_hybrid_visual()
│   ├── visual_embedder.py         # [NEW] ColPali/ColQwen2.5 client (omlx)
│   ├── maxsim.py                  # [NEW] MaxSim scoring + page→chunk granularity join policy
│   ├── fusion.py                  # [EXTENDED] Profile-conditional RRF fusion weights
│   └── config.py                  # [EXTENDED] Visual collection defaults + weight profiles
│
├── vision/                        # VLM integration (retained)
├── validators/                    # QA checks (retained; heuristics live alongside LLM)
│   ├── corruption_interceptor.py  # [RETAINED] — dual-write; deprecation gated on both-and-diff outcome
│   ├── token_validator.py
│   └── quality_filter_tracker.py
│
├── ocr/                           # OCR cascade (retained for non-LLM path)
├── chunking/                      # Chunking helpers (retained; consumed via UIR)
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
    │                    └───────┬───────┴──────┬───────┘
    │                            │              │
    │                    ┌───────┴───────┐      │
    │                    │ Guard stack   │      │
    │                    │ (6-layer)     │      │
    │                    └───────┬───────┘      │
    │                            │              │
    │                    ┌───────┴───────┐      │
    │                    │ both-and-diff │      │
    │                    │ comparison    │      │
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
RRF fusion (k=60, profile-conditional weights)
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

| Decision | Phase | Rationale | Cross-Reference |
|---|---|---|---|
| **Docling retained** as default PDF extractor; LLM sanitization added post-extraction | A-B | VLM-native parsing (MinerU/GOT-OCR2.0) is higher-risk and license-constrained (AGPL-3.0). Docling + LLM achieves the same Format improvement with lower risk. | Draft A Layer 2; Draft B Pillar 2; `DECISIONS.md` v2.10 chunker-quality ceiling |
| **LLM sanitization uses local GX10 FP8, not cloud** | B | $0 cost vs $30/rebuild on Dashscope. Privacy-preserving. Already operational (v2.14). | `DECISIONS.md` v2.14 local-LLM stack |
| **ColPali via omlx, not cloud vision API** | C | $0 per-query cost. LAN-local, privacy-preserving. Consistent with v2.13 embedder swap philosophy. | `DECISIONS.md` v2.13 embedder swap |
| **Profile-conditional fusion weights, NOT equal 3-way** | C | Equal weights implicitly weight text at 2/3 globally — hurts text-heavy docs (noise) AND deficit docs (underweight). Profile-conditional weights prevent both. | v2.12 Phase 2 RRF precedent |
| **Semantic-identity gate (not byte-for-byte)** for Phase A | A | Byte-identical is unachievable once structural flags change. Semantic-identity is achievable and meaningful. | `AGENT-VAL-01` |
| **Multi-layer guard stack** for LLM sanitization | B | Token-count ceiling alone misses numeric swaps, code changes, order reordering, and semantic inversion. Defense-in-depth required. | `DECISIONS.md` "Heal-Over for Encoding Corruption" |
| **Heuristics retained alongside LLM (dual-write)** for v3.0–v3.1 | B | LLM is probabilistic; heuristics are deterministic. Dual-write provides fallback, regression comparison, and prevents one-way-door lock-in. Retained until both-and-diff shows LLM strictly dominates. | This document §Fork-Back Triggers |
| **Granular schema versioning** (3.0.0/3.1.0/3.2.0) | A/B/C | Single jump conflates three independent consumer breaks. Granular versions isolate blame surfaces and enable incremental adoption. | `AGENTS.md` C9 |
| **Determinism policy for LLM sanitization** | B | temperature=0, fixed seed, deterministic sampling. Build-reproducibility test hashes corpus output. Required for Q8 (repeatable builds). | §7.4 |
| **Phase C-spike before Phase A implementation** | Pre-A | Visual retrieval is independent of UIR. Spike proves/disproves ColPali viability before any v3.0 code is written. | §4.2 |
| **No gate weakening** | All | Per `DECISIONS.md` "No gate weakening to make a failing run pass." | `DECISIONS.md` (2026-05-09) |

### 6.2 Fork-Back Triggers

These are numerical, measurable conditions that, if met, require reopening a decision documented above:

| Decision | Trigger | Action |
|---|---|---|
| **Docling retained vs VLM-native** | After two Phase B soak iterations, Format < 92% corpus-wide with LLM sanitization AND heuristic output is also < 92% (proving deficit is extraction-side, not sanitization-side) | Reopen VLM-native parsing evaluation. Run GOT-OCR2.0 or MinerU on the 5 deficit docs; compare extraction quality. |
| **Heuristics retained vs removed** | `both-and-diff` comparison across full corpus shows LLM output is strictly equal-or-better on every chunk for two consecutive soak iterations | Deprecate heuristic stack; remove in following cycle. |
| **Visual retrieval weights** | Sweep on deficit subset during Phase C shows a weight configuration outperforms profile-conditional defaults by ≥5pp on any axis | Adopt sweep-optimal weights as v3.0 defaults. |
| **ColPali vs alternative visual model** | Phase C-spike FAIL (visual retrieval does not recover correct page on ≥60% of text-missed queries) | Redirect to VLM-native parsing or evaluate an alternative visual embedding model (e.g., ColQwen2.5 if not already tested, or VisRAG). |
| **omlx co-residency** | ColPali + Qwen3-Embedding-8B cannot both fit in omlx server memory | Evaluate dedicated visual embedding endpoint on secondary hardware; or use cloud vision API as fallback (violates Q6 but may be acceptable if gated on deficit docs only). |

### 6.3 Decisions We May Regret (and How We'll Know)

This section exists because the external review correctly identified that Draft 0.1's "VLM-native deferred to post-v3.0" framing presented a one-way door as reversible. The following decisions are the most likely to be reversed within 6 months of v3.0 shipping:

1. **Docling + LLM sanitization as the permanent extraction path.**
   - **Why we'd regret it:** If Docling's bbox quality on the deficit docs is the real bottleneck (not text content quality), LLM sanitization cannot fix spatial coordinates. Visual retrieval inherits the same coordinate noise.
   - **How we'll know:** Phase C-spike results. If visual retrieval gains are <5pp despite clean text, bbox quality is the limiter → VLM-native parsing needed.
   - **Monitoring signal:** Bbox-IoU between Docling output and a VLM-native reference parse on the 5 deficit docs.

2. **Page-level visual retrieval granularity (not region-level).**
   - **Why we'd regret it:** The join policy (§3.4) propagates page scores to all chunks on that page. On dense magazine or form pages with 10+ distinct visual regions, page-level scoring loses precision.
   - **How we'll know:** Reranker fails to discriminate effectively on CarOK/Earthship-class docs despite correct page retrieval.
   - **Monitoring signal:** Reranker top-1 selection rate on visually-retrieved pages (if <50% of visually-retrieved pages have their correct chunk in the reranker's top-1, granularity is too coarse).

3. **Heuristics retained for one cycle only.**
   - **Why we'd regret it:** LLM model updates or prompt revisions introduce new failure modes that heuristics would have caught deterministically. One cycle may not be enough to see the full failure-mode surface.
   - **How we'll know:** `both-and-diff` comparison shows disagreements in cycle 1. If >5% of chunks differ, extend dual-write to cycle 2.
   - **Monitoring signal:** `both-and-diff` disagreement rate per soak.

---

## 7. Migration, Rollback & Operations

### 7.1 Schema Version

V3.0 uses granular schema versioning (not a single 3.0.0 jump):

| Version | Phase | What changes | Consumer impact |
|---|---|---|---|
| `3.0.0` | A | UIR introduced; structural flags populated; chunk content unchanged from v2.16 | Downstream consumers reading `chunk_id` or `content` see no difference. Structural flags are additive. Schema migration reader for v2.X JSONL → v3.0.0 reader. |
| `3.1.0` | B | LLM sanitization may change chunk content; `UIRChunk` provenance fields populated (`content_original`, `sanitizer_model_id`, etc.) | Consumers must accept that chunk content may differ from v2.16. Provenance fields enable audit trail. |
| `3.2.0` | C | Qdrant multi-vector collection for visual retrieval alongside text collections | Text retrieval collections unchanged. New visual collection is additive. |

**Downstream breakage explicitly enumerated:**
- Cached text embeddings against v2.X `chunk_id`s become invalid when content changes (v3.1.0).
- Retrieval regression fixtures are version-pinned; cross-version comparison requires fixture migration.
- Any application joining on `chunk_id` across runs loses identity if IDs are positionally derived.
- Synthetic soak baselines reset at each schema version; cross-version metric comparison is directional only.
- Observability queries keyed on `schema_version` need migration when version changes.

### 7.2 Rollback Paths

| Phase | Rollback | Procedure |
|---|---|---|
| A (UIR) | `git revert` | Semantic-identity output → no data migration needed |
| B (LLM sanitize) | `--sanitize-mode=heuristic` | Restore v2.16 behavior exactly; heuristics are retained, not removed |
| C (Visual index) | Drop visual collection | Text retrieval stack entirely unaffected; visual leg is additive + independent |
| D (Gates) | Gate-only change | No data migration; gate thresholds revert independently |

### 7.3 Corpus Migration

Each phase that changes chunk output or retrieval collections requires:
1. Rebuild corpus from source PDFs (~8-12h wall time for 34 docs)
2. Re-ingest to Qdrant text collections (~1-2h)
3. Phase C adds: render pages at 200 DPI + ColPali embed (~5h for 34 docs; scales with corpus size)
4. Synthetic soak against new collections (~1-2h)
5. Updated retrieval regression fixture (`tests/fixtures/retrieval_regression_v3_X.json`)

### 7.4 Determinism Policy

LLM sanitization is non-deterministic by default. To satisfy Q8 (repeatable builds):

- **Sanitization:** `temperature=0`, fixed random seed, deterministic sampling mode on the GX10 endpoint. If the endpoint does not support deterministic decoding at temperature=0, document the expected variance and add a hash-tolerance gate (chunk content hashes must match ≥99.5% across consecutive builds).
- **Build-reproducibility test:** A CI test that runs `mmrag-v2 batch --sanitize-mode=llm` on a 3-doc subset, hashes the output JSONL, and asserts the hash is stable within tolerance.
- **Heuristic mode:** Fully deterministic (no LLM). `--sanitize-mode=heuristic` output is byte-stable.
- **Off mode:** Fully deterministic. `--sanitize-mode=off` output is byte-stable.

### 7.5 Observability

When a visual retrieval misses or a sanitization corrupts a chunk, the following must be queryable:

- **Per-chunk lineage:** `UIRChunk` provenance fields (`extraction_engine_version`, `sanitizer_model_id`, `sanitizer_prompt_version`, `sanitization_status`) are written to `ingestion.jsonl` and mirrored in Qdrant payload.
- **Sanitization rejection log:** All chunks where any guard fired are logged with chunk ID, guard name, and rejection reason to `logs/sanitization_rejections_<timestamp>.jsonl`.
- **Graceful degradation log:** When GX10/omlx endpoint is unreachable, a summary log is emitted with endpoint name, unreachable duration, and affected chunk count.
- **Fusion trace (per query, opt-in):** When `--log-fusion-trace` is enabled, each query logs: per-leg scores before fusion, profile-conditional weights applied, fusion scores, reranker input/output. Required for debugging visual retrieval misses.
- **Soak provenance:** Synthetic soak output records schema_version, model versions, and sanitization mode for every judgment. This enables cross-version comparison without ambiguity.

### 7.6 Failure-Mode Behavior (Endpoint Unreachable)

| Component | Endpoint | When unreachable | Behavior |
|---|---|---|---|
| **LLM Sanitizer** | GX10 at `10.0.10.239:8000` | Pipeline falls back to heuristic sanitization | Log sentinel per chunk; build-level warning; no hard failure |
| **Text Embedder (dense)** | omlx at `10.0.10.246:8000` | Pipeline halts — no text retrieval without embeddings | Hard fail with clear error message; operator intervention required |
| **ColPali (visual)** | omlx at `10.0.10.246:8000` | Visual retrieval leg skipped for that query | Log per query; text-only retrieval proceeds; no hard failure |
| **ModernBERT (rerank)** | omlx at `10.0.10.246:8000` | Reranker skipped; top-K from fusion returned directly | Log per query; slight quality degradation; no hard failure |

---

## 8. Quality Gates (V3.0)

### 8.1 Universal Invariants (Unchanged from v2.X)

All checks from `docs/QUALITY_GATES.md` §"Universal Invariants" apply:
- `chunk_type` present (never null)
- `bbox` values in integer [0,1000]
- Non-empty content for text chunks
- `modality` present on every chunk
- `QA-CHECK-01` through `QA-CHECK-05`

### 8.2 V3.0-Specific Gates

| Gate | Phase | Threshold | Measurement |
|---|---|---|---|
| UIR semantic-identity | A | Identical content (modulo whitespace), chunk_type; flags additive; top-5 doc IDs unchanged | `diff` on JSONL content fields; regression fixture |
| `partial_code` flag coverage | A | 100% of cross-page code splits flagged | Corpus audit script |
| LLM guard-stack compliance | B | Zero chunks accepted with numeric mismatch, code-span change, or order swap | Guard unit tests + corpus-wide acceptance |
| LLM both-and-diff dominance | B | LLM output equal-or-better per chunk; no chunk where LLM degrades vs heuristic | `both-and-diff` comparison across full corpus |
| Format (soak judge) | B | ≥95% corpus-wide with LLM; no regression vs heuristic | `scripts/synthetic_soak.py` |
| Phase C-spike PASS | Pre-A | Visual retrieval recovers correct page on ≥60% of text-missed queries on ATZ_Elektronik_German | Spike report (numpy MaxSim, no production infra) |
| Recall@1 chunk (deficit docs) | C | ≥80% (from 67.8%) | Retrieval regression on 5 deficit docs |
| Recall@5 doc (deficit docs) | C | ≥98.6% maintained | Retrieval regression |
| Text-doc metrics (29 docs) | C | No regression on any axis | Full-corpus synthetic soak |
| Modality-aware soak axes | D | FORM/TABLE/DIAGRAM rubrics operational | `scripts/synthetic_soak.py --modality-aware` |
| Smoke matrix | All | 34/34 GATE_PASS + UNIVERSAL_PASS | `scripts/smoke_multiprofile.sh` |
| Blind-test (Greenhouse + ≥2 additional) | All | GATE_PASS + UNIVERSAL_PASS | Included in smoke matrix |
| Build reproducibility | B+ | Corpus output hash stable within 0.5% tolerance | CI test on 3-doc subset |

### 8.3 Carry-Forward from v2.16

| Limitation | Rationale |
|---|---|
| 1.4% Recall@5 doc residual | Likely judge edge cases; not a structural defect |
| ~5% Format residual after LLM sanitization | Acceptable ceiling; 100% Format is asymptotically unreachable without human review |
| Magazine image quality (composite layouts) | Rendered-region-crop deferred per v2.11 §3e; visual retrieval partially mitigates |
| EPUB engine | Deferred to post-v3.0; explicit acceptance criteria TBD in a future cycle plan |
| Cross-page table spanning | Tables spanning pages are harder than code spanning pages (column-alignment recovery); Phase A addresses code/paragraph cross-page splits; table spanning requires additional design work deferred to v3.1+ |

---

## 9. Appendix: Audit Commentary on Draft 0.1 External Review

### 9.1 Review Summary

Draft 0.1 received a 17-point structured audit from an external principal architect reviewer. The review identified: 5 architectural concerns, 4 risks requiring mitigation upgrades, 4 missing cross-cutting concerns, 3 phasing optimizations, and 10 greenlight conditions.

### 9.2 Changes in Draft 0.2 (By Review Finding)

| Review Finding | Draft 0.1 Issue | Draft 0.2 Resolution |
|---|---|---|
| **A1:** Layers not co-equal; L1+L2 are one subsystem, L3 independent, L4 is sidecar | Stacked 4-layer diagram implied false dependencies | Restructured to two peer subsystems (Ingestion Pipeline, Retrieval Backend) + sidecar Quality track (§3.1) |
| **A2:** UIR contract missing reading-order, hierarchy, flow-aware locator, lang, span/offset, provenance | Bare `bbox: List[int]`, flat `confidence: float`, no `content_original` | `Locator` type (bbox/flow_offset/dom_path); `ConfidenceBreakdown`; `reading_order`; `lang`; provenance block; `parent_element_id`/`parent_heading` (§3.2) |
| **A3:** Docling-retained decision lacks fork-back trigger | "Evaluate if Docling+LLM doesn't cross 99% Format" — no numerical trigger | Numerical fork-back trigger: Format < 92% after two soak iterations AND heuristic also < 92% → reopen VLM-native eval (§6.2) |
| **A4:** ColPali integration has granularity mismatch + reranker gap | No join policy; reranker assumed to work on visual hits | Page→chunk score propagation join policy; documented reranker behavior (§3.4) |
| **A4:** Equal 3-way RRF weights not defensible | "Equal weights as starting point" | Profile-conditional weights: visual=0 for PROSE, 0.4 for DIAGRAM/FORM, 0.3 for TABLE (§3.4) |
| **A5:** Schema version 2.7.0→3.0.0 too coarse | Single jump conflates 3 consumer breaks | Granular: 3.0.0 (UIR), 3.1.0 (sanitization provenance), 3.2.0 (visual vectors) (§7.1) |
| **B6:** Underrated risk: LLM sanitization deferral becoming permanent | Not in risk table | Added R2: "LLM sanitization deferral becomes permanent one-way door" (§2.3) |
| **B6:** Visual storage 10-100× underrated | Severity "Medium" | Upgraded to "High"; scaling math added (§2.3 R4) |
| **B7:** Edit-distance guard insufficient | Only token-count ceiling | Multi-layer guard stack: numeric/entity preservation, code-span hashing, order-preservation, token-level alignment, prompt/content delimiters (§3.3) |
| **B8:** Byte-for-byte identical unachievable | Phase A acceptance gate | Replaced with semantic-identity gate (§3.2, §8.2) |
| **B9:** B4 "Rip out heuristics" is highest-regret decision | Phase B task B4 | B4 removed. Heuristics retained alongside LLM (dual-write). Deprecation gated on both-and-diff dominance (§4 Phase B, §6.2) |
| **C10:** Missing: determinism, observability, failure-mode, provenance, cache invalidation, security, parallelism, i18n, test overfitting | Absent from Draft 0.1 | All addressed: determinism policy (§7.4), observability (§7.5), failure-mode behavior (§7.6), provenance on UIRChunk (§3.2), prompt injection guard (§2.3 R11), lang field (§3.2), holdout policy (§3.5) |
| **C12:** Phase requiring spike first | Not specified | Phase C-spike required before Phase A code (§4.2). Phase B prompt spike on 100-chunk sample recommended. |
| **D13:** Phases can be parallelized | Strict A→B→C→D | C-spike ∥ A; D1 ∥ A; B ∥ C-implementation (§4.1) |
| **D14:** Minimum viable measurement for Phase C | Not specified | Phase C-spike design: ATZ_Elektronik_German, numpy MaxSim, no production infra, 2-3 day time budget (§4.2) |
| **E16:** 10 greenlight conditions | — | All 10 addressed: (1) C-spike before A ✓; (2) semantic-identity gate ✓; (3) heuristics retained ✓; (4) profile-conditional weights ✓; (5) provenance fields ✓; (6) determinism policy ✓; (7) multi-layer guard stack ✓; (8) granular schema versioning ✓; (9) fork-back trigger for Docling vs VLM-native ✓; (10) failure-mode behavior ✓ |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---|---|---|---|
| 0.1 | 2026-05-25 | Claude Code (Opus 4.7) | Initial synthesis of Draft A + Draft B + governance audit. |
| 0.2 | 2026-05-25 | Claude Code (Opus 4.7) | Incorporate 17-point external architecture review. Restructured layer model; strengthened UIR contract; semantic-identity gate; heuristics retained (dual-write); Phase C-spike; granular schema versioning; expanded risk register; determinism/observability/failure-mode policies; fork-back triggers; regret-risk register; glossary. |

---

**END OF ARCHITECTURE_V3_DRAFT_0.2.md**