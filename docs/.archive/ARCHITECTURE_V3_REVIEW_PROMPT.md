**SYSTEM INSTRUCTIONS**

You are a principal software architect with 20+ years of experience in
document processing pipelines, multimodal RAG systems, and production ML
infrastructure. You are reviewing a draft architecture charter for a major
version bump of a production system. Be direct, specific, and technical.
Praise is only useful if it identifies *why* something is correct. Criticism
must include a concrete alternative or remediation.

---

**CONTEXT (supplied for review, not for comment)**

- **Project:** MM-Converter V2 — a multimodal document ingestion and
  retrieval pipeline. v2.16.0 is the current production release.
  1033 tests, 34-doc canonical corpus, hybrid retrieval (dense + sparse +
  RRF + cross-encoder rerank), schema version 2.7.0.
- **Hardware:** Apple Silicon (M-series), plus LAN-local GX10 (Nvidia GB10
  platform) FP8 inference
  endpoint and omlx server. Solo developer, 12-day convergence cycles.
- **Governance:** Strict architectural decision records (`docs/DECISIONS.md`),
  quality gates with a no-weakening rule, blind-test acceptance requirements.
- **Current ceilings:** Recall@5 doc 98.6% (near asymptote), Recall@1 chunk
  67.8% (hard ceiling from text-only retrieval), Format quality ~88-93%
  (coverage-revealed pre-existing OCR debt).

---

**TASK**

Review the architecture charter below. Answer every question in §REQUIRED
OUTPUT. Do not skip any. Provide your reasoning, not just conclusions.

---

**REQUIRED OUTPUT**

### A. ARCHITECTURAL SOUNDNESS (answer all)

1. Does the four-layer decomposition (UIR → LLM Sanitization → Visual
   Retrieval → Modality Gates) form a coherent whole? Would you reorder,
   merge, or split any layers?

2. Is the UIR interface contract (§3.2) sufficient to decouple extraction
   from chunking? What is missing from `UIRChunk` or `ConversionPlan` that
   real PDF/EPUB pipelines will need?

3. The document keeps Docling 2.86 as the extraction engine and adds LLM
   sanitization on top, deferring VLM-native parsing (MinerU/GOT-OCR2.0) to
   post-v3.0 evaluation. Is this the right call? Under what conditions would
   you reverse it?

4. Does the ColPali/MaxSim design (§3.4) integrate correctly with the existing
   hybrid retrieval stack (dense + sparse + RRF + ModernBERT rerank)? Are the
   3-way RRF fusion weights (equal) defensible as a starting point?

5. The document bumps `schema_version` from 2.7.0 to 3.0.0. Is this the
   right granularity for a schema version? What downstream breakage does
   this signal that the document doesn't mention?

### B. RISK ASSESSMENT (answer all)

6. Which risk in §2.3 do you consider **underrated** (higher probability or
   worse impact than the document estimates)? Why?

7. The LLM sanitization pass (§3.3) uses an edit-distance guard (>30% token
   change = reject) to prevent hallucination. Is this guard sufficient?
   What failure mode does it NOT catch?

8. Phase A requires "byte-for-byte identical output" to the v2.16 baseline
   as an acceptance gate. Is this achievable in a genuine UIR refactor?
   Where will it break?

9. What is the single highest-regret-risk decision in this document — the
   one most likely to be reversed within 6 months of v3.0 shipping?

### C. MISSING CONCERNS (answer all)

10. What architectural concern, quality attribute, or cross-cutting requirement
    is **absent** from this document that will cause trouble by Phase C?

11. The document defers magazine image quality and the 1.4% Recall@5 doc
    residual as carry-forwards (§8.3). Is anything else being implicitly
    deferred that should be made explicit?

12. If you were the solo developer executing this plan, which phase would you
    refuse to start without a spike/prototype first? What would that spike
    need to prove?

### D. PHASING & EXECUTION (answer all)

13. The 4-phase plan assumes sequential execution (A → B → C → D). Can any
    phases be parallelized without increasing risk? If so, which?

14. Phase C (Visual Retrieval) has the highest risk. Design a **minimum
    viable measurement** — the smallest experiment that proves or disproves
    the ColPali approach — that can run BEFORE the full Phase C build.

15. Each phase targets a 12-day convergence cycle. Which phase is most
    likely to overrun, and by how much? Why?

### E. OVERALL JUDGMENT

16. Would you greenlight this architecture for v3.0 execution? If yes, with
    what conditions? If no, what must change?

17. Rate the document on a 1-10 scale as:
    - A **decision-making tool** (can a team use it to make consistent
      implementation choices?)
    - A **communication artifact** (can a new engineer understand the system
      from it?)
    - A **risk management instrument** (does it honestly surface what could
      go wrong?)

---

**DOCUMENT TO REVIEW — `docs/ARCHITECTURE_V3_DRAFT_0.4.md`**

# MM-Converter V3.0: Architectural Charter — Draft 0.4

**Status:** DRAFT for review (synthesis of DRAFT_A + DRAFT_B + governance audit)
**Target release:** v3.0 (next major schema version after v2.X)
**Governance lineage:** `AGENTS.md` (Level-0 invariants) → `docs/DECISIONS.md` (canonical decisions) → `docs/ARCHITECTURE.md` (v2.X baseline) → this document (v3.0 target)
**Parent:** `docs/ARCHITECTURE.md` (v2.X production architecture — this document supersedes for v3.0)
**Audit basis:** `ARCHITECTURE_V3_DRAFT_A.md` (4-layer model), `ARCHITECTURE_V3_DRAFT_B.md` (3-pillar model), `docs/DECISIONS.md` (through v2.16.0), `docs/QUALITY_GATES.md`, `docs/AGENT_GOVERNANCE.md`

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
| **Chunk sanitization** | Regex + POS + per-profile rules | LLM-native semantic polish pass (local GX10 FP8 endpoint, $0 cost) |
| **Retrieval** | Flat-text embeddings (dense + sparse + RRF) | Hybrid text + late-interaction visual embeddings (ColPali/MaxSim) |
| **Quality evaluation** | Prose-calibrated format gates | Modality-aware gates (prose, form, table, diagram) |

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

### 2.2 Quality Attribute Scenarios

| ID | Quality Attribute | Scenario | Target |
|---|---|---|---|
| Q1 | **Recall@1 (chunk)** on complex docs | User queries German circuit diagram label | Close -12pp deficit (from 67.8% → ≥80%) |
| Q2 | **Format quality** | Chunk content judged by LLM | ≥95% corpus-wide (v3.0 target; 99% stretch) |
| Q3 | **Cross-page integrity** | Code block spans page boundary | Correct page attribution, no content truncation |
| Q4 | **Form/table fidelity** | Dutch inventory form (`CarOK`) | Modality-aware judge scoring; no false-negative prose penalties |
| Q5 | **End-to-end latency** | Retrieval (embed → search → rerank) | ≤3.0s p99 (kept at or below v2.16's ~1.6s for text-only; visual leg adds ≤1s) |
| Q6 | **Per-query cost** | Production retrieval | $0 (all models local/LAN) |
| Q7 | **Backward compatibility** | v2.X ingestion.jsonl | Consumable by v3.0 tooling; v3.0 outputs validated against v2.X schema shape where compatible |
| Q8 | **Repeatable builds** | Fresh checkout → full corpus rebuild | < 24h wall time (current: ~8-12h for 34 docs) |

### 2.3 Key Technical Risks

| Risk | Severity | Mitigation |
|---|---|---|
| ColPali multi-vector storage requires Qdrant version upgrade or migration | Medium | Phase C pre-flight: probe Qdrant MaxSim support on current instance; budget side-collection rebuild |
| LLM sanitization pass could hallucinate or alter technical content | High | Conservative prompt engineering; diff-against-original validation; edit-distance budget (mirrors refiner guardrails); opt-out flag |
| MinerU AGPL-3.0 license incompatible with project distribution | Medium (if adopted) | Containerize as isolated service; keep Docling as fallback extraction path (Draft A model) |
| VLM-native parsing (Pillar 2 of Draft B) changes chunk shape → invalidates retrieval regression fixtures | High | Phase as separate cycle AFTER UIR refactor stabilizes; maintain byte-identical output gate until explicitly released |
| Visual embeddings increase Qdrant storage 10-100× | Medium | Budget at design time; measure on corpus subset before full rebuild |

---

## 3. V3.0 Target Architecture

### 3.1 Architecture Overview (Four Interlocking Layers)

V3.0 is structured as four layers that can be phased independently. Layers 1-2 address the heuristic-patching and fragmentation limits; Layers 3-4 address the spatial-to-text and modality-aware evaluation limits.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        V3.0 ARCHITECTURE                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LAYER 4: MODALITY-AWARE QUALITY GATES (Evaluation)                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Profile-specific judge rubrics (prose | form | table | diagram)    │  │
│  │ Synthetic soak extended with visual-retrieval axes                  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                    │                                     │
│  LAYER 3: VISION-NATIVE RETRIEVAL (ColPali + MaxSim)                     │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Parallel visual vector index (Qdrant multi-vector)                 │  │
│  │ Patch-level embeddings (ColPali/ColQwen2.5)                        │  │
│  │ MaxSim late-interaction fusion with text-retrieval results          │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                    │                                     │
│  LAYER 2: LLM-NATIVE CHUNK SANITIZATION (Ingestion)                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Local FP8 LLM cleanup pass (RedHatAI/Qwen2.5-14B-Instruct-FP8)    │  │
│  │ Semantic polish: heal cross-page splits, sentence boundaries       │  │
│  │ Replaces brittle heuristics (POS, dropcap, label filters)          │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                    │                                     │
│  LAYER 1: TRUE UIR (Unified Data Contract)                                │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Format-agnostic ConversionPlan → UniversalDocument                 │  │
│  │ Decoupled ElementProcessor operates on UIR, not Docling DOM        │  │
│  │ Enables reliable cross-page state + partial_code flags             │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                    │                                     │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │              EXTRACTION ENGINES (retained, pluggable)              │  │
│  │  PDF (Docling 2.86)  │  EPUB (EbookLib)  │  Future: VLM-native    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Layer 1: True Universal Intermediate Representation (UIR Refactor)

**Directly addresses:** Limit 3 (Chunker-State Fragmentation), v2.15 Item #13 (parked)
**Governance anchor:** `docs/DECISIONS.md` — "Combined ceiling: UIR refactor + LLM cleanup together" (v2.10 chunker-quality ceiling); "v2.11 Carry-Forward §3c — PAUSED for user signoff"

**Design:**

The v2.X architecture has a partial UIR (`UniversalDocument`, `UniversalPage`, `Element`) defined in `src/mmrag_v2/universal/intermediate.py`, but the production pipeline does not actually route through it. `PdfConversionPlan`, `BatchProcessor`, and `DoclingPdfAdapter` each construct Docling options independently and the HybridChunker operates directly on Docling's `DoclingDocument` layout objects.

V3.0 elevates the UIR to the **single source of truth between extraction and chunking:**

1. **Format-Agnostic `ConversionPlan`:** `PdfConversionPlan` is elevated to a parent `ConversionPlan` class. All format-specific adapters (PDF, EPUB, future DOCX/HTML) produce a `ConversionPlan` that the pipeline consumes uniformly.

2. **Extraction Engines as Dumb Pipes:** `PDFEngine`, `EpubEngine` (and future engines) output a standardized `UniversalDocument` schema. They does NOT construct Docling options independently — the shared adapter/factory is the single construction site (preserving the v2.8 `test_no_raw_converter_invocation_outside_adapter` guard).

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

@dataclass
class UIRChunk:
    """Emitted by ElementProcessor; consumed by chunker + sanitizer."""
    modality: Modality
    content: str
    bbox: List[int]              # [x1, y1, x2, y2] in [0, 1000]
    page_number: int
    confidence: float            # 0.0–1.0
    extraction_method: str       # "docling_direct" | "ocr_tesseract" | "ocr_doctr" | "vlm_enrichment"
    structural_flags: Dict[str, bool]  # "partial_code", "cross_page_split", "orphan_label", etc.
    source_element_ids: List[str]      # Traceability back to extraction engine elements
    asset_ref: Optional[str] = None    # Path to extracted image/asset, if IMAGE modality

@dataclass
class ConversionPlan:
    """Format-agnostic extraction plan; subclasses add format-specific fields."""
    source_path: str
    file_type: str               # "pdf" | "epub" | "html"
    doc_id: str                  # First 12 chars of MD5 hash
    profile_type: str            # From ProfileClassifier
    modality_flags: Dict[str, bool]  # "is_scanned", "has_encoding_corruption", etc.
    extraction_strategy: str     # "digital_native" | "ocr_nuclear" | "ocr_forced"
    reading_order_strategy: str  # "docling_native" | "y_sort" | "y_sort_with_dropcap"
    batch_size: int = 10
    # ... format-specific fields in subclasses
```

**Acceptance gate:** The pipeline produces **byte-for-byte identical output** to the v2.16 baseline for all 34 docs, but with `partial_code` flags now emitting correctly for all profiles. This is a refactoring-only phase — no behavioral changes, no ML models introduced.

### 3.3 Layer 2: LLM-Native Chunk Sanitization

**Directly addresses:** Limit 2 (Heuristic Patching Ceiling)
**Governance anchor:** `docs/DECISIONS.md` — "Combined ceiling: UIR refactor + LLM cleanup together" (v2.10 chunker-quality ceiling, path 4: "LLM-clean every chunk on ingestion" estimated at $30/rebuild on Dashscope, now $0 via local GX10)

**Design:**

After Layer 1 ensures all chunks flow through the UIR, Layer 2 adds a **semantic polish pass** using the existing local LLM endpoint:

1. **Execution:** Raw extraction chunks are piped through `RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic` at `10.0.10.239:8000` (GX10 endpoint, already operational per `docs/DECISIONS.md` v2.14).

2. **Prompt contract:** The LLM receives the raw chunk content + surrounding context (previous/next chunk snippets, page breadcrumb) and is instructed to reconstruct the text into clean, well-formed Markdown:
   - Heal cross-page sentence and code-block splits
   - Normalize whitespace and punctuation
   - Remove extraction artifacts (drop-cap orphans, picture classification labels, OCR noise)
   - Preserve ALL factual content — no summarization, no paraphrasing
   - Edit-distance budget enforced (mirrors existing refiner guardrails in `src/mmrag_v2/refiner.py`)

3. **What it replaces:** The heuristic stack in `engines/docling_postprocess.py` (reading-order y-sort, drop-cap promotion, inline trailing heal, picture label suppression) AND the 4 multimodal validation layers (CorruptionInterceptor, POS Boundary Logic, Vision-Gated Hierarchy, Content-Type Classification) — the LLM subsumes their function natively.

4. **Cost:** $0 per rebuild (local GX10 FP8 endpoint). No cloud API calls on the sanitization path.

5. **Opt-out:** `--no-llm-sanitize` flag preserves v2.16 heuristic behavior for regression comparison.

**Risk mitigations:**
- **Hallucination guard:** Diff chunk content before/after sanitization. If >30% of tokens changed, reject and keep original content with a `[LLM_SANITIZE_REJECTED: edit ratio {pct}]` sentinel.
- **Content preservation:** Prompt explicitly forbids summarization, paraphrasing, or content deletion.
- **Latency budget:** Target <500ms/chunk at GX10 FP8 inference speed. Batch chunks per page to amortize overhead.
- **Regression: ** Full v2.16 corpus rebuild with sanitization enabled; compare Format scores against v2.16 baseline.

**Acceptance gate:** Synthetic soak Format scores cross 95% corpus-wide (target), with no regression on Recall, Relevance, or Faithfulness axes.

### 3.4 Layer 3: Vision-Native Retrieval (ColPali + MaxSim)

**Directly addresses:** Limit 1 (Spatial-to-Text Translation Deficit, -12pp Recall@1)
**Governance anchor:** `docs/DECISIONS.md` — v2.16 Phase 2 diagnosis; `docs/PLAN_V2.16.md` Item #11 (ColPali/VisRAG scoped out)
**Design basis:** Draft B's Pillar 3 (more technically detailed than Draft A's Layer 3)

**Design:**

The v2.16 retrieval stack (dense + sparse + RRF + ModernBERT reranker) operates entirely on text embeddings. For text-heavy documents, this stack approaches 98.6% Recall@5 doc. For visually-complex documents (diagrams, multi-column forms, scanned engineering drawings), text embeddings discard the 2D spatial information that makes the content meaningful.

V3.0 introduces a **parallel visual vector index** that operates alongside the text retrieval stack:

1. **Patch-Level Visual Embeddings (ColPali/ColQwen2.5):**
   - Each page (or page region for complex documents) is rendered as an image.
   - A Vision Transformer (ColPali, or its successor ColQwen2.5) divides the page into patches (e.g., 1030 patches per page at 448×448 resolution).
   - Each patch is embedded as a 128-dimensional vector.
   - The result: a page is represented as a **matrix of patch vectors**, not a single flat embedding.

2. **Qdrant Multi-Vector Storage:**
   - Qdrant is configured to support multi-vector collections (named vectors per patch position, or a single multi-vector payload).
   - Each document page stores its patch-vector matrix alongside the existing text-chunk embeddings.

3. **MaxSim Late-Interaction Retrieval:**
   - At query time, the query is embedded through the same ColPali vision model.
   - MaxSim (Maximum Similarity) computes: for each query token, find its most similar document patch, then sum those maximum similarities.
   - This allows the query "circuit diagram transistor" to match the visual patch that *shows* the transistor symbol — not just text that mentions it.

4. **Fusion with Text Retrieval:**
   - Visual retrieval produces a ranked list of pages/chunks.
   - Text retrieval (existing hybrid stack) produces its ranked list.
   - RRF or weighted fusion combines both into a single candidate set.
   - ModernBERT reranker operates on the combined set (text content + visual context metadata).

**End-to-end retrieval flow (V3.0):**

```
query
  ├─ dense  : omlx Qwen3-Embedding-8B-mxfp8 (4096-dim)     ← retained (v2.13)
  ├─ sparse : BM25                                         ← retained (v2.12)
  └─ visual : ColPali/ColQwen2.5 patch vectors             ← NEW (v3.0)
  → RRF fusion (k=60, equal weights 3-way)
  → ModernBERT rerank                                     ← retained (v2.12)
  → top-5 return
```

**Resource budget:**
- Patch-vector storage: ~128 dimensions × 1030 patches × N pages. For the 34-doc corpus (~10,000 pages): ~5 GB.
- Visual embedding latency: target <1s/page at omlx server inference speed.
- Qdrant multi-vector support: verify on current Qdrant version; may require version upgrade or side-collection.

**Acceptance gate:** The -12pp Recall@1 deficit on the 5 complex engineering documents (`ATZ_Elektronik_German`, `Python_Cookbook`, `IRJET_Modeling_of_Solar_PV`, `Earthship_Vol1`, `CarOK_voorraadtelling`) is closed. Target: Recall@1 chunk ≥80% (from 67.8%), Recall@5 doc ≥98.6% maintained.

### 3.5 Layer 4: Modality-Aware Quality Gates

**Directly addresses:** CarOK form-class false-negative Format penalties, profile-specific evaluation
**Governance anchor:** `docs/QUALITY_GATES.md` — "Form / Invoice Acceptance Class" (v2.8); `docs/DECISIONS.md` — "v2.12 Phase 0 Outcome" (CarOK form-shape decision); Draft A's Layer 4

**Design:**

The v2.X quality gates are calibrated primarily for prose documents. Structured data (forms, tables, inventory sheets) is penalized by prose-calibrated judges because it doesn't "read like a paragraph." V3.0 extends the quality evaluation framework:

1. **Profile-Specific Judge Rubrics:**
   - `PROSE` (default): relevance, fluency, formatting — the existing axes.
   - `FORM`: data integrity (key-value completeness), field accuracy, no unstructured prose requirement.
   - `TABLE`: structural fidelity (row/column preservation), numeric accuracy.
   - `DIAGRAM`: visual description quality, label accuracy, spatial relationship capture.

2. **UIR Modality Tag Driving Judge Selection:**
   - The `UIRChunk.modality` field (set by Layer 1) determines which judge rubric applies.
   - `FORM` and `TABLE` chunks skip prose-fluency checks; `DIAGRAM` chunks skip OCR-text accuracy checks.

3. **Visual Retrieval Quality Axes:**
   - New soak axes for Layer 3: **Visual Relevance** (did the retrieved visual patch match the query intent?), **Spatial Fidelity** (did the patch encoding preserve layout information?)
   - Existing axes (Recall, Relevance, Faithfulness, Format) extended to account for visual retrieval contributions.

4. **Strict Gate Extension:**
   - `FORM_AUDIT_PASS` (existing v2.8) extended to `TABLE_AUDIT_PASS` and `DIAGRAM_AUDIT_PASS`.
   - Universal invariants (bbox range, modality present, non-empty content) still apply across all classes — no waivers.

**Acceptance gate:** `CarOK_voorraadtelling` Format score no longer penalized for non-prose content shape. Modality breakdown visible in soak reports. No false-negative quality failures on structured data.

---

## 4. Implementation & Phasing Strategy

V3.0 is sequenced to respect solo-dev 12-day convergence cycles while maintaining a green test suite. Each phase produces a releasable artifact (tagged `v3.0-rcN` or similar).

### Phase A: UIR Foundation (Cycle 3.0)

**Scope:** Layer 1 only. No ML changes. No behavioral changes.

| Task | Description | Acceptance |
|---|---|---|
| A1 | Elevate `PdfConversionPlan` to parent `ConversionPlan` | All existing tests pass without modification |
| A2 | Refactor extraction engines to output `UniversalDocument` | Byte-for-byte identical JSONL output vs v2.16 baseline |
| A3 | Decouple chunker from `DoclingDocument` → operate on UIR | `partial_code` flags emit correctly; cross-page splits attributed to correct pages |
| A4 | Rip out duplicate Docling option construction sites | `test_no_raw_converter_invocation_outside_adapter` expanded to cover v3.0 paths |
| A5 | Corpus-wide rebuild + strict gate | 34/34 PASS (or documented v2.16-equivalent deferrals) |

**Risk:** Lowest-risk phase. Pure refactoring. Rollback is `git revert`.

### Phase B: LLM Sanitization (Cycle 3.1)

**Scope:** Layer 2. Requires Phase A complete.

| Task | Description | Acceptance |
|---|---|---|
| B1 | Wire GX10 FP8 endpoint into UIR output phase | Sanitization harness operational; `--no-llm-sanitize` flag functional |
| B2 | Design + validate sanitization prompt | Prompt passes negative tests (no hallucination, no summarization, no content loss) |
| B3 | Implement edit-distance guard | Chunks with >30% token change rejected with sentinel |
| B4 | Rip out legacy heuristic patches (`docling_postprocess.py`) | Heuristic stack removed; LLM sanitization covers all previously-patched defect classes |
| B5 | Corpus-wide rebuild + synthetic soak | Format ≥95% corpus-wide; no regression on other axes |

**Risk:** Medium. LLM may introduce subtle content alterations. Mitigated by edit-distance guard + diff-against-original validation + opt-out flag for regression comparison.

### Phase C: Visual Retrieval (Cycle 3.2)

**Scope:** Layer 3. Requires Phase B complete (clean chunks to embed visually).

| Task | Description | Acceptance |
|---|---|---|
| C1 | Probe Qdrant multi-vector support | Confirm MaxSim compatibility; plan migration or side-collection |
| C2 | Deploy ColPali/ColQwen2.5 on omlx server | Visual embedding endpoint operational; latency <1s/page |
| C3 | Build parallel visual index | Corpus pages embedded as patch-vector matrices |
| C4 | Implement MaxSim retrieval + fusion with text stack | `retrieve_hybrid_visual()` functional; 3-way RRF fusion |
| C5 | Synthetic soak on complex-doc subset | Recall@1 chunk ≥80% on the 5 deficit docs; text-doc metrics maintained |

**Risk:** Highest-risk phase. Multi-vector storage may require Qdrant version upgrade. Visual model deployment may exceed omlx server memory budget. Mitigated by incremental rollout (single doc → deficit subset → full corpus).

### Phase D: Modality-Aware Gates (Cycle 3.3)

**Scope:** Layer 4. Can partially overlap with Phases B-C.

| Task | Description | Acceptance |
|---|---|---|
| D1 | Implement profile-specific judge rubrics | FORM, TABLE, DIAGRAM rubrics operational in synthetic soak |
| D2 | Extend strict gate for modality classes | `FORM_AUDIT_PASS`, `TABLE_AUDIT_PASS`, `DIAGRAM_AUDIT_PASS` in `qa_full_conversion.py` |
| D3 | Add visual retrieval quality axes | Visual Relevance, Spatial Fidelity axes in soak reports |
| D4 | Full-corpus v3.0 acceptance run | Smoke matrix: all rows GATE_PASS + UNIVERSAL_PASS; blind-test document (Greenhouse) included |

**Risk:** Low. Extends existing v2.8 form-audit precedent. No pipeline changes.

---

## 5. Component Architecture (Detailed)

### 5.1 Module Map (V3.0 Target)

```
src/mmrag_v2/
├── universal/                     # [EXPANDED] True UIR layer
│   ├── intermediate.py            # UniversalDocument, UniversalPage, UIRChunk
│   ├── conversion_plan.py         # Parent ConversionPlan + format-specific subclasses
│   ├── element_processor.py       # [REFACTORED] Operates on UIR, not Docling DOM
│   ├── router.py                  # Format detection → engine routing (retained)
│   └── quality_classifier.py      # ConfidenceNormalizer (retained)
│
├── engines/                       # Format-specific extraction
│   ├── base.py                    # FormatEngine ABC (retained)
│   ├── pdf_engine.py              # [REFACTORED] Consumes shared adapter; outputs UniversalDocument
│   ├── docling_adapter.py         # [REFACTORED] Single Docling construction + invocation site
│   ├── docling_postprocess.py     # [DEPRECATED → removed in Phase B]
│   └── epub_engine.py             # [NEW/FUTURE] EPUB → UniversalDocument (optional)
│
├── sanitization/                  # [NEW] LLM-native chunk sanitization
│   ├── __init__.py
│   ├── llm_sanitizer.py           # GX10 FP8 endpoint client + prompt management
│   ├── edit_guard.py              # Diff-based hallucination/rejection guard
│   └── prompts.py                 # Sanitization prompt templates
│
├── retrieval/                     # [EXPANDED] Visual retrieval
│   ├── pipeline.py                # [EXTENDED] retrieve_hybrid_visual()
│   ├── visual_embedder.py         # [NEW] ColPali/ColQwen2.5 client (omlx)
│   ├── maxsim.py                  # [NEW] MaxSim scoring + fusion logic
│   └── config.py                  # [EXTENDED] Visual collection defaults
│
├── vision/                        # VLM integration (retained)
│   ├── vision_manager.py
│   ├── vision_prompts.py
│   └── ocr_hint_engine.py
│
├── validators/                    # QA checks (retained; heuristics removed in Phase B)
│   ├── corruption_interceptor.py  # [DEPRECATED → removed in Phase B]
│   ├── token_validator.py
│   └── quality_filter_tracker.py
│
├── ocr/                           # OCR cascade (retained for non-LLM path)
├── chunking/                      # Chunking helpers (retained; consumed via UIR)
├── schema/                        # Output schema (extended for v3.0)
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
    │                              LLMSanitizer (Phase B)
    │                              ┌── edit-distance guard
    │                              │
    │                              ▼
    │                              Cleaned chunks
    │                              │
    ▼                              ▼
HybridChunker (UIR-native)  ←──────┘
    │
    ▼
IngestionChunk[]  ──→  ingestion.jsonl + assets/
    │
    ▼
Qdrant ingest (text + visual)
    ├── dense  : omlx text embed → mmrag_v3__dense
    ├── sparse : BM25 → mmrag_v3__sparse
    └── visual : ColPali patches → mmrag_v3__visual  (Phase C)
```

### 5.3 Retrieval Flow (V3.0 Production)

```
QUERY
    │
    ├─→ dense  : omlx Qwen3-Embedding-8B-mxfp8 (4096-dim)
    ├─→ sparse : BM25 (token match)
    └─→ visual : ColPali patch vectors → MaxSim
    │
    ▼
RRF fusion (k=60, equal 3-way weights)
    │
    ▼
ModernBERT rerank (top-25 → top-5)
    │
    ▼
Top-5 results (text chunks + visual page references)
```

---

## 6. Design Decisions & Cross-References

| Decision | Phase | Rationale | Cross-Reference |
|---|---|---|---|
| **Docling retained** as default PDF extractor; LLM sanitization added post-extraction | A-B | Draft B's VLM-native parsing (MinerU/GOT-OCR2.0) is higher-risk and license-constrained (AGPL-3.0). Draft A's approach (keep Docling + LLM-clean) achieves the same Format improvement with lower risk. VLM-native parsing can be evaluated as a Phase B+ alternative if Docling+LLM doesn't reach 99% Format. | Draft A Layer 2; Draft B Pillar 2; `DECISIONS.md` v2.10 chunker-quality ceiling |
| **LLM sanitization uses local GX10 FP8, not cloud** | B | $0 cost vs $30/rebuild on Dashscope. Privacy-preserving. Already operational (v2.14). | Draft A Layer 2; `DECISIONS.md` v2.14 local-LLM stack |
| **ColPali via omlx, not cloud vision API** | C | $0 per-query cost. LAN-local, privacy-preserving. Consistent with v2.13 embedder swap philosophy. | Draft B Pillar 3; `DECISIONS.md` v2.13 embedder swap |
| **3-way RRF fusion (dense + sparse + visual)** | C | Equal weights as starting point; tune after soak measurement. Keeps text-retrieval path intact for text-heavy docs. | Draft B Pillar 3; v2.12 Phase 2 RRF precedent |
| **Phase A requires byte-for-byte identical output** | A | Ensures UIR refactor introduces zero behavioral regressions. Draft A's Phase A acceptance gate. | Draft A Phase A; `AGENT-VAL-01` |
| **LLM edit-distance guard mirrors refiner guardrails** | B | Proven pattern from `src/mmrag_v2/refiner.py`. Keeps LLM from hallucinating or over-editing. | `DECISIONS.md` "Heal-Over for Encoding Corruption" |
| **VLM-native parsing (MinerU/GOT-OCR2) deferred to post-v3.0 evaluation** | — | Higher risk (license, chunk-shape stability). Evaluate if Docling+LLM doesn't cross 99% Format. | Draft B Pillar 2; risk table §2.3 |
| **No gate weakening** | All | Per `DECISIONS.md` "No gate weakening to make a failing run pass." If a phase doesn't meet its acceptance gate, it carries forward with documented rationale. | `DECISIONS.md` (2026-05-09); `AGENT-GOVERNANCE.md` |

---

## 7. Migration & Rollback Strategy

### 7.1 Schema Version

V3.0 bumps `schema_version` from `2.7.0` to `3.0.0`. This is the first schema bump since v2.7 (all v2.8–v2.16 changes were behavioral, not schema changes). The schema bump signals:
- `UIRChunk` is the new internal representation
- `partial_code` and other structural flags are now reliably populated
- Chunk shape may differ from v2.X due to LLM sanitization (Phase B) and visual chunk emission (Phase C)

Consumer warning: downstream RAG adapters that key on `chunk_id` or `schema_version` for cross-version mapping must rebuild from v3.0 outputs.

### 7.2 Rollback Paths

| Phase | Rollback | Procedure |
|---|---|---|
| A (UIR) | `git revert` | Byte-for-byte identical output → no data migration needed |
| B (LLM sanitize) | `--no-llm-sanitize` flag | Restore v2.16 heuristic behavior; re-ingest with flag |
| C (Visual index) | Drop visual collection | Text retrieval stack unaffected; visual leg is additive |
| D (Gates) | Gate-only change | No data migration; gate thresholds revert independently |

### 7.3 Corpus Migration

Each phase that changes chunk output (B, C) requires:
1. Rebuild corpus from source PDFs (existing scripts; ~8-12h wall time)
2. Re-ingest to Qdrant (existing scripts; ~1-2h for text, ~5h for visual)
3. Synthetic soak against new collections (existing scripts; ~1-2h)
4. Updated retrieval regression fixture (`tests/fixtures/retrieval_regression_v3_X.json`)

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
| UIR byte-identical output | A | 100% match vs v2.16 baseline | `diff <(python -c '...') <(python -c '...')` |
| `partial_code` flag coverage | A | 100% of cross-page code splits flagged | Corpus audit script |
| LLM edit-distance compliance | B | <30% token change per chunk | `llm_sanitizer.edit_guard` |
| Format (soak judge) | B | ≥95% corpus-wide | `scripts/synthetic_soak.py` |
| Recall@1 chunk (complex docs) | C | ≥80% (from 67.8%) | Retrieval regression on 5 deficit docs |
| Recall@5 doc (complex docs) | C | ≥98.6% maintained | Retrieval regression |
| Modality-aware soak axes | D | FORM/TABLE/DIAGRAM rubrics operational | `scripts/synthetic_soak.py --modality-aware` |
| Smoke matrix | All | 34/34 GATE_PASS + UNIVERSAL_PASS | `scripts/smoke_multiprofile.sh` |
| Blind-test (Greenhouse) | All | GATE_PASS + UNIVERSAL_PASS | Included in smoke matrix |

### 8.3 Carry-Forward from v2.16

The following v2.16 documented limitations are carried forward without resolution in v3.0:

| Limitation | Rationale |
|---|---|
| 1.4% Recall@5 doc residual | Likely judge edge cases; not a structural defect |
| ~5% Format residual after LLM sanitization | Acceptable ceiling; 100% Format is asymptotically unreachable without human review |
| Magazine image quality (composite layouts) | Rendered-region-crop deferred per v2.11 §3e; visual retrieval (Layer 3) partially mitigates |

---

## 9. Appendix: Audit Commentary on Draft A vs Draft B

### 9.1 Structural Comparison

| Dimension | Draft A | Draft B | This Document (Draft 0.1) |
|---|---|---|---|
| **Layers/Pillars** | 4 (UIR, LLM Sanitize, Vision Retrieval, Modality Gates) | 3 (UIR, VLM Parsing, Visual Retrieval) | 4 (synthesizes both: Draft A's structure + Draft B's technical depth) |
| **Docling posture** | Keep + add LLM sanitization | Replace with VLM-native (MinerU/GOT-OCR2) | Keep + LLM sanitize (A); VLM-native deferred to post-v3.0 evaluation |
| **Phasing** | 3 phases (Core, LLM, Visual) | 3 cycles (UIR, VLM Swap, Visual) | 4 phases (UIR, LLM, Visual, Gates) with explicit acceptance gates |
| **Risk profile** | Conservative, incremental | Aggressive, higher-risk | Conservative core (A) with aggressive paths gated on measured outcomes |
| **Quality gates** | Layer 4 dedicated to modality-aware gates | Not addressed | Layer 4 retained from Draft A; essential for CarOK-class false negatives |
| **Component detail** | Light | Moderate (ColPali mechanics) | Comprehensive (§5 component map, data flow, module map) |
| **Governance traceability** | Implicit | Implicit | Explicit cross-references to DECISIONS.md, QUALITY_GATES.md, AGENT_GOVERNANCE.md |
| **Rollback strategy** | Not addressed | Not addressed | §7 — per-phase rollback procedures |
| **Resource budgets** | Mentioned (GX10, $0) | Mentioned (containerization) | §2.2 (quality scenarios), §3.4 (visual storage budget) |
| **Schema version** | Not addressed | Not addressed | §7.1 — explicit 2.7.0 → 3.0.0 bump |

### 9.2 Strengths of Each Draft

**Draft A strengths:**
- Clear, concise narrative (83 lines) — easy to read and reason about
- Laser-focused on v2.16's empirically-diagnosed limits
- Pragmatic phasing with byte-identical output gate
- Concrete local infrastructure references (GX10, omlx, 10.0.10.239:8000)
- Layer 4 (Modality-Aware Quality Gates) is essential and unique to Draft A

**Draft B strengths:**
- Technical specificity on ColPali/MaxSim mechanics (patch-level embeddings, 128-dim vectors, 1030 patches)
- Acknowledges licensing issues (AGPL-3.0 for MinerU)
- Mentions specific state-of-the-art models (GOT-OCR 2.0, Qwen2.5-VL, ColQwen2.5)
- More explicit about Docling replacement (Pillar 2)
- Clean 3-cycle deployment narrative

### 9.3 Gaps Addressed in Draft 0.1

| Gap | Present in | Resolution |
|---|---|---|
| No explicit governance cross-references | Both | §0 (Read-order), §6 (Decision rationale table), §8 (Quality gates) |
| No rollback/migration strategy | Both | §7 — per-phase rollback, schema version bump, corpus migration |
| No resource budget | Both | §2.2 (quality attribute scenarios), §3.4 (visual storage budget) |
| No component-level interface contracts | Both | §3.2 (UIR contract in Python), §5 (module map + data flow) |
| No risk analysis | Both (Draft B partially via AGPL note) | §2.3 (risk table with severity + mitigation) |
| No quality attribute scenarios | Both | §2.2 (Q1-Q8 with measurable targets) |
| Draft B missing modality-aware gates | Draft B | Layer 4 retained from Draft A; essential for CarOK-class defects |
| Draft A light on ColPali mechanics | Draft A | §3.4 incorporates Draft B's technical detail |
| No explicit distinction: Docling retained vs replaced | Both (contradict each other) | §6 decision table — Docling retained (Draft A approach); VLM-native deferred |
| No schema version discussion | Both | §7.1 — 2.7.0 → 3.0.0 bump with consumer warning |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---|---|---|---|
| 0.1 | 2026-05-25 | Claude Code (Opus 4.7) | Initial synthesis of Draft A + Draft B + governance audit. 4-layer architecture, 4-phase implementation, explicit governance traceability. |

---

**END OF ARCHITECTURE_V3_DRAFT_0.1.md**