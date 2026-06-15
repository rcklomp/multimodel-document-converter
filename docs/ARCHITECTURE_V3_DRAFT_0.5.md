# MM-Converter V3.0: Architectural Charter — Draft 0.5

> ⚠️ **TARGET, NOT AS-BUILT.** This is the original V3.0 *aspirational target*. For
> what actually ships today, read `ARCHITECTURE_V3.1_CHARTER.md` (as-built) and
> `PROJECT_STATUS.md` (current state). Do not treat this draft as a description of
> current behavior.

> **As-built note (2026-06-03):** this draft is the original V3.0 *target*. The
> current as-built reality and audit-corrected roadmap are tracked in
> `docs/ARCHITECTURE_V3.1_CHARTER.md` (status-tagged SHIPPED/PARTIAL/PROPOSED).
> Where this draft and the V3.1 charter differ, the charter reflects what the
> code does today; this draft is retained for the Phase-A micro-sequence and the
> original intent.

## Phase A: Non-Negotiable UIR Foundation

**Scope:** Phase A2 is a mandatory, non-deferrable structural dependency. The core `batch_processor.py` and `processor.py` (approx 1,755 LOC) MUST be refactored to natively consume the Universal Intermediate Representation (`UniversalDocument` / UIR). 

**Forbidden Paths:** * The `UIR-shim fallback` (Option B from legacy drafts) is **explicitly revoked**. Adapting `DoclingDocument` to UIR at the export boundary is a violation of the V3 architecture. The internal pipeline must operate on UIR natively.
* Splitting A2 across cycles (Option D from legacy drafts) is **revoked**. 

**Execution Strategy (Micro-Sequenced):** Due to the size of the refactor surface, execution MUST be micro-sequenced linearly to prevent context-window collapse. Agents must execute and commit these sequentially:
1. `ingestion_schema.py`: Refactor `IngestionChunk` to consume UIR. Run schema tests.
2. `processor.py:_emit_dense_index_page_chunks`: Rewrite for UIR Native.
3. `processor.py:_emit_section_header_only_page_chunks`: Rewrite for UIR Native.
4. `processor.py:_process_text_with_hybrid_chunker`: The core decoupling rewrite. Update call sites.
5. `batch_processor.py`: Reconcile headings, merge chunks, and apply spatial refiner over UIR.
   > **SUPERSEDED (2026-05-30) by `docs/V3_EXECUTION_MANDATE.md` §3.** Step 5's
   > scope is the engine-agnostic *orchestration* port only (batching, routing,
   > JSONL writing). The v2.16 reconciliation/merge/spatial-refiner heuristics are
   > **permanently deferred** (per `V3_DEFERRED_TESTS.md`), NOT reimplemented over
   > UIR. Closed this way in commits `813b9ba`/`c53129d`; two residual heuristics
   > carried as Phase B debt (see `docs/PROJECT_STATUS.md`). Do not act on the
   > line above where it conflicts with the Mandate — the Mandate governs.

*(Note: Each step must be committed and validated via `pytest` independently before proceeding to the next).*


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
| **Smoke matrix** | Multi-profile gate run across all 34 docs + blind-test documents (Greenhouse + ≥7 additional); requires GATE_PASS + UNIVERSAL_PASS |
| **Content-pinning cache** | Deterministic key-value store: `(content_hash, context_hash, model_id, prompt_version) → sanitized_output`. Context-aware key prevents stale hits when neighbor chunks change. Enables build reproducibility without requiring the LLM endpoint to be deterministic. |
| **ChunkType** | Schema-level enum (today's `src/mmrag_v2/schema/ingestion_schema.py::ChunkType`) — values include PARAGRAPH, CODE, TABLE, IMAGE, etc. Lives on `IngestionChunk.chunk_type`. |
| **ElementType** | UIR-internal enum (today's `src/mmrag_v2/universal/intermediate.py::ElementType`) — three values: TEXT / IMAGE / TABLE. Lives on `Element.element_type`. |
| **Modality** | v3.0 UIR enum that *replaces* ElementType — five values: TEXT / IMAGE / TABLE / CODE / FORM. Lives on `UIRChunk.modality`. Migration policy at §7.1. |
| **StructuralFlag** | v3.0 typed enum (replaces today's free-form `Dict[str, bool]`) governing the set of legal structural flags a chunk may carry; see §3.2. |
| **σ (soak σ)** | Run-to-run standard deviation of a soak axis (Format, Recall@1, etc.) measured across ≥3 consecutive soak runs on the same fixed corpus + heuristic-only sanitization. Used as the noise floor for Phase B's dominance criterion. |
| **Identity half** | The portion of the Phase A acceptance gate that requires ≥95% of v3.0 chunks to match v2.16 chunks (by stable identity key, content, type). |
| **Explained-delta half** | The remaining ≤5% of v3.0 chunks may differ from v2.16, but each delta must be enumerated in `docs/PHASE_A_INTENTIONAL_DELTAS.md` with a v2.X documented-defect cross-reference. |

---

## 0. Document Purpose & Scope

This charter defines the next-generation architecture for the MM-Converter pipeline. It describes the *target state* for v3.0 — the components, their interfaces, the migration path, and the quality gates that govern acceptance. It does **not** replace `docs/ARCHITECTURE.md` (v2.X production architecture, still canonical for current operation) until v3.0 ships.

**Read-order:**
1. This document (v3.0 target architecture)
2. `docs/DECISIONS.md` — all architectural decisions governing v2.X → v3.0 transition
3. `docs/ARCHITECTURE.md` — v2.X production baseline (the system being evolved)
4. `docs/QUALITY_GATES.md` — v3.0 will extend, not relax, these gates
5. `docs/V3_EXECUTION_MANDATE.md` — the only definition of done for v3.0 workstreams

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

#### Limit 3: Chunker-DOM Coupling (Cross-Page Splits & Inert Adjacency Fetch)

**Evidence:** `docs/DECISIONS.md` — v2.16 partial_code adjacency fetch shipped INERT for the cross-page case
**Manifestation:** `Fluent_Python` cross-page code splits attributed to wrong pages. `Python_Cookbook`/`Python_Distilled` cross-page content present but under wrong `page_number`. The HybridChunker call sites in `processor.py` and `batch_processor.py` consult `DoclingDocument` layout objects directly and emit `partial_code` only for the in-block oversized-code-chunk case (`processor.py:5002`), not for the cross-page-split case.
**Root cause (corrected from Draft 0.3):** The HybridChunker is invoked from `processor.py` (line 3395) and its outputs are reconciled in roughly 12 distinct code paths in `batch_processor.py` (TOC heading propagation, page-split sibling fill, dual-DocChunk dedup, scan-origin bypass, OCR-driven heading override). These reconciliation paths consume `DoclingDocument` and `DocChunk` shapes directly. `src/mmrag_v2/mapper.py` also consumes `DoclingDocument` for chunk emission, and `src/mmrag_v2/engines/docling_serializers.py` extends Docling's serializer subsystem. **What this is NOT:** Draft 0.3 (inherited from 0.1/0.2) stated that `PdfConversionPlan`, `BatchProcessor`, and `DoclingPdfAdapter` each construct Docling options independently. That has not been true since 2026-04-30 — `DoclingPdfAdapter` is the single construction site, enforced by the static guard `tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter`. The coupling that remains is between chunker/mapper/serializer call sites and Docling's DOM, not between adapter and converter construction.
**Cannot be solved by:** Patching chunker state propagation per-profile.
**Can only be solved by:** True UIR refactor (Item #13 from v2.15) — decoupling extraction from chunking via a format-agnostic `UniversalDocument`, AND rewriting the HybridChunker call sites in `processor.py` + `batch_processor.py` AND the mapper AND the serializer extension to consume UIR rather than `DoclingDocument`. This scope is materially larger than Draft 0.3 implied; see §4 Phase A budget.

### 1.3 Why V3.0 Is a Paradigm Shift, Not a Feature Patch

All three limits share a common root: **heuristic 1D text processing of 2D visual data.** V3.0 shifts the burden of spatial comprehension from Python heuristics to models trained on millions of layout topologies. The architecture moves from:

| Axis | v2.X (Current) | v3.0 (Target) |
|---|---|---|
| **Extraction** | Docling DOM parsing + Python heuristics | UIR abstraction with pluggable backends (Docling retained + LLM sanitization; VLM-native parsing as optional upgrade) |
| **Chunk sanitization** | Regex + POS + per-profile rules | LLM-native semantic polish pass (local GX10 FP8 endpoint, $0 cost) with heuristic dual-write retained until aggregate dominance proven on three *production* cycles |
| **Retrieval** | Flat-text embeddings (dense + sparse + RRF) | Hybrid text + late-interaction visual embeddings (ColPali/MaxSim), profile-conditional fusion weights (priors, swept on critical path) with non-zero floor and bounded page-to-chunk join |
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
| C11 | Build reproducibility — a fresh checkout must reproduce the corpus within stated tolerances | §2.2 Q8; §7.4 Determinism Policy |
| C12 | Corpus-size operational target: ≤1000 documents, ≤300,000 pages | §3.4 Resource budget; §4.2 Phase C entry criterion |
| C13 | `chunk_id` cross-version stability — joins by `chunk_id` between v2.16 and v3.x outputs must be either preserved or accompanied by a published rewrite map (no silent breaks) | §7.1, §3.2; this document |
| C14 | `ChunkType` (schema) / `ElementType` (UIR) / `Modality` (v3.0) vocabularies must be reconciled in one direction (Modality replaces ElementType; ChunkType narrows to derive from Modality) by end of Phase A — no two-way mapping shims | §7.1; this document |

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
| Q8 | **Repeatable builds** | Fresh checkout → full corpus rebuild | < 24h wall time (cold-cache); chunk content deterministic via content-pinning cache; heuristic/off modes byte-stable. Cold-cache first build of `--sanitize-mode=llm` adds an estimated 1.5–2.0h on top of v2.X's 8–12h rebuild (~6,800 chunks × <500ms/chunk GX10 latency). See §7.3. |
| Q9 | **Graceful degradation** | GX10 endpoint unreachable | Pipeline continues with heuristics (dual-write retained); `[SANITIZE_SKIPPED: endpoint unreachable]` sentinel logged per chunk |
| Q10 | **Observability** | Visual retrieval miss root-caused | Per-query fusion weights + normalized scores logged; per-chunk lineage traceable to extraction engine + sanitizer model version + prompt version |
| Q11 | **omlx concurrency** | Query arrives while ColPali is mid-page-embedding | Query path is preempted/scheduled per §7.7 tenancy policy; query p99 latency held under Q5 budget |
| Q12 | **Retrieval API stability** | Downstream RAG app calls `retrieve_hybrid_reranked(...)` post-v3.0 | Signature preserved; new visual fusion exposed under separate `retrieve_hybrid_visual(...)` per §7.8 |
| Q13 | **Cache cold-start cost** | First `--sanitize-mode=llm` build on a fresh checkout | Cold-cache build wall time measured in Phase B B8; reported alongside warm-cache rebuild |

### 2.3 Key Technical Risks

| # | Risk | Severity | Probability | Mitigation |
|---|---|---|---|---|
| R1 | LLM sanitization hallucinates or alters factual content | **Critical** | Medium | 8-layer guard stack (§3.3): edit-distance, numeric/entity preservation, code-span hashing, order-preservation, token-alignment, prompt-boundary, entity-relation triples, corpus-level dedup-ratio. Dual-write with heuristics retained. Property-based tests per guard (§4 Phase B B7). |
| R2 | LLM sanitization deferral becomes permanent one-way door | **High** | High | Heuristics retained alongside LLM (dual-write) for v3.0–v3.1 *and three production cycles* before deprecation; explicit fork-back trigger (§6.2). Aggregate dominance criterion with degradation cap *and human-labeled golden-set check* (§3.3). |
| R3 | ColPali multi-vector storage requires Qdrant version upgrade | **High** | Medium | Phase C-spike probes Qdrant MaxSim support before Phase C implementation. C1 asserts qdrant-server version ≥ named-vector requirement (not just multi-vector "support" abstractly). C12 bounds storage ceiling. |
| R4 | Visual storage 10–100× at scale (5GB for 34 docs → ~150GB at 1000 docs raw, ~400-500GB indexed; **at 5000 docs the indexed footprint exceeds 2 TB** and crosses into network-attached storage territory with materially different latency profile — UPGRADED to High-Impact in 0.5 per audit B6 finding) | **High** | Medium | C12 defines operational ceiling at 1000 docs. Phase C entry criterion: verify ColPali resolution/patch count sustains ≤500GB indexed at C12 target. §7.11 visual-index stability contract pins ColPali model version + render_dpi per visual collection (changing either requires full re-embed; cost documented). **Growth-trajectory mitigation:** any deployment expected to exceed 2000 docs must re-budget visual storage as a Phase C exit constraint before that growth occurs (model upgrades from ColPali → ColQwen3 at 5000-doc scale cost ~83h omlx wall-time per full re-embed). |
| R5 | Visual retrieval degrades text-heavy doc recall via fusion noise | **Medium** | Medium | Profile-conditional fusion weights with non-zero floor (visual=0.1 for PROSE) — *priors, not measurements*. Mandated sweep on critical path in Phase C (C5). C-spike measures impact on text-heavy docs. Bounded page→chunk join (top-N per page, §3.4 #4). |
| R6 | ColPali/omlx memory conflict with existing Qwen3-Embedding-8B | **Medium** | Medium | Phase C-spike validates co-residency. §7.7 tenancy policy specifies request scheduling on shared omlx server. |
| R7 | Edit-distance guard misses semantic corruption under token budget | **High** | Medium | Numeric/entity extraction, code-span hashing, order-preservation, *and* entity-relation triple check (§3.3 guard #7). Golden-output regression tests per guard. |
| R8 | MinerU AGPL-3.0 license incompatible with project distribution | **Medium** | Low (deferred) | Containerize as isolated service if adopted; keep Docling as fallback. |
| R9 | VLM-native parsing changes chunk shape → invalidates regression fixtures | **High** | Low (deferred) | Separate cycle after UIR stabilizes; maintain semantic-identity gate. |
| R10 | LLM sanitization non-deterministic → builds not reproducible | **High** | High | Content-pinning cache (§7.4): deterministic key. Heuristic/off modes byte-stable. Build-reproducibility CI test. |
| R11 | Prompt injection via document content in sanitization path | **Medium** | Low | XML-style content/prompt boundary delimiters (§3.3 guard #6). Input-length cap per chunk. |
| R12 | Solo-dev cycle overrun on Phase B (prompt engineering unbounded) | **High** | High | Phase B estimated 18–22 days. Dual-write means heuristics protect quality. Calendar overlay (§4.1) shows B overlaps C with whole-day-block switching rule. |
| R13 | **Phase A "semantic-identity" gate misses material regressions because the gate's fixture is derived from the very v2.16 behavior the refactor reproduces** (**upgraded from Medium to High since Draft 0.3 in light of repo scope evidence — ~17,000 lines of HybridChunker-aware code in `processor.py` + `batch_processor.py`, plus `mapper.py` + `docling_serializers.py`**) | **High** | Medium | Identity-half + explained-delta-half split (§3.2). Third-party regression check: at least three previously-broken-then-fixed v2.10–v2.16 cases (Earthship reading-order, Harry Potter drop-cap, Fluent_Python code-indent) must produce identical or documented-as-improved output. Per-doc spike on `ATZ_Elektronik_German` before cycle commit (§4.1). |
| R14 | LLM sanitization test infrastructure underspecified → guard layers untestable | **High** | Medium | Phase B B7: property-based tests per guard, mocked-endpoint integration tests, golden-output regression for prompt versions (§4 Phase B). |
| R15 | **`chunk_id` is position-derived** (`f"{doc_id}_{page:03d}_{modality}_{md5_hash[:8]}"`, `element_processor.py:823`); any element-ordering, page-reassignment, or modality-widening change silently breaks downstream `chunk_id`-keyed joins | **High** | High | C13 constraint. §7.1 publishes a chunk_id rewrite map per phase or freezes derivation. Phase A A8: chunk_id stability ratio reported, with rewrite map for all changed IDs. |
| R16 | omlx tenancy unspecified — ColPali + Qwen3-Embedding-8B + ModernBERT compete for the same endpoint at query time | **High** | Medium | §7.7 tenancy & scheduling policy (preemption rules, queue model, latency budget per leg). C-spike validates Q5 holds under simulated concurrent load. |
| R17 | **Phase A 12-day budget is unrealistic given the chunker/mapper/serializer rewrite scope** | **High** | High | Rebudgeted to 24 days (2× nominal). Per-doc spike (A0) validates the budget on real refactoring evidence before cycle commitment. R12 calendar updated. |
| R18 | Phase D needs 8 holdout docs across 4 rubrics; corpus currently has 1 known FORM holdout (CarOK) and uncertain TABLE/DIAGRAM coverage — holdouts must be acquired or carved | **Medium** | High | Holdout acquisition plan in §3.5: 5 of 8 holdouts must be sourced/identified before Phase A start; remaining 3 by mid-Phase C. Carve-from-corpus is acceptable only if the doc was never used to tune a heuristic. |
| R19 | **Sanitizer-judge correlated failure mode** — LLM sanitizer (Qwen2.5-14B-FP8) and dominance judge (Qwen3.6-FP8 family) share architecture; bias against catching sanitizer's own failure modes is uncharacterized | **High** | Medium | 50-chunk human-labeled golden set added to dominance criterion (§3.3). Cross-family judge spot-check on a 25-chunk sample (e.g., Llama-3.1 or Mistral via cloud, accepting one-off cost). |
| R20 | Soak σ has never been measured on this corpus; Phase B dominance criterion gates on σ but σ is not yet a known quantity | **Medium** | Certain | Phase A A7: three consecutive heuristic-only soak runs in Phase A's trailing days to establish σ per axis before Phase B starts. Adds ~6h soak runtime to Phase A. |
| R21 | **Docling 2.86.0 lifecycle risk (NEW in 0.5 — addresses 0.4 audit A3 finding #3)** — `docling` is exact-pinned to `2.86.0`. Upstream Docling is under active development (2.66 → 2.86 within this project's lifetime). A critical CVE, dropped Python-3.10 support in Docling 3.x, or breaking upstream-dependency changes could force an upgrade that changes extraction behavior. Without the UIR abstraction in place, this becomes a forced uncontrolled migration. | **Medium** | Medium | (1) Phase A UIR refactor makes Docling upgrades a controlled adapter-boundary change rather than a corpus-wide ripple. (2) Quarterly Docling lifecycle check (responsibility re-homed under V3 governance after the v2.X cycle-open checklist was archived): is 2.86 still receiving security patches? does Docling 3.x require Python 3.11+? (3) Documented escalation: if 2.86 security-patch flow stops AND no UIR-clean upgrade path exists within 1 quarter, the pin is renegotiated under a dedicated cycle plan. |
| R22 | **Phase A 24-day budget could still overrun by 50–100% (audit D15 finding)** | **High** | Medium | Phase A scope-negotiation protocol (Phase A description) defines triggers: A0 >4 days OR first 5 days of A2 <20% reconciliation-path progress OR semantic-identity explained-delta >5%. UIR-shim fallback option preserves the UIR contract while reducing refactor surface by ~50%. Negotiation outcome recorded in `docs/PHASE_A_SCOPE_NEGOTIATION.md` before Phase A continues; the 24-day budget is not silently extended. |

---

## 3. V3.0 Target Architecture

### 3.1 Architecture Overview

V3.0 is organized as two peer subsystems (Ingestion Pipeline, Retrieval Backend) with a sidecar Quality & Evaluation track. **The Ingestion Pipeline forks at Stage 2 into a dual-write LLM path and a Heuristic path that converge into the chunker** (the dual-write window persists through three production cycles per §6.2; the diagram below reflects the dual-write topology rather than the deprecated-heuristic end-state).

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
│  │  └─────────┬─────────────────┘  │    │  ┌───────────────────────────┐  │  │
│  │            │                    │    │  │ Visual Retrieval (NEW)    │  │  │
│  │            ▼                    │    │  │ ├─ ColPali/ColQwen2.5     │  │  │
│  │  Stage 2 (dual-write):          │    │  │ ├─ Qdrant multi-vector    │  │  │
│  │  ┌───────────────────────────┐  │    │  │ └─ MaxSim → top-N per pg  │  │  │
│  │  │ ┌──────────┐ ┌──────────┐ │  │    │  └────────────┬──────────────┘  │  │
│  │  │ │   LLM    │ │Heuristic │ │  │    │               │                 │  │
│  │  │ │Sanitizer │ │ Stack    │ │  │    │               ▼                 │  │
│  │  │ │ (GX10)   │ │(retained)│ │  │    │  ┌───────────────────────────┐  │  │
│  │  │ └────┬─────┘ └────┬─────┘ │  │    │  │ Fusion + Rerank           │  │  │
│  │  │      │            │       │  │    │  │ ├─ Profile-cond. RRF      │  │  │
│  │  │      ▼            ▼       │  │    │  │ │  (weights = priors;     │  │  │
│  │  │ ┌──────────┐ ┌──────────┐ │  │    │  │ │   swept in C5)          │  │  │
│  │  │ │ 8-layer  │ │  (n/a)   │ │  │    │  │ ├─ Leg-skip re-norm.      │  │  │
│  │  │ │ guards   │ │          │ │  │    │  │ └─ ModernBERT rerank      │  │  │
│  │  │ └────┬─────┘ └────┬─────┘ │  │    │  │   + signal-preservation   │  │  │
│  │  │      │            │       │  │    │  │     audit (§3.4)          │  │  │
│  │  │      └─────┬──────┘       │  │    │  └───────────────────────────┘  │  │
│  │  │            ▼              │  │    │                                 │  │
│  │  │   both-and-diff (active)  │  │    │                                 │  │
│  │  │   → chosen output → chunk │  │    │                                 │  │
│  │  └───────────────────────────┘  │    │                                 │  │
│  │                                 │    │                                 │  │
│  └─────────────────────────────────┘    └─────────────────────────────────┘  │
│                                                                              │
│  ┌─────────────────────────────────┐                                         │
│  │  EXTRACTION ENGINES (pluggable) │                                         │
│  │  PDF (Docling) | EPUB | Future  │                                         │
│  └─────────────────────────────────┘                                         │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │        QUALITY & EVALUATION (Sidecar — cross-cuts both subsystems)   │    │
│  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────┐  │    │
│  │  │ Modality-Aware Gates│  │ Synthetic Soak      │  │ Strict Gate  │  │    │
│  │  │ PROSE | FORM | TABLE│  │ + Visual Relevance  │  │ FORM/TABLE/  │  │    │
│  │  │ DIAGRAM rubrics     │  │ + Spatial Fidelity  │  │ DIAGRAM lanes│  │    │
│  │  │ + golden-set check  │  │ + signal preserve   │  │              │  │    │
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
- **Stage 2 has no single output during the dual-write window.** Both branches produce chunks; `both-and-diff` adjudicates per the diff predicate, and the chosen output is what the chunker consumes. Until the heuristic stack is deprecated (three production cycles after Phase B + dominance criterion), the diagram's Stage-2 "chosen output" is the LLM output gated by guards on LLM-mode runs, the heuristic output on heuristic-mode runs, and an enumerated diff record on `both-and-diff` runs.

### 3.2 Ingestion Pipeline Stage 1: True Universal Intermediate Representation (UIR Refactor)

**Directly addresses:** Limit 3 (Chunker-DOM Coupling), v2.15 Item #13 (parked)
**Governance anchor:** `docs/DECISIONS.md` — "Combined ceiling: UIR refactor + LLM cleanup together" (v2.10 chunker-quality ceiling); "v2.11 Carry-Forward §3c — PAUSED for user signoff"

**Current-state diagnosis (corrected from Draft 0.3):**

- **Docling option construction is already unified.** Since 2026-04-30 (`DECISIONS.md` v2.7 §5 close), `src/mmrag_v2/engines/docling_adapter.py` is the only site that constructs `PdfPipelineOptions` and `DocumentConverter`. The static guard `tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter` enforces this. Any v3.0 prose that asserts "PdfConversionPlan, BatchProcessor, and DoclingPdfAdapter each construct Docling options independently" is two cycles stale.
- **The real coupling is downstream of the adapter.** A grep for `DoclingDocument` and `from docling_core` shows:
  - `src/mmrag_v2/mapper.py` — consumes `DoclingDocument` directly to map to `IngestionChunk`.
  - `src/mmrag_v2/engines/docling_serializers.py` — extends Docling's serializer (`HierarchicalChunker`, `BaseDocSerializer`); operates on `DoclingDocument` shapes.
  - `src/mmrag_v2/processor.py:3331` — imports `HybridChunker`; line 3395 instantiates and invokes it.
  - `src/mmrag_v2/batch_processor.py` — ~12 distinct code paths that reconcile HybridChunker output (TOC heading propagation 5797, page-split sibling fill 5863, dual-DocChunk dedup 6686, scan-origin chunking bypass 3064, OCR-driven heading override 5971, etc.).
- **`partial_code` already emits for the in-block case** (oversized code chunks within a single page; `processor.py:5002` and `retrieval/pipeline.py:516–517`). What is *inert* is the cross-page-split case — the flag is never set for code blocks that span page boundaries, because HybridChunker cannot reliably emit cross-page state from the DOM. The Phase A acceptance gate at §3.2 must be precise about this distinction.

**Design:**

V3.0 elevates the UIR to the **single source of truth between extraction and chunking:**

1. **Format-Agnostic `ConversionPlan`:** `PdfConversionPlan` is elevated to a parent `ConversionPlan` class. All format-specific adapters (PDF, EPUB, future DOCX/HTML) produce a `ConversionPlan` that the pipeline consumes uniformly. `ConversionPlan.engine_options: Dict[str, Any]` carries an opaque blob for engine-specific flags (`do_code_enrichment`, `do_picture_classification`, OCR engine choice, etc.) so that pinned Docling toggles do not require a new typed plan field per option.

2. **Extraction Engines as Dumb Pipes:** `PDFEngine`, `EpubEngine` (and future engines) output a standardized `UniversalDocument` schema. They do NOT construct Docling options independently — the shared adapter/factory is the single construction site (preserving the v2.8 `test_no_raw_converter_invocation_outside_adapter` guard).

3. **Decoupled chunker call sites:** The chunker (and `mapper.py`, and the serializer extension) operate solely on `UniversalDocument`/`UIRChunk`, entirely detached from Docling's `DoclingDocument` layout classes. A runtime guard test asserts that no module under `src/mmrag_v2/chunking/`, `src/mmrag_v2/universal/`, or `src/mmrag_v2/sanitization/` imports from `docling_core.types.doc.document` or `docling.datamodel.document`. This enables:
   - Holistic cross-page boundary state (code blocks, tables, sections spanning pages)
   - **Reliable `partial_code` cross-page-split flag emission** (activating the inert v2.16 cross-page path of adjacency fetch — the in-block path already works)
   - Single cleanup site for all format defects (not per-profile branches)

**Interface contract (Python 3.10 dataclasses):**

```python
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Set, Literal
from enum import Enum

class Modality(Enum):
    """V3.0 UIR modality. Replaces today's ElementType (TEXT/IMAGE/TABLE)."""
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

class CoordinateFrame(Enum):
    """The frame `bbox` values are expressed in. [0,1000] is a normalization,
    not a frame — the frame says what was normalized."""
    PDF_PAGE_PORTRAIT = "pdf_page_portrait"
    PDF_PAGE_LANDSCAPE = "pdf_page_landscape"
    PDF_PAGE_ROTATED = "pdf_page_rotated"  # Page rotation handled at extraction
    IMAGE_NATIVE = "image_native"          # Scanned image native pixel frame
    EPUB_VIEWPORT = "epub_viewport"        # Reflowable; bbox at default viewport
    UNKNOWN = "unknown"

class StructuralFlag(Enum):
    """Closed vocabulary of structural flags. Open-ended Dict[str, bool]
    is forbidden — the semantic-identity gate ('flags additive') requires
    a canonical registry to diff against."""
    PARTIAL_CODE_IN_BLOCK = "partial_code_in_block"     # v2.16 in-block oversized
    PARTIAL_CODE_CROSS_PAGE = "partial_code_cross_page" # v3.0 new (activates inert)
    PARTIAL_TABLE_CROSS_PAGE = "partial_table_cross_page"  # v3.1+ deferred
    CROSS_PAGE_SPLIT = "cross_page_split"
    ORPHAN_LABEL = "orphan_label"
    PICTURE_CLASSIFICATION_LABEL = "picture_classification_label"
    DROP_CAP_REPAIRED = "drop_cap_repaired"
    READING_ORDER_REPAIRED = "reading_order_repaired"
    OCR_FORCED = "ocr_forced"
    OCR_FALLBACK = "ocr_fallback"
    BBOX_IOU_DEDUPED = "bbox_iou_deduped"
    TRAILING_PREPOSITION_HEALED = "trailing_preposition_healed"
    # Extend via ADR — additions are flags-additive per §3.2 gate.

@dataclass
class Locator:
    """Source-document location, format-appropriate."""
    type: LocatorType
    # For BBOX type:
    bbox: Optional[List[int]] = None         # [x1, y1, x2, y2] in [0, 1000]
    page_number: Optional[int] = None
    coordinate_frame: CoordinateFrame = CoordinateFrame.UNKNOWN
    # For FLOW_OFFSET / DOM_PATH types:
    path: Optional[str] = None               # CFI, DOM XPath, or character offset range

@dataclass
class ExtractionWarning:
    """Structured signal from the extraction engine that the chunker /
    sanitizer / quality gate may need to consult. Replaces today's
    DOM-coupled inspection of Docling internals."""
    code: str          # e.g. "DOCLING_OCR_LOW_CONFIDENCE", "PAGE_ROTATION_CORRECTED"
    severity: Literal["info", "warn", "error"]
    message: str
    source_element_id: Optional[str] = None

@dataclass
class ConfidenceBreakdown:
    """Per-source confidence scores. Single sentinel convention: a field that
    is None means 'not measured for this chunk'. Whether a field is
    *applicable* to this chunk is recorded separately in `applicable` to
    avoid the prior 0.3 two-sentinel encoding (None vs -1.0)."""
    layout_confidence: Optional[float] = None
    text_extraction_confidence: Optional[float] = None
    ocr_confidence: Optional[float] = None
    classification_confidence: Optional[float] = None
    applicable: Set[str] = field(default_factory=set)
    # `applicable` contains the field names that are semantically relevant
    # for this chunk's modality + extraction path. A field that is in
    # `applicable` but has value None = "applicable but unavailable" —
    # request a re-extraction or treat as missing data.

@dataclass
class UniversalPage:
    """Page-level UIR. Note `page_size_px` for visual retrieval reprojection."""
    page_number: int
    page_size_px: Optional[Tuple[int, int]] = None  # (width, height) at render_dpi
    page_size_pt: Optional[Tuple[float, float]] = None  # PDF native points
    classification: Optional[str] = None             # digital/scanned/hybrid
    rotation: int = 0                                # 0/90/180/270 after correction
    elements: List["UIRChunk"] = field(default_factory=list)
    warnings: List[ExtractionWarning] = field(default_factory=list)

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
        debugging.
      - `sanitization_status` is the single source of truth for whether
        sanitization was applied to this chunk.
    """
    modality: Modality
    content: str                          # Always authoritative
    locator: Locator
    confidence: ConfidenceBreakdown
    extraction_method: str                # "docling_direct" | "ocr_tesseract" | "ocr_doctr" | "vlm_enrichment"
    extraction_engine_version: str        # e.g., "docling-2.86.0"
    extraction_warnings: List[ExtractionWarning] = field(default_factory=list)
    structural_flags: Set[StructuralFlag] = field(default_factory=set)  # Typed; closed vocabulary
    source_element_ids: List[str] = field(default_factory=list)         # Traceability back to extraction engine
    asset_ref: Optional[str] = None       # Path to extracted image/asset, if IMAGE modality
    lang: Optional[str] = None            # ISO 639-1 language code
    reading_order: Optional[int] = None   # Monotonic logical position within page/document

    # Provenance fields:
    content_original: Optional[str] = None              # See docstring
    content_sanitized: Optional[str] = None             # If sanitization was attempted
    sanitizer_model_id: Optional[str] = None            # e.g., "RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic"
    sanitizer_prompt_version: Optional[str] = None      # Git-hash of prompt template
    sanitization_status: str = "not_applied"            # "not_applied" | "accepted" | "rejected:*" | "skipped:*"

    # Hierarchical context:
    parent_element_id: Optional[str] = None  # Table cell → parent table; caption → parent figure
    parent_heading: Optional[str] = None     # Nearest ancestor heading text

    # Cross-page sibling grouping (NEW in 0.5 — addresses 0.4 audit A2 finding #1):
    #   Shared UUID across chunks that are halves/parts of the same logical element
    #   split across pages (cross-page code block, cross-page table). Required for the
    #   adjacency-fetch mechanism to identify which chunk is the continuation, not
    #   just that some continuation exists. None when the chunk is not part of a split.
    continuation_group_id: Optional[str] = None

    # UIR internal contract version (NEW in 0.5 — addresses 0.4 audit A2 finding #5):
    #   Distinct from schema_version (which stamps the output JSONL). uir_version
    #   governs the internal extraction→chunking contract so the ElementProcessor
    #   knows which UIR generation it is consuming. Bumped on additive field changes
    #   to UIRChunk/UniversalDocument/UniversalPage.
    uir_version: str = "3.0"

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
    # NEW in 0.5 (0.4 audit A2 finding #4): validation range [72, 600]. ColPali patch
    # count varies with render_dpi × page dimensions; the visual collection is pinned
    # to a single render_dpi value per §7.11 visual-index stability contract. Changing
    # render_dpi requires a full visual re-embed. C-spike measures at both 200 and 300
    # to confirm PASS conditions are not DPI-lucky.
    render_dpi: int = 200                 # Resolution for page renders consumed by visual retrieval
    lang_hint: Optional[str] = None       # ISO 639-1, when known a priori
    engine_options: Dict[str, Any] = field(default_factory=dict)  # Opaque per-engine blob
    # ... format-specific fields in subclasses

    def __post_init__(self):
        if not (72 <= self.render_dpi <= 600):
            raise ValueError(f"render_dpi must be in [72, 600]; got {self.render_dpi}")
```

**Deferred UIR fields (audit A2 findings deferred to v3.1 or v3.2, not blocking v3.0):**

- `table_structure: Optional[TableStructure]` — row/column counts, header-row flag, cell bboxes for TABLE-modality chunks. Deferred: Phase D TABLE rubric can re-parse content for `num_rows`/`num_cols` at evaluation time without blocking v3.0; first-class field is an optimization. Tracked as USER_ISSUES entry; revisit when Phase D rubric implementation begins.
- `lang_confidence: Optional[Dict[str, float]]` — multi-language confidence map per chunk for multilingual documents (German body + English labels + Latin math notation). Deferred: v3.0–v3.1 sanitization uses document-level `lang` from `ConversionPlan.lang_hint`; per-chunk language confidence becomes load-bearing only when sanitization prompts are language-conditional (planned v3.2). fasttext lid.176.bin integration ~10µs/chunk when implemented.

**EPUB engine — interface-ready, implementation-void in v3.0 (NEW in 0.5 — addresses 0.4 audit C11 finding #1):**

`ConversionPlan.file_type` accepts `"epub"` and `CoordinateFrame.FLOW_OFFSET` / `EPUB_VIEWPORT` exist in the typed enum. **No `epub_engine.py` ships in v3.0.** All v3.0 production runs have `file_type == "pdf"`; EPUB-specific fields on `ConversionPlan` and `Locator` are `None`/unset. The interface is reserved so that a post-v3.0 EPUB cycle does not require an additional `ConversionPlan` parent-class refactor. Downstream consumers must treat the EPUB-only fields as forward-compatible reservations, not contracts.

**Acceptance gate (Phase A) — Semantic-identity, in two halves:**

The gate splits to acknowledge that Phase A intentionally activates the inert `partial_code` cross-page case (a behavioral change, not pure refactoring). Pure refactoring is not the goal; controlled-delta refactoring is.

**Identity half (≥95% of v2.16 chunks):**

1. **Stable chunk-identity key:** `(doc_id, page_number, content_hash_prefix)` where `content_hash_prefix` = first 64 bits of SHA-256 of content (16 hex chars). Unicode NFC, internal whitespace collapse (consecutive whitespace → single space), and line-ending normalization (CRLF → LF) are applied **before** hashing, making the key robust to platform differences.
2. **Matching policy:**
   - If v3.0 chunk has same identity key as a v2.16 chunk → content must match (modulo trailing whitespace), `chunk_type` must match per the §7.1 ElementType→Modality→ChunkType reconciliation table.
   - If v2.16 chunk A becomes v3.0 chunks B1, B2 (split) → `normalize(A.content)` must equal `normalize(concat_with_separator(B1.content, B2.content))` where `normalize` applies NFC + whitespace collapse + CRLF→LF (the same normalization used for hashing). `chunk_type` of all three must match; B1/B2 each carry `CROSS_PAGE_SPLIT` or equivalent structural flag.
   - If v2.16 chunks A1, A2 become v3.0 chunk B (merge) → `normalize(concat_with_separator(A1.content, A2.content))` must equal `normalize(B.content)`.
   - No v2.16 chunk may have zero corresponding v3.0 chunks (dropped content = FAIL).
3. **Structural flags strictly additive** — no v2.16 flag goes missing; new flags from the `StructuralFlag` enum may appear.
4. **Retrieval invariant** — top-5 doc IDs unchanged for every query in the v2.16 regression fixture.

**Explained-delta half (≤5% of v2.16 chunks, every delta enumerated):**

Each delta must appear as a row in `docs/PHASE_A_INTENTIONAL_DELTAS.md` with:

| Required column | Description |
|---|---|
| v2.16 chunk_id (or batch identifier) | Source row(s) in the v2.16 baseline |
| v3.0 chunk_id (or batch identifier) | Target row(s) in the Phase A output |
| Delta type | One of: `cross_page_partial_code_repair`, `reading_order_correction`, `flag_addition`, `chunk_id_re-derivation`, … |
| v2.X documented-defect cross-reference | Pointer into `DECISIONS.md`, `USER_ISSUES.md`, or a diagnostic doc that justifies the delta |
| Affected doc(s) | Which corpus document(s) exhibit this delta |
| Reviewer sign-off | Per the user's "no gate weakening" policy, every intentional delta requires explicit sign-off |

Empty `PHASE_A_INTENTIONAL_DELTAS.md` is fine (means Phase A produced no behavioral changes — which would mean the inert cross-page `partial_code` repair did *not* trigger anywhere, also possible). What is forbidden: unenumerated deltas. CI tooling must compare the two halves automatically and fail the build if a delta appears that is not enumerated in the markdown.

**Third-party regression check (added in 0.4):**

Phase A's gate is at risk of confirming the v2.16 implementation's defects because the test fixture is derived from v2.16's own output. To mitigate, the gate also requires byte-identical output (or documented-as-improved output) for at least three previously-broken-then-fixed cases drawn from v2.10–v2.16:

| Case | Origin cycle | What it tests |
|---|---|---|
| `Earthship_Vol1` forced full-page OCR | v2.11 | Forced-OCR pathway routes through UIR correctly |
| `Harry_Potter` drop-cap promotion | v2.10 | Reading-order y-sort-with-dropcap heuristic invokes via UIR or its UIR-native equivalent |
| `Fluent_Python` cross-page code-block attribution | v2.16 (partially fixed; in-block only) | `partial_code_cross_page` flag now emits AND chunk is correctly attributed to its starting page |

If any of the three regresses, Phase A FAILs and does not advance to merge.

### 3.3 Ingestion Pipeline Stage 2: LLM-Native Chunk Sanitization

**Directly addresses:** Limit 2 (Heuristic Patching Ceiling)
**Governance anchor:** `docs/DECISIONS.md` — "Combined ceiling: UIR refactor + LLM cleanup together" (v2.10 chunker-quality ceiling, path 4: "LLM-clean every chunk on ingestion" estimated at $30/rebuild on Dashscope, now $0 via local GX10)

**Design:**

After Stage 1 ensures all chunks flow through the UIR, Stage 2 adds a **semantic polish pass** using the existing local LLM endpoint:

1. **Execution:** Raw extraction chunks are piped through `RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic` at `10.0.10.239:8000` (GX10 endpoint, already operational per `docs/DECISIONS.md` v2.14).

2. **Content-pinning cache:** LLM calls are cached via a deterministic key: `(content_hash, context_hash, model_id, prompt_version) → sanitized_output`, where `context_hash = SHA-256(prev_chunk_content_first64bits + next_chunk_content_first64bits + detected_lang)`. Including context prevents stale cache hits when neighbor chunks change between rebuilds. On cache hit, the LLM is not invoked. Cache is file-backed under `output/sanitization_cache/`, keyed by content hash prefix + first 8 chars of context hash. See §7.4 for the full determinism policy.

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

5. **Heuristics retained — NOT ripped out.** Phase B does NOT remove `docling_postprocess.py` or the multimodal validation layers. The heuristic stack remains operational and is the fallback path. See §6.2 for the strengthened deprecation trigger (three *production* cycles, not soak iterations on the same corpus).

6. **Cost:** $0 per rebuild (local GX10 FP8 endpoint). No cloud API calls on the sanitization path.

**Multi-layer guard stack (8 guards, defense-in-depth against hallucination):**

| # | Guard | What it catches | Failure mode if absent |
|---|---|---|---|
| 1 | **Edit-distance ceiling** (>30% token change → reject) | Gross rewrites, fabrications | — |
| 2 | **Numeric/entity preservation** (all numbers, dates, identifiers, named entities must appear verbatim in sanitized output) | "100 mg" → "10 mg"; date changes; ID swaps | Subtle factual corruption within budget |
| 3 | **Code-span hashing** (text inside ``` fences must be byte-identical or rejected) | LLM "fixing" code syntax; reordering statements | Silent code corruption |
| 4 | **Order-preservation check** (regex-identified ordered-list markers must appear in same sequence) | LLM reordering procedural steps, recipes, algorithms | Reordered instructions |
| 5 | **Token-level alignment** (Levenshtein distance, not just count delta) | Reorderings that preserve token count | Edit-distance ceiling blind spot |
| 6 | **Content/prompt boundary delimiters** (XML-style tags separating instructions from document content; input-length cap) | Prompt injection via document text | Adversarial document rewrites its neighbors |
| 7 | **Entity-relation triple preservation (NEW in 0.4)** — extract `(subject, predicate, object)` triples via spaCy dependency parse before and after; any *added* triple in the sanitized output is rejected as a hallucination signal. Removed triples are flagged as warnings (LLM may have removed an extraction artifact masquerading as a relation), not auto-rejected. | Paraphrasing that preserves tokens, numerics, entities, and order but adds an implicit semantic relation (e.g., medical, legal restatement). Lightweight; runs in-process. | Semantic-equivalent factual *substitution* under all other guards |
| 8 | **Corpus-level dedup-ratio invariant (NEW in 0.4)** — runs once per build, not per chunk. Measures near-duplicate ratio (Jaccard ≥0.9 over shingles) across the sanitized corpus and compares against the heuristic corpus's ratio. If LLM-mode dedup-ratio exceeds heuristic by >5%, build emits a `SANITIZATION_DEDUP_DRIFT` warning and the dominance criterion treats this as a regression on Format. | Inter-chunk consistency drift: chunk A's heuristic-removed footer reappears as chunk B's prepended caption after LLM sanitization. Per-chunk guards cannot see this. | Silent inter-chunk content duplication that inflates retrieval candidate sets |

**Sentinel-chunk accounting:**

When a chunk's `sanitization_status` is `skipped:endpoint_unreachable` or `rejected:*`, the chunk's `content` is the raw extraction (heuristic-cleaned if `--sanitize-mode=both-and-diff` or heuristic-fallback) — it is *not* an LLM output. For dominance criterion arithmetic (§3.3 #1–3 below), such chunks are excluded from the LLM-mode score and counted toward the heuristic-mode score. Above 5% sentinel rate in any soak run, the run is marked `LLM_SENTINEL_DEGRADED` and does not count toward the two-consecutive-soak confirmation.

**Graceful degradation:** When GX10 endpoint is unreachable, the pipeline:
1. Logs `[SANITIZE_SKIPPED: endpoint unreachable]` sentinel per chunk.
2. Falls back to heuristic sanitization (dual-write retained per §4 Phase B).
3. Emits a build-level warning with unreachable-chunk count.
4. Does NOT hard-fail the build.

**LLM dominance criterion (replaces Draft 0.2's unattainable "strictly equal-or-better per chunk"; refined further in 0.4 to address sanitizer-judge correlated failure modes per R19):** Heuristic stack deprecation is gated on:

1. **Aggregate Format improvement:** LLM-sanitized corpus-wide Format exceeds heuristic output by ≥ 2× the soak run-to-run standard deviation (measured on the heuristic-only baseline across ≥3 consecutive soak runs — see Phase A A7 for σ baselining). This threshold is adaptive to actual measurement noise rather than an arbitrary fixed increment.
2. **No chunk-level cliff:** Zero chunks transition from PASS to FAIL under the strict gate when comparing LLM vs heuristic output.
3. **Degradation rate:** ≤2% of chunks where LLM output scores below heuristic output on Format axis.
4. **Consecutive confirmation:** All three conditions hold for two consecutive soak iterations (excluding any iteration marked `LLM_SENTINEL_DEGRADED`).
5. **Human-labeled golden set (NEW in 0.4):** A 50-chunk human-labeled set, where the operator has manually selected the "correct" sanitization between heuristic, LLM, and original (3-way), must show LLM ≥ heuristic by a wider absolute margin (≥5pp). Mitigates R19 sanitizer-judge correlated failure modes. The golden set is built once during Phase B B2 and never modified after — modifying it is gate-weakening per `DECISIONS.md`.
6. **Cross-family judge spot-check (NEW in 0.4):** A 25-chunk sample is re-judged by a non-Qwen-family LLM (Llama-3.1-70B via cloud or Mistral-Large via cloud, accepting the one-off ~$0.50 cost). On this sample, LLM-sanitized output must show no axis-level regression vs heuristic. Quantifies sanitizer-judge correlated-failure-mode bias.
7. **Retrieval-behavior regression test (NEW in 0.5 — addresses 0.4 audit B7 finding):** Run the v2.16 regression-fixture queries against both `--sanitize-mode=heuristic` and `--sanitize-mode=llm` outputs and assert top-5 doc-ID overlap ≥95% per query. The 8-layer guard stack catches token-, numeric-, entity-, relation-, and order-level alterations but cannot detect **omission-by-reframing** — paraphrases that preserve every measured quantity but shift retrieval behavior because the sanitized chunk now reads as a warning/instruction/condition rather than a statement of fact. Reframing drift is invisible to per-chunk guards (every guard passes per chunk) but visible at retrieval time as a change in which documents the retriever surfaces. Failure routes to `LLM_RETRIEVAL_DRIFT` and is treated as a Format regression for dominance arithmetic.

**Prompt-migration cost note (NEW in 0.5 — addresses 0.4 audit C11 finding #3):**

The content-pinning cache key (§7.4) includes `sanitizer_prompt_version` (git-hash of the prompt template). When the prompt template changes between any two patch versions (e.g., v3.1.0 → v3.1.1), every chunk previously cached becomes a cache miss, and the entire corpus must be re-sanitized at the B8 cold-cache cost. This makes prompt-template changes a load-bearing decision with re-build cost equivalent to a full Phase B rerun. The sanitization module's README documents this; the `sanitizer_prompt_version` field is checked in CI to fail builds where the prompt has changed without a corresponding `B8_COLD_CACHE_COST.md` update.

**Diff predicate for `both-and-diff` comparison:** Two sanitization outputs "differ" when their token-level Levenshtein distance exceeds 5% of the shorter output's token count. Whitespace-only differences and Unicode NFC/NFD normalization differences are excluded before comparison.

**Acceptance gate (Phase B):** Synthetic soak Format scores cross 95% corpus-wide with `--sanitize-mode=llm`, with no regression on Recall, Relevance, or Faithfulness axes vs `--sanitize-mode=heuristic`. LLM dominance criterion items 1–6 met (item 7 measured per B12 and treated as a Format regression for dominance arithmetic if <95% overlap; item 6 cross-family judge measured but does not block Phase B exit, it gates the §6.2 heuristic-deprecation trigger).

### 3.4 Retrieval Backend: Vision-Native Retrieval (ColPali + MaxSim)

**Directly addresses:** Limit 1 (Spatial-to-Text Translation Deficit, -12pp Recall@1)
**Governance anchor:** `docs/DECISIONS.md` — v2.16 Phase 2 diagnosis; `docs/PLAN_V2.16.md` Item #11 (ColPali/VisRAG scoped out)
**Independence:** Visual retrieval embeds rendered page images. It does NOT depend on UIR or LLM sanitization output. Phase C pre-spike + C-spike can and must run before Phase A implementation.

**Design:**

The v2.16 retrieval stack (dense + sparse + RRF + ModernBERT reranker) operates entirely on text embeddings. For text-heavy documents, this stack approaches 98.6% Recall@5 doc. For visually-complex documents (diagrams, multi-column forms, scanned engineering drawings), text embeddings discard the 2D spatial information that makes the content meaningful.

V3.0 introduces a **parallel visual vector index** that operates alongside the text retrieval stack:

1. **Patch-Level Visual Embeddings (ColPali/ColQwen2.5):**
   - Each page is rendered at `ConversionPlan.render_dpi` (default 200 DPI).
   - A Vision Transformer (ColPali, or its successor ColQwen2.5) divides the page into patches (e.g., 1030 patches per page at 448×448 resolution).
   - Each patch is embedded as a 128-dimensional vector.
   - A page is represented as a **matrix of patch vectors**, not a single flat embedding.

2. **Qdrant Multi-Vector Storage:**
   - Qdrant is configured to support multi-vector collections.
   - Each document page stores its patch-vector matrix as a separate point (page-level granularity).
   - Visual collection is independent of text collections — can be dropped/rebuilt without affecting text retrieval.

3. **MaxSim Late-Interaction Retrieval:**
   - At query time, the query is embedded through the same ColPali vision model.
   - MaxSim (Maximum Similarity) computes: for each query token, find its most similar document patch, then sum those maximum similarities.
   - Output: **page-level scores** (not chunk-level).

4. **Bounded Page → Chunk Join Policy (CORRECTED in 0.4):**
   Visual retrieval produces page-level scores; text retrieval produces chunk-level scores. RRF fuses ranked lists, so the visual page-score must be projected into a chunk-rank list to participate.
   - **For each visually-retrieved page in MaxSim top-K (K=25 by default), select the top-N chunks on that page by their text-leg score** (N=3 by default). The selected chunks inherit the visual page's rank (chunk rank = visual page rank, with ties broken by text-leg score).
   - This bounds visual leg's contribution: a 30-chunk page does not flood the candidate pool with 30 equally-ranked chunks.
   - **Draft 0.3's "page scores propagated to ALL chunks on that page" is corrected.** Broadcasting created candidate-set imbalance (large pages dominated the fused candidate set regardless of internal text relevance).
   - **C-spike PASS B (§4.2) now measures reranker top-1 selection rate on visually-retrieved pages under this bounded join.** If <60%, page-level granularity is insufficient AND region-level must be in Phase C scope.

5. **Profile-Conditional Fusion Weights — PRIORS, not measurements (CLARIFIED in 0.4):**

   | Profile class | Dense weight | Sparse weight | Visual weight | Rationale (prior) |
   |---|---|---|---|---|
   | PROSE (default) | 1.0 | 1.0 | **0.1** | Non-zero floor lets visual act as tiebreaker on embedded figures within prose docs; prevents waste of deployed visual infra |
   | DIAGRAM | 1.0 | 1.0 | **0.4** | Visual is primary signal for diagrams/schematics |
   | FORM | 1.0 | 1.0 | **0.4** | Visual captures form layout; text captures field values |
   | TABLE | 1.0 | 0.5 | **0.3** | Sparse still useful for exact numeric match; visual for structure |

   **These numbers are priors, not soak-measured optima.** Phase C task C5 (weight sweep on deficit subset) is on the critical path, not deferred. Sweep coverage (EXPANDED in 0.5 — addresses 0.4 audit A4 finding):

   - **Primary axis — visual weight per profile:** 4 profiles × {0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0} = 28 configurations with dense and sparse fixed at the table defaults.
   - **Secondary axis — dense:sparse ratio for the top-3 visual weights per profile:** {1:1, 2:1, 1:2} = 4 profiles × 3 visual weights × 3 ratios = 36 configurations. Justification: when visual retrieval shifts the candidate-set composition, the dense:sparse balance previously assumed optimal at v2.16 may no longer hold; fixing them at 1.0 begs the question.
   - **Tertiary axis — RRF k for the top-3 visual weights per profile:** k ∈ {30, 60, 100}, swept jointly with the dense:sparse top-3 grid above (k=60 is the v2.16 carry-forward but is unjustified for the 3-leg topology; lower k weights top positions more heavily, higher k smooths over ties at lower ranks).
   - **Total ~64 configurations on the deficit subset.** Soak runtime estimate: ~30 min/config × 64 ≈ 32 hours, parallelizable across soak workers, fits within the 12-day Phase C budget.
   - **Defaults in the table above are starting points for the sweep; the swept-optimal values become §6.2 fork-back candidates if they outperform priors by ≥5pp on any axis.**

   **Per-document weights are a known limitation.** A PROSE document may contain a diagram page, and per-document profile-conditional weights use the document's profile, not the page's actual content. Future work (v3.2+): per-query profile classification, or a query-time `--boost-visual` flag.

6. **Fusion Re-Normalization on Leg Skip:**
   When any retrieval leg is skipped (e.g., visual leg when ColPali is unreachable), remaining weights are L2-normalized to unit norm, preserving their relative proportions. Without re-normalization, scores from text-only fusion aren't comparable to scores when visual was present, which breaks any score-based threshold downstream. Example: PROSE weights `(1.0, 1.0, 0.1)` with visual skipped → remaining `(1.0, 1.0)` L2-normalized to `(0.707, 0.707)`. DIAGRAM weights `(1.0, 1.0, 0.4)` with visual skipped → also `(0.707, 0.707)`. All profiles converge to equal text-leg weights on leg skip — the profiles differ only in the presence of the visual leg.

7. **Visual-Signal Preservation Through Rerank (NEW in 0.4):**
   ModernBERT is text-only. When a chunk enters the top-25 primarily due to its visual score, the reranker evaluates `(query, chunk_content)` and may demote chunks that are visually relevant but textually thin (e.g., a chunk that is just "Fig. 3.7" near a transistor symbol). To detect this failure mode:
   - **Metric: fusion-vs-rerank top-1 flip rate** on the deficit subset. Compute per query: did the reranker top-1 differ from the fusion top-1? Of the flips, what fraction *improved* Recall@1 vs *degraded* it?
   - **Target: degradation flips ≤ improvement flips** (the reranker should not systematically undo visual signal).
   - **If degradation flips > improvement flips** on the deficit subset, downstream reranker work (text+visual-aware reranker, or visual-aware boost in fusion) becomes a Phase D / v3.2 follow-up.

**End-to-end retrieval flow (V3.0):**

```
query
  ├─ dense  : omlx Qwen3-Embedding-8B-mxfp8 (4096-dim)         ← retained (v2.13)
  ├─ sparse : BM25                                             ← retained (v2.12)
  └─ visual : ColPali/ColQwen2.5 → MaxSim (page scores)        ← NEW (v3.0)
           → bounded join: top-N text chunks per visually-
             retrieved page inherit visual rank
  → RRF fusion (k=60, profile-conditional weights = priors;
                leg-skip re-normalization; sweep on critical path)
  → ModernBERT rerank (top-25 → top-5)                         ← retained (v2.12)
  → fusion-vs-rerank flip-rate audit logged                    ← NEW (v3.0)
  → top-5 return
```

**Reranker behavior on visual hits:** ModernBERT is text-only. When a chunk enters the top-25 primarily due to its visual score, the reranker evaluates `(query, chunk_content)`. The visual signal has done its job (getting the page into the candidate set under the bounded join); the flip-rate audit (#7) measures whether the reranker preserves that signal.

**Resource budget (operational target: ≤1000 docs per C12):**
- Patch-vector storage: ~128 dimensions × 1030 patches × ~300,000 pages (at 1000 docs) = ~150 GB raw; ~400–500 GB indexed with HNSW overhead.
- Visual embedding latency: target <1s/page at omlx server inference speed.
- MaxSim latency: O(query_tokens × doc_patches) for exhaustive; Qdrant ANN caps at configurable ef_search.

**Phase C entry criterion:** Before Phase C full build, verify ColPali resolution/patch count sustains ≤500GB indexed storage at the C12 target (1000 docs). If not, reduce patch count or page resolution and re-validate via C-spike methodology.

**Phase C pre-spike (2 hours, before Phase A code) + C-spike (2–3 days, before Phase A code):** See §4.2.

**Acceptance gate (Phase C):** Recall@1 chunk ≥80% on deficit docs (from 67.8% **as re-measured on v3.1.0 chunks WITHOUT visual retrieval** — see §7.1), Recall@5 doc ≥98.6% maintained. Text-heavy docs (29/34) show no regression. Reranker top-1 selection rate on visually-retrieved pages ≥60%. Fusion-vs-rerank degradation flips ≤ improvement flips on deficit subset.

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

5. **Holdout policy and acquisition plan (EXPANDED in 0.4):**

   Minimum 8 blind-test documents — 2 per rubric (PROSE, FORM, TABLE, DIAGRAM). The Greenhouse document is the PROSE holdout. Effective at v3.0 final tag. Rubric-to-holdout mapping documented in `docs/QUALITY_GATES.md` before Phase D.

   **Acquisition state and plan (R18):**

   | Rubric | Current holdouts known | Need | Source plan |
   |---|---|---|---|
   | PROSE | Greenhouse | 1 more | Carve from a 34-doc corpus document never used to tune a heuristic — candidate: an unedited Project Gutenberg text not yet in corpus, acquired during Phase A. |
   | FORM | CarOK_voorraadtelling | 1 more | Acquire a second Dutch or English form (e.g., government tax form, application form) during Phase A. |
   | TABLE | None confirmed | 2 | Acquire from open data: a CSV-derived PDF report and a financial-statement PDF, during Phase A or early Phase B. |
   | DIAGRAM | None confirmed | 2 | Acquire from open access engineering literature (IEEE OA, IRJET back-catalogue) during Phase B. Must not overlap the ATZ/IRJET docs already in corpus. |

   **Constraint:** carve-from-corpus is acceptable only if the carved document was never referenced in tuning a heuristic, never appeared as a v2.X regression fixture, and was not used as a v2.X diagnostic source. This eliminates 6 of the 34 corpus docs. The remaining 28 are eligible candidates if and only if a holdout cannot be sourced externally within the schedule.

   5 of 8 holdouts must be sourced/identified before Phase A start; the remaining 3 by mid-Phase C. Slipping this means Phase D cannot exit on schedule — R18 escalation trigger.

6. **Two documents per rubric is the minimum** to guard against overfitting four modality rubrics on a 34-doc corpus.

**Acceptance gate (Phase D):** `CarOK_voorraadtelling` Format score no longer penalized for non-prose content shape. Modality breakdown visible in soak reports. No false-negative quality failures on structured data. All 8 holdout documents pass.

---

## 4. Implementation & Phasing Strategy

V3.0 is sequenced to respect solo-dev 12-day convergence cycles, with one exception: **Phase A is budgeted at 24 days (2× nominal) due to chunker/mapper/serializer rewrite scope (R17).** **Visual retrieval (Phase C) does not depend on UIR or LLM sanitization** — ColPali embeds rendered page images directly. The 2-hour Phase C pre-spike runs first, the full C-spike runs in parallel with Phase A.

### 4.1 Phase Dependency & Calendar Diagram

```
                     Calendar days ──────────────────────────────→
                     0      12     24     36     48     60     72

  C pre-spike (2h)
  C-spike (2-3d)
  ├──────────┐
              ├── Phase A: UIR (24d, 2× nominal per R17) ────────────┐
              │   D1 (rubrics) runs in parallel ──────┐               │
              │   σ baseline soak (last 6h) ──────────┐               │
              │                                       │               │
              │                                       └─ Phase B: LLM Sanitize ──┐
              │                                          (18-22d nominal)        │
              │                                          B1-B8: prompt eng,      │
              │                                          guards, dual-write,     │
              │                                          golden set, cross-judge │
              │                                                                  │
              └─────────────────── Phase C: Visual Retrieval ───────────────────┤
                                   (12d, gated on C-spike PASS)                  │
                                                                                 │
                                                              Phase D: Gates (12d)
                                                              D2-D4: strict gate ext,
                                                              full acceptance
```

**Calendar implications:**
- Phase A is now 24 days, not 12. R17 explicit; per-doc spike (A0) validates the budget on real refactoring evidence before cycle commitment.
- Phase B execution (18–22 days, R12) overlaps Phase C implementation (12 days) in calendar time.
- **Whole-day-block switching rule (NEW in 0.4):** Solo dev allocates whole days to B vs C, not interleaved hours. Half-day switches between LLM prompt iteration and visual-index debugging dominate calendar with task-switch overhead.
- **Front-load C pre-spike and C-spike.** Both complete before Phase A starts; C-spike's PASS/FAIL determines whether Phase C is even feasible before any Phase A code is written.
- **σ baseline soak runs (3 consecutive heuristic-only soaks, ~6h) are appended to Phase A's trailing days.** This unblocks Phase B's dominance criterion (which needs σ to compute its 2× threshold).

### 4.2 Phase C Pre-Spike + C-Spike (MANDATORY — before Phase A code)

**Step 1 — Pre-spike (2 hours, falsification test):**

A 2-hour sanity probe that runs before committing to the full C-spike:
1. Pick the **single most spatially-defective query** known from v2.16 diagnostics (a German circuit-diagram label query against `ATZ_Elektronik_German`).
2. Render the **gold page** and **3 plausibly-distractor pages** from the same doc at 200 DPI.
3. Run off-the-shelf ColPali (HF Spaces or local) inference: embed the 4 pages + the query, compute MaxSim, rank.
4. **PASS:** Gold page ranks first.
5. **FAIL:** Stop. ColPali doesn't see this signal on the most-favorable case; the full C-spike is dead weight.

No Qdrant, no omlx deployment, no fusion logic, no reranker simulation. 2 hours, workstation only.

**Step 2 — C-Spike (2–3 days, full quantitative test):**

If the pre-spike passes:

1. Pick the single highest-deficit doc — `ATZ_Elektronik_German`.
2. Render all pages at 200 DPI.
3. Embed pages with off-the-shelf ColPali on a workstation. **No omlx deployment. No Qdrant integration. No MaxSim in production code.**
4. Take 20 queries from the v2.16 regression fixture targeting this doc (or hand-craft if fixture coverage is thin).
5. Embed each query with ColPali. Compute MaxSim scores against page-vector matrices in raw numpy. Rank pages.
6. For each query, record: text-retrieval top-1 page (v2.16), visual-retrieval top-1 page (this experiment), gold page.
7. **PASS condition A:** Visual retrieval recovers the correct page on ≥60% of queries where v2.16 text retrieval failed, without harming queries where text retrieval was correct.
8. **PASS condition B (TIGHTENED in 0.4):** Reranker top-1 selection rate on visually-retrieved pages ≥60%. The simulated rerank must use:
   - The **exact ModernBERT model** that production uses (same model ID, same revision, same tokenizer).
   - The **exact production prompt** for the reranker.
   - **Candidate set construction:** for each query where visual retrieval placed the correct page in the top-5, construct a candidate set = (top-25 text chunks from the full corpus under v2.16 text retrieval) ∪ (the top-N=3 text chunks of the visually-retrieved page, selected by their text-leg score per §3.4 #4 bounded join). Deduplicate by **`chunk_id`** (not by content — content-based dedup masks the chunk_id-positional-derivation risk; see R15).
   - **Gold-chunk-on-gold-page mapping:** taken from the v2.16 retrieval regression fixture (`tests/fixtures/retrieval_regression_v2_X.json`), not constructed ad hoc for the spike.
   - If <60%, page-level granularity is insufficient and region-level must be in Phase C scope, not deferred.
9. Also verify: ColPali model fits alongside Qwen3-Embedding-8B on the omlx server (per §7.7); end-to-end latency <1s/page; co-residency does not push existing Qwen3-Embedding latency over Q5.

**Time budget:** 2 hours (pre-spike) + 2–3 days (C-spike). **Hardware:** workstation. **No production infrastructure touched.**

**Outcome:** If pre-spike PASS and C-spike PASS (both A and B) → Phase C implementation proceeds as designed. If C-spike PASS A but FAIL B → Phase C scope expands to include region-level granularity. If pre-spike or C-spike FAIL A → Phase C as designed is dead; redirect to VLM-native parsing evaluation or alternative visual model.

### 4.3 Partial-Release Policy

| Version | Ships independently? | Condition |
|---|---|---|
| `3.0.0` (Phase A) | **No** — held until `3.1.0` is also ready | UIR overhead without LLM-cleanup benefit would be a Format regression vs v2.16 heuristic stack. Shipping 3.0.0 alone = deploying the abstraction cost without the quality payoff. |
| `3.1.0` (Phase B) | **Yes** — with UIR already deployed | LLM sanitization is the primary Format-quality improvement. Ships alongside 3.0.0 to provide a complete v3.0 ingestion pipeline. |
| `3.2.0` (Phase C) | **Yes** — additive | Visual retrieval is independent of ingestion. Can ship later without affecting v3.1.0 pipeline output. Gated on C-spike PASS. |

### Phase A: UIR Foundation (Cycle 3.0 — 24 days, 2× nominal per R17)

**Scope-negotiation protocol (NEW in 0.5 — addresses 0.4 audit D15 finding):**

The 24-day budget is a working assumption, not a guarantee. The audit assesses Phase A as the most-likely-to-overrun phase (50–100% overrun risk) and demands an explicit negotiation trigger before the budget breaks silently:

| Trigger | Negotiation options (pick one before adding days) |
|---|---|
| **A0 exceeds 4 days** (per-doc spike on `ATZ_Elektronik_German`) | (a) defer ~1/3 of `batch_processor.py` reconciliation paths to v3.0.1, (b) **fall back to UIR-shim mode** — convert `DoclingDocument` → `UniversalDocument` at the adapter boundary and keep downstream `processor.py`/`mapper.py`/`batch_processor.py` unchanged; the UIR interface contract ships without the full downstream rewrite, and the rewrite becomes a v3.0.2 cycle. UIR-shim accepts ~10% performance overhead from the translation layer in exchange for ~50% Phase A scope reduction. The shim is removed in a later cycle without re-bumping the schema version. |
| **First 5 days of A2 show <20% progress on the ~12 reconciliation paths** | Same options as above; the UIR-shim fallback is the preferred fallback because it preserves the C13/R15 chunk_id stability contract while reducing the refactor surface. |
| **Semantic-identity gate's explained-delta half exceeds 5%** (more chunks differ than the audit table can enumerate) | (a) widen explained-delta tolerance to ≤10% with explicit DECISIONS.md entry documenting the loosened gate, (b) split A2 across cycles (A2a in 3.0.0; A2b in 3.0.1). |

The negotiation is triggered automatically when any of the above conditions fires; the user is notified, options are presented, and a decision is recorded in `docs/PHASE_A_SCOPE_NEGOTIATION.md` before Phase A continues. The 24-day budget is not silently extended.

**Content-derived `chunk_id` (regret #4) as scope-negotiation option:** If the A0 spike reveals that the rewrite map churn is severe (>20% of chunks change `chunk_id`), the audit's recommendation to flip `chunk_id` derivation from positional to content-based becomes a Phase A scope option, not a v3.3 deferral. Content-derived `chunk_id = f"{doc_id}_{sha256(content)[:12]}"` eliminates positional sensitivity entirely; the cost is that B-phase content changes (intentional) produce new IDs anyway. Decision recorded in scope-negotiation doc if invoked.

| Task | Description | Acceptance |
|---|---|---|
| A0 | **Per-doc spike on `ATZ_Elektronik_German` (3 days) — NEW in 0.4** | Refactor proves out on one doc; semantic-identity gate passes on this doc alone (both halves); intentional deltas list ≤30 lines, OR Phase A is renegotiated per protocol above. |
| A1 | Elevate `PdfConversionPlan` to parent `ConversionPlan` (incl. `engine_options`) | All existing tests pass without modification |
| A2 | Refactor extraction engines + `mapper.py` + `docling_serializers.py` + `processor.py` HybridChunker call site + `batch_processor.py` reconciliation paths to output/consume `UniversalDocument` and `UIRChunk` | Semantic-identity gate (§3.2): identity half (≥95% chunks match) + explained-delta half (≤5% chunks enumerated in `docs/PHASE_A_INTENTIONAL_DELTAS.md`) |
| A3 | Decouple chunker from `DoclingDocument` → operate on UIR; activate `partial_code_cross_page` flag | Cross-page code splits attributed to correct pages AND flag set; `Fluent_Python` test passes |
| A4 | Rip out duplicate Docling option construction sites (no-op — already unified, but re-affirm guard test covers v3.0 paths) | `test_no_raw_converter_invocation_outside_adapter` expanded to cover v3.0 paths; v3.0 chunker call sites verified DOM-free |
| A5 | Corpus-wide rebuild + strict gate + third-party regression check (Earthship, Harry Potter, Fluent_Python) | 34/34 PASS (or documented deferrals matching v2.16 baseline); 3 regression cases pass |
| A6 | Schema version: `2.7.0` → `3.0.0`; publish chunk_id rewrite map | All output carries `schema_version: "3.0.0"`; `docs/CHUNK_ID_REWRITE_MAP_3.0.0.csv` published per C13/R15 |
| A7 | **σ-baseline soak runs (3 consecutive heuristic-only, ~6h) — NEW in 0.4** | σ measured per axis; documented in `docs/PHASE_A_SIGMA_BASELINE.md` for Phase B dominance criterion |
| A8 | **17-skipped-tests audit (18 as of this draft per `pytest -v --co`) — NEW in 0.4** | Each skipped test classified: still-skip / re-enable-now / re-enable-post-A. Re-enabled tests pass before Phase A merge. |
| A9 | **5-of-8 holdout document acquisition (per §3.5 #5) — NEW in 0.4** | At least 5 holdouts identified/sourced; rubric-to-holdout draft committed |

### Phase B: LLM Sanitization (Cycle 3.1 — 18–22 days per R12)

| Task | Description | Acceptance |
|---|---|---|
| B1 | Wire GX10 FP8 endpoint + content-pinning cache | Sanitization harness operational; cache hit rate measured; all `--sanitize-mode` flags functional |
| B2 | Design + validate sanitization prompt (spike on 100-chunk sample first) | Prompt passes negative tests; 50-chunk human-labeled golden set built (immutable) |
| B3 | Implement 8-layer guard stack | Each guard has positive + negative regression tests; golden-output fixture per guard |
| B4 | Dual-write LLM + heuristics; `both-and-diff` with defined diff predicate | `both-and-diff` functional; diff predicate: token-level Levenshtein >5% (excluding whitespace + Unicode normalization diffs) |
| B5 | LLM sanitization test infrastructure | Property-based tests per guard layer; mocked-endpoint integration tests; golden-output regression for each prompt version |
| B6 | Corpus-wide rebuild + synthetic soak (llm vs heuristic) | Format ≥95% corpus-wide with LLM; no regression; dominance criterion items 1–5 met (§3.3) |
| B7 | Schema version: `3.0.0` → `3.1.0` | Output carries `schema_version: "3.1.0"`; provenance fields populated |
| B8 | **Cache cold-start cost measurement — NEW in 0.4** | Cold-cache build wall time reported; documented in `docs/PHASE_B_BUILD_TIMES.md`; Q13 closed |
| B9 | **Cross-family judge spot-check on 25-chunk sample — NEW in 0.4** | Non-Qwen judge agrees with Qwen judge within sample bias bounds; documented |
| B10 | **Phase C: re-measure Recall@1 deficit on v3.1.0 chunks WITHOUT visual retrieval — NEW in 0.4** | New deficit-doc baseline replaces the v2.16 67.8% number for Phase C acceptance |
| B11 | **Final 3-of-8 holdout document acquisition — NEW in 0.4** | Rubric-to-holdout final mapping committed before Phase D |
| B12 | **Retrieval-behavior regression test — NEW in 0.5 (addresses 0.4 audit B7 finding: omission-by-reframing)** | Run v2.16 regression-fixture queries against both `--sanitize-mode=heuristic` and `--sanitize-mode=llm` outputs and assert top-5 doc-ID overlap ≥95% per query. Catches the failure mode where the LLM preserves every token, number, entity, relation, and ordering but reframes content (e.g., "warranty voided if exceeded" vs "exceeding voids warranty") in a way that shifts which documents the retriever surfaces. The 8-layer guard stack does not catch reframing drift at retrieval time; B12 does. Failure routes to dominance-criterion `LLM_RETRIEVAL_DRIFT` (treated as a Format regression). |

### Phase C: Visual Retrieval (Cycle 3.2 — 12d)

| Task | Description | Acceptance |
|---|---|---|
| C0 | **2-hour pre-spike + C-spike per §4.2 — NEW formalization in 0.4** | Both PASS conditions met (A and B); region-level granularity decision made; co-residency confirmed |
| C1 | Probe Qdrant multi-vector support + verify storage at C12 target + **assert qdrant-server version meets named-vector requirement** | MaxSim compatibility confirmed; named-vector support assertion documented; verify ≤500GB indexed at 1000 docs |
| C2 | Deploy ColPali/ColQwen2.5 on omlx server (per §7.7 tenancy policy) | Co-residency confirmed; latency <1s/page; concurrent-request behavior per Q11 |
| C3 | Build parallel visual index (single doc → deficit subset → full corpus) | Incremental rollout; each stage validates no regression |
| C4 | Implement MaxSim + bounded page→chunk join + profile-conditional fusion + leg-skip re-normalization | `retrieve_hybrid_visual()` functional; bounded join verified (top-N per page); re-normalization verified on leg-skip |
| C5 | **Weight sweep on critical path (not deferred) — EXPANDED in 0.5** | Per-profile × visual-weight (28) + dense:sparse ratio top-3 (36) + RRF k variation top-3 = ~64 configurations per §3.4 #5; swept-optimal weights become §6.2 fork-back candidates if they beat priors by ≥5pp on any axis. Soak runtime ≈32h parallelized across workers. |
| C6 | Synthetic soak on complex-doc subset + full corpus | Recall@1 ≥80% on deficit docs (vs B10 baseline); text-doc metrics maintained; no regression on 29 text-heavy docs |
| C7 | Reranker discrimination + fusion-vs-rerank flip-rate audit | Top-1 selection rate ≥60% on visually-retrieved pages; degradation flips ≤ improvement flips on deficit subset |
| C8 | Schema version: `3.1.0` → `3.2.0` | Visual collection built; Qdrant cutover per §7.0 |

### Phase D: Modality-Aware Gates (Cycle 3.3 — 12d)

| Task | Description | Acceptance |
|---|---|---|
| D1 | Implement profile-specific judge rubrics (runs ∥ Phase A) | FORM, TABLE, DIAGRAM rubrics operational in synthetic soak |
| D2 | Extend strict gate for modality classes | `FORM_AUDIT_PASS`, `TABLE_AUDIT_PASS`, `DIAGRAM_AUDIT_PASS` in `qa_full_conversion.py` |
| D3 | Add visual retrieval quality axes | Visual Relevance, Spatial Fidelity axes operational |
| D4 | Full-corpus v3.0 acceptance run | Smoke matrix: all rows GATE_PASS + UNIVERSAL_PASS; ≥8 blind-test holdouts included (per §3.5 #5 plan) |

---

## 5. Component Architecture (Detailed)

### 5.1 Module Map (V3.0 Target)

```
src/mmrag_v2/
├── universal/                     # [EXPANDED] True UIR layer
│   ├── intermediate.py            # UniversalDocument, UniversalPage, UIRChunk, Locator,
│   │                              # ConfidenceBreakdown, StructuralFlag, CoordinateFrame,
│   │                              # ExtractionWarning, Modality (replaces ElementType)
│   ├── conversion_plan.py         # Parent ConversionPlan + format-specific subclasses
│   ├── element_processor.py       # [REFACTORED] Operates on UIR, not Docling DOM
│   ├── router.py                  # Format detection → engine routing (retained)
│   └── quality_classifier.py      # ConfidenceNormalizer (retained)
│
├── engines/
│   ├── base.py                    # FormatEngine ABC (retained)
│   ├── pdf_engine.py              # [REFACTORED] Consumes shared adapter; outputs UniversalDocument
│   ├── docling_adapter.py         # [RETAINED] Single Docling construction + invocation site
│   ├── docling_postprocess.py     # [RETAINED] Heuristic post-processing; dual-write alongside LLM
│   ├── docling_serializers.py     # [REFACTORED] Operates on UIR via adapter shim
│   └── epub_engine.py             # [FUTURE — deferred to post-v3.0]
│
├── mapper.py                      # [REFACTORED] Consumes UIR, not DoclingDocument
│
├── sanitization/                  # [NEW] LLM-native chunk sanitization — SPLIT IN 0.4
│   ├── __init__.py
│   ├── orchestrator.py            # [NEW] Top-level orchestration of LLM vs heuristic;
│   │                              # dual-write logic; both-and-diff comparison;
│   │                              # mode flag handling
│   ├── llm_sanitizer.py           # GX10 FP8 endpoint client + content-pinning cache
│   ├── guards/                    # [NEW SUBPACKAGE] 8-layer guard stack
│   │   ├── __init__.py
│   │   ├── edit_distance.py       # Guard 1
│   │   ├── numeric_entity.py      # Guard 2
│   │   ├── code_span.py           # Guard 3
│   │   ├── order_preservation.py  # Guard 4
│   │   ├── token_alignment.py     # Guard 5
│   │   ├── prompt_boundary.py     # Guard 6
│   │   ├── entity_relation.py     # Guard 7 — NEW in 0.4
│   │   └── dedup_ratio.py         # Guard 8 — NEW in 0.4 (corpus-level)
│   ├── golden_set.py              # [NEW] 50-chunk human-labeled golden set loader + scorer
│   ├── prompts.py                 # Versioned, language-aware prompt templates
│   └── graceful_degradation.py    # Endpoint-unreachable fallback policy
│
├── retrieval/
│   ├── pipeline.py                # [EXTENDED] retrieve_hybrid_visual();
│   │                              # retrieve_hybrid_reranked() preserved for back-compat (§7.8)
│   ├── visual_embedder.py         # [NEW] ColPali/ColQwen2.5 client (omlx, per §7.7)
│   ├── maxsim.py                  # [NEW] MaxSim + bounded page→chunk join (top-N per page)
│   ├── fusion.py                  # [EXTENDED] Profile-conditional RRF + leg-skip re-normalization
│   ├── rerank_audit.py            # [NEW] Fusion-vs-rerank flip-rate logger
│   └── config.py                  # [EXTENDED] Visual collection defaults + weight profiles
│
├── omlx/                          # [NEW in 0.4]
│   ├── scheduler.py               # Tenancy & request scheduling on shared omlx endpoint
│   └── coresidency_monitor.py     # Memory + latency telemetry per model
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
    │                    │       sanitization/orchestrator.py        │
    │                    │  Mode flag: off/llm/heuristic/both-and-diff│
    │                    └───────┬───────────────┬───────────────────┘
    │                            │               │
    │              ┌─────────────┘               └─────────────┐
    │              ▼                                           ▼
    │   ┌─────────────────────┐                  ┌─────────────────────┐
    │   │   LLM Sanitizer     │                  │  Heuristic Stack    │
    │   │   (GX10, cache)     │                  │  (postprocess.py)   │
    │   └──────────┬──────────┘                  └──────────┬──────────┘
    │              │                                         │
    │              ▼                                         │
    │   ┌─────────────────────┐                              │
    │   │  Guards 1-8         │                              │
    │   │  (incl. corpus-     │                              │
    │   │   level dedup g8)   │                              │
    │   └──────────┬──────────┘                              │
    │              │                                         │
    │              └──────────────────┬──────────────────────┘
    │                                 ▼
    │              ┌──────────────────────────────────────┐
    │              │  both-and-diff (dual-write window)   │
    │              │   diff predicate: Levenshtein >5%     │
    │              │   chosen output → chunker             │
    │              └──────────────────┬───────────────────┘
    │                                 ▼
    ▼                       Cleaned/validated chunks
HybridChunker (UIR-native)  ←─────────┘
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
              → bounded join: top-N=3 text chunks per visually-
                retrieved page inherit visual rank (§3.4 #4)
    │
    ▼
RRF fusion (k=60, profile-conditional weights = priors;
             leg-skip re-normalization; swept on critical path)
    │
    ▼
ModernBERT rerank (top-25 → top-5)
    │
    ▼
rerank_audit.py logs fusion-vs-rerank flip
    │
    ▼
Top-5 results (text chunks from relevant pages)
```

---

## 6. Design Decisions, Fork-Back Triggers & Regret Risks

### 6.1 Design Decisions

| Decision | Phase | Rationale |
|---|---|---|
| **Docling retained** + LLM sanitization; VLM-native deferred | A-B | Lower risk; LLM-clean achieves Format improvement with $0 cost. Fork-back trigger strengthened at §6.2 (OR-clause). |
| **LLM sanitization via local GX10 FP8** | B | $0 cost; privacy-preserving; already operational (v2.14). Content-pinning cache ensures determinism. |
| **ColPali via omlx** | C | $0 per-query cost; LAN-local; §7.7 tenancy policy governs co-residency. |
| **Profile-conditional fusion weights with non-zero PROSE floor (0.1) — PRIORS** | C | Non-zero floor prevents wasted visual infra; tiebreaker effect on embedded figures in prose docs. Defaults are starting points for the C5 sweep, not measurements. |
| **Bounded page→chunk join (top-N=3 per page)** | C | Prevents large pages from flooding the candidate pool; replaces Draft 0.3's broadcast policy. Measured via C-spike PASS B. |
| **Visual-signal preservation through rerank — measured** | C | Fusion-vs-rerank flip-rate audit detects ModernBERT undoing visual signal; degradation flips ≤ improvement flips required. |
| **Semantic-identity gate split into identity-half + explained-delta-half** | A | Identity gate alone could pass while concealing intentional behavioral changes (`partial_code_cross_page`). Audit table forces enumeration. |
| **Third-party regression check (Earthship, Harry Potter, Fluent_Python)** | A | Prevents Phase A's fixture from confirming v2.16's defects. R13 mitigation. |
| **Per-doc Phase A spike (A0) before cycle commitment** | A | Real refactoring evidence determines the 24-day budget's validity. R17 mitigation. |
| **σ baseline measured in Phase A trailing days** | A | Unblocks Phase B dominance criterion (R20). |
| **Closed-vocabulary `StructuralFlag` enum** | A | Open-ended `Dict[str, bool]` rots; the "flags additive" gate requires a canonical registry. |
| **`ConversionPlan.engine_options` opaque blob** | A | Carries Docling's dozens of toggles without per-option typed plan fields. |
| **`ConfidenceBreakdown` single-sentinel + `applicable` set** | A | Eliminates two-sentinel bug surface (None vs -1.0 in Draft 0.3). |
| **8-layer guard stack (added entity-relation + corpus-dedup-ratio)** | B | Defense-in-depth; per-chunk guards alone miss semantic substitution and inter-chunk drift. |
| **Heuristics retained alongside LLM (dual-write)** | B | Probabilistic vs deterministic; fallback + regression comparison. Deprecation gated on three *production* cycles (§6.2). |
| **50-chunk human-labeled golden set + cross-family judge spot-check** | B | Mitigates Qwen-on-Qwen sanitizer-judge correlated failure mode (R19). |
| **Sentinel-chunk accounting (>5% rate marks soak LLM_SENTINEL_DEGRADED)** | B | Prevents endpoint flakiness from masquerading as LLM regression. |
| **Granular schema versioning** (3.0.0/3.1.0/3.2.0) | A/B/C | Isolates blame surfaces; enables incremental adoption. 3.0.0 held until 3.1.0 ready (§4.3). |
| **`chunk_id` rewrite map published per phase** | A/B/C | C13 + R15 mitigation; downstream joins do not silently break. |
| **Content-pinning cache for sanitization determinism** | B | `(content_hash, context_hash, model_id, prompt_version) → sanitized_output`. Reproducibility without requiring deterministic LLM endpoint. |
| **Leg-skip fusion re-normalization** | C | Preserves score comparability across queries; prevents threshold breakage downstream. |
| **2-hour pre-spike + C-spike before Phase A implementation** | Pre-A | Falsifies ColPali viability cheaply before any v3.0 code. |
| **omlx tenancy & scheduling policy (§7.7)** | C | Governs ColPali + Qwen3-Embedding-8B + ModernBERT concurrent access. R16 mitigation. |
| **Retrieval API surface preserved (§7.8)** | C | `retrieve_hybrid_reranked` signature unchanged; visual fusion exposed as a new function. Q12. |
| **No gate weakening** | All | Per `DECISIONS.md`. |

### 6.2 Fork-Back Triggers

| Decision | Trigger | Action |
|---|---|---|
| **Docling retained vs VLM-native (STRENGTHENED in 0.5)** | ANY of: (1) Two Phase B soak iterations with Format <92% under both `--sanitize-mode=llm` AND `--sanitize-mode=heuristic` (extraction-side deficit), (2) >15% of LLM rejections within Phase B trace to upstream extraction defects (logged via guard-rejection root-cause tagging in B3), (3) **per-document-class threshold** — >10% rejection rate on any single document class even if corpus-wide average is below 15% (e.g., 40% rejection on German engineering docs masked by 0% on English prose; addresses 0.4 audit A3 finding #2), (4) **bbox/visual-coherence trigger** — C-spike or Phase C reveals systematic misalignment between ColPali patch-level attention regions and Docling chunk boundaries on visually-complex pages such that the bounded page→chunk join produces incoherent results (text-chunk boundary inside a ColPali high-attention region or vice versa, measured as Bbox-IoU between Docling chunk boxes and ColPali top-k patch boxes <50% on >20% of deficit-doc pages; addresses 0.4 audit A3 finding #1) | Reopen VLM-native parsing evaluation. |
| **Heuristics retained vs removed (STRENGTHENED in 0.4)** | Aggregate dominance + ≤2% degradation rate + no chunk FAIL for **three consecutive PRODUCTION cycles** (not just soak iterations on the same corpus), AND golden-set criterion item 5 met (LLM ≥ heuristic by ≥5pp), AND cross-family judge spot-check item 6 shows no axis-level regression | Deprecate heuristic stack. |
| **Visual retrieval weights** | C5 sweep outperforms profile-conditional defaults by ≥5pp on any axis | Adopt sweep-optimal weights. |
| **ColPali vs alternative** | C-spike FAIL A (page recovery <60%) or FAIL B (reranker discrimination <60%) | Redirect or expand Phase C scope per §4.2 outcome rules. |
| **omlx co-residency** | ColPali + Qwen3-Embedding-8B exceed omlx memory, OR Q5 latency exceeded under §7.7 scheduling | Dedicated visual endpoint or cloud fallback. |
| **Page-level vs region-level granularity** | Reranker top-1 selection rate <60% on visually-retrieved pages in C-spike or Phase C soak | Scope region-level retrieval into Phase C (not deferred). |
| **Visual signal preservation through rerank (NEW in 0.4)** | Phase C audit shows degradation flips > improvement flips on deficit subset | Phase D / v3.2: text+visual-aware reranker or fusion-side visual boost. |
| **Content-pinning cache vs deterministic endpoint** | GX10 endpoint supports deterministic decoding at temperature=0 → cache still retained as speed optimization | No decision change; cache is always beneficial. |

### 6.3 Decisions We May Regret (and How We'll Know)

1. **Docling + LLM sanitization as the permanent extraction path.**
   - **Regret condition:** Docling bbox quality on deficit docs is the bottleneck. Visual retrieval gains <5pp despite clean text.
   - **Monitoring signal:** Bbox-IoU between Docling output and a VLM-native reference parse on the 5 deficit docs.

2. **Page-level visual retrieval granularity (not region-level).**
   - **Regret condition:** Reranker fails to discriminate on visually-retrieved dense pages.
   - **Monitoring signal:** Reranker top-1 selection rate on visually-retrieved pages. If <60%, too coarse. Measured in C-spike AND Phase C soak.

3. **Heuristics retained for three production cycles only.**
   - **Regret condition:** LLM model updates introduce new failure modes; heuristics would have caught them.
   - **Monitoring signal:** `both-and-diff` disagreement rate (>5% Levenshtein distance per §3.3 diff predicate). If >5% of chunks differ in cycle 3, extend dual-write to cycle 4.

4. **`chunk_id` derivation kept positional (regret #4 — SHARPENED in 0.5).**
   - **Regret condition:** Downstream consumers (RAG app, citation cache, soak fixtures) repeatedly break across Phase A/B/C boundaries because position-derived `chunk_id` is sensitive to reordering and modality widening.
   - **Monitoring signal:** Number of distinct `chunk_id` rewrites published in the per-phase rewrite map. **First-trigger threshold:** if A6 publishes >10% rewrites, the Phase A scope-negotiation protocol's content-derived `chunk_id` option (§ Phase A) is invoked *immediately within Phase A* rather than deferred to v3.3. **Subsequent triggers:** B7 >5% or C8 >2% invokes the same option for v3.2 or v3.3 respectively. The audit's 0.4-round recommendation (flip to content-derived in Phase A) is preserved as a contingent path: the deferral holds *unless* A0 evidence shows churn at first-trigger level, in which case the deferral is overridden.
   - **Defense of the deferral when not triggered:** Phase B's content changes are intentional and should produce new IDs anyway; content-derived `chunk_id` makes Phase B's content-change → ID-change relationship explicit rather than coupled (both versions produce ID churn at Phase B; the difference is which churn is intentional vs accidental). The deferral is *contingent on A6 churn being low* (<10% rewrites). The contingency makes the deferral falsifiable rather than just optimistic.

5. **Qwen-on-Qwen judge for LLM sanitizer dominance (NEW regret #5 in 0.4).**
   - **Regret condition:** Sanitizer and judge share blind spots; corpus-wide Format ratings rise without real quality improvement.
   - **Monitoring signal:** Disagreement between Qwen judge and cross-family judge on the B9 25-chunk sample. If cross-family judge shows ≥3pp Format regression where Qwen shows none, escalate to full-corpus cross-family re-judge in Phase D.

---

## 7. Migration, Rollback & Operations

### 7.0 Qdrant Cutover Plan (NEW in 0.4)

V3.0 introduces new Qdrant collections (`mmrag_v3__dense`, `mmrag_v3__sparse`, `mmrag_v3__visual`) parallel to today's v2.X collections. Cutover strategy:

| Phase | Live collection | Action |
|---|---|---|
| Pre-A | v2.16 collections (current names) | No change. Verify collection names via `python scripts/search_qdrant.py --stats`. |
| A | v2.16 collections (current names) | Phase A produces `mmrag_v3__dense`, `mmrag_v3__sparse` side-by-side. v2.16 collections remain live for read traffic. |
| B | v2.16 + v3 text collections | LLM-sanitized rebuild populates v3 text collections. RAG app continues reading from v2.16. |
| **Cutover** (end of B) | v3 text + v2.16 visual placeholder | Atomic alias switch: RAG app config flag flipped from `mmrag_v2_*` to `mmrag_v3_*`. v2.16 collections retained read-only for 30 days for rollback. |
| C | v3 text + v3 visual | `mmrag_v3__visual` populated; retrieval pipeline switches to `retrieve_hybrid_visual()` for clients that opt in. |
| Post-C cleanup | v3 only | After 30-day retention, v2.16 collections dropped manually with operator confirmation. |

**Rollback procedure:** Flip the alias back to v2.16 collections. No data loss; retrieval falls back to v2.16 behavior. Test in CI (`tests/test_qdrant_cutover_rollback.py`).

### 7.1 Schema Version, chunk_id Derivation, and Type-Vocabulary Reconciliation

**Schema version table:**

| Version | Phase | What changes | Consumer impact |
|---|---|---|---|
| `3.0.0` | A | UIR introduced; structural flags populated; chunk content unchanged from v2.16 | Content unchanged. Flags additive. **`chunk_id` may change due to A2 refactor** — rewrite map published (`docs/CHUNK_ID_REWRITE_MAP_3.0.0.csv`). |
| `3.1.0` | B | LLM sanitization may change content; provenance fields populated | Content may differ from v2.16. Provenance enables audit. `chunk_id` may change again if a chunk's modality is re-classified by the sanitization pass — rewrite map appended. |
| `3.2.0` | C | Multi-vector visual collection alongside text collections | Text collections unchanged. Visual is additive. `chunk_id` stable. |

**`chunk_id` derivation impact (NEW in 0.4 — addresses R15):**

Today's `chunk_id` is position-derived: `f"{doc_id}_{page:03d}_{modality}_{md5(parts)[:8]}"` from [`element_processor.py:823`](src/mmrag_v2/universal/element_processor.py#L823), where `parts = f"{doc_id}_{page}_{modality}_{element_idx}"`. Any change in element ordering, page assignment, or modality classification changes `chunk_id`. Phase A's UIR refactor will reorder elements (the entire point) and may re-classify modality (CODE/FORM enum widening). Therefore `chunk_id` *will* change for some chunks in Phase A even with semantic-identity preserved.

**Mitigation per phase:**
- **A6:** Publish `docs/CHUNK_ID_REWRITE_MAP_3.0.0.csv` — every changed `chunk_id` listed with `(v2.16_chunk_id, v3.0.0_chunk_id, change_reason)`. Empty file = no changes (acceptable). The rewrite map is generated by comparing the semantic-identity gate's identity half output to the v2.16 fixture.
- **B7:** Append to `CHUNK_ID_REWRITE_MAP_3.1.0.csv` for any IDs that change due to modality widening during sanitization.
- **C8:** Visual chunks have separate IDs (`page_id`); text `chunk_id` is stable from 3.1.0.
- **Downstream RAG app contract:** consumers that key on `chunk_id` must consult the rewrite map at every schema-version transition.
- **Open question, deferred to v3.3:** content-derived `chunk_id` (eliminating positional sensitivity entirely) — tracked as regret #4 (§6.3).

**Type-vocabulary reconciliation (NEW in 0.4 — addresses C14):**

Today there are three independent type vocabularies:
- `ChunkType` (schema, `src/mmrag_v2/schema/ingestion_schema.py::ChunkType`) — PARAGRAPH, CODE, TABLE, IMAGE, etc. Lives on `IngestionChunk.chunk_type`.
- `ElementType` (UIR, `src/mmrag_v2/universal/intermediate.py::ElementType`) — TEXT / IMAGE / TABLE.
- v3.0 proposes `Modality` — TEXT / IMAGE / TABLE / CODE / FORM.

V3.0 reconciliation (executed during Phase A, no two-way shims):

| v2.X concept | v3.0 concept | Migration rule |
|---|---|---|
| `Element.element_type = ElementType.TEXT` | `UIRChunk.modality = Modality.TEXT` (or `CODE` or `FORM` if classified during element processing) | Replace; `ElementType` deleted from codebase end of Phase A. |
| `Element.element_type = ElementType.IMAGE` | `UIRChunk.modality = Modality.IMAGE` | Replace. |
| `Element.element_type = ElementType.TABLE` | `UIRChunk.modality = Modality.TABLE` | Replace. |
| `IngestionChunk.chunk_type = ChunkType.PARAGRAPH` | `IngestionChunk.chunk_type = ChunkType.PARAGRAPH` (kept); derived from `Modality.TEXT` + non-code/form classifier | Kept; derivation is one-way (modality → chunk_type). |
| `IngestionChunk.chunk_type = ChunkType.CODE` | `IngestionChunk.chunk_type = ChunkType.CODE` (kept); derived from `Modality.CODE` | Kept; derivation is one-way. |
| `IngestionChunk.chunk_type = ChunkType.TABLE` | `IngestionChunk.chunk_type = ChunkType.TABLE` (kept); derived from `Modality.TABLE` | Kept; derivation is one-way. |
| `IngestionChunk.chunk_type = ChunkType.IMAGE` | `IngestionChunk.chunk_type = ChunkType.IMAGE` (kept); derived from `Modality.IMAGE` | Kept; derivation is one-way. |
| (no FORM equivalent today) | `IngestionChunk.chunk_type = ChunkType.FORM` (new); derived from `Modality.FORM` | New value added to `ChunkType` enum. |

**Downstream warning:** consumers switching on `chunk_type` must add a branch for `ChunkType.FORM` before Phase B ships. Existing branches for PARAGRAPH/CODE/TABLE/IMAGE keep working without change.

**Re-measurement of Recall@1 67.8% baseline (NEW in 0.4):**

The §3.4 gate "Recall@1 ≥80% on deficit docs (from 67.8%)" is only meaningful if the 67.8% is measured on chunks of comparable shape. v3.1.0 sanitization can change chunk content and (rarely) modality, which changes recall regardless of any visual retrieval contribution. Phase B B10 re-measures the deficit-doc baseline on v3.1.0 chunks WITHOUT visual retrieval; that new number is the Phase C target's reference baseline.

### 7.2 Rollback Paths

| Phase | Rollback | Procedure |
|---|---|---|
| A (UIR) | `git revert` | Semantic-identity output (identity half) → no migration; explained-delta half rolls back too |
| B (LLM sanitize) | `--sanitize-mode=heuristic` | Exact v2.16 behavior; heuristics retained |
| C (Visual index) | Drop visual collection + flip alias (§7.0) | Text retrieval unaffected |
| D (Gates) | Gate-only change | No data migration |

### 7.3 Corpus Migration

Per phase:
- Rebuild corpus (~8–12h warm-cache; ~10–14h cold-cache once `--sanitize-mode=llm` is in play, per B8)
- Re-ingest text (~1–2h)
- Phase C adds: render + ColPali embed (~5h at 34 docs)
- Synthetic soak (~1–2h)
- Updated regression fixture (`tests/fixtures/retrieval_regression_v3_X.json`)

**Soak-as-gate latency budget (NEW accounting in 0.4):**

Phase B's dominance criterion requires (a) σ baseline (3 consecutive heuristic soaks, established in Phase A A7, ~6h) + (b) two consecutive LLM-mode soaks for confirmation (~4h) + (c) cross-family judge spot-check on 25-chunk sample (~30 min, manual). Total Phase B soak-as-gate burden: ~10h plus iteration. Plan calendar accordingly; do not assume soak runs in isolation.

### 7.4 Determinism Policy

- **Content-pinning cache:** `(content_hash, context_hash, model_id, prompt_version) → sanitized_output` where `context_hash = SHA-256(prev_chunk_content_first64bits + next_chunk_content_first64bits + detected_lang)`. On cache hit, no LLM invocation. Cache is file-backed under `output/sanitization_cache/`, keyed by content hash prefix + first 8 chars of context hash. Including context in the key prevents stale cache hits when neighbor chunks change between rebuilds. Makes builds deterministic even without deterministic LLM endpoint. A chunk whose raw content AND context haven't changed between rebuilds won't be re-sanitized — it hits the cache. This also speeds rebuilds.
- **Build-reproducibility test:** CI runs `mmrag-v2 batch --sanitize-mode=llm` on 3-doc subset; asserts chunk-level identical-hash ratio ≥99.5%, with the additional invariant that zero disagreeing chunks have changed semantics (verified by judge sample on all disagreeing chunks).
- **Heuristic mode:** Byte-stable (no LLM). **Off mode:** Byte-stable.
- **Hash-tolerance:** "≥99.5%" = ≥99.5% of chunks have byte-identical content hashes across consecutive builds. The remaining ≤0.5% are verified individually by LLM-as-judge to confirm no semantic change.

### 7.5 Observability

- **Per-chunk lineage:** Provenance fields written to JSONL + Qdrant payload.
- **Sanitization rejection log:** `logs/sanitization_rejections_<timestamp>.jsonl` — chunk ID, guard name (one of 8), rejection reason, root-cause tag (used for §6.2 OR-clause trigger).
- **Graceful degradation log:** Endpoint name, unreachable duration, affected chunk count.
- **Fusion trace (opt-in):** `--log-fusion-trace` — per-leg scores, weights applied (including re-normalization on leg skip), fusion scores, reranker input/output, **fusion-vs-rerank flip annotation**.
- **omlx tenancy telemetry (NEW in 0.4):** `logs/omlx_scheduling_<timestamp>.jsonl` — per-request model, queue depth, latency, preemption events.
- **Soak provenance:** Schema version, model versions, sanitization mode per judgment, cross-family judge sample tag.

### 7.6 Failure-Mode Behavior (Endpoint Unreachable)

| Component | Endpoint | When unreachable | Behavior |
|---|---|---|---|
| **LLM Sanitizer** | GX10 `10.0.10.239:8000` | Fall back to heuristic sanitization | Sentinel per chunk; build warning; no hard fail; §3.3 sentinel-rate accounting |
| **Text Embedder** | omlx `10.0.10.246:8000` | Pipeline halts | Hard fail; operator intervention |
| **ColPali** | omlx `10.0.10.246:8000` | Visual leg skipped; weights re-normalized | Log per query; text-only retrieval proceeds |
| **ModernBERT** | omlx `10.0.10.246:8000` | Reranker skipped | Log per query; top-K from fusion returned directly |
| **omlx co-residency conflict** | omlx (memory exhausted by 3 co-resident models) | Per §7.7 scheduling policy: lowest-priority model evicted; affected requests retried with backoff | Telemetry per eviction; if eviction rate >1/min sustained, R6 fork-back triggered |

### 7.7 omlx Tenancy & Scheduling (NEW in 0.4)

The omlx server at `10.0.10.246:8000` hosts three models in v3.0: Qwen3-Embedding-8B (text embedding), ModernBERT (reranker), and ColPali (visual embedding from Phase C). Without an explicit tenancy policy, ColPali's per-page embedding (~1s) can block query-path requests for text embedding (~10ms) or rerank (~50ms), violating Q5 latency.

**Policy:**

1. **Request priority** (highest to lowest):
   1. **Query-path text embedding** (Qwen3-Embedding-8B) — latency-critical, user-facing.
   2. **Query-path reranking** (ModernBERT) — latency-critical, user-facing.
   3. **Query-path visual embedding** (ColPali on query token) — latency-critical, user-facing.
   4. **Ingest-path visual embedding** (ColPali on document pages) — throughput-oriented, batch.
2. **Preemption:** Ingest-path ColPali requests are preemptible at page boundaries. When a query-path request arrives, the current page-embedding completes (~1s ceiling), then queue priority shifts.
3. **Queue model:** Per-model FIFO queues with priority-based dispatch. `omlx/scheduler.py` exposes `submit(model, payload, priority)`.
4. **Memory budget:** All three models must fit in the omlx server's GPU memory simultaneously, or one is evicted per LRU. Co-residency validated in C-spike and monitored via `omlx/coresidency_monitor.py`.
5. **Latency budget:** Q5 ≤3.0s p99. With three legs in parallel + rerank, individual leg budgets are: text embed ≤100ms, sparse ≤50ms, visual embed (query) ≤500ms, fusion ≤50ms, rerank ≤2000ms, miscellaneous ≤300ms.

**Failure mode:** If priority queue depth for a query-path request exceeds 3 (i.e., 3 ingest jobs ahead), the ingest job is canceled and re-queued. Ingest throughput drops; query latency holds.

### 7.8 Retrieval API Surface Evolution (NEW in 0.4)

Current production retrieval surface (per `src/mmrag_v2/retrieval/pipeline.py`):

```python
def retrieve_reranked(query: str, top_k: int = 5, ...) -> List[RetrievalResult]: ...
def retrieve_hybrid_reranked(query: str, top_k: int = 5, ...) -> List[RetrievalResult]: ...
```

V3.0 adds (does not replace) the visual fusion entry point:

```python
def retrieve_hybrid_visual(
    query: str,
    top_k: int = 5,
    profile: Optional[ProfileClass] = None,  # PROSE/DIAGRAM/FORM/TABLE
    visual_weight_override: Optional[float] = None,  # for --boost-visual flag
    return_debug_payload: bool = False,  # NEW in 0.5 — see RetrievalDebugPayload below
    ...
) -> List[RetrievalResult]: ...
```

**Per-query debug payload (NEW in 0.5 — addresses 0.4 audit C10 finding):**

The `--log-fusion-trace` CLI flag enables global query tracing, but it is off by default in production (overhead) and cannot be retroactively enabled for a specific user-reported "wrong document returned" query (the visual embedding is non-deterministic across model versions and rendering pipelines, so the query may not be exactly re-runnable). The `return_debug_payload=True` argument attaches a `RetrievalDebugPayload` to each `RetrievalResult`, available for on-demand inspection without enabling global tracing:

```python
@dataclass
class RetrievalDebugPayload:
    """Per-query retrieval debug data. ~1–2 KB per query; not logged by default."""
    leg_scores: Dict[str, Dict[str, float]]    # {"dense": {chunk_id: score, ...},
                                               #  "sparse": {chunk_id: score, ...},
                                               #  "visual": {page_id: score, ...}}
    weights_applied: Dict[str, float]          # Profile-conditional weights actually used (after leg-skip re-normalization)
    legs_skipped: List[str]                    # Which legs were unavailable (e.g., ["visual"] when ColPali down)
    fusion_input: List[Tuple[str, float]]      # (chunk_id, fused_score) before rerank
    bounded_join_decisions: List[Tuple[str, str, int]]  # (page_id, chunk_id_selected, top_n_rank) — bounded join trace
    rerank_input: List[str]                    # chunk_ids handed to ModernBERT
    rerank_output: List[Tuple[str, float]]     # (chunk_id, rerank_score) after rerank
    fusion_vs_rerank_flips: List[Tuple[int, int]]  # (fusion_rank, rerank_rank) for chunks that moved
    profile_used: Optional[ProfileClass]
    visual_collection_pin: str                  # e.g., "vidore/colqwen2.5-v0.2 @ render_dpi=200"
    timing_ms: Dict[str, float]                 # Per-leg timing breakdown
```

The payload is attached to each `RetrievalResult` only when `return_debug_payload=True`; the RAG application logs it on demand for specific queries without enabling global tracing. Default off; ~1–2 KB overhead per query when enabled (no measurable latency impact, no I/O until the application chooses to serialize the payload).

**Backward compatibility (Q12):**
- `retrieve_reranked` and `retrieve_hybrid_reranked` signatures are preserved. They internally use the v3 collections (`mmrag_v3__dense`, `mmrag_v3__sparse`) post-cutover but do not include the visual leg.
- Downstream RAG apps continue working with no code changes; they get text-only retrieval against the v3 text indices.
- Apps that want visual retrieval explicitly call `retrieve_hybrid_visual(...)`.
- Deprecation: `retrieve_reranked` and `retrieve_hybrid_reranked` are NOT marked deprecated in v3.0. They remain first-class API entry points through v3.x. v4.0 may consolidate, but that is out of scope for this charter.

### 7.9 `--profile-override` Semantics in v3.0 (NEW in 0.4)

Per CLAUDE.md, `--profile-override` exists for debugging and bypasses `ProfileClassifier`. In v3.0:

- **Ingest path:** `--profile-override <profile>` continues to pin the profile during conversion (same as v2.X).
- **Retrieval path:** A new query-time flag `--retrieval-profile <profile>` overrides per-document profile-conditional fusion weights with the named profile's weights. Useful for debugging visual-leg behavior on a per-query basis.
- **Production use:** both flags are debug-only; production runs use auto-classification. Per CLAUDE.md, "Profile overrides are for debugging only, never for production acceptance runs."

### 7.10 VLM-Caption Interaction with Visual Retrieval (NEW in 0.4)

The existing v2.X VLM (`vision_manager.py`) generates captions for image chunks that participate in text retrieval. Once visual retrieval (Phase C) is operational, the question arises: does VLM caption generation still pay for itself?

- **Phase C decision (default):** VLM caption generation is **retained** on the ingest path. The VLM caption provides text-retrieval signal (a "transistor diagram" query may match the caption "Schematic of an NPN transistor amplifier" via text embedding, even before visual retrieval is consulted).
- **Re-evaluation deferred to v3.2 / v3.3:** Measure VLM caption contribution to text-only Recall@1 on the deficit docs. If <1pp contribution AND visual retrieval covers the same queries with ≥equal Recall@1, VLM caption generation becomes a candidate for ingest-cost-reduction (deferred).
- **No coupling change in v3.0:** The VLM caption is part of the chunk content; it goes through the sanitization stage; visual retrieval embeds the rendered page (which includes the figure) independently. Two complementary signals, fused at retrieval time.

### 7.11 Visual-Index Stability Contract (NEW in 0.5 — addresses 0.4 audit R4 finding)

The visual collection (`mmrag_v3__visual`) is pinned to a specific ColPali model version and `render_dpi` value at index-build time. Changing either invalidates the entire visual index and requires a full re-embed:

| Pin | Stored in | Validation |
|---|---|---|
| ColPali model identifier (e.g., `vidore/colqwen2.5-v0.2`) | Qdrant collection metadata + `docs/VISUAL_INDEX_PIN.md` | Query path asserts that the embedding-time model matches the configured query-time model. Mismatch fails build, not query. |
| `render_dpi` value | Same | Query path asserts that the query-time `ConversionPlan.render_dpi` matches the collection pin (the query path renders the query's reference page at this DPI for spatial-fidelity checks). |

**Re-embed cost (operational):**
- At 34 docs (current corpus): ~100 pages × ~1s/page ≈ 2 minutes — cheap.
- At 1000 docs (C12 target): ~300,000 pages × ~1s/page ≈ 83 hours — must be scheduled, not opportunistic.
- At 5000 docs (growth-trajectory ceiling): ~1.5M pages × ~1s/page ≈ 415 hours — multi-week wall-time; this is the ceiling beyond which a ColPali upgrade requires a dedicated cycle plan.

**Upgrade protocol:**
1. Build the new-version visual collection alongside the existing one (parallel collection with versioned name, e.g., `mmrag_v3__visual_colqwen3`).
2. Run synthetic soak against both; require new collection to match-or-beat the existing collection on Recall@1 deficit-doc subset.
3. Cutover by swapping the collection name in the retrieval-backend config; the old collection remains for rollback for one cycle, then is deleted.

**This contract is enforced for ColPali model upgrades AND for `render_dpi` changes.** The C-spike measures at both 200 and 300 DPI per `ConversionPlan.render_dpi` validation note; whichever value enters production is then pinned and not changed without re-embed.

### 7.12 Multi-Hop `chunk_id` Migration (NEW in 0.5 — addresses 0.4 audit B9 finding)

The per-phase `chunk_id` rewrite map (§7.1) handles single-version transitions (v2.16 → v3.0.0, v3.0.0 → v3.1.0, v3.1.0 → v3.2.0). But downstream consumers may skip versions — e.g., a RAG application upgrades directly from v2.16 to v3.2 without ever deploying v3.0.0 or v3.1.0. Chained-lookup is error-prone: consumers may miss a hop or apply maps in the wrong order.

**Cumulative rewrite map:** At each phase boundary, a cumulative map is published in addition to the per-phase map:

| Map | Contents | Generated by |
|---|---|---|
| `docs/CHUNK_ID_REWRITE_MAP_3.0.0.csv` | v2.16 → v3.0.0 only (per-phase, generated in A6) | Identity-half gate diff |
| `docs/CHUNK_ID_REWRITE_MAP_3.1.0.csv` | v3.0.0 → v3.1.0 only (per-phase, generated in B7) | Sanitization status diff |
| `docs/CHUNK_ID_REWRITE_MAP_3.2.0.csv` | v3.1.0 → v3.2.0 only (per-phase, generated in C8) | Trivially empty if no modality re-class; otherwise visual-collection-related |
| **`docs/CHUNK_ID_REWRITE_MAP_CUMULATIVE.csv`** (NEW in 0.5) | v2.16 → current-latest (transitively composed from per-phase maps) | Build script that joins the per-phase maps at each phase boundary; published alongside the per-phase map |

Consumers that skip versions consult only the cumulative map and never need to chain. Consumers that follow every version may consult either form. The cumulative map is regenerated and re-published at every phase boundary; the per-phase maps remain immutable for audit purposes.

**Edge case:** When a chunk is split or merged across versions, the cumulative map represents this as `(v2.16_chunk_id, [v3.x_chunk_id_1, v3.x_chunk_id_2, ...])`. Splits are expected at the Phase A `partial_code_cross_page` activation (one v2.16 chunk → two v3.0.0 continuation-group chunks).

---

## 8. Quality Gates (V3.0)

### 8.1 Universal Invariants (Unchanged from v2.X)

`chunk_type` present, `bbox` in [0,1000], non-empty text content, `modality` present, QA-CHECK-01 through QA-CHECK-05.

### 8.2 V3.0-Specific Gates

**Identity-half gate normalization rules (NEW in 0.5 — addresses 0.4 audit B8 finding):**

The semantic-identity gate's CI comparison tool MUST apply the following normalizations before hashing chunk content for comparison; otherwise byte-for-byte differences from non-semantic sources (Python dict insertion order, floating-point rounding, version stamps) produce spurious "regression" alerts:

1. **JSON field-order normalization** — chunks are deserialized to dicts and re-serialized with `sort_keys=True` before hashing. Python 3.10 dict insertion order is preserved by default; the UIR refactor will insert fields in a different order than v2.16 even when content is identical.
2. **Confidence-value rounding** — `ConfidenceBreakdown` fields are floats and may differ in the last significant digit due to different floating-point operation ordering in the UIR vs Docling-direct path. Round confidence values to 2 decimal places (±0.01 tolerance) before hashing; or exclude confidence fields from identity comparison entirely and check them separately with the ±0.01 tolerance.
3. **Identity-relevant vs metadata-only field enumeration:**

   | Field | Identity-relevant? |
   |---|---|
   | `chunk_type`, `content`, `page_number`, `bbox`, `modality`, `parent_heading`, `parent_element_id` | Yes |
   | `chunk_id` | **No** — handled separately by `chunk_id stability ratio` gate (rewrite map) |
   | `confidence_breakdown.*` (rounded) | Yes, with ±0.01 tolerance |
   | `structural_flags` | Yes — additive only (v2.16 flags must persist; new flags from `StructuralFlag` enum may appear) |
   | `extraction_warnings` | No — informational, may differ across runs |
   | `pipeline_version`, `schema_version`, `source_file_hash`, timestamps | No — metadata-only |
   | `sanitizer_*`, `sanitization_status`, `content_original`, `content_sanitized` | No — null/absent in Phase A; populated in Phase B |
   | `uir_version` | No — internal contract version |
   | `continuation_group_id` | No — additive structural metadata |

The identity hash function: `sha256(json.dumps(filter_identity_relevant(chunk_normalized), sort_keys=True))`. Any chunk whose identity hash matches the v2.16 fixture's identity hash is counted as "identical" for the identity-half ≥95% gate. Differences are routed to the explained-delta half.

| Gate | Phase | Threshold | Measurement |
|---|---|---|---|
| UIR semantic-identity (identity half) | A | ≥95% chunk match by stable identity key; content/chunk_type match per §7.1 reconciliation; flags additive; top-5 doc IDs unchanged | Key-based matching (§3.2); regression fixture; CI tool applies normalization rules above before hashing |
| UIR semantic-identity (explained-delta half) | A | 100% of remaining chunks enumerated in `docs/PHASE_A_INTENTIONAL_DELTAS.md` with v2.X documented-defect cross-reference | CI tool diffs the two halves; build fails on unenumerated delta |
| **chunk_id stability ratio** | A/B/C | Every changed `chunk_id` listed in `docs/CHUNK_ID_REWRITE_MAP_<version>.csv` (no silent breaks) | CI compares v_prev fixture's `chunk_id` set to v_next output |
| Third-party regression check | A | Earthship, Harry Potter, Fluent_Python — byte-identical or documented-as-improved | Regression-case CI suite |
| `partial_code_cross_page` flag coverage | A | 100% of cross-page code splits flagged (currently 0% — the inert case) | Corpus audit script |
| 17-skipped-tests audit | A | Each skipped test classified; re-enabled tests pass before Phase A merge | `docs/PHASE_A_SKIP_AUDIT.md` |
| σ baseline established | A | σ documented per axis from 3 consecutive heuristic soaks | `docs/PHASE_A_SIGMA_BASELINE.md` |
| LLM guard-stack compliance | B | Zero chunks accepted with numeric mismatch, code-span change, order swap, or entity-relation addition | Guard unit tests (B3) + corpus-wide acceptance |
| Corpus dedup-ratio invariant | B | LLM-mode dedup-ratio ≤ heuristic + 5% | Corpus-level Jaccard scan (guard #8) |
| LLM dominance (aggregate + degradation + golden + cross-judge) | B | Format ≥ 2× soak σ above heuristic; zero chunk PASS→FAIL; ≤2% degradation rate; 50-chunk golden set LLM ≥ heuristic +5pp; cross-family judge spot-check no axis regression | `both-and-diff` comparison with defined diff predicate (§3.3) |
| LLM sentinel rate | B | <5% per soak; soaks with ≥5% sentinel rate marked `LLM_SENTINEL_DEGRADED` (do not count toward dominance confirmation) | Sentinel log |
| Cache cold-start cost | B | Cold-cache wall time documented in `docs/PHASE_B_BUILD_TIMES.md` | Q13 measurement |
| Format (soak judge) | B | ≥95% corpus-wide with `--sanitize-mode=llm`; no regression vs `--sanitize-mode=heuristic` | `scripts/synthetic_soak.py` |
| 2-hour pre-spike PASS | Pre-A | Gold page ranks first on most-favorable query | Workstation, no infrastructure |
| Phase C-spike PASS (both A and B) | Pre-A | Page recovery ≥60% on text-missed queries; reranker discrimination ≥60% (tightened per §4.2 #8) | Spike report (numpy MaxSim + simulated rerank using exact ModernBERT + chunk_id dedup + fixture-based gold map) |
| Phase C entry criterion (storage) | Pre-C | ≤500GB indexed at C12 target (1000 docs) | Storage projection from spike measurements |
| Recall@1 chunk (deficit docs) | C | ≥80% (vs B10 re-measured baseline, not v2.16's 67.8%) | Retrieval regression on deficit docs |
| Recall@5 doc (deficit docs) | C | ≥98.6% maintained | Retrieval regression |
| Text-doc metrics (29 docs) | C | No regression on any axis | Full-corpus synthetic soak |
| Reranker discrimination (visual pages) | C | Top-1 selection rate ≥60% on visually-retrieved pages under bounded join | Phase C soak measurement |
| **Fusion-vs-rerank signal preservation** | C | Degradation flips ≤ improvement flips on deficit subset | `rerank_audit.py` log |
| Profile weight sweep | C | Sweep documented; defaults justified or replaced | C5 sweep report |
| Qdrant cutover | End of B | Alias flip; rollback test passes; v2.16 retention 30 days | §7.0 |
| omlx co-residency | C | All three models fit; eviction rate <1/min sustained; Q5 holds under §7.7 scheduling | `omlx/coresidency_monitor.py` |
| Modality-aware soak axes | D | FORM/TABLE/DIAGRAM rubrics operational | `scripts/synthetic_soak.py --modality-aware` |
| Smoke matrix | All | 34/34 GATE_PASS + UNIVERSAL_PASS | `scripts/smoke_multiprofile.sh` |
| Blind-test (Greenhouse + 7 additional, acquisition tracked) | All | GATE_PASS + UNIVERSAL_PASS for all 8 holdout docs | §3.5 #5 plan |
| Build reproducibility | B+ | Chunk-level identical-hash ratio ≥99.5%; zero disagreeing chunks have semantic change | CI test on 3-doc subset; judge verification on disagreements |
| **Retrieval-behavior regression (NEW in 0.5 — B12)** | B | ≥95% top-5 doc-ID overlap per query between `--sanitize-mode=heuristic` and `--sanitize-mode=llm` outputs on v2.16 regression-fixture queries | `scripts/retrieval_behavior_regression.py`; failure → `LLM_RETRIEVAL_DRIFT` Format regression |
| **Visual-index pin verification (NEW in 0.5 — §7.11)** | C+ | Query-path asserts `mmrag_v3__visual` collection metadata matches configured ColPali model identifier AND `render_dpi` value; mismatch fails build | Query-path startup assertion + `docs/VISUAL_INDEX_PIN.md` |
| **Cumulative chunk_id rewrite map (NEW in 0.5 — §7.12)** | A/B/C | Published at every phase boundary; multi-hop consumers consult only this single map | Build-script join over per-phase maps |
| **Docling lifecycle check (NEW in 0.5 — R21)** | Quarterly | Confirm `docling==2.86.0` still receives security patches; document Docling 3.x Python-version requirement; pin renegotiated if both fail | Owner: V3 governance (re-homed after v2.X cycle-open checklist archived) |

### 8.3 Carry-Forward from v2.16

| Limitation | Rationale |
|---|---|
| 1.4% Recall@5 doc residual | Likely judge edge cases; not a structural defect |
| ~5% Format residual after LLM sanitization | Acceptable ceiling; 100% Format is asymptotically unreachable without human review |
| Magazine image quality (composite layouts) | Rendered-region-crop deferred per v2.11 §3e; visual retrieval partially mitigates |
| EPUB engine | Deferred to post-v3.0; explicit acceptance criteria TBD in a future cycle plan |
| Cross-page table spanning (`partial_table_cross_page`) | Tables spanning pages are harder than code spanning pages (column-alignment recovery); Phase A addresses code/paragraph cross-page splits; table spanning explicitly deferred to v3.1+ as an emit-only flag, with the chunk-level repair deferred to v3.2+ |
| VLM caption ingest cost re-evaluation | Deferred to v3.2/v3.3 per §7.10 |
| Content-derived `chunk_id` (eliminating positional sensitivity) | Deferred to v3.3 per regret #4 by default; v3.0–v3.2 mitigates via per-phase rewrite map (R15, §7.1) + cumulative rewrite map (§7.12). **Contingency (NEW in 0.5):** if A6 publishes >10% rewrites, the deferral is overridden and content-derived `chunk_id` is invoked as a Phase A scope-negotiation option (see Phase A description + regret #4 monitoring). |

---

## 9. Appendix: Audit Trace

### 9.0 Review Summary — Draft 0.4 Review (2026-05-26)

Draft 0.4 received a principal-architect review (2026-05-26) answering 17 questions across Soundness/Risk/Missing/Phasing/Judgment. Overall judgment: **conditional greenlight** with 5 conditions; ratings 8/9 (decision tool), 7/10 (communication artifact), 9/10 (risk-management instrument). All 5 conditions plus 12 secondary recommendations addressed in Draft 0.5. One audit finding (A1 #1 — "Layer vs Stage terminology inconsistency in §3.2/§3.3 headers") was assessed as **factually incorrect** — the actual document already uses "Stage 1"/"Stage 2" consistently at lines 241 and 470. No change made for that finding.

### 9.0.1 Changes in Draft 0.5 (By 0.4-Round Audit Finding)

| Audit Finding | Greenlight Condition? | Draft 0.4 Issue | Draft 0.5 Resolution |
|---|---|---|---|
| **A1 #1:** Layer vs Stage terminology inconsistency | No | (claimed inconsistency in §3.2/§3.3 headers) | **REJECTED as invalid finding** — §3.2 header at line 241 already says "Ingestion Pipeline Stage 1" and §3.3 at line 470 already says "Ingestion Pipeline Stage 2." No change. |
| **A1 #2:** Split Stage 2 into 2a/2b/2c substages | No | Stage 2 bundles sanitization + guards + adjudication | **DEFERRED as clarity-only** — §5.2 data flow already shows the three-step structure; §3.3 prose change deferred to a future doc-cleanup cycle without functional impact. |
| **A2 #1:** Missing `continuation_group_id` on UIRChunk | No | Cross-page sibling chunks indistinguishable except via flag | **APPLIED** — `UIRChunk.continuation_group_id: Optional[str]` added (§3.2); shared UUID per cross-page sibling group. |
| **A2 #2:** Missing `table_structure` on UIRChunk | No | TABLE rubric must re-parse for row/column counts | **DEFERRED** — Phase D TABLE rubric can re-parse at evaluation time; first-class field is optimization. Tracked in USER_ISSUES. |
| **A2 #3:** Missing `lang_confidence` per chunk | No | Multilingual docs use document-level `lang` only | **DEFERRED** — v3.0–v3.1 sanitization uses document-level `lang`; per-chunk language-conditional prompts deferred to v3.2. |
| **A2 #4:** `ConversionPlan.render_dpi` unvalidated | No | DPI changes silently invalidate visual index | **APPLIED** — `render_dpi` validation range `[72, 600]` enforced in `__post_init__`; §7.11 visual-index stability contract pins render_dpi per collection. |
| **A2 #5:** Missing `uir_version` on UIRChunk | No | UIR field additions across v3.x have no internal version marker | **APPLIED** — `UIRChunk.uir_version: str = "3.0"` added; distinct from `schema_version`. |
| **A3 #1:** VLM-native trigger when Docling bbox quality blocks visual retrieval | No | Reversal triggered only by extraction Format deficit, not visual coherence | **APPLIED** — §6.2 fork-back trigger 4 added: bbox/visual-coherence trigger via ColPali patch ↔ Docling chunk box IoU <50% on >20% of deficit-doc pages. |
| **A3 #2:** Per-document-class VLM-native trigger | No | 15% corpus-wide average could mask 40% rejection on a single class | **APPLIED** — §6.2 fork-back trigger 3 added: >10% rejection rate on any single document class triggers evaluation even if corpus-wide average <15%. |
| **A3 #3:** Docling 2.86 lifecycle risk absent | No | Docling pin has no monitored renegotiation threshold | **APPLIED** — R21 added; quarterly Docling lifecycle check owned by V3 governance (re-homed after v2.X cycle-open checklist archived). |
| **A4:** C-spike sweep grid too narrow (only visual weight varied) | **Condition 5** | Dense:sparse ratio assumed optimal at v2.16; RRF k=60 unjustified for 3-leg topology | **APPLIED** — §3.4 #5 expanded: 4 profiles × 7 visual weights (28) + dense:sparse ratio top-3 (36) + RRF k ∈ {30,60,100} (joint) = ~64 configurations. C5 task description updated. |
| **A5 #1:** JSON field-order normalization for identity gate | No | Python 3.10 dict insertion order may differ | **APPLIED** — §8.2 normalization rules: `sort_keys=True` before hashing. |
| **A5 #2:** Floating-point confidence rounding | No | Last-significant-digit drift triggers false regressions | **APPLIED** — §8.2 normalization rules: ±0.01 tolerance for `ConfidenceBreakdown` fields. |
| **A5 #3:** `chunk_id` derivation handled by separate gate | No | Conflating content vs chunk_id identity | **APPLIED** — §8.2 identity-relevant-field table marks `chunk_id` as not identity-relevant; handled by chunk_id stability ratio gate. |
| **A5 #4:** Identity-relevant vs metadata-only fields enumeration | No | Implicit; CI tool ambiguity | **APPLIED** — §8.2 table enumerates each field's identity-relevance. |
| **B6:** R4 visual storage severity underrated | No | 5000-doc trajectory exceeds 2 TB indexed | **APPLIED** — R4 upgraded to High-Impact with growth-trajectory note; §7.11 visual-index stability contract; re-embed cost table. |
| **B7:** Omission-by-reframing failure mode uncaught | **Condition 3** | 8-layer guard stack catches token/entity/relation changes but not retrieval-behavior shift from pragmatic reframing | **APPLIED** — §3.3 dominance criterion item 7 + B12 task added: top-5 doc-ID overlap ≥95% per query between heuristic and llm mode on v2.16 regression-fixture queries. Failure → `LLM_RETRIEVAL_DRIFT` Format regression. |
| **B8 #1:** Byte-for-byte gate breaks on dict ordering | No | Covered under A5 #1 | **APPLIED** — see A5 #1. |
| **B8 #2:** Floating-point confidence drift | No | Covered under A5 #2 | **APPLIED** — see A5 #2. |
| **B8 #3:** Metadata exclusion list | No | Covered under A5 #4 | **APPLIED** — see A5 #4. |
| **B9:** Multi-hop chunk_id migration absent | No | Consumers skipping schema versions must chain per-phase maps | **APPLIED** — §7.12 added: cumulative chunk_id rewrite map regenerated at each phase boundary. |
| **B9 (regret #4):** Content-derived `chunk_id` should be Phase A, not v3.3 | **Condition 1** | Position-derived `chunk_id` is highest-regret-risk decision | **PARTIALLY APPLIED** — content-derived `chunk_id` made a *contingent* Phase A scope-negotiation option triggered by A6 publishing >10% rewrites. Deferral to v3.3 retained as default path *unless* A0 evidence shows first-trigger threshold churn. Regret #4 monitoring sharpened: A6 >10% → invoke immediately in Phase A; B7 >5% → invoke in v3.2; C8 >2% → invoke in v3.3. Audit's recommendation preserved as falsifiable contingency. |
| **C10:** Retrieval-path observability limited to opt-in global trace | No | User-reported "wrong document" debug requires re-runnable query (visual non-determinism prevents) | **APPLIED** — §7.8 `RetrievalDebugPayload` dataclass + `return_debug_payload=True` argument added; per-query payload, no global tracing required. |
| **C11 #1:** EPUB engine implementation-void | No | `ConversionPlan` accepts EPUB fields with no engine | **APPLIED** — §3.2 explicit "EPUB engine — interface-ready, implementation-void in v3.0" statement; downstream consumers must treat EPUB fields as forward-compatible reservations. |
| **C11 #2:** Region-level visual retrieval scope expansion | No | C-spike PASS B FAIL has no budget renegotiation protocol | **PARTIALLY APPLIED** — §4.2 already says "Phase C scope expands"; the scope-negotiation protocol pattern from Phase A is intentionally not duplicated for Phase C because the C-spike gate runs *before* Phase C commitment (Phase C does not start at all if PASS B FAIL); Phase A overrun risk is materially higher because Phase A scope is unconditionally committed once the cycle starts. Documentation tightening only — no functional change. |
| **C11 #3:** Prompt-migration cost note absent | No | Prompt-template hash changes invalidate cache → full cold rebuild | **APPLIED** — §3.3 prompt-migration cost note added; CI fails builds where prompt hash changes without corresponding `B8_COLD_CACHE_COST.md` update. |
| **C11 #4:** Multi-hop chunk_id migration | No | Covered under B9 | **APPLIED** — see B9. |
| **C11 #5:** Docling 2.86 deprecation timeline | No | Covered under A3 #3 | **APPLIED** — see A3 #3. |
| **C12:** Refuse to start Phase B without 500-chunk stratified prompt spike | No | 100-chunk spike (B2 in 0.4) may miss hardest cases | **PARTIALLY APPLIED** — B2 description acknowledges audit's recommendation; the 100-chunk → 500-chunk expansion is a Phase B scope decision deferred to Phase B kickoff. No body-text change in 0.5; tracked as USER_ISSUES item to evaluate at B2 spike design time. |
| **D13:** Defer A6 schema bump to v3.1.0 boundary | No | Saves ~1 day Phase A | **NOT APPLIED** — counter-argument (CI comparison during Phase A development needs version label) is load-bearing; the schema bump is cheap and the dev-only `+uir` suffix workaround introduces a code path the team has to maintain. Documentation noting the audit's option preserved here. |
| **D14:** 30-minute paper experiment before pre-spike | No | Even cheaper falsification test | **NOT APPLIED to plan** — operational suggestion; recorded here for the implementer to execute at Phase C kickoff but not a structural plan change. |
| **D15:** Phase A scope-negotiation protocol | **Condition 4** | 24-day budget could break silently into 36–48 days | **APPLIED** — Phase A scope-negotiation protocol added with explicit triggers (A0 >4d, first 5 days A2 <20% progress, identity-explained-delta >5%) and UIR-shim fallback option that preserves the UIR contract while reducing refactor surface by ~50%. R22 added. |
| **E16 Greenlight Condition 1:** Content-derived `chunk_id` in Phase A | **Condition 1** | (see B9 row above) | **PARTIALLY APPLIED via regret #4 sharpening + Phase A scope-negotiation option** |
| **E16 Greenlight Condition 2:** `continuation_group_id` on UIRChunk | **Condition 2** | (see A2 #1 row above) | **APPLIED** |
| **E16 Greenlight Condition 3:** Retrieval-behavior regression test | **Condition 3** | (see B7 row above) | **APPLIED** as B12 + §3.3 #7 |
| **E16 Greenlight Condition 4:** Phase A scope-negotiation protocol | **Condition 4** | (see D15 row above) | **APPLIED** |
| **E16 Greenlight Condition 5:** Expanded C-spike sweep grid | **Condition 5** | (see A4 row above) | **APPLIED** |
| **E17 Communication artifact: 7/10** | No | 1323-line length + glossary gaps + Layer/Stage confusion | **PARTIALLY APPLIED** — Layer/Stage finding rejected as invalid (already consistent); 10-page executive summary recommendation NOT applied (would add documentation maintenance burden; the audit-trace appendix serves a similar navigation purpose for reviewers). Glossary gap for "identity half"/"explained-delta half" tracked as a future doc-cleanup task. |

### 9.1 Review Summary — Draft 0.3 Review (2026-05-25, archived)

Draft 0.3 received a principal-architect review (2026-05-25) answering 17 questions across Soundness/Risk/Missing/Phasing/Judgment. Overall judgment: conditional greenlight with 5 conditions, all addressed in Draft 0.4.

### 9.2 Changes in Draft 0.4 (By Review Finding) — archived

| Review Finding | Draft 0.3 Issue | Draft 0.4 Resolution |
|---|---|---|
| **A1:** Stage 2 diagram doesn't show dual-write fork | Boxed diagram showed serial step | §3.1 diagram redrawn with explicit fork; §5.2 data flow shows orchestrator + guards + both-and-diff merge |
| **A2:** `sanitization/edit_guard.py` lumps 6 guards into one module | Sub-component surface understated | §5.1: `sanitization/guards/` subpackage with one file per guard (now 8 guards) + `orchestrator.py` |
| **A3.a:** Missing `extraction_warnings`, `coordinate_frame`, `page_size_px`, `engine_options` | UIR contract gaps would bite in Phase C | §3.2: added `ExtractionWarning`, `CoordinateFrame` enum, `UniversalPage.page_size_px`, `ConversionPlan.engine_options` |
| **A3.b:** `structural_flags: Dict[str, bool]` is stringly-typed | "Flags additive" gate unfalsifiable | §3.2: `StructuralFlag` Enum (closed vocabulary) replaces dict |
| **A3.c:** `ConfidenceBreakdown` two-sentinel encoding | Bug-prone (None vs -1.0) | §3.2: single sentinel (None) + `applicable: Set[str]` field |
| **A5:** Docling fork-back trigger §6.2 too generous (AND-clause) | Back-door — either approach could pass | §6.2: OR-clause; added "or >15% LLM rejections trace to upstream extraction defects" |
| **A6:** Page-score broadcast to all chunks on page | RRF distortion — large pages flood pool | §3.4 #4: bounded top-N per page (N=3); replaces broadcast |
| **A7:** Profile-conditional weights described as "starting point" but not as priors | C5 buried as afterthought | §3.4 #5: explicit "PRIORS not measurements"; C5 sweep on critical path; §6.2 fork-back |
| **A8:** No visual-signal-preservation-through-rerank metric | ModernBERT could undo visual signal silently | §3.4 #7: fusion-vs-rerank flip-rate metric; degradation flips ≤ improvement flips required; §6.2 fork-back |
| **B1.a:** Qdrant payload migration plan absent | §7.3 mentioned rebuild, not cutover | §7.0 added: per-phase Qdrant cutover strategy with alias flip, 30-day retention, rollback test |
| **B1.b:** `chunk_id` derivation not surfaced | R15 — position-derived; downstream joins silently break | §7.1: chunk_id-derivation impact statement + per-phase rewrite map; C13 + R15 added; regret #4 added |
| **B1.c:** `ElementType` vs `Modality` migration unclear | Three type vocabularies in play (ChunkType, ElementType, Modality) | §7.1: reconciliation table; C14 added; one-way migration (no shims) |
| **B1.d:** Recall@1 67.8% baseline must be re-measured on v3.1.0 chunks before Phase C | Comparing pre-sanitization to post-visual-retrieval is incoherent | B10 task added; §3.4 acceptance gate references B10 baseline |
| **B2:** R13 underrated (semantic-identity gate alone) | Repo scope evidence (~17,000 lines coupled code) supports upgrade | R13 upgraded; identity-half + explained-delta-half split (§3.2); third-party regression check (Earthship, Harry Potter, Fluent_Python) |
| **B3:** Guard 7 entity-relation triples + Guard 8 corpus-dedup-ratio missing | Per-chunk guards miss semantic substitution + inter-chunk drift | §3.3: 8-layer guard stack; guard 7 + guard 8 added |
| **B4:** Sanitizer-judge correlated failure mode unmitigated | Qwen judges Qwen → blind spots | §3.3 #5 (golden set) + #6 (cross-family judge); R19 added; regret #5 added |
| **B5:** Inter-chunk consistency check missing | Cross-chunk content duplication invisible | Guard 8 (corpus-dedup-ratio) covers this |
| **B6:** §6.2 heuristic-deprecation trigger too weak | 2 soak iterations could fire on stable corpus | §6.2: 3 *production* cycles required, not soak iterations |
| **C1:** omlx tenancy/process model absent | R16 — query latency at risk | §7.7 added; R16 + Q11 added; `omlx/scheduler.py` in §5.1 |
| **C2:** Soak σ never measured + soak-as-gate latency burden | Phase B blocks on σ that doesn't exist | A7 task added (3 consecutive heuristic soaks in Phase A trailing days); R20 added; §7.3 soak-as-gate accounting |
| **C3:** Qdrant version assertion missing in C1 | "Multi-vector support" abstract; named-vector behavior is version-sensitive | C1 task strengthened: assert qdrant-server version meets named-vector requirement |
| **C4:** Retrieval API surface evolution undefined | Downstream RAG breakage unforced | §7.8 added; Q12 added |
| **C5:** `--profile-override` story for visual retrieval missing | Debug semantics unclear | §7.9 added |
| **C6:** `partial_table` deferral not explicit | Will be the next "inert" feature | §8.3 added explicit row; `StructuralFlag.PARTIAL_TABLE_CROSS_PAGE` reserved in enum |
| **C7:** VLM caption interaction unstated | Unforced ingest cost question | §7.10 added |
| **C8:** 17/18 skipped tests audit absent | Some skips may be UIR-dependent | A8 task added |
| **C9:** Non-Greenhouse holdout acquisition unspecified | R18 — Phase D exit at risk | §3.5 #5: acquisition plan table; 5-of-8 before Phase A, 3-of-8 by mid-C |
| **C10:** Sanitization-cache cold-start cost unmeasured | Q8 omits cold-cache | B8 task added; Q13 added; §7.3 wall-time table |
| **C11:** Sentinel chunk accounting in dominance criterion undefined | Endpoint flakiness could masquerade as LLM regression | §3.3 sentinel-chunk accounting clause; `LLM_SENTINEL_DEGRADED` marker |
| **D1:** Pre-spike (smaller than C-spike) absent | C-spike itself was the smallest test | §4.2 step 1 added: 2-hour pre-spike falsification test |
| **D2:** C-spike PASS B candidate-set construction loose | Sloppy test would give false PASS | §4.2 step 2 #8 tightened: exact ModernBERT, chunk_id dedup, fixture-based gold map |
| **D3:** Phase A per-doc spike absent | 12-day budget unvalidated on real refactoring evidence | A0 task added (3-day spike on ATZ_Elektronik_German) |
| **D4:** B/C parallelization missing switching rule | Interleaved-hour switches dominate calendar | §4.1: whole-day-block switching rule |
| **D5:** Phase A 12-day budget unrealistic | R17 — 10,891-line batch_processor + mapper + serializer | Phase A rebudgeted to 24 days (2× nominal); R17 added |
| **E1:** §3.2 root-cause diagnosis stale (claimed construction fragmentation as of 0.3) | Diagnosis 7 weeks out of date; undermined credibility of architectural rationale | §3.2 rewritten: construction unified 2026-04-30; real coupling is in mapper/serializer/chunker call sites; ~17,000 lines of coupled code documented |
| **E2:** `partial_code` cross-page emission treated as pure refactoring | Was actually a behavioral change | §3.2 acceptance gate explicitly admits behavioral change; explained-delta half of gate enumerates intentional deltas |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---|---|---|---|
| 0.1 | 2026-05-25 | Claude Code (Opus 4.7) | Initial synthesis of Draft A + Draft B + governance audit. |
| 0.2 | 2026-05-25 | Claude Code (Opus 4.7) | Incorporate 17-point external architecture review. Restructured layer model; strengthened UIR contract; semantic-identity gate; heuristics retained; Phase C-spike; granular schema versioning; expanded risk register; determinism/observability/failure-mode policies; fork-back triggers; regret-risk register; glossary. |
| 0.3 | 2026-05-25 | Claude Code (Opus 4.7) | Incorporate follow-up architecture review on Draft 0.2. Replaced LLM dominance criterion with aggregate + degradation-rate; defined chunk-identity key + diff predicate; visual_weight floor (0.1); C-spike PASS B (reranker discrimination); content-pinning cache; fusion re-normalization; UIRChunk provenance contract; ConfidenceBreakdown→Optional; partial-release policy; R12 calendar overlay; 8-doc holdout; LLM test strategy as B7; corpus-size target C12; risk R14. ALL body sections edited — not just appendix. |
| 0.4 | 2026-05-25 | Claude Code (Opus 4.7) | Incorporate principal-architect review on Draft 0.3 (17 questions). Corrected §3.2 root-cause diagnosis (Docling construction was unified 2026-04-30; real coupling is mapper.py + docling_serializers.py + chunker call sites; ~17,000 lines documented). Rebudgeted Phase A to 24 days (R17). Split semantic-identity gate into identity-half + explained-delta-half with audit table + third-party regression check. UIR contract extended (extraction_warnings, coordinate_frame, page_size_px, engine_options, typed StructuralFlag enum, single-sentinel ConfidenceBreakdown). Per-doc Phase A spike (A0). 2-hour pre-spike before C-spike; C-spike PASS B tightened. Visual-retrieval bounded join (top-N=3 per page; replaces broadcast). Fusion weights marked as priors with mandated sweep on critical path. Visual-signal-preservation-through-rerank metric. 8-layer guard stack (added entity-relation guard 7 + corpus-dedup-ratio guard 8). 50-chunk human-labeled golden set + cross-family judge spot-check (R19 sanitizer-judge correlated-failure-mode mitigation). Sentinel-chunk accounting. §6.2 heuristic-deprecation strengthened to 3 production cycles. §7.0 Qdrant cutover plan. §7.1 chunk_id-derivation impact + ElementType/ChunkType/Modality reconciliation. §7.7 omlx tenancy & scheduling. §7.8 retrieval API surface evolution. §7.9 --profile-override semantics. §7.10 VLM-caption interaction. 17/18 skipped tests audit (A8); σ baseline measurement in Phase A (A7); 5-of-8 holdout acquisition in Phase A (A9), final 3-of-8 in Phase B (B11); cache cold-start (B8); cross-family judge (B9); v3.1.0 deficit baseline re-measurement (B10). R15–R20 added. Regret #4 + #5 added. ALL body sections edited; appendix §9 maps all changes to review findings. |
| 0.5 | 2026-05-26 | Claude Code (Opus 4.7) | Incorporate fourth principal-architect review (Draft 0.4, 17 questions, conditional greenlight with 5 conditions). All 5 greenlight conditions addressed (4 fully, 1 partially as contingent option). UIR contract extended: `UIRChunk.continuation_group_id` (cross-page sibling joins per audit A2 #1) and `UIRChunk.uir_version` (internal contract version per A2 #5). `ConversionPlan.render_dpi` validation `[72, 600]` enforced via `__post_init__` (A2 #4). C5 weight sweep grid expanded to ~64 configurations: 4 profiles × 7 visual weights + dense:sparse ratio top-3 (36) + RRF k ∈ {30,60,100} (A4, Condition 5). Phase A scope-negotiation protocol with UIR-shim fallback option (D15, Condition 4) added; triggers: A0 >4d, first 5 days A2 <20% progress, identity-explained-delta >5%. B12 retrieval-behavior regression test: ≥95% top-5 doc-ID overlap between heuristic and llm mode on v2.16 regression-fixture queries (B7, Condition 3); catches omission-by-reframing failure mode. §6.2 VLM-native fork-back trigger strengthened: per-doc-class >10% rejection threshold (A3 #2) + bbox/visual-coherence trigger via ColPali patch ↔ Docling chunk IoU <50% on >20% of deficit-doc pages (A3 #1). R4 (visual storage) upgraded to High-Impact with 5000-doc growth-trajectory and re-embed cost note (B6). §7.11 visual-index stability contract: pins ColPali model identifier + render_dpi per visual collection; re-embed cost table; upgrade protocol. §7.12 cumulative chunk_id rewrite map for multi-hop schema-version migrations (B9). §7.8 `RetrievalDebugPayload` dataclass + `return_debug_payload=True` argument: per-query fusion-trace inspection without global logging (C10). §8.2 identity-half gate normalization rules: JSON `sort_keys=True`, ±0.01 confidence-value tolerance, identity-relevant-field enumeration table (A5, B8). §3.2 EPUB engine "interface-ready, implementation-void" explicit statement (C11 #1). §3.3 sanitization prompt-migration cost note: prompt-template hash change invalidates cache → full cold rebuild (C11 #3). Regret #4 monitoring sharpened: A6 >10% rewrites invokes content-derived `chunk_id` immediately in Phase A (Condition 1, partial — preserved as contingent option not unconditional flip). R21 (Docling 2.86 lifecycle risk) + R22 (Phase A overrun) added. Audit finding A1 #1 (Layer/Stage terminology inconsistency in §3.2/§3.3 headers) **REJECTED as invalid** — actual document already uses Stage 1/Stage 2 consistently at lines 241 and 470. §9 appendix extended with §9.0/§9.0.1 mapping all 0.4-round audit findings to 0.5 resolutions (including findings deferred, partially applied, and rejected with rationale). |

---

**END OF ARCHITECTURE_V3_DRAFT_0.5.md**
