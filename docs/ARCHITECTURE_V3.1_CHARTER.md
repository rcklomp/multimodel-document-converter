# Architecture Charter: MM-Converter V3.1

**Date:** 2026-06-03
**Status:** Working charter (as-built + roadmap). NOT a "finished system" claim.
**Author:** Claude (Opus 4.8), grounded in the repo at commit `b44724b` and this
cycle's measured telemetry. External citations verified 2026-06-03 (see Section 10).

**Status legend used throughout:**
`[SHIPPED]` exists in the code and is test-covered ·
`[PARTIAL]` exists but incomplete or carries known debt ·
`[PROPOSED]` designed here, not yet built.

This charter supersedes the Gemini V3.0 / V3.1 drafts (which contained factual
errors and capability overclaims) and updates the target in
`docs/ARCHITECTURE_V3_DRAFT_0.5.md` with shipped reality. Promote to canonical per
`docs/V3_EXECUTION_MANDATE.md` only after the Section 9 acceptance gates pass.

---

## 1. Executive Summary (honest framing)

V3 is **vision-native extraction ADDED to the v2 pipeline, not a replacement of
it.** The shipping system is a two-engine hybrid: a VLM extracts visually-complex
or code-dense pages, while `DoclingFastEngine` (CPU, OCR-off) handles prose pages
for cost. Docling therefore remains live, and so do its caveats (stripped code
indentation, per-page layout inconsistency, placeholder images). Pretending V3
"replaced" V2 is the mistake that produced the previous drafts.

V3's value is real but bounded. In a head-to-head soak (same embedder, reranker,
GX10 judge, seed 7), V3 beat the v2.16 baseline on every axis:

| Metric | V3 | v2.16 | delta |
|---|---:|---:|---:|
| Recall@1 (chunk) | 77.3% | 54.5% | +22.8 pp |
| Recall@5 (chunk) | 95.5% | 63.6% | +31.9 pp |
| Recall@5 (doc) | 100.0% | 86.4% | +13.6 pp |
| Relevance | 84.1% | 72.7% | +11.4 pp |
| Format (judge TRUSTWORTHY) | 97.7% | 95.5% | +2.2 pp |
| Faithfulness | 84.1% | 61.4% | +22.7 pp |

**Caveat that must travel with these numbers:** this was an 11-document overlap
sampled from a budget-truncated soak (16 of 43 docs), single seed. It is
**directional evidence, not a full-corpus result.** v2.16 was never at a
"mathematical limit"; V3 simply has more headroom. (The 98.6% figure that appeared
in the Gemini drafts was a v2.12 number on a different fixture.)

The instability seen this cycle was **integration debt, not an engine fault**: the
VLM's richer, more dynamic output kept hitting V2-era contracts that fail closed.
The fix is structural (Section 2), and most of it has shipped. What remains is
completing the vocabulary migration, retiring or hardening the Docling lane, and
validating at corpus scale.

---

## 2. Organizing Principle: the Fail-Open Invariant

The root cause of every V3 failure this cycle was the same shape: the extraction
engine was swapped (Docling geometric -> VLM semantic) but the surrounding
contracts (schema, vocabulary, router, resilience model) were left on V2-era
assumptions, so richer VLM output hit boundaries that **fail closed** - they
crash, reject, strip, or silently drop.

**Invariant:** every extraction-to-disk boundary MUST fail open - generate,
fit, carry-through, or degrade-with-provenance - never crash, reject, or silently
drop. Each boundary is enforced by an executable contract test.

### 2.1 Boundary Register (the invariant made concrete)

| # | Boundary | V2-era assumption (fail-closed) | Fail-open contract (V3.1) | Enforcing test | Status |
|---|---|---|---|---|---|
| B1 | `from_uir` asset_ref (QA-CHECK-05) | Docling emits a binary image per picture | Materialize a bbox crop -> `asset_ref`; never reject | `test_v3_asset_materializer.py` | `[SHIPPED]` |
| B2 | `visual_description` 400 cap | OCR-era short captions | Truncate at producer; full text stays in `content` | `test_v3_asset_materializer.py` | `[SHIPPED]` |
| B3 | `ElementType` vocabulary | 3 values: text/image/table | Smuggle code/form as TEXT + `promoted_modality`; promote to `Modality.CODE/FORM` in the chunker; unknown -> TEXT + `original_vlm_type` provenance | `test_v3_vlm_code_form.py` | `[SHIPPED]` |
| B4 | VLM transport failure | Docling is local CPU; no network failure class | `VlmInfraError` hard-fails the batch (no silent Docling fallback); soak harness pauses-and-polls to recover | `test_v3_circuit_breaker.py`, `test_v3_resilient_breaker.py` | `[SHIPPED]` |
| B5 | empty-content asset chunk | text chunks always carry content | guard empty-content asset chunks before qdrant ingest | (regression for `b44724b`) | `[SHIPPED]` |
| B6 | VLM bbox accuracy | coordinates are trustworthy | adapter projects raw px -> `[0,1000]` and clamps; crop-audit flags edge-overflow + blank crops | `test_v3_asset_materializer.py` (crop-audit) | `[SHIPPED]`, residual risk in 3.3 |
| B7 | router code blind spot | object presence implies visual complexity | monospace-ratio >= 0.10 routes code-as-text to the VLM | router monospace guard | `[SHIPPED]` (`2a60a99`) |

### 2.2 Vocabulary migration (Charter §7.1)

`ElementType` is the legacy 3-value extraction vocabulary; `Modality` is the
5-value target (`TEXT, IMAGE, TABLE, CODE, FORM`). The migration is **one-way:
ElementType is to be retired, not widened** - the contract test
`test_modality_distinct_from_elementtype` enforces this. The current interim is
smuggle-and-promote (B3). `[PARTIAL]` Completion = the extraction adapter emits
`Modality` directly and `ElementType` is removed; tracked as a roadmap item
(Section 9), not done.

### 2.3 Validate on the hardest content class

Simple cases prove nothing about structural integrity. A schema fix smoked only
on a 1-page form passed while the first code-heavy document crashed in 30s. Every
boundary change MUST be validated against the hardest relevant class (code-heavy
manuals, dense multi-column magazines, degraded scans), not the easiest available
doc.

---

## 3. As-Built Architecture

Canonical flow (V3 default for `BatchProcessor.process_pdf`):
`HybridEngine.extract()` -> `UniversalDocument` (UIR) -> `chunk_universal_document()`
-> `materialize_visual_assets()` -> `IngestionChunk.from_uir()` -> JSONL.

### 3.1 Universal Intermediate Representation (UIR) `[SHIPPED]`
Engines are pure mappers that emit a `UniversalDocument` (`src/mmrag_v2/universal/
intermediate.py`), decoupling extraction from chunking. Today only a **PDF** engine
exists (fitz-based). `[PROPOSED]` ePub: the UIR contract is format-agnostic, so an
ePub engine is a defined extension point - but it is NOT built (both V3 engines
hardcode `fitz.open` / `file_type="pdf"`). Do not represent ePub as shipped.

### 3.2 Cost-Optimized Hybrid Router `[SHIPPED]`
`HybridEngine._classify_page` (`src/mmrag_v3/engines/router.py`) uses fast PyMuPDF
pre-flight signals to choose an engine per page:
- tables / images / drawings>threshold -> VLM;
- monospace-char ratio >= `0.10` -> VLM (rescues code-as-text from Docling, B7);
- otherwise -> `DoclingFastEngine`.

**Honest caveat:** the router is a *geometric/lexical heuristic gating a semantic
engine*. The monospace signal closes the code blind spot, but a page that is
visually complex without raster objects or monospace text can still be misrouted
to Docling. `[PROPOSED]` Hardening signal: text-block count / column detection via
`page.get_text("blocks")` as an additional VLM trigger - build only if the AIOS
smoke (Section 9) shows residual code/layout loss.

### 3.3 Vision-Native Extraction `[SHIPPED]`
`VlmNativeEngine` (`src/mmrag_v3/engines/vlm_native.py`) renders each routed page
to PNG and prompts a VLM (default Qwen3-VL-8B) for strict UIR JSON. The prompt
mandates: per-element type incl. `code`/`form`, exact indentation preservation for
code (markdown fences), key-value/markdown for forms, and chart-to-data
transcription for data visualizations.

- **BBox handling:** the VLM returns raw pixel coordinates; the Python adapter
  deterministically projects them into the normalized `[0,1000]` frame
  (`COORD_SCALE`) and clamps to range. **Residual risk (formally accepted):**
  projection + crop-audit bound two failure modes (out-of-range overflow, blank
  crops) but CANNOT detect a plausible, in-range, semantically-WRONG crop
  (interior misplacement). A crop of the wrong figure passes both gates. This is a
  known coverage boundary, not a solved problem.

### 3.4 Smuggle-and-Promote + Provenance `[SHIPPED]`
Per B3: code/form traverse the legacy enum as TEXT with a `promoted_modality` tag,
promoted to `Modality.CODE`/`FORM` in the chunker; unknown VLM types degrade to
TEXT with an `original_vlm_type` provenance marker (`3d5d9e5`) so no signal is
silently lost.

### 3.5 Asset Materialization + Crop-Audit `[SHIPPED]`
`src/mmrag_v2/universal/asset_materializer.py` is the single source of truth shared
by the batch path and the soak harness (so they cannot diverge - that divergence
caused the 0/18 crucible failure). It crops IMAGE/TABLE bbox regions to PNG and
sets `asset_ref`. Crop-audit emits per-crop `CropHealth` signals
(`is_full_page_fallback`, `is_edge_clamped`, `is_low_information` - the last reuses
the v2.9 blank definition `std<10 and (mean>250 or mean<5)`) and raises
`QA_WARN_CROP_DRIFT` above a 15% document drift rate, recorded in `meta.json`.

### 3.6 The Docling Lane `[SHIPPED]` + retained debt `[PARTIAL]`
`DoclingFastEngine` (`src/mmrag_v3/engines/docling_fast.py`, the sole V3 docling
import boundary) serves prose pages. It retains V2's caveats. The untracked
`scripts/postprocess_markdown.py` is a *symptom-level band-aid* for Docling's
per-page layout inconsistency (abbreviation pairs classed TEXT on one page / TABLE
on the next), figure placeholders, and mid-sentence page-break merges. Debt, named
honestly. See Section 8 (retire-or-harden decision).

### 3.7 Schema / Serialization `[SHIPPED]`
`IngestionChunk.from_uir` is the V3 emission boundary. `content` is canonical and
unmutated (the `_strip_c0_controls` validator preserves `\n` and `\t`, so code
indentation survives). `from_uir` already supports all 5 modalities; QA-CHECK-05
requires `asset_ref`+`spatial.bbox` only for IMAGE/TABLE.

---

## 4. Resilience & Operations `[SHIPPED]`

- **Circuit breaker:** `VlmInfraError` (transport timeout / connection refused /
  502/503/504/408) hard-fails rather than silently degrading to Docling; semantic
  errors (empty content, malformed JSON, non-retryable 4xx, 429, 500) still fall
  back per-page. The engine stays fail-fast; resilience policy lives only in the
  harness, so the production CLI never silently blocks.
- **Resilient pause-and-poll** (`scripts/v3_batch_ingest.py
  ::_process_with_resilience`): on infra failure, poll `GET /v1/models` every 60s
  and resume on recovery. **Two bounded guards** (both required): a 30-minute
  continuous-unreachability ceiling AND a `max_resume_attempts` cap (default 5)
  for the "HTTP-200-but-inference-dead" flap that the ceiling alone cannot catch.
  `--strict-breaker` restores instant-fail for short attended runs.
- **Gates:** `scripts/smoke_production.sh` (`SMOKE_PRODUCTION_PASS`, mandatory
  pre-merge for any extraction-path change) and `scripts/qa_full_conversion.py`
  (`QA_PASS`/`QA_WARN`/`QA_FAIL`).

---

## 5. Hardware Topology & Throughput Budget

Segregate work by its binding constraint. VLM decode is **memory-bandwidth-bound,
not compute-bound**.

| Role | Host | Why | Status |
|---|---|---|---|
| VLM extraction | M5 Max (mlx-vlm, 4/8-bit) | bandwidth-rich unified memory (est. >=546 GB/s) + fast prefill | `[SHIPPED]` |
| LLM-as-judge / soak scoring | GX10 / GB10 (vLLM, Qwen2.5-14B-FP8) | stable FP8 text inference; bandwidth-starved for VLM but fine for the judge | `[SHIPPED]` |
| Embedding + rerank | omlx-server (Mac Mini) Qwen3-Embedding-8B + ModernBERT | local, LAN | `[SHIPPED]` |
| Vector store | Qdrant | dense + sparse collections | `[SHIPPED]` |

**Bandwidth rationale (corrects the "discrete GPU" error):** the GB10 (DGX Spark)
is a *unified-memory* machine (128GB LPDDR5X, ~273 GB/s), the same architecture
class as the M5 - not a discrete GPU. Its 11.3 tok/s BF16 8B decode is ~66% of the
~17 tok/s theoretical ceiling that 273 GB/s allows (one weight-read per token); it
is bandwidth-starved, and post-fix it is *stable*, not OOM-crashing. The M5
advantage is bandwidth + prefill, giving a **measured ~2.7x on comparable pages,
within an honest ~2.6-3.8x range** (not the "7x" of the drafts).

**Throughput budget (`[PARTIAL]` - estimate, not validated at scale):** measured
floor is ~49 s/VLM-page (doc `0013`, a 1-page form, n=1 - dense pages with ~17k
-token prompts run slower). Only routed-to-VLM pages incur this; prose pages are
sub-second on Docling. A ~600-page crucible is therefore a multi-hour run and an
11,000-page Grand Soak is multi-day. **The Grand Soak has NOT been run; the largest
validated run to date is a single-document smoke.** Budget VLM page-hours before
committing.

---

## 6. Retrieval

### 6.1 Existing stack `[SHIPPED]`
V3 ingestion feeds the retrieval stack built across v2.11-v2.14, which the Gemini
drafts ignored: local omlx **Qwen3-Embedding-8B** dense vectors + **BM25** sparse,
fused via **RRF**, reranked by a local **ModernBERT** cross-encoder, with **HyDE**
query expansion. ColPali must integrate with this, not replace it.

### 6.2 ColPali late-interaction reranker `[PROPOSED]`
For visually-dense queries where text embedding is insufficient (engineering
schematics, heavily degraded scans), add ColPali (Faysse et al., arXiv:2407.01449)
as a **targeted late-interaction reranker, not a primary index.** Rationale: ColPali
stores ~1024 patch vectors/page (PaliGemma 32x32 grid) x 128-dim (~256KB/page raw);
at 11k pages that is ~11M multi-vectors, and Qdrant multi-vector + MaxSim latency
makes a blanket primary index cost-prohibitive.

Design constraints (must be satisfied before adoption):
- **Scope:** invoked only for queries classified visual-intent, over a candidate
  set already narrowed by the text stack (6.1) - a reranker, not a retriever.
- **Cost control:** apply patch/token pooling (~2-3x vector reduction at minor
  recall cost) and quantify the storage + p95 latency budget on real hardware.
- **Host:** name the GPU that holds the ColQwen/ColPali model (unspecified today).
- **Fusion:** define how MaxSim scores combine with the RRF+ModernBERT scores.

Do not build Pillar 6.2 until 6.1 is shown insufficient on a measured visual-query
set; otherwise it is the next un-budgeted seam.

---

## 7. Modality-Aware Quality Gates `[PROPOSED]`

Uniform RAG rubrics penalize structured output for lacking prose. The judge rubric
must switch on the chunk's `modality` (the real 5-value enum, not "prose"):
- `TEXT`: linguistic faithfulness + relevance (current default).
- `TABLE` / `FORM`: schema adherence + key-value mapping; ignore prose fluidity.
- `CODE`: **indentation fidelity + syntax preservation** - code was the dominant
  failure class this cycle, so this rubric is mandatory, not optional.

Status: the judge harness (`scripts/synthetic_soak.py`) exists `[SHIPPED]`; the
modality-switched rubric matrix is design intent `[PROPOSED]`.

---

## 8. Risk Register & Open Questions

| Risk | Severity | Mitigation / decision needed |
|---|---|---|
| VLM JSON invalid on dense pages (NEW, 2026-06-02 soak) | High - BLOCKER | Whole-page strict-JSON truncates/malforms on dense layouts (Combat Aircraft magazine: ~25/43 pages -> Docling fallback). No `finish_reason=length` detection today. Fix: raise/handle `max_completion_tokens`, guided/constrained JSON decoding, or per-region extraction. The "one JSON per page" design has a density ceiling. |
| VLM bbox crop drift / interior misplacement (3.3) | High - MEASURED 40-50% | Confirmed in the 2026-06-02 soak: `QA_WARN_CROP_DRIFT` fired on 5 of 8 docs (forms/scans/tables). crop-audit flags it but extraction produces it. No longer "accept-and-monitor" - a confirmed blocker. Needs bbox-fidelity work (higher-fidelity coords / per-region crops) or a semantic crop-vs-description check. |
| Router misroute residue (3.2) | Med | Monospace closes code; broad layout complexity open. Measure via AIOS smoke before building a heuristic. |
| Docling-lane debt (3.6) | Med | Decide: retire the lane (route everything to VLM, eat the cost) OR harden it (fold `postprocess_markdown` into the engine + add a per-page-consistency check). Currently a band-aid. |
| Grand Soak (5) | High - RUN + HALTED 2026-06-02 | Stopped at doc 9/17: pipeline does not meet requirements on dense docs (JSON-invalid -> Docling fallback; 40-50% crop drift). The long tail (~20 books) was excluded by `--max-pages 200`. Do not re-run until the two extraction blockers above are fixed. See `docs/paper/FINDINGS_LOG.md` 2026-06-02. |
| Single-point M5 dependency | Med | Extraction has one bandwidth-rich host. Define a fallback (OpenRouter cloud VLM) and its cost ceiling. |
| ColPali cost if adopted (6.2) | Med | Gated behind the 6.2 constraints; do not adopt blind. |
| ElementType migration half-done (2.2) | Low | Smuggle-and-promote is stable interim; complete the one-way migration to remove the seam class. |

---

## 9. Roadmap & Definition of Done

"Done" is gate-defined, not vibe-defined.

1. **AIOS code-extraction smoke** (immediate next move; needs M5 up). Run a
   code-heavy doc through the soak path; inspect the Docling-routed pages for code
   loss and the `crop_audit` block. This measures the B7/router residual.
2. **Run + validate the crucible** (18 docs) end-to-end: every doc `status=ok`,
   `QA_PASS`, crop-audit within threshold. Only then is the extraction path
   "validated," not just "green in unit tests."
3. **Retire-or-harden the Docling lane** (Section 8 decision).
4. **Complete the ElementType -> Modality migration** (2.2).
5. **Modality-aware CODE/TABLE/FORM judge rubric** (Section 7).
6. **(Conditional) ColPali reranker** (6.2), only if 6.1 is measured insufficient.
7. **Grand Soak** with a budgeted page-hour ceiling and the resilient breaker.

A component is "done" when its boundary contract test (Section 2.1) is green AND it
has passed on the hardest relevant content class (2.3) AND the run-level gate
(`SMOKE_PRODUCTION_PASS` + `QA_PASS`) holds.

---

## 10. External References (verified 2026-06-03)

| Source | What it actually is | How V3.1 uses it |
|---|---|---|
| ColPali - Faysse et al., [arXiv:2407.01449](https://arxiv.org/abs/2407.01449) | Late-interaction multi-vector retrieval over image patches (PaliGemma, ~1024 patches/page, 128-dim, MaxSim) | Foundation for the PROPOSED visual reranker (6.2) |
| GOT-OCR 2.0 - [arXiv:2409.01704](https://arxiv.org/abs/2409.01704) | 580M end-to-end model (VitDet + Qwen-0.5B), raw image -> markdown/LaTeX, no cascade | Validates the vision-native thesis (3.3) |
| LlamaParse Cost Optimizer / Auto Mode ([LlamaIndex](https://www.llamaindex.ai/blog/optimize-parsing-costs-with-llamaparse-auto-mode)) | Per-page tier routing run in parallel (standard vs Premium/Agentic) | Industry precedent for the cost router (3.2) |
| MinerU - [arXiv:2409.18839](https://arxiv.org/pdf/2409.18839) | Pipeline backend is a layout-detection + OCR **cascade** (doclayout_yolo + PP-OCRv5 + formula/table models); MinerU2.5 adds a decoupled VLM | The cascade approach V3 moves AWAY from (not a vision-native exemplar) |
| HierFinRAG - [MDPI Informatics 13(2):30](https://www.mdpi.com/2227-9709/13/2/30) | Table-text GNN + symbolic-neural fusion for FinQA arithmetic | Suggestive support for routing numeric/tabular queries to a symbolic engine. CAVEAT: 1-month-old, lower-tier venue - verify before adopting |
| DGX Spark GB10 ([NVIDIA](https://docs.nvidia.com/dgx/dgx-spark/hardware.html), [LMSYS](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)) | Unified 128GB LPDDR5X, 273 GB/s; token-gen is bandwidth-limited | Backs the bandwidth-segregation rationale (5) |

External benchmarks are not a substitute for measuring on this corpus and hardware.
Every adoption above is gated on a local measurement (Sections 6.2, 9).
