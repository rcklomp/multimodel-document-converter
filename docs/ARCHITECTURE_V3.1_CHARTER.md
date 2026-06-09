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

**Terminology (reconciling "fail-open" with the "FAIL-CLOSED" extraction ladder):**
each individual boundary "fails OPEN" (degrades rather than crashing); the extraction
ladder as a whole (`src/mmrag_v3/processor.py::extract`, `processor.py` docstring
"FAIL-CLOSED") is "fail-CLOSED" against silent DATA LOSS - it never lets an
engine/network failure zero a text-bearing page. Same don't-lose-data goal from
opposite ends; the two terms are not in conflict. (NOTE: the ladder's actual
3-tier behavior + provenance keys are documented in PROJECT_STATUS and will be folded
into Section 3/4 here per `docs/PLAN_EXTRACTION_FIDELITY_V1.md` Phase 5; that plan
also corrects this section's circuit-breaker description, which the shipped ladder
superseded.)

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
| B8 | per-batch heading reset (cluster B) | heading context is per-call | carry the last active heading across batch boundaries; a real in-page/TOC heading still overrides the carry | `test_chunk_universal_document_contract.py` | `[SHIPPED]` (`71aeed1`) |
| B9 | TOC-cell sanitizer drops empty-content chunks (cluster D) | only text chunks reach it | preserve non-text / empty-content chunks (it deleted ALL images); empty-text dropped at the canonical boundary | `test_toc_cell_marker_sanitizer.py` | `[SHIPPED]` (`dd4a758`) |
| B10 | asset render/encode failure discards the batch (cluster A) | the crop always renders | crop encode -> full-page fallback; drop any asset-less IMAGE/TABLE BEFORE `from_uir` so no render-fail re-triggers the QA-CHECK-05 batch-discard | `test_v3_b2_reextraction.py` | `[SHIPPED]` (`7b1871b`, `de1af9d`) |
| B11 | separator-less / corrupt pipe table (cluster C) | the extractor emits valid grids | repair at the engine-agnostic chunker chokepoint; guarded (escaped-pipe, ragged-bail, title-tolerance, single-dash) so it never ships a gate-passing corrupt grid | `test_table_markdown.py` | `[SHIPPED]` (`b032a29`, `de1af9d`) |
| B12 | no-VLM image (cluster D) | every image is described inline | retain as a documented ID-only fallback (`vision_status=no_vlm`); describe POST-conversion via `enrich_image_chunks_v29.py`; gate advisory, not failure | `test_qa_image_gate_calibration.py`, `test_tiny_icon_filter.py` | `[SHIPPED]` (`dd4a758`) |

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
`mmrag_v3.extract()` (the router picks the engine: `MineruQwenHybridEngine` by
default when `MINERU_ENDPOINT` is set, else the legacy `HybridEngine`) ->
`UniversalDocument` (UIR) -> `chunk_universal_document()` (heading carry-forward
across batches, B8) -> `materialize_visual_assets()` -> `IngestionChunk.from_uir()`
-> JSONL. Image DESCRIPTION is a separate POST-conversion enrichment step (3.5a).

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

**Default extractor: MinerU2.5 (`MineruNativeEngine`, 2026-06-05).** After the
VLM evaluation (`docs/PLAN_VLM_EVAL.md`), MinerU2.5 (`opendatalab/MinerU2.5-*-1.2B`)
replaced Qwen3-VL-8B as the default extraction engine. It is a two-stage document
VLM (global layout detector -> per-region recognition) whose detector bboxes are
reliable - structurally resolving Blocker B crop drift (§9.1) and reading dense
tables Qwen emptied. `MineruNativeEngine` (`src/mmrag_v3/engines/mineru_native.py`)
renders each page with PyMuPDF and drives a MinerU server over HTTP via the light,
lazy-imported `mineru_vl_utils` http-client (the model stays in an ISOLATED server
- mlx on the M5 / vLLM on the GX10 - never in the mmrag-v2 env). It emits a flat
element list `{type, bbox[0,1], content, merge_prev}`; the converter projects bbox
to `[0,1000]`, maps MinerU's 13-type vocabulary onto the 3-value `ElementType`
(code smuggled as TEXT per B3), folds `merge_prev` continuations, and transcodes
MinerU's HTML tables into Markdown grids (the pipeline R2 contract).

**Default route is the MinerU+Qwen-for-code hybrid (`MineruQwenHybridEngine`,
2026-06-06)**, not pure MinerU: when `MINERU_ENDPOINT` is set, code-dense pages
(monospace ratio >= 0.10) route to Qwen (clean indentation, R3 1.00) and every
other page to MinerU2.5 (tables/layout) - neither engine alone passes a code-heavy
doc. Pure MinerU via `USE_MINERU_ENGINE=1`; the legacy Docling+VLM `HybridEngine`
is the no-`MINERU_ENDPOINT` fallback. Corpus-validated: the full 16-doc crucible is
16/16 clean QA_PASS post-enrichment, `leak=0` (2026-06-08).

Separator-less / corrupt pipe tables from EITHER engine are repaired at the
engine-agnostic chunker chokepoint (`universal/table_markdown.py`, B11), not per
adapter, so the MinerU HTML transcode and the Qwen pipe-markdown both converge.

The Qwen vision-native path below is RETAINED as an alternative
(`USE_VLM_ENGINE=1`) and inside the `HybridEngine` per-page router:

`VlmNativeEngine` (`src/mmrag_v3/engines/vlm_native.py`) renders each routed page
to PNG and prompts a VLM (Qwen3-VL-8B) for strict UIR JSON. The prompt
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

### 3.5a Multimodal Image Policy + Enrichment Lane `[SHIPPED]` (2026-06-08)
Image DESCRIPTION is a POST-conversion step, not conversion-time (the conversion
path only uses the VLM for full-page-guard verification). IMAGE chunks are always
retained: with `--vision-provider none` they ship as documented ID-only fallbacks
(`vision_status=no_vlm`, asset filename), which the strict gate treats as an
advisory (`IMAGE_NO_VLM`), not a failure - but only with a real `asset_ref` (B12).
Descriptions are produced by `scripts/enrich_image_chunks_v29.py`, now
env-pointable at a LOCAL OpenAI-compatible VLM (`MMRAG_ENRICH_PROVIDER/MODEL/
BASE_URL`; DashScope cloud is the unchanged default). Two hygiene filters run in
the export chain, BOTH behind a page-coverage guard so neither can manufacture
`MISSING_PAGES`: `_filter_tiny_icon_images` (icon/glyph regions, rendered <96px
both dims AND <1.5KB) and `_promote_or_drop_empty_tables` (empty-content tables;
the only-chunk-on-page case is PROMOTED to IMAGE, keeping the rendered crop).

### 3.6 The Docling Lane `[SHIPPED]` + retained debt `[PARTIAL]`
`DoclingFastEngine` (`src/mmrag_v3/engines/docling_fast.py`, the sole V3 docling
import boundary) serves prose pages. It retains V2's caveats: Docling's per-page
layout inconsistency (abbreviation pairs classed TEXT on one page / TABLE on the
next), figure placeholders, and mid-sentence page-break merges. A symptom-level
band-aid (`scripts/postprocess_markdown.py`, untracked, V2-era, operated on
Docling *markdown* not the V3 JSONL) was **removed 2026-06-09** (PR review flag:
footgun, in-place-overwrite default, never wired into the V3 path). The
underlying debt is unchanged - see Section 8 (retire-or-harden decision); if the
harden path is chosen, fold the fix into the engine with a per-page-consistency
check, not a post-hoc markdown repair.

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
| VLM JSON invalid on dense pages (NEW, 2026-06-02 soak) | RESOLVED for the default path (MinerU, 2026-06-05) | Was: whole-page strict-JSON truncates/malforms on dense layouts. MinerU2.5 (now default, §3.3) does NOT emit whole-page JSON - it emits per-element structured output with built-in anti-repetition, so the density ceiling does not apply; the dense CarOK spreadsheet Qwen emptied now yields a full Markdown table. The A1-A4 scaffolding remains for the retained Qwen path. |
| VLM bbox crop drift / interior misplacement (3.3) | RESOLVED for the default path (MinerU, 2026-06-05) | Was: MEASURED 40-50% drift on the 2026-06-02 soak. MinerU2.5's two-stage layout DETECTOR emits reliable region bboxes (the §5 research precedent), structurally fixing the crop drift; the 7/7 corpus soak passed crop-audit (`PLAN_VLM_EVAL` §14). The semantic in-range-but-wrong-crop residual (3.3) still applies in principle but was not observed. |
| Router misroute residue (3.2) | Med | Monospace closes code; broad layout complexity open. Measure via AIOS smoke before building a heuristic. |
| Docling-lane debt (3.6) | Med | Decide: retire the lane (route everything to VLM, eat the cost) OR harden it (a per-page-consistency check folded INTO the engine). The untracked `postprocess_markdown` band-aid was removed 2026-06-09; the underlying per-page-inconsistency debt is unaddressed, not masked. |
| Grand Soak (5) | High - RUN + HALTED 2026-06-02 | Stopped at doc 9/17: pipeline does not meet requirements on dense docs (JSON-invalid -> Docling fallback; 40-50% crop drift). The long tail (~20 books) was excluded by `--max-pages 200`. Do not re-run until the two extraction blockers above are fixed. See `docs/paper/FINDINGS_LOG.md` 2026-06-02. |
| Single-point M5 dependency | Med | Extraction has one bandwidth-rich host. Define a fallback (OpenRouter cloud VLM) and its cost ceiling. |
| ColPali cost if adopted (6.2) | Med | Gated behind the 6.2 constraints; do not adopt blind. |
| ElementType migration half-done (2.2) | Low | Smuggle-and-promote is stable interim; complete the one-way migration to remove the seam class. |

---

## 9. Roadmap & Definition of Done

Cycle status (UPDATED 2026-06-06): the 2026-06-02 Grand Soak HALTED at doc 9/17 on
the two §8 blockers (dense magazines + forms/scans; see `docs/paper/FINDINGS_LOG.md`
2026-06-02). The 2026-06-04 VLM-eval pivot then **resolved both blockers
structurally** by replacing the prose/visual extractor: **MinerU2.5 is the chosen
extractor** (two-stage layout detector -> per-region recognition: reliable detector
bboxes fix Blocker B crop drift; structured per-element JSON fixes Blocker A
malformation), shipped as the default `MineruQwenHybridEngine` route. So §9.1 below
is **SUPERSEDED** (kept for history), and the roadmap now leads with **§9.2**: the
remaining gate is re-running the FULL crucible (18 docs, long tail) on the MinerU
default to confirm the pivot holds at corpus scale. Validated so far: 6/6 golden +
7/7 cross-category corpus soak QA_PASS (default route). Full eval/decision record:
`docs/PLAN_VLM_EVAL.md` §10-16.

### 9.1 Remediation plan for the two blockers (SUPERSEDED 2026-06-06 by the MinerU pivot - history)

**SUPERSEDED:** these items targeted the OLD Docling+Qwen path; the MinerU pivot
(above) resolved Blockers A and B structurally, so none of A1-A5 / B1-B3 were
built as written. Retained as the design reasoning that justified the pivot
(A5's per-region precedent IS MinerU's two-stage design). Items below are
historical, not active work.

**Blocker A - VLM emits invalid JSON on dense pages** (truncation + malformation
-> mass Docling fallback; ~58% of pages on the one magazine reached):

- **A1. Detect truncation (cheap, do first).** `vlm_provider.describe` checks HTTP
  200 + empty content but NOT `finish_reason`. Treat `finish_reason == "length"`
  as a typed truncation signal (today it is silently a `json.loads` failure ->
  fallback): escalate the token budget once and retry, then hand to A4.
- **A2. Adaptive output budget.** Raise `max_completion_tokens` (the overnight run
  used 8192; the soak likely hit the 4096 default) and scale it with a cheap
  per-page element-density estimate. Bounded, to respect the self-hosted-OOM
  constraint already noted in the provider.
- **A3. Guided JSON decoding (structural fix - guarantees parseable output).**
  Upgrade the provider from the `response_format=json_object` hint to a full
  `json_schema`-constrained decode of the UIR element schema. Verified available
  on BOTH endpoints: mlx-vlm `/v1/chat/completions` supports OpenAI-compatible
  `json_schema` structured outputs, and vLLM supports `guided_json` (xgrammar
  default backend). Eliminates the malformation class. Caveat: a constrained
  decode can still hit the token cap (valid-but-incomplete), so A3 pairs with
  A1+A4, it does not replace them.
- **A4. Bounded JSON repair (fail-open last resort).** If a response is still
  truncated/malformed, repair to the last complete element (drop the trailing
  partial) and keep the N complete elements instead of discarding the whole page
  to Docling. Partial VLM extraction beats full Docling fallback on a dense page -
  the fail-open invariant applied at this boundary.
- **A5. Per-region (decoupled) extraction for dense pages (design fix).** Replace
  whole-page extraction with a coarse layout pass then per-region content
  extraction, bounding each call's output and removing the density ceiling.
  Precedent: MinerU2.5's decoupled two-stage (global layout on a thumbnail ->
  targeted high-res per-region recognition). Costs more VLM calls; build only if
  A1-A4 do not clear the dense-doc fallback rate.

**Blocker B - VLM bbox crop drift 40-50%** (hallucinated coordinates -> garbage
crops on forms/scans/tables):

- **B1. Prefer deterministic bboxes for cropping (structural fix).** The VLM is
  good at semantics, bad at coordinates. When a page also yields detectable
  objects, crop from the GEOMETRIC bbox (PyMuPDF `get_images()` / `find_tables()`
  or the Docling-layout bbox) and use the VLM only for the description/content.
  Trust VLM coordinates only when no deterministic source exists.
- **B2. Crop-audit as a re-extraction trigger (fail-open).** Today crop-audit only
  WARNS. Make a flagged crop (edge-clamp / blank) fall back to the deterministic
  bbox or a full-page render for that asset, so a garbage crop is never persisted.
- **B3. Higher render DPI / stronger model** - secondary, only if B1+B2 leave
  residual drift.

**Acceptance criteria (these gate the soak re-run):**
- A: on a dense magazine (Combat Aircraft / PCWorld), whole-page JSON fallback
  rate near 0 (down from ~58%), and zero SILENT truncation (A1 makes it typed).
- B: crop-audit drift back under the 15% threshold on the forms/scans/tables that
  measured 40-50%.
- Re-run the bounded crucible subset to `QA_PASS` + crop-audit within threshold
  BEFORE any Grand Soak.

### 9.2 Broader roadmap (after the blockers clear)

1. Re-run + validate the crucible (18 docs) end-to-end: every doc `status=ok`,
   `QA_PASS`, crop-audit within threshold.
2. Retire-or-harden the Docling lane (§8 decision).
3. Complete the ElementType -> Modality migration (§2.2).
4. Modality-aware CODE/TABLE/FORM judge rubric (§7).
5. (Conditional) ColPali reranker (§6.2), only if §6.1 is measured insufficient.
6. Grand Soak with a budgeted page-hour ceiling and the long tail included - only
   after 9.1 passes its acceptance criteria.

### 9.3 Definition of done

A component is "done" when its boundary contract test (§2.1) is green AND it has
passed on the hardest relevant content class (§2.3) AND the run-level gate
(`SMOKE_PRODUCTION_PASS` + `QA_PASS`) holds. The pipeline "meets requirements"
only when a full crucible run (dense magazines + forms/scans included) clears
9.1's acceptance criteria - the bar the 2026-06-02 soak failed.

---

## 10. External References (verified 2026-06-03)

| Source | What it actually is | How V3.1 uses it |
|---|---|---|
| ColPali - Faysse et al., [arXiv:2407.01449](https://arxiv.org/abs/2407.01449) | Late-interaction multi-vector retrieval over image patches (PaliGemma, ~1024 patches/page, 128-dim, MaxSim) | Foundation for the PROPOSED visual reranker (6.2) |
| GOT-OCR 2.0 - [arXiv:2409.01704](https://arxiv.org/abs/2409.01704) | 580M end-to-end model (VitDet + Qwen-0.5B), raw image -> markdown/LaTeX, no cascade | Validates the vision-native thesis (3.3) |
| LlamaParse Cost Optimizer / Auto Mode ([LlamaIndex](https://www.llamaindex.ai/blog/optimize-parsing-costs-with-llamaparse-auto-mode)) | Per-page tier routing run in parallel (standard vs Premium/Agentic) | Industry precedent for the cost router (3.2) |
| MinerU - [arXiv:2409.18839](https://arxiv.org/pdf/2409.18839) | Pipeline backend is a layout-detection + OCR **cascade** (doclayout_yolo + PP-OCRv5 + formula/table models); MinerU2.5 adds a decoupled two-stage VLM (global layout -> per-region recognition) | v1 cascade is what V3 moves AWAY from; MinerU2.5's decoupled two-stage is the precedent for the A5 per-region fix (§9.1) |
| Guided JSON decoding - [vLLM structured outputs](https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html), [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) | Constrained decoding to a JSON schema: vLLM `guided_json` (xgrammar default backend); mlx-vlm OpenAI-compatible `json_schema` structured outputs | The A3 structural fix for Blocker A (§9.1); verified available on both M5 (mlx-vlm) and GX10 (vLLM) |
| HierFinRAG - [MDPI Informatics 13(2):30](https://www.mdpi.com/2227-9709/13/2/30) | Table-text GNN + symbolic-neural fusion for FinQA arithmetic | Suggestive support for routing numeric/tabular queries to a symbolic engine. CAVEAT: 1-month-old, lower-tier venue - verify before adopting |
| DGX Spark GB10 ([NVIDIA](https://docs.nvidia.com/dgx/dgx-spark/hardware.html), [LMSYS](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)) | Unified 128GB LPDDR5X, 273 GB/s; token-gen is bandwidth-limited | Backs the bandwidth-segregation rationale (5) |

External benchmarks are not a substitute for measuring on this corpus and hardware.
Every adoption above is gated on a local measurement (Sections 6.2, 9).
