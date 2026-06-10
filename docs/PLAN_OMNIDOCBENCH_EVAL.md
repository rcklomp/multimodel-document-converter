# PLAN_OMNIDOCBENCH_EVAL - Ground-Truth Fidelity Benchmark Workstream

Status: PROPOSED (2026-06-07) -- Phase 0 underway, see Section 12.
Owner: extraction + QA
Depends on / pairs with: PLAN_GATE_QUALITY_V1 (this is its ground-truth half).
Trigger: stand this up so it is ready WHEN acceptance testing for the
gate-quality iteration begins (see Section 8).

## 0. Why

Every quality signal we have measures the converter against ITSELF (structural
self-consistency in the gates) or against a human read (the 2026-06-07 audit).
PLAN_GATE_QUALITY_V1 Section 9 names the gap explicitly and puts it out of
scope: "Source-fidelity (output vs PDF ground truth) at scale. No labeled ground
truth exists." OmniDocBench (opendatalab, Apache-2.0) is exactly that missing
labeled ground truth: 1,651 expert-annotated PDF pages with real fidelity
metrics. It gives us an objective, third-party yardstick we cannot build
ourselves.

## 1. What OmniDocBench provides

- 1,651 annotated PDF pages, 10 doc types (academic, financial, newspaper,
  textbook, handwritten, PPT, exam, magazine, research report, book - a near
  superset of our crucible classes), 5 layout types, 5 language types.
- Block + span annotations: layout bbox, text, tables (HTML/LaTeX), formulas
  (LaTeX), reading order; page/text/table attribute tags.
- Metrics: normalized Edit Distance, BLEU/METEOR (text), TEDS (tables), CDM
  (formulas, render-based), COCO mAP/mAR (layout).
- Pure benchmark + toolkit, no models. Leaderboard ranks 50+ systems incl.
  MinerU, PaddleOCR-VL, granite-docling, GPT-4o, Gemini.
- Download: HuggingFace `opendatalab/OmniDocBench`. Run:
  `python pdf_validation.py --config configs/end2end.yaml`.

## 2. The scoping line that matters: fidelity, not retrieval value

OmniDocBench scores PARSING FIDELITY (does the Markdown match the page). It does
NOT score RETRIEVAL VALUE (is this chunk useful for RAG). A folio correctly
transcribed scores well in OmniDocBench and is still junk in our index. So:

- OmniDocBench and PLAN_GATE_QUALITY_V1 are ORTHOGONAL and complementary. One
  measures transcription accuracy vs ground truth; the other measures chunk
  hygiene / retrieval value on our own docs.
- OmniDocBench is an OFFLINE selection + regression tool (needs ground truth),
  NOT a per-conversion production gate.

Use it for: objective extractor selection, fidelity regression, and the
table/formula/reading-order metrics we entirely lack today. Do NOT use it to
replace the retrieval-value gates.

## 3. Integration architecture

OmniDocBench evaluates per-page Markdown whose filenames match the GT page image
names. Our pipeline is PDF-in / JSONL-out per DOCUMENT. Two adapters bridge that,
plus strict environment isolation.

### 3a. Input adapter (their pages -> our pipeline)
OmniDocBench ships page IMAGES, our pipeline takes PDF paths. Wrap each
benchmark page image as a single-page PDF (img2pdf) and run our standard
pipeline on it (batch_size is irrelevant for one page). Resolve during Phase 0
whether the HF dataset also ships source PDFs we can feed directly.

### 3b. Output adapter (our JSONL -> their Markdown) - the crux
Render each document's `ingestion.jsonl` to one Markdown file per page, named
`<gt_image_name>.md`, by joining chunks in `reading_order`:
- headings -> `#`/`##` from `hierarchy.level`
- text -> paragraphs
- tables -> our Markdown grid (already produced)
- formulas -> LaTeX if present
- (SUPERSEDED by 12.1#1: "join chunks in reading_order" -- reading_order is not
  a schema field; reconstruct from JSONL line order. See Section 12.)
- DECISION (image handling): OmniDocBench end-to-end Markdown does NOT carry our
  long VLM visual_descriptions. Emitting them would INFLATE edit distance against
  ground truth. So for the OmniDocBench adapter, render image regions as the GT
  convention (a placeholder / caption only), NOT the enriched description. This
  is the inverse of our RAG output and must be a dedicated render mode, not the
  production exporter. Confirm the exact GT image convention in Phase 0.

### 3c. Environment isolation (hard requirement)
OmniDocBench needs TeX Live 2025, ImageMagick 7.x, Ghostscript (for the CDM
formula metric). Per project policy (VLM/standalone services run in their own
env, never co-installed into mmrag-v2), the eval runs in its OWN environment or
the provided Docker image `ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204`.
The adapter (our side, producing Markdown) runs in mmrag-v2; the scoring (their
side) runs in the isolated OmniDocBench env. If we skip the formula CDM metric we
can avoid the TeX stack entirely (most of our corpus is not formula-heavy).

## 4. Scoping decisions

- ENGLISH-FIRST. Our corpus is English + German + Dutch; OmniDocBench covers
  English, Simplified Chinese, EN-CN mixed, and others (no German/Dutch). Run
  with `filter: language: english` first. The Chinese subset is still useful for
  one targeted question (Section 6, item 4).
- METRIC SUBSET v1: text Edit_dist + BLEU/METEOR, table TEDS, reading_order
  Edit_dist. Defer formula CDM (and its TeX stack) to a later phase unless the
  academic-doc results demand it.
- MATCH METHOD: start with `quick_match` (their robust default with
  truncation/merge), compare against `simple_match` for sensitivity.

## 5. Concrete workflow

1. Provision the isolated eval env (Docker preferred) and download the HF dataset
   + `OmniDocBench.json`.
2. Build `scripts/omnidocbench_adapter.py` (mmrag-v2 side): page-image -> 1-page
   PDF -> our pipeline -> per-page `<name>.md` (GT-image render mode, 3b).
3. Write `configs/omnidocbench_end2end.yaml`: `ground_truth.data_path`,
   `prediction.data_path` (our adapter output), `match_method: quick_match`,
   `filter.language: english`, metric subset (Section 4).
4. Run `python pdf_validation.py --config configs/omnidocbench_end2end.yaml` in
   the isolated env; capture the per-category report (text / table / reading
   order edit distance + TEDS).
5. Record the baseline in `docs/paper/FINDINGS_LOG.md`.

## 6. Use cases (in priority order)

1. **Validate the extractor choice objectively.** We chose MinerU2.5 + Qwen-for-
   code off our own crucible read. Benchmark MinerU vs PaddleOCR-VL vs
   granite-docling vs Qwen3-VL on OmniDocBench - all four are already served on
   the M5 box (`/v1/models`). Confirm or challenge the choice on TEDS/edit
   distance instead of subjective inspection.
2. **Pipeline-vs-stock-MinerU.** Run our FULL pipeline (chunking + dedup +
   sanitizers + the new gate-quality fixes) through the adapter and compare to
   MinerU's published leaderboard score. Tells us whether our post-processing
   ADDS or SUBTRACTS fidelity vs raw extraction - a question we currently cannot
   answer.
3. **Metrics we lack.** TEDS for tables (vs our "has a `|---|` row" check), a
   reading-order metric (we have none), formula fidelity (the mislabelled-
   equation issue).
4. **Quantify the CJK-hallucination bias.** The `会` heading on the English HP
   cover plausibly stems from MinerU2.5's Chinese-doc bias. Running the English
   vs Simplified-Chinese subsets would quantify any EN-vs-CN fidelity gap.

## 7. Calibration tie-in to PLAN_GATE_QUALITY_V1

Where useful, calibrate the gate-quality advisory thresholds against
OmniDocBench's LABELED edit distance rather than only our unlabeled crucible:
e.g., confirm that pages our `text_garble_ratio` flags actually have high GT edit
distance, and that our `cross_page_dupe_ratio` does not penalize legitimate
multi-page table headers (which OmniDocBench's TEDS treats as correct). This
turns two of our heuristics from "looks right" into "correlated with ground
truth."

## 8. How this plugs into the next iteration's acceptance

The gate-quality iteration's acceptance becomes a TWO-AXIS gate:
- Retrieval-value axis (our crucible + the new advisory metrics): chunk hygiene
  on our own docs.
- Fidelity axis (OmniDocBench, English subset): transcription accuracy vs ground
  truth, must not regress vs the Phase-0 baseline.

To be ready in time, the prep phases below run EARLY in the next iteration so the
benchmark is wired before acceptance starts. PLAN_GATE_QUALITY_V1 Section 10
(Acceptance) carries a forward-reference to this fidelity axis.

## 9. Phasing

- Phase 0 (prep, do FIRST next iteration): isolated env + dataset + both
  adapters + baseline run of the CURRENT pipeline on the English subset. Output:
  a recorded baseline (text edit distance, TEDS, reading order) before any
  gate-quality fixes land.
- Phase 1: extractor bake-off (Section 6 item 1) to objectively confirm/revisit
  MinerU2.5 + Qwen.
- Phase 2: re-run after the gate-quality fixes (F1-F7) to prove fidelity did not
  regress (Section 6 item 2), and wire the Phase-0 baseline as the acceptance
  fidelity floor (Section 8).
- Phase 3 (optional): add formula CDM + the EN-vs-CN bias study (Section 6 item
  4) if the academic-doc results warrant the TeX stack.

## 10. Risks / open questions

- Does the HF dataset ship source PDFs, or only page images? (Decides 3a.) -
  resolve in Phase 0.
- Exact GT Markdown image convention (placeholder vs caption vs omitted)? -
  resolve in Phase 0; drives the 3b render mode.
- TeX/ImageMagick/Ghostscript footprint - mitigated by Docker + deferring CDM.
- Domain gap: no German/Dutch in OmniDocBench, so two of our classes get no
  ground-truth coverage. OmniDocBench is a fidelity SAMPLE, not full coverage;
  the crucible still owns those classes.
- Effort: bounded but non-trivial (two adapters + env). Apache-2.0, no license
  friction.

## 11. Acceptance (of this workstream)

- Isolated eval env reproducible (Docker or documented), dataset pinned.
- Both adapters built and tested; our current pipeline produces a clean
  OmniDocBench run on the English subset with a recorded baseline.
- Baseline (text edit distance, TEDS, reading order) logged in FINDINGS_LOG.md.
- PLAN_GATE_QUALITY_V1 acceptance references the fidelity floor.

## 12. Phase 0 Execution (2026-06-09)

Status: STRATIFIED BASELINE DONE (2026-06-09). Infra DONE (isolated `omnidocbench`
conda env, repo cloned at `~/omnidocbench-eval/OmniDocBench`). Dataset downloaded
(1651 pages, 755 English; no source PDFs ship -> input adapter required). Both
adapters built + tested: `scripts/omnidocbench_adapter.py` (select/build-pdfs/run/
render, standalone, R4-clean) and `configs/omnidocbench_end2end.yaml`. Open
questions R1/R5/R6/R7 resolved against the scorer source (see FINDINGS_LOG
2026-06-09). Stratified 128-page English baseline scored (text ED 0.251 / reading
ED 0.249 / table TEDS 0.669) and recorded with all five caveat labels; F1<->abandon
sanity pass clean. REMAINING: full 755-page English run (~8h, sequential),
then Phase 1 extractor bake-off.

### 12.1 Grounding corrections (verified against the repo + demo, 2026-06-09)

Three facts override assumptions in Sections 3-5 above:

1. **`reading_order` is not a schema field at all.** It is absent from
   `ingestion_schema.py` entirely (not present-but-null). The output adapter (3b)
   MUST reconstruct reading order from **JSONL line order** (chunks are emitted in
   document order). Section 3b's "join chunks in reading_order" describes a field
   that does not exist. (R1: verify emission order == GT `order` on the smoke page
   before scaling.)
2. **First JSONL line is a doc-metadata header** (`object_type:
   ingestion_metadata`, carrying `total_pages` / `source_file` / `doc_id`). The
   adapter MUST skip it and key real chunks off `metadata.page_number`.
3. **GT convention is concrete.** `OmniDocBench.json` is a list of page objects
   `{layout_dets, extra, page_info}`. Match on `page_info.image_path`
   (`<name>.jpg`); the authoritative language is `page_info.page_attribute.language`
   (assert on this, NOT the `_eng_` filename); block labels are
   `layout_dets[].category_type` (`title/text/abandon/figure/table/formula`),
   reading order is `layout_dets[].order`. Heading depth in OUR output comes from
   `metadata.hierarchy.level` (1-5) -- do NOT use `breadcrumb_path` depth, it is
   polluted with synthetic entries (`'Page 1'`, `'[RECOVERED]'`). Demo
   predictions wrap markdown in a ```` ```markdown ```` fence -- confirm whether
   the scorer strips it before the full run.

### 12.2 Steps (each with a verification check)

1. **Download + characterize.** `snapshot_download("opendatalab/OmniDocBench",
   repo_type="dataset")` into `~/omnidocbench-eval/data/` (in the `omnidocbench`
   env). Verify: JSON parses; count `language=="english"` pages; cross-tab vs
   `_eng_` filename to quantify disagreement; confirm no source PDFs ship.
2. **Input adapter** (`scripts/omnidocbench_adapter.py`, input half): image ->
   1-page PDF (`img2pdf`, lossless) -> `process <pdf> --batch-size 10
   --vision-provider none`. Verify: smoke ONE code/table-bearing English page
   end-to-end; chunks produced, `page_number==1`, no crash. Routing decision (R2)
   resolved AFTER this smoke -- inspect the lane + score, then decide whether a
   `--profile-override` comparison run is worth the credits.
3. **Output adapter** (output half): JSONL -> one `<image_basename>.md` per page.
   Group by `page_number`, order by JSONL line order (12.1#1). Render:
   - heading -> `#`x`hierarchy.level` ONLY when `level` is non-null. **`level` is
     null on ~43-53% of chunks** (verified: a chunk gets a level only when
     `breadcrumb_path` is populated). NULL-LEVEL FALLBACK: render as a paragraph,
     never as `#`x`None`. Do NOT derive depth from `breadcrumb_path` length (it
     carries synthetic `Page N`/`[RECOVERED]` tails, 12.1#3).
   - text -> paragraph; table -> existing markdown grid; code -> fenced.
   - image -> see R6 below. The omit-vs-caption call (12.2 Step 3 said "OMIT" vs
     Section 3b's "placeholder/caption") MUST be resolved against
     `OmniDocBench.json` `figure` blocks FIRST: if GT figure blocks contribute
     scored caption text, OMIT is penalized for content we have; if figures are
     unscored/excluded, OMIT is correct. This section supersedes 3b once resolved.

   Verify: eyeball one `.md` vs the GT page rendered from `layout_dets[].text` in
   `order`; resolve the ```` ```markdown ```` wrapper question AND the figure-caption
   scoring question in the same pass.
4. **Config + run.** Copy `end2end_notex.yaml`; set `ground_truth.data_path`,
   `prediction.data_path`, `match_method: quick_match`, `filter.language:
   english`. Verify: math pages EXCLUDED not scored-as-zero (else they drag
   edit-distance); run completes; per-category text edit-dist + TEDS +
   reading-order present; scored page count == English subset count.
5. **Record baseline** in `docs/paper/FINDINGS_LOG.md`: three metrics per
   category + all five 12.3 caveat labels verbatim.
6. **F1 <-> abandon sanity pass** on ~3 docs: diff our F1 furniture drops vs GT
   `abandon` blocks; confirm disagreements are directional noise, not extraction
   errors. Log the finding.

### 12.3 Baseline caveat labels (bake into FINDINGS_LOG verbatim)

1. Synthetic image-PDF, scanned-lane routing -- NOT native-PDF quality.
2. F1 <-> abandon directional, not exact -- sanity-passed first.
3. no-CDM config: formulas unscored (excluded, not penalized).
4. English subset = `language` attr AND `_eng_` filename (asserted on attr).
5. VLM descriptions omitted -- this is a TEXT/TABLE fidelity baseline, not
   multimodal value-add.

### 12.4 Issue / risk register

- R1: `reading_order` null -> verify JSONL emission order == GT `order` on the
  smoke page before scaling.
- R2: synthetic image-PDF routes everything down the scanned lane -> label
  baseline accordingly; routing-override comparison decided after the Step 2 smoke.
- R3: F3 CJK rule misfires on CN leakage -> English-subset-only sidesteps; assert
  on language attr.
- R4: adapter is new code but must NOT touch the extraction path -> keep it a
  standalone script, no imports into `batch_processor`/`uir_chunker`; run
  `scripts/smoke_production.sh` (must print `SMOKE_PRODUCTION_PASS`).
- R5: ```` ```markdown ```` fence handling unknown -> resolve in Step 3 verify
  before the full run.
- R6: `hierarchy.level` is null on ~43-53% of chunks -> Step 3 null-level fallback
  (render as paragraph, never `#`x`None`); do NOT use breadcrumb depth. Pre-full-run.
- R7: image OMIT (12.2) vs placeholder/caption (3b) unreconciled -> resolve
  against GT `figure` blocks BEFORE building the output adapter; an omit that
  drops scored caption text inflates edit-distance on content we already have.
  Pre-full-run, same pass as R5.

## 13. Phase 1 Execution (2026-06-09) -- extractor bake-off

Status: RAN 2026-06-09, INCONCLUSIVE — blocked by M5 serving/integration faults
(see FINDINGS_LOG 2026-06-09 "Phase 1 extractor bake-off"). Only docling_fast and
qwen3vl ran cleanly; mineru/hybrid invalidated by degraded M5 MinerU2.5 serving
(empty content-step + `broadcast_shapes` 500s), paddleocr by a strict-JSON engine
mismatch (returns Markdown), granite by a server load failure. The MinerU re-run is
parked on a M5 serving fix (or a GX10 vLLM MinerU endpoint). UNPARKED 2026-06-10:
the GX10 vLLM MinerU endpoint (`http://10.0.10.239:8001`, served id
`MinerU2.5-2509-1.2B`) is live, probe-validated (5/5 page classes, 0 500s, batches
at k=4), and is the DECIDED MinerU serving home (DECISIONS.md 2026-06-10) - the
`broadcast_shapes` fault is deterministic to the mlx stack and routed around by
substitution. The re-run is PLAN_EXTRACTION_FIDELITY_V1 Phase 1, governed by the
13.4 pre-registered rule. Harness
`scripts/omnidocbench_bakeoff.py` (standalone, R4-clean) drives prep/smoke/run/
score/report and is correct — the faults are downstream of it.

### 13.1 Premises verified (M5 box 10.0.10.235:8000, `/v1/models`, 2026-06-09)

All four bake-off models are served. Direct endpoint probes (NOT pipeline runs, to
avoid contending with the in-flight baseline):
- **Qwen3-VL-8B-Instruct-8bit** -- returns clean Markdown + tables (wrapped in a
  ```` ```markdown ```` fence the scorer strips). VLM route, proven.
- **PaddleOCR-VL-1.5-8bit** -- returns Markdown + tables via the generic OpenAI
  chat API; bake-off viable through the VLM engine with a different model id.
- **MinerU2.5-2509-1.2B-bf16** -- rich layout detection matching GT structure
  (header/table/table_caption/text/title/list/page_number); `extract()` clean.
- **granite-docling-258M-mlx** -- DEFERRED. Server cannot load it (HTTP 500
  "Unrecognized image processor"); also emits DocTags not Markdown -> would need a
  dedicated adapter. Re-add when both are resolved. (Not silently dropped.)

### 13.2 Engine routes (env per `src/mmrag_v3/processor.py:extract` precedence)

| route | env | endpoint shape |
|---|---|---|
| docling_fast | `USE_DOCLING_FAST=1` | local (Phase 0 baseline) |
| mineru | `USE_MINERU_ENGINE=1` + `MINERU_ENDPOINT`/`MINERU_MODEL` | base URL `http://10.0.10.235:8000` (NO /v1), full id `mlx-community/MinerU2.5-2509-1.2B-bf16` |
| qwen3vl | `USE_VLM_ENGINE=1` + `VLM_NATIVE_*` | `http://10.0.10.235:8000/v1`, `mlx-community/Qwen3-VL-8B-Instruct-8bit` |
| hybrid | `MINERU_ENDPOINT` + `VLM_NATIVE_*` (default route, no force flag) | both above -- our shipped default |
| paddleocr | `USE_VLM_ENGINE=1` + `VLM_NATIVE_MODEL=...PaddleOCR-VL-1.5-8bit` | `.../v1` |

### 13.3 Execution plan (runs AFTER the full-755 baseline completes)

Subset: 44 pages (6 per `data_source` from the stratified set; note/research_report
n=1). Workspaces under `~/omnidocbench-eval/bakeoff/<engine>/`, sharing the
already-built 1-page PDFs by symlink; GT subset `bakeoff/gt_bakeoff.json`.
Harness mechanics (prep -> score -> report) validated end-to-end on the
already-rendered docling preds (text ED 0.277 / reading 0.307 / TEDS 0.441 on the
44-subset). Sequence: smoke-first one page through `mineru` (most complex
integration) -> per-engine `run`+`render`+`score` (contiguous per engine so the M5
mlx server swaps models only 4x) -> `report`. Record the comparison table + the
engine-choice verdict (confirm/revisit MinerU2.5+Qwen) in FINDINGS_LOG. The verdict
is issued per the PRE-REGISTERED rule in 13.4 - never post-hoc from the table.

### 13.4 Pre-registered decision rule (2026-06-10; mirror of
PLAN_EXTRACTION_FIDELITY_V1 Section 7.2, which is canonical)

Added after the Round-1 audit of PLAN_EXTRACTION_FIDELITY_V1 (findings A1/A3): the
INCONCLUSIVE first run showed the bake-off can fail by competitor forfeit, and
neither this plan nor `omnidocbench_bakeoff.py` defined any margin or variance
treatment, leaving the verdict open to post-hoc rationalization.

- **Fixed paired set:** expand the 44-page shakeout subset toward 150-200 stratified
  pages; every engine INCLUDING the baseline runs the SAME set. The full-755
  baseline (text ED 0.301 / TEDS 0.563) is a corpus reference, never a comparator
  for subset runs (the 44-page docling numbers vs full-755 are cross-set and carry
  no comparative meaning). Per-class claims need n >= 10 pages of that class.
- **Paired statistics:** per-page paired deltas, bootstrap 95% CI on the mean delta;
  report worst-K per-page deltas per class alongside means.
- **The rule:** pipeline-primary CONFIRMED iff paired mean text-ED delta improves by
  >= 0.02 with CI excluding zero, AND table TEDS does not regress (CI excludes a
  regression > 0.02), AND no per-class paired mean regresses by > 0.05, AND the
  internal-corpus axis (PLAN_EXTRACTION_FIDELITY_V1 Section 7.3, validated signals
  only) does not regress. REFUTED iff the same holds for the VLM hybrid. Anything
  else: INCONCLUSIVE, recorded, default does not move.
- **Verdict eligibility (engine-health guard):** per-engine page-level request
  failures are logged; any engine exceeding 2% failures (after retry) invalidates
  every comparison involving it. A run with a forfeiting/unhealthy candidate is a
  DRY RUN - harness shakeout, no verdict authority.
- Margins are pre-registered; changing them after verdict-eligible data exists
  requires a recorded USER decision with rationale.
