# PLAN_OMNIDOCBENCH_EVAL - Ground-Truth Fidelity Benchmark Workstream

Status: PROPOSED (2026-06-07)
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
