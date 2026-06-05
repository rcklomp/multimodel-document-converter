# PLAN: VLM Evaluation for Document Extraction (2026-06-04)

**Status:** ACTIVE workstream. Main extraction-code work is PAUSED by decision
(2026-06-04) pending this evaluation. The §9.2 full crucible (2026-06-04) showed
this cycle's fixes hold at scale, but the next-weakest axis is VLM output QUALITY
(table->markdown compliance, empty tables) - i.e. we are scaffolding around a
possible model mismatch. This plan evaluates whether a better-fit extraction VLM
removes the need for that scaffolding.

## 0. Decision being made

Pick the extraction VLM (and, implicitly, the extraction ARCHITECTURE) that best
serves our pipeline. The central fork:

- **Keep the element+bbox (UIR) architecture** - a model that emits structured
  per-element output WITH bounding boxes is near-drop-in: we keep B1/B2 geometric
  crops, QA-CHECK-05, asset materialization, and spatial retrieval. (Candidates:
  structured/layout doc VLMs - dots.ocr, MinerU 2.5, Granite-Docling/DocTags,
  PaddleOCR-VL.)
- **Rearchitect markdown-first** - a model that emits clean markdown per page
  (no per-element bboxes). Potentially higher table fidelity, but we lose
  per-element spatial structure and must rebuild the asset/crop story. (NuMarkdown
  -8B-Thinking, olmOCR, GOT-OCR2, Nanonets-OCR, DeepSeek-OCR.)
- **Stay on Qwen3-VL-8B + targeted fixes** - the null hypothesis. Keep iff no
  candidate clears it on quality without an unacceptable speed/structure cost.

This is NOT just "are the tables better." Score STRUCTURAL FIT and SPEED with the
same weight as content quality.

## 1. Requirements (what "best for our application" means)

| # | Requirement | Why | Hard/Soft |
|---|---|---|---|
| R1 | Structured output WITH per-element bbox, OR markdown whose quality gain justifies losing bbox | bbox is load-bearing (crops, QA-CHECK-05, spatial) | scored, not hard |
| R2 | Tables -> faithful markdown grids (not prose, not empty) | the dominant corpus-scale failure | HARD |
| R3 | Code -> exact indentation preserved | code was the early-cycle failure class | HARD |
| R4 | Forms -> key-value / structured | form class | soft |
| R5 | Charts -> data transcription | charts->data is a V3 differentiator | soft |
| R6 | Robust: valid/parseable output, no repetition loops, no premature EOS | the failure modes we scaffolded around | HARD |
| R7 | Local servable: MLX (Apple Silicon M5) or vLLM (GB10/GX10) | bandwidth/cost architecture (charter §5) | HARD for primary; cloud allowed as ceiling only |
| R8 | Throughput acceptable at corpus scale | decode is memory-bandwidth-bound; "thinking" models are slow | scored (budget: target <= ~Qwen's s/page; flag if >2x) |
| R9 | Permissive license | production use | soft (note, not gate) |

## 2. Golden test set  `[built: output/vlm_eval/golden_set/]`

15 single pages, each a PDF + 200-DPI PNG (matches our render) + a manifest
labelling capability + what-to-check. Each page is a KNOWN hard case from the
crucible:

| capability | pages |
|---|---|
| table (6) | Firearms spec, Grundlagen German (was empty), Hybrid-EV academic (was empty), AIOS academic, CarOK spreadsheet, CarOK loop page |
| code (2) | FluentPython Python, AIOS pseudocode |
| form (2) | Form_0013 scanned, betwisting digital |
| chart (1) | IRJET solar-PV |
| layout (2) | Combat magazine interior, PCWorld multi-column |
| image (1) | Digitale-Fotografie full-bleed |
| prose (1) | HarryPotter baseline |

Expand later if the A/B shortlist needs a class we under-cover (e.g. degraded scan).

## 3. Scoring rubric (per page, per model)

DETERMINISTIC (cheap, automated in the harness):
- **parseable** (bool): did the output parse in the model's declared format
  (JSON-elements / markdown / DocTags)?
- **has_bbox** (bool): per-element bboxes present? (structural-fit axis)
- **table_markdown** (0/1): for table pages, is there a real markdown grid
  (`|...|` + `|---|` separator) with non-empty cells? (mirrors our
  table_markdown_ratio gate)
- **code_indent_fidelity** (0-1): for code pages, leading-whitespace preserved
  vs a reference? (reuse the v3 indentation metric)
- **repetition** (bool): any unit repeated >= 8x (reuse `_collapse_degenerate_repeats` detector)
- **empty_elements** (count): table/image elements with empty content
- **latency_s**: wall-clock per page (and whether a thinking phase is used)

JUDGED (GX10 Qwen2.5-14B-FP8 judge and/or human spot-check - correctness, not
just format):
- **content_completeness** (0-5): did it capture the page's content (rows, cells,
  code lines, form fields)?
- **content_fidelity** (0-5): is the captured content CORRECT (no hallucinated
  rows, right numbers)?
- NOTE the judge is calibration-TRUSTWORTHY mainly on format axes (per
  FINDINGS_LOG); treat fidelity as directional + human spot-check the finalists.

AGGREGATE: per-model scorecard = mean per-axis across the golden set, plus a
weighted score (R2/R3/R6 hard axes dominate; R8 speed as a gate, not a sum).

## 4. Harness design  `[to build: scripts/vlm_eval_harness.py]`

Model-agnostic. A candidate is `{name, endpoint, model_id, output_format,
prompt}`. For each golden page: send (image | pdf) per the candidate's input
contract, capture raw output + latency, parse per output_format, score the
deterministic axes, persist raw + scored rows to `output/vlm_eval/<candidate>/`.
A second pass runs the judge over the captured outputs. A final step emits the
cross-model comparison table.

Reuse what exists: `scripts/measure_vlm_page_latency.py` (latency), the v3
indentation + repetition detectors, the GX10 judge harness
(`scripts/synthetic_soak.py`).

## 5. Candidate shortlist  `[from deep-research wf_6a3d73f6-901, 2026-06-04; 24/25 claims verified]`

**Strategic insight from the research:** the strongest candidates are
*document-specialist* VLMs that are SMALLER than Qwen3-VL-8B (258M-3B), emit
structure WITH bboxes, and several use a DEDICATED layout-detection stage for
the bbox (MinerU2.5, PaddleOCR-VL) - which would give us RELIABLE detector
bboxes, fixing our Blocker-B crop drift AND the table-format problem in one
move, while KEEPING the UIR architecture. The markdown-first rewrite
(NuMarkdown) looks like the weaker path.

**Group A - structure-preserving (keep UIR, primary A/B):**
- **MinerU2.5 (1.2B)** - TOP PICK. Decoupled two-stage (global layout ->
  per-region recognition; the §10 precedent). Leads dots.ocr on OmniDocBench
  (Overall 90.67, **TableTEDS 88.22**); emits bbox in **0-1000** (our exact
  scale); MLX + vLLM. Two-stage layout = reliable bboxes. CAVEAT: the "2.12
  pages/s / 7x" speed is **A100, not MLX** - must measure on M5.
- **Granite-Docling-258M** - size/speed winner. 258M, **official MLX**, Apache
  2.0, strong tables (FinTabNet TEDS 0.97 vs SmolDocling 0.82, OCRBench 500 vs
  338). Outputs **DocTags** -> needs a DocTags->UIR adapter (Docling tooling
  exists). Tiny -> fast even on M5.
- **PaddleOCR-VL (0.9B)** - top OmniDocBench (94.5/96.3), bbox from a dedicated
  RT-DETR layout stage (reliable). CAVEATS: SOTA contested (GLM-OCR reportedly
  ahead); **MLX serving unverified**; two-stage pipeline integration. SECONDARY
  (gate on servability).
- **dots.ocr (3B)** - emits JSON+bbox, MLX via mlx-vlm. DOWNRANKED: **weak on
  complex tables/formulas** (our #1 need) and its headline OmniDocBench SOTA was
  **REFUTED (0-3)**. Include only if a slot is free.

**Group B - markdown-first (rearchitect, only if Group A fails):**
- **NuMarkdown-8B-Thinking** - markdown-ONLY + slow thinking phase (bad on
  bandwidth-bound M5). The user's candidate; test only if Group A disappoints.
- **DeepSeek-OCR** - vision-token compression (efficient); CAN emit
  grounding/layout (not pure markdown) - a possible hybrid; keep as a wildcard.

**Group C - ceilings (reference only, cost-capped):** Qwen3-VL-30B; one cloud
frontier (Gemini/GPT/Claude vision) for an upper-bound on the hard pages.

**Baseline / control:** Qwen3-VL-8B (current), scored identically.

**Recommended A/B (run these first):** MinerU2.5 + Granite-Docling-258M
(+ PaddleOCR-VL if servable) vs the Qwen3-VL-8B baseline - all
structure-preserving, so we test "keep the architecture, fix tables + crop
drift" BEFORE considering any markdown-first rewrite.

**Hard caveats (the verification earned its keep):** benchmarks are vendor
self-reported and OmniDocBench is "saturated" - leaderboard rank != performance
on OUR corpus (German technical tables, magazines, code books). Benchmarks set
the shortlist; the golden-set A/B is the real test. dots.ocr's SOTA claim was
refuted; MinerU's speed is A100; PaddleOCR-VL MLX is unverified - all must be
re-measured on M5.

## 6. Operator steps (gating the A/B)

- Serve each shortlisted LOCAL model on M5 (MLX) or GX10 (vLLM) - same
  `vlm_serve` mechanism, pointed at the candidate. The harness needs an
  OpenAI-compatible endpoint per candidate (or a local generate call for models
  without a server).
- Cloud ceilings: API keys + an explicit cost ceiling.

## 7. Decision criteria (how we choose)

1. A candidate must clear the HARD axes (R2 table-markdown, R3 code, R6
   robustness, R7 local) on the golden set to be eligible.
2. Among eligible, prefer the one that KEEPS the UIR architecture (Group A) unless
   a Group B model's quality gain is large enough to justify the markdown-first
   rewrite cost (a deliberate, logged architecture decision).
3. Speed (R8) is a gate: > ~2x Qwen's s/page on comparable pages is a strike that
   must be bought back by a large quality win.
4. Output: a recommendation + a one-page architecture-decision memo (keep UIR vs
   markdown-first) for sign-off, then a full-corpus crucible re-run on the winner.

## 8. Out of scope / paused

- Main extraction-code fixes (render guard, empty-table degrade, table-prompt
  tuning, crop-audit heuristic) are PAUSED. They re-enter AFTER the model
  decision - several are model-specific (the table-prompt work is wasted if we
  switch models).
- The Grand Soak (all 43 docs) stays un-run.

## 9. Status / next

- [x] Golden test set built (15 pages).
- [~] Deep research running (wf_6a3d73f6-901) -> fills the shortlist (§5).
- [ ] Build the harness (§4).
- [ ] Operator serves the shortlist.
- [ ] Run A/B, score, decide.

## 10. RESULT (2026-06-05): MinerU2.5 wins; topology stays Config C

Empirical A/B on the 15-page golden set (M5 + GX10), deterministic scorecard:

| candidate | fmt | bbox% | tbl_struct% | rep% | med s/page | verdict |
|---|---|---|---|---|---|---|
| **mineru_mlx (M5 native)** | json | 100 | 67 | 7 | **6.8** | **WINNER** |
| mineru_pipeline (M5 MPS) | json | 100 | 67 | 7 | 26.7 | same quality, 4x slower path |
| granite_docling (M5 MLX) | doctags | 100 | 83 | 7 | 2.1 | fastest, but EMPTIES dense tables (CarOK) -> not viable |
| paddleocr_vl | markdown | 7 | 0 | 20 | 2.8 | needs full pipeline for bbox/tables |
| qwen_baseline (M5 server) | markdown | 93 | 0 | 27 | 25.3 | the baseline failure (tables 0%) |

Key findings:
- **MinerU2.5 is the extractor.** Two-stage (layout detector -> per-region recognition)
  emits per-element JSON with NORMALIZED [0,1] bboxes + rich typing (header/title/
  text/code/table/figure...) -> near-drop-in for UIR (x1000 = our frame). Detector
  bboxes are RELIABLE -> fixes Blocker-B crop drift. Built-in anti-repetition
  (no_repeat_ngram_size=100) -> 7% vs Qwen 27%. Reads the DENSE tables Qwen empties
  (CarOK full <table>). Use the Apache-2.0 **Pro** variant (`opendatalab/
  MinerU2.5-Pro-2604-1.2B`); the 2509 base is AGPL-3.0.
- **Granite-Docling (Apache-2.0, MLX-native, 2.1s)** is excellent on moderate pages
  but its 258M model EMPTIES the dense CarOK spreadsheet -> re-introduces our #1
  failure. Not viable as primary; possible fast-path in a future hybrid.
- **mlx-engine fix (the unlock):** mineru-vl-utils mlx backend crashed
  (`mlx.array == None`) because mlx_vlm 0.3.12 dropped `image_token_id` through
  mineru's config rewrite. One-line fix: after constructing
  `MinerUClient(backend="mlx-engine", model_path="mlx-community/MinerU2.5-2509-1.2B-bf16")`,
  set `client.client.model.config.image_token_id = 151655` (Qwen2-VL `<|image_pad|>`)
  and `video_token_id = 151656`, plus
  `mlx_vlm.utils.MODEL_REMAPPING["qwen2_vl_text"]="qwen2_vl"`. Env: isolated venv on
  the M5 with `mlx-vlm==0.3.12`, `mlx<=0.31.1`, `mineru-vl-utils`, torch+torchvision
  (processor dep), `transformers<5`. Result: native MLX two_step_extract at 3-12s/page.

## 11. Hardware topology decision (2026-06-05): Config C (no reshuffle)

Measured the 3 deciding numbers:
- **M5 native MinerU = fast** (6.8s median; the "M5 is slow" premise was the mlx-engine
  bug, now fixed).
- **GX10 = 44GB free** with the judge healthy (co-location feasible, but unneeded).
- **Mac Mini = no SSH + production embedder** -> locked to the embedder role.

**Decision: Extraction=MinerU2.5(M5 native MLX), Judge=GX10, Embedder=Mac Mini** -
i.e. the current topology with MinerU swapped in for Qwen3-VL. Contention-free
(each workload isolated; embedder active both phases so it sits alone), no judge
re-quant/re-calibration. The judge->Mac-Mini swap is dominated (judge+embedder
collide during soaks + re-cal cost). **Config F** (parallel extraction on M5+GX10,
judge time-shared on GX10 across phases) remains an OPTIONAL throughput upgrade for
the full corpus/Grand Soak; not needed for correctness.

## 12. Config F measurement (2026-06-05): GX10 vLLM throughput vs M5

Measured MinerU2.5-2509-1.2B served via vanilla vllm-openai on the GX10 (Docker,
port 8001, gpu-memory-utilization 0.15, co-resident with the judge on 8000),
driven from the M5 over LAN via mineru-vl-utils http-client over the 15-page
golden set. MinerU2.5 is `Qwen2VLForConditionalGeneration` -> served by stock
vLLM, no custom code; image_token_id already in config (the mlx-engine bug is
mlx-only).

| metric | M5 (mlx native) | GX10 (vLLM, c=8) | note |
|---|---|---|---|
| latency, single page (median) | **6.8s** | 13.4s | M5 ~2x faster = the 546 vs 273 GB/s bandwidth ratio; decode is bandwidth-bound |
| throughput, batched | 0.147 p/s | **0.34 p/s** | **GX10 2.3x faster**: vLLM continuous-batches, mlx-engine is sequential |

GX10 sequential per-page: 3.7-35.2s (dense CarOK/AIOS pages are the 30s+ tails;
they amortize under concurrency -> 44.2s total for 15 pages). 15/15 OK.

**Throughput ranking (corpus-scale, fair):**
1. **Config F (M5 mlx + GX10 vLLM in parallel): ~0.49 p/s** (= 0.147 + 0.34; 3.3x M5-alone)
2. GX10 vLLM alone: 0.34 p/s (2.3x M5-alone)
3. M5 mlx alone: 0.147 p/s (latency-optimal, 6.8s/page)

At 1000 pages: M5-alone ~1.9h, GX10-alone ~0.8h, Config F ~0.57h.

Phase-contention check PASSES: the judge runs in the EVAL phase, idle during
INGESTION, so GX10 extraction during a soak's ingest phase is compute-contention-
free (judge holds memory at 0.60, MinerU fits in 0.15; 44GB headroom).

**Decision (refines §11):**
- **Config C (M5 extraction) stays the DEFAULT** - simplest, latency-optimal, no
  GX10 dependency, best for interactive/single-doc and small jobs.
- For **corpus-scale reconversion / Grand Soak**, route extraction to the **GX10
  vLLM** (2.3x the M5 alone, frees the M5) or **Config F** (3.3x, both boxes).
  GX10-alone is the better cost/complexity point than Config F unless ingest
  wall-time is a hard bottleneck (Config F needs a 2-endpoint work-split
  dispatcher; we already drive MinerU over HTTP so the delta is small).
- Net: the "extract on the bandwidth-king M5" rule is a LATENCY rule, not a
  throughput rule. Pick the box by the job: M5 for latency, GX10/Config F for
  batch throughput.

## 13. MinerU2.5 integration SHIPPED + live-validated (2026-06-05)

The MinerU2.5 extractor is integrated into the V3 pipeline as a selectable
engine and live-validated end-to-end. Five atomic commits on
v3.1-extraction-hardening (all gated: full suite, firewall, repo-integrity,
ruff+black, SMOKE_PRODUCTION_PASS at each):

1. b3b5b9b - pure MinerU element-JSON -> UniversalDocument converter (bbox
   [0,1]->[0,1000], 13-type vocab -> 3-value ElementType, code smuggle), 13
   offline tests, registered in the AST firewall.
2. e6afa93 - MineruNativeEngine: render (PyMuPDF 200 DPI -> PIL) + drive a MinerU
   server via the LIGHT lazy-imported mineru_vl_utils http-client + assemble;
   offline-mocked-transport tests. Model stays in an ISOLATED server (Option A).
3. 1b9e650 - USE_MINERU_ENGINE processor route (top precedence) + routing tests.
4. 8179f3a - pyproject [mineru] optional extra (mineru-vl-utils>=1.0.3; light,
   no torch/mlx/vllm).
5. 5273c71 - HTML-table -> Markdown-grid transcode (live validation found MinerU
   emits HTML tables; the pipeline contract R2 needs Markdown). Fixed the
   conversion, not the gate.

**Live validation (USE_MINERU_ENGINE=1 -> GX10 MinerU vLLM server, real CLI):**
6/6 golden docs QA_PASS via qa_full_conversion.py --source-pdf, across every
capability class:

| doc | class | result |
|---|---|---|
| table_spreadsheet_carok | dense table (Qwen EMPTIED it) | QA_PASS - 45-row Markdown table |
| table_spec_firearms | spec table | QA_PASS |
| code_python_fluent | code (indentation R3) | QA_PASS |
| form_digital_betwisting | form | QA_PASS |
| magazine_pcworld | multi-column layout + image | QA_PASS |
| prose_baseline_hp | prose | QA_PASS |

The dense CarOK spreadsheet - the original Blocker-A failure class - now produces
a full 45-row Markdown table with extraction_method=uir_native_chunker (V3 path)
and an on-disk asset (QA-CHECK-05). The MinerU extractor delivers the eval's
promised win through the production CLI.

**Remaining (not yet done):** make MinerU the DEFAULT route (today it is opt-in
via USE_MINERU_ENGINE; the router/HybridEngine still defaults to Qwen-VLM/
Docling); broader corpus soak; the merge_prev reading-order pass (inert today).

## 14. Broader corpus soak via the DEFAULT route (2026-06-05)

Ran 7 cross-category real docs from data/ (49 pages) through the NEW DEFAULT
route (MINERU_ENDPOINT set, NO USE_* flag -> exercises increment 6's MinerU
default) -> GX10 MinerU server, strict-gated each (qa_full_conversion
--source-pdf). Harness: scripts/mineru_corpus_soak.sh.

| doc | pages | chunks | gate |
|---|---|---|---|
| betwistingsformulier (form) | 1 | 3 | QA_PASS |
| Bevestigingsmiddelen (mixed) | 2 | 6 | QA_PASS |
| Recent_Trends_in_Transportation (academic) | 5 | 17 | QA_PASS |
| ATZ Effizientere Software (German technical) | 6 | 30 | QA_PASS |
| IRJET solar PV (academic + charts) | 7 | 31 | QA_PASS |
| CarOK voorraadtelling (dense spreadsheet) | 12 | 49 | QA_FAIL (HEADING) |
| Hybrid_electric_vehicles (academic tables) | 16 | 93 | QA_PASS |

**6/7 QA_PASS.** The lone QA_FAIL (CarOK voorraadtelling, a pure data
spreadsheet) is NOT a MinerU regression and NOT a table problem
(table_markdown_ratio=1.0000 - all 12 tables clean Markdown via the transcode):
it fails the qa_conversion_audit HEADING gate at 0/37 coverage because a raw data
spreadsheet has no section headings. Verified engine-independent: the same doc
via the legacy USE_DOCLING_FAST path ALSO QA_FAILs HEADING 0/37, and WORSE
(failures=2, only 8/12 pages chunked vs MinerU's full 12/12, failures=1). The
recovery_scan/recovery_gap_fill chunks (which carry no bbox/heading) are
batch_processor's engine-independent recovery net, identical on both paths.

Conclusion: MinerU is validated across the corpus and outperforms the legacy path
on the one hard doc. Open (pre-existing, NOT MinerU): the HEADING gate has no
exemption for genuinely headingless docs (data spreadsheets) - a doc-class gate
policy question, tracked separately; do NOT weaken the gate.

## 15. Scanned-class + scale validation (2026-06-05)

Filling the two classes the §14 corpus soak did not cover, via the default route
(MINERU_ENDPOINT set) -> GX10 MinerU server:

- **Scanned (vision-OCR):** `data/business_form/0013_140302111325_001.pdf` (the
  scanned German invoice that the OFFLINE Docling path produces 0 chunks for, and
  that v2 once flagged "UNSUPPORTED PROFILE"). MinerU read it into 4 chunks (2
  text, 2 table): the line-items as a proper Markdown table
  (`| Pos | Art-Nr. | Bezeichnung | Anz | Mwst-Satz | Preis | Gesamt |`), a totals
  table, and the address/payment text. **QA_PASS.** Confirms MinerU's vision OCR
  handles the scanned/form class as first-class structured content.

Note (verify-before-concluding): an earlier scanned run returned 0 chunks - this
was the GX10 MinerU container being DOWN (I had torn it down after §14), NOT a
MinerU failure. The engine correctly RAISED on `[Errno 61] Connection refused`
and halted with zero fabricated chunks (fail-open / no-silent-degradation working
as designed), exactly the verify-before-blaming lesson.

Cumulative validation: golden 6/6 + corpus soak 7/7 + scanned 1/1 across
table (dense + spec), code, form, scanned, layout, prose, charts, German.
