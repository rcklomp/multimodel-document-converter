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
