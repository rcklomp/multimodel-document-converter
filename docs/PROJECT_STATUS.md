# Project Status

Last updated: 2026-06-06 (PR #4 code-review hardening: 5 review findings fixed; MinerU+Qwen-for-code hybrid SHIPPED as the default route; corpus-validated)

## Current state (2026-06-06) - MinerU+Qwen-for-code hybrid is the default route, corpus-validated

The 2026-06-04 VLM-evaluation pivot CONCLUDED: **MinerU2.5 is the chosen
extractor** and is now integrated into the V3 pipeline as a selectable, and
default, engine. It resolves the §9.1 blockers structurally (two-stage layout
detector -> per-region recognition: reliable detector bboxes fix Blocker B crop
drift; structured per-element output fixes Blocker A) rather than via more Qwen
scaffolding. Full eval + decision record: `docs/PLAN_VLM_EVAL.md` §10-14.

Shipped this cycle (branch `v3.1-extraction-hardening`, all gated per commit -
full suite, firewall, repo-integrity, ruff+black, SMOKE_PRODUCTION_PASS):
- `src/mmrag_v3/engines/mineru_native.py` - pure MinerU-element-JSON -> UIR
  converter (bbox [0,1]->[0,1000], 13-type vocab -> 3-value ElementType + code
  smuggle, merge_prev continuation-fold, HTML-table -> Markdown transcode) +
  `MineruNativeEngine` (renders pages, drives a MinerU server via the light lazy
  `mineru_vl_utils` http-client; model stays in an isolated server). `b3b5b9b`,
  `e6afa93`, `6ff83b3`.
- `USE_MINERU_ENGINE` route + the **default flip**: `mmrag_v3.processor` defaults
  to MinerU when `MINERU_ENDPOINT` is set (else legacy `HybridEngine`; never hard-
  breaks), plus a `USE_HYBRID_ENGINE` escape hatch. `1b9e650`, `600e055`.
- pyproject `[mineru]` optional extra (light: mineru-vl-utils, no torch/mlx/vllm).
  `8179f3a`.
- HEADING-gate correctness: a `tabular_document` skip for genuinely-headingless
  table-dominant docs (data spreadsheets), with a non-leak guard. `2799dd0`.
- **MinerU+Qwen-for-code hybrid default (2026-06-06):** `MineruQwenHybridEngine`
  supersedes the pure-MinerU default - code-dense pages (monospace ratio >= 0.10)
  route to Qwen (clean indentation, R3 1.00), every other page to MinerU
  (tables/layout). Live AIOS QA_PASS 35/35. `2be91a4`, `cfd3709`. Record:
  `DECISIONS.md` "MinerU+Qwen-for-code hybrid is the default extraction route".

**Validation:** 6/6 golden docs QA_PASS (table/code/form/layout/prose), and a
7-doc cross-category corpus soak through the DEFAULT route = **7/7 QA_PASS**
(after the HEADING fix). The dense CarOK spreadsheet Qwen EMPTIED now yields a
45-row Markdown table on the V3 path. Topology measured: M5 wins single-page
latency (6.8s vs GX10 13.4s), GX10 vLLM wins batched throughput 2.3x (vLLM
batches; mlx is sequential) - so M5 for latency, GX10/Config F for throughput.

**Also validated (2026-06-05):** SCANNED class - MinerU reads the scanned invoice
0013 (0 chunks on the offline Docling path) into structured Markdown tables ->
QA_PASS (`PLAN_VLM_EVAL` §15). SCALE - AIOS 35pg academic+pseudocode: surfaced and
FIXED the code-fencing gap (`a47f0c4`: MinerU emits code unfenced; converter now
fences it, R3) (`PLAN_VLM_EVAL` §16).

**Resolved since (2026-06-06):**
- R3 code-indentation gate redesign SHIPPED + signed: the dead
  `modality=="text"`-only seam in BOTH gate scripts is replaced by the shared
  `scripts/_code_quality.py` (`qa_conversion_audit.py` hard, `qa_semantic_fidelity.py`
  advisory). Strictly more enforcement; AIOS now correctly FAILs on CODE. See
  `DECISIONS.md` "R3 Code-Indentation Gate Redesign".
- MinerU dense-code ceiling resolved on the default route by the MinerU+Qwen-for-code
  hybrid (code-dense pages -> Qwen at fidelity 1.00).
- PR #4 code-review hardening (5 findings, 4 fixes - findings 1-2 are two breakages
  in one diagnostic script): `measure_vlm_page_latency.py` repointed to the new
  shared helpers; R3 single-line-collapse blind spot closed; recovery chunks now
  get infix step-number repair; redundant per-page text parse removed. No
  gate/assertion weakened. See `DECISIONS.md` "PR #4 code-review hardening".

**Open / deferred (none a default ship-blocker):**
- Sparse-code residual (NARROWED 2026-06-06): a CODE BLOCK on a mostly-prose page
  whose page-average mono ratio sits under the 0.10 router threshold now routes to
  Qwen via block-aware routing (`page_has_code_block`, commit `16dc097`; table-
  guarded; DECISIONS "Block-aware routing for sub-threshold code blocks"). The
  remaining residual is only SCATTERED INLINE code (no contiguous mono run) on a
  sub-threshold page, which still goes to MinerU-1.2B and can be mangled; the R3
  gate (incl. the single-line-collapse fix) flags it. A per-region code lane for
  the inline case is unbuilt ([[project_open_issues]]).
- The recovery_scan/gap_fill chunks (batch_processor's engine-independent net)
  carry no bbox - pre-existing, out of MinerU scope.
- AIOS HEADING 79.7% (just under 0.80) - a genuine borderline coverage signal,
  left as-is (not over-firing; the doc has real headings).

## PIVOT (2026-06-04): evaluate the extraction VLM before more scaffolding

Decision: PAUSE main extraction-code work and run a thorough evaluation of
candidate document-extraction VLMs. Rationale: the §9.2 full crucible showed this
cycle's fixes hold at scale, but the next-weakest axis is VLM output QUALITY
(table->markdown compliance, empty tables) - i.e. we may be scaffolding around a
model mismatch. Rather than build more corrective layers around Qwen3-VL-8B,
evaluate whether a better-fit model (esp. document-specialist VLMs that emit
structure+bboxes, or markdown-first models) removes the need. Plan + rubric +
golden test set: `docs/PLAN_VLM_EVAL.md`. The central fork is keep-UIR
(structured+bbox, near-drop-in) vs markdown-first (rearchitect). Main-code fixes
flagged on 2026-06-04 (render guard, empty-table degrade, table-prompt tuning,
crop-audit heuristic) are deferred behind the model decision - several are
model-specific.



## Current state (2026-06-04) - bounded crucible subset MEETS §9.1 acceptance

The bounded crucible subset - one doc per class that failed the 2026-06-02 soak
(CombatAircraft dense-magazine interior, CarOK spreadsheet/tables, Form_0013
scanned form, FluentPython code) - now passes end-to-end through the shipping
HybridEngine on M5:

- **4/4 docs `status=ok`, 0 failed, 0 Docling fallbacks, all `CROP_AUDIT_PASS`.**
- QA gate: CarOK / FluentPython / Form `QA_PASS`; Combat `QA_PASS_WITH_ADVISORIES`
  (0 failures; one `ASSET_TINY` advisory on small logo crops).
- Acceptance criteria met: dense-page whole-page JSON fallback ~0 (was ~58%);
  crop-audit drift cleared (was 40-50%).

Five fixes this cycle closed the gap, each one a deeper layer of the SAME 8B-VLM
repetition pathology surfaced only by running the hardest pages (charter §2.3):
1. page-dim propagation into chunk spatial metadata (`297a384`) - the hard TABLE
   structural fail;
2. within-page text dedup (`a7ddc07`) + completion to exact-any-length
   (`77aa2ec`) - chunk-level repetition (DUPLICATE_LONG_TEXT + the stricter
   universal byte-equal gate + the page-chunk outlier);
3. VLM sampling repetition penalty (`0b15a48`) - source-side curb;
4. partial-first-element salvage (`93dac05`) - premature-EOS mid-table truncation
   (the 5 CarOK Docling fallbacks);
5. degenerate-repetition collapse (`64cf109`) - within-cell loops (CarOK's
   13,955-char looped row -> TABLE_CORRUPTION).
Earlier in the cycle: the `VLM_NATIVE_TIMEOUT` 180->600s fix and the
json_schema-off self-hosted default (see the 2026-06-03 entry below).

## §9.2 full crucible (2026-06-04) - this cycle's fixes HELD; a new layer surfaced

Ran a 16-doc crucible across all 8 categories (big docs sliced to ~25 interior
pages, small full; 285 pages, 224 VLM-routed; 3.9h on M5, resilient breaker).

**This cycle's fixes held at corpus scale:** 0 Docling fallbacks across all 224
VLM pages, 0 within-page repetition dupes, salvage/timeout/dedup all stable.
**10/15 extracted docs pass** (8 QA_PASS + 2 QA_PASS_WITH_ADVISORIES).

**The broader corpus surfaced a NEW layer of work (next cycle, needs
prioritization) - all in VLM output QUALITY, not the fixed failure modes:**
1. `table_markdown_ratio < 0.80` (all 5 fails: Firearms 0%, Grundlagen 0%,
   Hybrid-EV 17%, AIOS 63%, FluentPython 75%) - the VLM does not always format
   tables as markdown grids. The dominant new failure.
2. Empty TABLE content (AIOS 2/8, Grundlagen 1/1, Hybrid-EV 1/6) - a table
   region detected but emitted with empty content -> TABLE_CORRUPTION /
   table_placeholder_ratio.
3. Crop drift on full-bleed / photo content (DigitaleFotografie 42% edge-clamp,
   Hybrid-EV 68% edge-clamp, HarryPotter 33% blank) - likely crop-audit
   edge-clamp FALSE POSITIVES on legitimate full-bleed art (a photography book),
   plus blank crops on prose pages. Needs a heuristic review, not necessarily a
   crop fix.
4. 1 empty image visual_description (Grundlagen).
5. 1 hard render crash: Kimothi `FzErrorArgument: Invalid bandwriter header
   dimensions` (fitz, in-run/transient; all 25 pages render fine in isolation).
   Fail-open render guard is the clear robustness fix.

**Next:** prioritize the table-format/empty-element/crop-heuristic work (some
needs prompt changes + live re-validation), add a render guard, then re-run the
crucible. The Grand Soak (all 43 docs / 12,215 pages) remains multi-day and
un-run. Live artifacts: output/crucible_full_run/ + output/crucible_full_src/
(gitignored).

## Prior state (2026-06-03 PM - blocker remediation shipped)

## Current state (2026-06-03 PM) - Charter §9.1 remediation SHIPPED

The two Grand-Soak blockers (Charter §8/§9.1) are remediated on branch
`v3.1-extraction-hardening` (pushed to origin), seven atomic commits, each gated
on `pytest tests/ -q` (1395 passed / 99 skipped), `tests/test_repo_integrity.py`,
ruff/black on changed files, and `scripts/smoke_production.sh`
(`SMOKE_PRODUCTION_PASS`, offline).

- **Blocker A (VLM invalid JSON on dense pages) - CLEARED.** A1 typed truncation
  detection (`finish_reason=length` + one budget escalation +
  `VlmTruncationError`); A2 adaptive per-page output budget (floor 8192, cap
  16384); A4 bounded JSON repair (keep the N complete elements); A3
  json_schema/guided_json constrained-decode capability + fail-open 400
  strip-retry. Live M5 check (Combat Aircraft dense magazine, json_schema off):
  dense-page Docling fallback **~58% -> 0**, 0 truncation, real VLM extraction
  (`uir_native_chunker`) at ~88 s/page.
- **Blocker B (bbox crop drift 40-50%) - REMEDIATED.** B1 prefers a deterministic
  geometric bbox (`get_image_info`/`find_tables`) for the crop; B2 re-renders a
  drift-flagged crop to the full page before persisting (garbage crop never
  written; `reextracted` flag). Live: 5 image assets materialized, 1 B2
  re-extraction fired.
- **json_schema default = OFF for self-hosted (live-evidence correction).**
  mlx-vlm accepts json_schema but its constrained decode is too slow on dense
  pages (180s timeout). json_schema/guided_json stay opt-in via
  `VLM_NATIVE_STRUCTURED_OUTPUT` for vLLM + xgrammar. See `docs/DECISIONS.md`
  + `docs/paper/FINDINGS_LOG.md` (2026-06-03).
- **A5 (per-region) NOT built** - gated on A1-A4 not clearing the fallback rate;
  they cleared it.
- **Dense-page TIMEOUT failure - RESOLVED (2026-06-04).** A follow-up per-page
  measurement of Combat Aircraft INTERIOR pages corrected an over-optimistic
  read: at the 180s default, dense interior pages lost ~46% (median 265s, 5/13
  fully timed out). Root cause was a budget x decode-speed mismatch - 8192-token
  pages need ~248s on M5, which 180s guillotined - NOT a hang and NOT predicted
  by image density. Fix shipped: `VLM_NATIVE_TIMEOUT` wired into `from_env` +
  default raised 180 -> 600s + read-timeout retries capped at 1 (B4 intact).
  Re-measure: **13/13 ok, 0 loss**. A5 / DPI cut / density-keyed batch sizing are
  unnecessary for correctness (nothing hangs). See DECISIONS.md + FINDINGS_LOG
  2026-06-04. NOTE: this corrects the framing above - A1-A4 cleared Blocker A's
  JSON-VALIDITY half; this timeout failure was a separate dense-page issue.
- **Crucible re-run / Grand Soak: still NOT run** - re-run the bounded crucible
  subset (`batch_size=1` for dense docs, structured output off) to confirm the
  acceptance criteria at corpus scale before any Grand Soak.

## Prior state (2026-06-03 AM)

Layer-1 current-state doc. For the as-built architecture + roadmap read
`docs/ARCHITECTURE_V3.1_CHARTER.md`; for decision history `docs/DECISIONS.md`;
for the engineering log `docs/paper/FINDINGS_LOG.md`. The only definition of done
lives in `docs/V3_EXECUTION_MANDATE.md`.

## Current state (2026-06-03)

- **V3.1 extraction hardening is COMMITTED** on branch `v3.1-extraction-hardening`
  (pushed to origin). Commits, in order: `c6c2105` (schema-compliant VLM
  extraction + resilient breaker + code/form mapping), `2a60a99` (router code
  blind spot closed via monospace heuristic), `3d5d9e5` (VLM provenance /
  `original_vlm_type`), `b44724b` (empty-content asset-chunk qdrant guard).
- **Tests:** full suite 1355 passed / 99 skipped / 0 failed; offline
  `SMOKE_PRODUCTION_PASS` (`scripts/smoke_production.sh`).
- **Authoritative architecture:** `docs/ARCHITECTURE_V3.1_CHARTER.md` (as-built +
  roadmap, status-tagged). The 0.5 draft is the original V3.0 *target*; the
  charter is the corrected *current reality* (V3 is an additive hybrid, not a
  Docling replacement).
- **Latest run (2026-06-02): Grand Soak HALTED at doc 9/17 - the pipeline does
  NOT meet requirements on the documents V3 targets.** The AIOS code-extraction
  smoke passed (25 VLM pages, 0 fallback, crop-audit PASS), but the soak then
  exposed two blocking extraction failures (full analysis in
  `docs/paper/FINDINGS_LOG.md`, 2026-06-02 entry):
  - **(A) VLM emits invalid JSON on dense pages -> mass Docling fallback.** 34
    page-level `json.loads` failures (mostly truncated mid-`content`); Combat
    Aircraft magazine fell back to Docling on ~25 of 43 pages - the
    layout-mangling path V3 exists to replace.
  - **(B) VLM bbox crop drift 40-50%** on forms/scans/tables (`QA_WARN_CROP_DRIFT`
    fired on 5 of 8 completed docs). Clean born-digital docs (AIOS, the hybrid
    review, Form_0013) passed.
  Throughput (~20-60 s/VLM-page) plus `--max-pages 200` also left the entire
  long tail (~20 large books) unattempted.
- **Real next work** is the extraction layer, not "run another soak": (1) VLM
  JSON validity on dense pages (raise/handle `max_completion_tokens`, detect
  `finish_reason=length`, consider guided JSON decoding or per-region
  extraction); (2) bbox fidelity to cut the 40-50% crop drift. See charter §8.

## What shipped this cycle (V3.1 extraction hardening)

The previous crucible soak produced 0/18 valid baselines - a `from_uir` schema
breach, not the M5 outage it was first blamed on. The fixes (all boundary
contracts now fail open; see charter §2.1):

1. **Schema compliance.** `src/mmrag_v2/universal/asset_materializer.py` crops
   IMAGE/TABLE bbox regions to PNG and sets `asset_ref` (fixes the QA-CHECK-05
   mass failure); shared by the batch path and the soak so they cannot diverge.
   Plus producer-side `visual_description` truncation in `from_uir`. Gates NOT
   weakened. Proven on real M5 VLM output.
2. **Crop-audit.** Per-crop drift signals (full-page-fallback / edge-clamp /
   low-information) + doc-level `QA_WARN_CROP_DRIFT` at 15%, in `meta.json`.
3. **Circuit breaker + resilient soak.** `VlmInfraError` hard-fails on infra
   errors instead of silently falling back to Docling; the soak harness defaults
   to a pause-and-poll breaker (poll 60s; hard-fail after a 30-min ceiling OR 5
   per-doc infra failures); `--strict-breaker` for attended runs.
4. **VLM code/form.** `ElementType` stays 3-value (Charter §7.1; contract test
   unchanged); code/form are smuggled as TEXT + a `promoted_modality` tag and
   promoted to `Modality.CODE`/`FORM` in the chunker; unknown types degrade to
   TEXT with an `original_vlm_type` provenance marker, never crash the page.
5. **Router Boundary-1 closed.** `src/mmrag_v3/engines/router.py` adds a
   monospace-char-ratio signal (>= 0.10) so code-as-text pages route to the VLM
   instead of being stripped by Docling.

## V3 foundation (complete)

- **Phase A native-UIR refactor COMPLETE** (`813b9ba`): `batch_processor.py` is
  engine-agnostic on input and emission boundaries (extraction via
  `mmrag_v3.extract()` -> `UniversalDocument` -> `chunk_universal_document()` ->
  `IngestionChunk.from_uir()`); zero docling imports; legacy OCR/layout lanes
  deleted.
- **Phase C vision-native extraction SHIPPED**: `HybridEngine` router +
  `VlmNativeEngine` + `DoclingFastEngine`; AST firewall `tests/test_v3_security.py`
  (13/13).
- **PLAN_V3.1**: P1 (single extraction path) + P2 (UIR-native HEADING coverage)
  landed; P3 (kept heuristics adopted + guarded: `AGENT-SPATIAL-20`,
  `_merge_mid_sentence_chunks`), P4 (geometric boundary-repair bridge deprecated
  for VLM-native), P5 (`smoke_production.sh` anti-rot gate) shipped. Detail in
  `docs/DECISIONS.md` and `docs/paper/FINDINGS_LOG.md`.

## Active model / endpoint state

- **VLM extraction:** code default is OpenRouter `qwen/qwen3-vl-8b-instruct`; this
  cycle ran the local **M5** (`macbook-pro-m5.lan:8000`, mlx
  `Qwen3-VL-8B-Instruct-8bit`) for bandwidth (charter §5). Override via
  `VLM_NATIVE_ENDPOINT` / `VLM_NATIVE_MODEL` / `VLM_NATIVE_API_KEY`.
- **Judge / soak scoring:** GX10 (GB10) `RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic`
  at `http://10.0.10.239:8000`.
- **Embedder + reranker:** omlx-server `http://10.0.10.246:8000`
  (`Qwen3-Embedding-8B-mxfp8` 4096-dim + `gte-reranker-modernbert-base-mlx`).
- **Vector store:** Qdrant.

## Retrieval stack

dense (omlx Qwen3-Embedding-8B) + sparse (BM25) -> RRF fusion -> ModernBERT
rerank -> top-5. A ColPali visual late-interaction reranker is **PROPOSED, not
built** (charter §6.2): adopt only if the text stack is measured insufficient on
visual-dense queries.

## Engine override env vars

| Env var | Effect |
|---|---|
| (none) | `MineruQwenHybridEngine` when `MINERU_ENDPOINT` set (Qwen for code-dense pages, MinerU elsewhere); else legacy `HybridEngine` |
| `USE_MINERU_QWEN_HYBRID=1` | force the MinerU+Qwen-for-code hybrid |
| `USE_MINERU_ENGINE=1` | all pages through `MineruNativeEngine` (pure MinerU escape hatch) |
| `USE_VLM_ENGINE=1` | all pages through `VlmNativeEngine` |
| `USE_HYBRID_ENGINE=1` | force the legacy cost-optimizer `HybridEngine` |
| `USE_DOCLING_FAST=1` | all pages through `DoclingFastEngine` (offline, deterministic) |
| `VLM_DRAWINGS_THRESHOLD=N` | router treats `> N` drawings as visual (default 10) |

## Must-respect constraints

- Python 3.10 only; batch size <= 10 pages; `docling` exact-pin 2.86.0.
- BBoxes: integer `[0,1000]` (`COORD_SCALE`).
- No filename- or document-specific rules; profile overrides are debug-only.
- Acceptance requires `GATE_PASS` + `UNIVERSAL_PASS` across the smoke matrix, and
  `SMOKE_PRODUCTION_PASS` for any V3 extraction-path change.
- Default-route dependency: the hybrid default needs BOTH servers (GX10 MinerU +
  M5 Qwen) for any code-bearing doc; a Qwen transport outage trips the circuit
  breaker and HALTS the doc (correctness over availability, no MinerU fallback).
  Use `USE_MINERU_ENGINE=1` for MinerU-only availability. See `DECISIONS.md`
  "MinerU+Qwen-for-code hybrid is the default extraction route" -> Operational
  envelope.

## Test command

```
pytest tests/ -q            # 1355 passed / 99 skipped / 0 failed
bash scripts/smoke_production.sh   # -> SMOKE_PRODUCTION_PASS (offline)
```

## Archived history

All v2.10-v2.16 plans, telemetry, calibration reports, quality snapshots,
diagnostics, prior handovers, and dated audit/smoke one-offs are quarantined
under `docs/.archive/`; `.aiignore` blocks agent reads there. Do not reference
archived paths from active docs.
