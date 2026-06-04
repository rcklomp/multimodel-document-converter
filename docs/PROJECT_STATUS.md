# Project Status

Last updated: 2026-06-03 (PM - blocker remediation shipped)

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
| (none) | HybridEngine cost-optimizer routing (default) |
| `USE_VLM_ENGINE=1` | all pages through `VlmNativeEngine` |
| `USE_DOCLING_FAST=1` | all pages through `DoclingFastEngine` (offline, deterministic) |
| `VLM_DRAWINGS_THRESHOLD=N` | router treats `> N` drawings as visual (default 10) |

## Must-respect constraints

- Python 3.10 only; batch size <= 10 pages; `docling` exact-pin 2.86.0.
- BBoxes: integer `[0,1000]` (`COORD_SCALE`).
- No filename- or document-specific rules; profile overrides are debug-only.
- Acceptance requires `GATE_PASS` + `UNIVERSAL_PASS` across the smoke matrix, and
  `SMOKE_PRODUCTION_PASS` for any V3 extraction-path change.

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
