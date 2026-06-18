# Project Status

Last updated: 2026-06-18 (branch `feat/omnidocbench-phase0`, pushed). Code-book coverage
expanded (14 code books converted + ingested; production dense ~37k / sparse ~26k pts),
plus extraction-reliability hardening and a code-fidelity audit (see the deferred note).

> **DEFERRED to post-first-production-release (user-directed 2026-06-18):** the VLM
> (mlx_vlm.server / Qwen3-VL-8B on the M5) is the recurring reliability/quality liability -
> intermittent per-request handler wedge (mitigated by a client hard deadline +
> retry-on-fresh-connection, `4bfd7c8`/`b8160aa`), non-deterministic double-transcription of
> page furniture into code chunks (mitigated by `_strip_code_furniture`), and weaker
> non-Python (C/C++) code fidelity (unaddressed). Tackle the VLM properly (server
> logging+watchdog / more robust serving path / model+prompt work) as an improvement only
> once the first production-level release is achieved. Details in the open-issues memory backlog.

Earlier (2026-06-15) two things closed: the residual-burndown plan, and a measured
RAG/retrieval investigation that reframed where the real bottleneck is.

## Current state (2026-06-15) - residual burndown DONE; RAG measured; the lever is "feed top-10", not conversion

**Residual burndown (`PLAN_FIDELITY_ORACLE_FIRST_V1` Section 3') COMPLETE.** WS1a
content-emptiness advisory (`e653fac`), WS1b extraction-ladder verdict signal -
laddered code hard-FAILs (`8e954c1`), WS2a fullwidth code token-corruption scrub
(`ef5a377`), WS2b Adedeji thin-strip cull (`254ec07`); WS2c OCR-on-fallback DROPPED
(already tried, drove the V3 pivot); WS2d prose-into-code CLOSED as a non-issue
(measured 0.0%); WS3 render tail PROVEN (cap1600 best on the dense academic class,
`85dfc9b`). All with frozen fixtures, suite green, offline `SMOKE_PRODUCTION_PASS`.

**The reframe + RAG measurement (the bigger finding).** The "done" bar is "the RAG
works well" - a RETRIEVAL bar, not a transcription bar. Measured end-to-end on 514
code-inclusive queries over the 29 ingested docs (`output/v3_soak_code/`, fully local
oMLX + Qdrant; gen+judge on GX10 14B):
- **Conversion is NOT the bottleneck** (the data said so twice). Doc-level retrieval is
  strong (R@5-doc 91.6%); the soft spot is chunk-level precision.
- **The answer-quality lever is FEED THE LLM TOP-10 CHUNKS, not top-5**: +4.9pp answer
  correctness (56.8 -> 61.7), German-safe, one-line gen-config change.
- **Three "clever" levers were ruled out by measurement before shipping:** empty image
  chunks (+0pp), rerank-score-sort (-10pp), and hybrid/BM25 (no gain over plain top-10,
  and a German regression). The BM25-index persistence fix (`e3467bc`) was still a real
  bug fix - hybrid was silently broken - but it is not needed for the answer win.
- **The one remaining retrieval item:** ~6% of queries never retrieve the right document
  even at top-100 (an embedder / query-expansion problem). This is the next task.

**Housekeeping (2026-06-15):** repo sanitation pass - archived done plans + dead
old-phase scripts to `.archive/`; reconciled stale docs; the findings log is being
condensed for faster session onboarding. Tests untouched (all 146 sound).

See `[[project_retrieval_findings]]` / `[[project_oracle_first_pivot]]` (memory) for detail.

## Current state (2026-06-13) - WP-A P2 re-attempt: chunker contiguity is the WRONG LEVER; residual re-scoped

The registered "chunker contiguity" item (PLAN_F1 WP-A) is **BUILT, correct, and
non-regressing, but it is NOT the lever for Chaubal's 0.85** - the prior
Phase-1-closure diagnosis ("15/26 of Chaubal's residual fails are one code block
split across interleaved figure/table chunks") is **FALSIFIED**.

- `_coalesce_code_blocks` shipped in `uir_chunker._chunk_page` (`c6c1f4c`+`9866832`,
  12 unit tests): when code Elements on a page form one logical block (prev ends
  open `:`/`\`/unclosed-bracket/unterminated-docstring, or next starts mid-body)
  across an interleaved figure/table/prose Element, the code segments merge into
  one chunk and the non-code is emitted after. Per page only; never bridges a page
  boundary; integer bbox union; heading-carry + table/form untouched. Full suite
  **1690 passed / 100 skipped / 0 failed**; WS-B negatives green; offline
  `SMOKE_PRODUCTION_PASS`.
- **P2 re-attempt FAIL -> STOP (registered rule).** Re-extract + `f1_oracle`:
  FluentPython 15pg slice QA_PASS indent 1.00 (no-regression), Jungjun 0.669
  (coalesced 0x, swing is fresh-VLM-extraction variance not a chunker regression -
  the prior 0.895 stands on its extraction), **Chaubal 0.828 < 0.85 (coalesced 0x)**.
  Per the rule: no floor weakening, no book swap; code books EXCLUDED from WP-C
  ingestion.
- **Why 0x.** The 129 same-page CODE-[noncode]-CODE interleavings in Chaubal are
  **REPL/notebook transcripts** (Jupyter `[17]:` inputs / `[t17]: tensor(...)`
  outputs around prose), NOT one contiguous block split by a figure - genuinely
  separate snippets the continuation predicate correctly declines to merge. The
  residual is further dominated by **engine token corruption** (`=`->`\(\equiv\)`,
  CJK garbage, fullwidth punctuation).
- **RE-SCOPED REGISTER ITEM (was "chunker contiguity"):** Chaubal's P2 0.85 needs
  a **REPL/notebook-transcript-aware code handler + engine token-corruption repair**
  (de-LaTeX `\(\equiv\)`, CJK/fullwidth scrub) - a NEW item distinct from chunker
  segmentation. Keep the contiguity fix (net-positive, fires on the genuine
  contiguous-split class, 0x here, no regression). Owner/scope TBD.

## Current state (2026-06-12) - PLAN_F1 Phase 1 closed (scoped); chunker fragment-merge registered

PLAN_F1 Phase 1 (deterministic text-layer code lane) is **closed, scoped**:
- **P1 VALIDATED.** On Jungjun (a born-digital x-indented code book) the lane +
  signal recalibration + code-content repairs take repair-touched judgeable-Python
  `ast.parse` from 0.000 to **0.90-0.92** (>= the unchanged 0.85 oracle floor).
  Devlin reclassified OUT as a flat-source defect (no indentation in the PDF at
  all - a new P0 population, like Earthship/Adedeji).
- **P2 BLOCKED-BY-CHUNKER.** Chaubal (P2) tops out at ~0.78: its code is clean and
  correctly indented, but ~58% of the residual `ast.parse` fails are code blocks
  the CHUNKER split across an interleaved image/table/prose chunk (15 of 26), which
  no post-hoc adjacent-code merge can heal without breaking document order. The
  one-iteration general fragment-merge helped Jungjun (0.90->0.92) but was neutral
  on Chaubal and was reverted (it is a band-aid; the fix belongs in the chunker).
- **REGISTERED ITEM (pipeline-wide chunker):** the chunker must not split a single
  code block across a non-code (figure/table) chunk - keep code-block segments
  contiguous, or merge code fragments that bracket an interleaved non-code chunk.
  This is a chunker-segmentation fix, not code-hygiene; it gates re-attempting P2's
  0.85 and applies corpus-wide. Owner/scope TBD.
- Shipped this phase (committed, not pushed): modality-seam fix (`c95950b`),
  `text_native_code` signal + recalibration + Mechanism B + the `f1_oracle`
  (`31a0ee0`/`05ac421`), code-content repairs - smart quotes, open-string wraps,
  fence-strip, nbsp normalize (`f713954`/`51ccef9`). No gate/fixture weakening;
  Workstream B negatives green throughout; suite 1678.

## Current state (2026-06-11 PM) - Phase 5: spec reconciled + bounded corpus re-ingested

`PLAN_EXTRACTION_FIDELITY_V1` Phase 5 is DONE. WP-1 rewrote the Layer-0 spec to the
Phase 4 reality (charter §4 retry-first + fail-closed ladder + rollback hatch; QUALITY_GATES
advisory fidelity gate; F1-F9 dispositioned - `DECISIONS.md` "Phase 5 - spec rewrite
closure"). WP-2 re-extracted a 12-doc bounded crucible subset (12/12 `mineru_qwen_hybrid`,
`degraded=0`, 0/724 laddered) and dense-ingested 3338 points into `mmrag_v3__qwen3_local` on
the LOCAL Mini Qdrant (`127.0.0.1:6333`, omlx 4096-dim). A per-process conda-env routing
fault to the M5/GX10 (EHOSTUNREACH - VPN `utun` scoped-route) was bridged with a localhost
relay (`scripts/phase5_relay.py`); the permanent fix is the VPN split-tunnel. The BM25 sparse
twin `mmrag_v3__bm25_sparse` (1854 pts, `scripts/phase5_ingest_bm25.py`) is also DONE - RRF
fuses by chunk_id, validated 20/20. Deferred: full-corpus reconciliation (~32 docs). See
`HANDOVER_PHASE5_REEXTRACT_REPORT.md`.

## Prior state (2026-06-11 PM) - Phase 4 complete: the hybrid is the formal production default

`PLAN_EXTRACTION_FIDELITY_V1` Phase 4 is DONE (controls + evidence + records; no
engine code changed - the default route already selected `MineruQwenHybridEngine`
when `MINERU_ENDPOINT` is set). Evidence + records: DECISIONS.md "Phase 4 - the
MinerU+Qwen hybrid is the production default", FINDINGS_LOG 2026-06-11 (Phase 4),
gitignored `HANDOVER_PHASE4_REPORT.md`.

- **Shadow window (WP-A): the flip is JUSTIFIED.** 16-doc crucible, identical
  15-page slices per doc, both arms via the shipping CLI (`--vision-provider none`).
  arm A = interim default (`USE_DOCLING_FAST=1`), arm B = the hybrid config. arm B
  regresses NO doc and is strictly better on 5: QA_WARN+QA_FAIL rate arm A 25%
  (4/16 QA_FAIL) vs **arm B 0%**. The 4 arm-A failures are real docling content
  losses (CarOK spreadsheet HEADING 0/37 table-flatten; Firearms + DigitaleFotografie
  HEADING 0/0, zero text on image/scan pages with `do_ocr=False`; HarryPotter dropped
  page 12). arm B: 0 ladder-served pages, 0 leak; known class gaps (tables/scans/code)
  all favour B.
- **Formalized config:** route `mineru_qwen_hybrid` = GX10 MinerU `:8001`
  (`MinerU2.5-2509-1.2B`) + M5 Qwen `:8000` (`Qwen3-VL-8B-Instruct-8bit`, code lane)
  + cap1600 render, default precedence (no force flag). Runtime prerequisite
  installed: the `[mineru]` extra `mineru-vl-utils` (was missing in the Mini env;
  its absence silently ladders every non-code page to docling).
- **Rollback (WP-B, pre-named):** revert to `USE_DOCLING_FAST=1` if, over any 10
  consecutive production docs, QA_WARN+QA_FAIL > 20pp (arm-B baseline 0%) OR
  ladder-served > 2% of pages (baseline 0%). The env-var routing stays alive through
  Phase 5 (pinned by `test_docling_fast_overrides_mineru_default`).
- **Re-extraction policy (WP-B, written; execution USER-SCHEDULED):** stale = any
  prior JSONL whose provenance is not (hybrid + GX10 + cap1600); re-extract via
  `scripts/rebaseline_v3.py`; Qdrant re-ingestion is user-scheduled (production
  collections on the M1 docker, not this box). Nothing re-extracted/ingested tonight.
- **Interim default retired** from "production default" to tier-2 of the fail-closed
  ladder; the option-2 scanned/image-only exclusion DIES with the flip (hybrid OCRs
  scans on the primary path). The laddered-scan-still-blank residual stays the Phase 3
  OCR-on-fallback candidate.
- **Validation (WP-C):** 7 routing tests pass (default + rollback paths);
  `SMOKE_FULL=1` with the exact flipped env -> `SMOKE_PRODUCTION_PASS`. Two-axis
  advisory baseline: hybrid Phase 1 fidelity text-ED 0.2212 / TEDS 0.7933.
- **Phase 5 (Layer-0 spec rewrite) NOT done** - charter/mandate/QUALITY_GATES edits
  are Phase 5. Phase 3 scope (quality-risk arbitration, specialist re-extraction,
  content-empty health guard, OCR-on-fallback net) untouched.

## Current state (2026-06-11) - Phase 1 complete: hybrid validated, Phase 2 settled by evidence, Phase 4 greenlit

The Phase 1 two-corpus bake-off ran verdict-ELIGIBLE (158-page OmniDocBench fixed
set + 6-doc internal corpus, 4 registered candidates, paired bootstrap stats per
the pre-registered Section 7.2 rule) and was RATIFIED by the user 2026-06-11. See
DECISIONS.md "Phase 1 outcome RATIFIED" + FINDINGS_LOG 2026-06-11:
- **Pipeline-vs-hybrid: INCONCLUSIVE by construction** (identical on a code-free
  benchmark; +0.0001). The default does not move on it.
- **Pure VLM-primary REFUTED** (hybrid > Qwen3-VL: text-ED +0.0346, TEDS +0.1745,
  CIs exclude 0). **Pure pipeline-primary REFUTED for code** (R3 0.300 vs 0.947).
- **The MinerU+Qwen hybrid is non-dominated on every measured class.** Phase 2 is
  settled by the same evidence: ONE specialist lane (Qwen-for-code), already
  implemented. **Phase 4 (formalize hybrid as production default, with shadow
  window + rollback + re-extraction policy) is greenlit and is the next milestone.**
- Baseline-provenance CORRECTION: the 0.301/0.563 full-755 baseline came from the
  OCR-enabled legacy offline route, NOT `USE_DOCLING_FAST=1` (do_ocr=False, proven
  content-empty on image-only input). Interim-default disposition RESOLVED
  (option 2): keep `USE_DOCLING_FAST=1` with a documented scanned/image-only
  exclusion, time-boxed to the Phase 4 flip; tier-2-net OCR fix registered for
  Phase 3.
- R3 reconciliation: the PRODUCTION schema prompt preserves code indentation on
  Qwen (0.950); the render-sweep's indentation loss was a PROMPT property.

## Prior state (2026-06-10) - rev. 4 prerequisites shipped; interim default + serving home + render cap decided

`PLAN_EXTRACTION_FIDELITY_V1` rev. 4 Phases 0A/0B/0.5 are DONE (two unattended
runs, 2026-06-10; evidence in FINDINGS_LOG + the gitignored
`HANDOVER_MORNING_REPORT*.md`). Shipped: MinerU retry-before-fallback
(`6a352da`+`66d2a08`, 22 tests), Section 5.4 provenance consumers wired to the
JSONL + QA advisory block (`bcfac2b`), seeded-fault blindness report (`d046583`
- text-ED is BLIND to code-indentation loss; junk-presence gate signals are
BLIND to all omission faults), Phase 0A n=44 render sweep + 3-way MinerU
serving probe, and the `VLM_RENDER_MAX_PX` longest-side render cap (default
1600, env rollback).

**Decisions ratified 2026-06-10 (see DECISIONS.md "Phase 0B interim default +
MinerU serving home + cap1600 render"):** (1) INTERIM production default = the
offline floor (`USE_DOCLING_FAST=1`) with an initial production-level
acceptance definition; (2) MinerU serving home = GX10 vLLM
(`http://10.0.10.239:8001`, served id `MinerU2.5-2509-1.2B`) - mlx MinerU
serving deprecated (deterministic `broadcast_shapes` on magazine/form pages,
concurrency collapse); (3) cap1600 INTERIM render for the VLM lane (dpi200 was
measured fidelity-HARMFUL: repetition loops on dense pages, text-ED 0.411 vs
0.081). **Phase 0 is effectively complete by substitution** - all bake-off
candidate engines now have a healthy serving path, so Phase 1 is
verdict-ELIGIBLE pending its entry gates (Section 7.2 health logging + the
150-200 page set; the fixed 44-page subset is the down payment).

**Test state:** `pytest tests/` = 1624 passed / 99 skipped / **0 failures**
(fully green; the long-standing `test_v3_vlm_code_form` contract conflict was
user-adjudicated to the F4 fenced contract, `4f20801`).
`SMOKE_PRODUCTION_PASS` (offline) on the branch tip.

## Prior state (2026-06-09) - OmniDocBench fidelity benchmark + fail-closed extraction

Branch `feat/omnidocbench-phase0` (NOT pushed). Two workstreams shipped on top of
the 2026-06-08 crucible baseline:

**Fail-closed extraction ladder (`fcd4207`, `e73fbd8`, + working-tree hardening).**
`mmrag_v3.processor.extract()` no longer depends on a remote GPU server staying
healthy. It degrades through three tiers, each serving only the pages the tier
above could not:
- tier 1 - selected engine (MineruQwenHybridEngine default when `MINERU_ENDPOINT`
  is set, else legacy HybridEngine; forced via `USE_*` env flags),
- tier 2 - offline `DoclingFastEngine` (no network),
- tier 3 - PyMuPDF native-text terminal tier (no model/network).

The served lane + outcome are stamped on `doc.metadata.extra`
(`extraction_engine` / `extraction_fallback` / `extraction_degraded_pages` /
`extraction_recovered_pages` / `extraction_fallback_reason`) and logged.
Guarantee scope (honest): no page that HAS an extractable text layer is zeroed by
an engine/server/network failure; a cheap text-layer probe means genuinely blank
pages pay zero fallback cost; the terminal tier never fabricates text. Out of
scope: a scanned page where OCR also fails. Working-tree hardening (this session)
broadened degeneracy detection to zero-element pages, backstopped the whole-doc
failure path with the terminal tier, and added the blank-page cost gate (3 tests).

**OmniDocBench Phase 0+1 (`746ea31`..`e970e54`).** Ground-truth fidelity benchmark
(adapters + stratified baseline). Full 755-page English baseline: text
edit-distance 0.301 / TEDS 0.563. Phase 1 extractor bake-off harness shipped but
**INCONCLUSIVE** - the M5 mlx MinerU2.5 server threw intermittent
`broadcast_shapes` 500s (serving faults, not a fidelity verdict); paddle/granite
adapters deferred. See `docs/PLAN_OMNIDOCBENCH_EVAL.md`.

**Test state (honest):** `pytest tests/` = 1616 passed / 99 skipped on the
committed branch tip, plus **1 KNOWN failure** -
`tests/test_v3_vlm_code_form.py::test_code_smuggles_as_text_promotes_to_code_modality`.
Diagnosed (2026-06-10): it is a CONTRACT CONFLICT, not an implementation bug. Its
`assert ic.content == code` (verbatim, UNFENCED) was superseded by the deliberate
F4 fencing contract (`eeffcff` "code MUST be fenced"), now pinned by the passing
`tests/test_code_fencing_f4.py` + the `code_fence_consistency` guard. No
implementation can satisfy both contracts; which one wins needs USER adjudication
(proposed requirement change in `HANDOVER_MORNING_REPORT.md`). Left failing per
the don't-weaken-a-test rule. `ruff` clean; `SMOKE_PRODUCTION_PASS` (offline).
(RESOLVED 2026-06-10: user adjudicated to the F4 fenced contract, `4f20801` -
see Current state above.)

## Prior state (2026-06-08) - full 16-doc crucible CLEAN; clusters A/C/B/D fixed; multimodal image policy shipped

The full-crucible Grand Soak (16 docs) that closed the MinerU+Qwen cycle surfaced
5 gate failures in 4 root-cause clusters. All four are fixed, corpus-validated,
and committed on branch `fix/crucible-clusters-acd-b` (7 commits, NOT yet pushed;
each gated `SMOKE_PRODUCTION_PASS`). Final result: the full 16-doc crucible is
**16/16 clean QA_PASS** post-enrichment, `leak=0`, 0 hard fallbacks. Each fix was
a SYSTEMIC bug, not a one-doc patch:

- **A - asset-render fail-open** (`7b1871b`): a MuPDF PNG crash during cosmetic
  crop materialization discarded the whole batch's extracted text (Kimothi
  HEADING 20%->92%). Crop encode now falls back to a full-page render and asset
  rendering can never abort a batch. Hardened (`de1af9d`): asset-less IMAGE/TABLE
  chunks are dropped before `from_uir` so no render-fail path discards the batch.
- **C - engine-agnostic table separator** (`b032a29`): MinerU AND Qwen emit
  separator-less pipe tables (FluentPython table 0.75->1.00). Repair lives at the
  engine-agnostic chunker chokepoint (`universal/table_markdown.py`). Hardened
  (`de1af9d`): escaped-pipe split, ragged-bail (no silent column-shift), leading-
  title/trailing-caption tolerance, single-dash-data-not-a-separator.
- **B - cross-batch heading carry-forward** (`71aeed1`): heading assignment reset
  per batch, so a batch whose chapter title was only a glued running header went
  null (HarryPotter 62%->98%, CombatAircraft 79%->100%). The last heading is
  threaded across batch boundaries; a real in-page/TOC heading still overrides it.
- **D - multimodal no-VLM image policy** (`dd4a758`): a TOC-cell sanitizer was
  silently deleting EVERY image chunk crucible-wide (image content has no text),
  orphaning image-only pages into MISSING_PAGES. Fixed, and the converter's no-VLM
  behavior is now defined (below).

**Multimodal image policy (locked - see `DECISIONS.md`):** images are always
retained. With `--vision-provider none` they ship as documented ID-only fallbacks
(`vision_status=no_vlm`, asset filename as description), treated by the strict
gate as a documented advisory (`IMAGE_NO_VLM`) not a failure; a run-time warning
fires for image-dense no-VLM runs. Image DESCRIPTION is a separate POST-conversion
step (`scripts/enrich_image_chunks_v29.py`), now env-pointable at a LOCAL VLM
(`MMRAG_ENRICH_PROVIDER` / `MMRAG_ENRICH_MODEL` / `MMRAG_ENRICH_BASE_URL`; the
DashScope cloud default is unchanged). Validated end-to-end: ~237 images across 16
docs described by the local M5 Qwen, 0 hard fallbacks, all -> clean QA_PASS.

**Chunk hygiene added (cluster D + hardening):** `_filter_tiny_icon_images` drops
icon/glyph-class image regions (rendered <96px in BOTH dims AND <1.5KB) behind a
page-coverage guard; `_promote_or_drop_empty_tables` drops empty-content tables
but PROMOTES the only-chunk-on-page case to IMAGE (keeps the crop, no
MISSING_PAGES, no TABLE_CORRUPTION).

**Next iteration (plans written, not started):** the same crucible passed every
gate yet a manual content audit found gate-INVISIBLE defects (furniture as
chunks/headings, text-as-image survivors, CJK garbage headings, cover garble).
- `docs/PLAN_GATE_QUALITY_V1.md` - spatial-first advisory metrics paired with
  extraction-side fixes (fix-and-guard), the `AGENT-GATE-PROGRESSION` advisory-to-
  hard protocol (now in AGENTS.md), crucible-calibrated regression fixtures.
- `docs/PLAN_OMNIDOCBENCH_EVAL.md` - ground-truth fidelity benchmark (the gate
  plan measures retrieval value on OUR docs; this measures transcription fidelity
  vs labeled ground truth - 1651 annotated pages, TEDS/edit-distance). PROPOSED as
  a two-axis acceptance; the fidelity axis is ADVISORY (offline selection/regression
  gate, benchmark-gated per `AGENT-GATE-PROGRESSION`), NOT a wired per-conversion
  floor (F5). The recorded hybrid regression baseline is the 158-page fixed set
  text-ED 0.2212 / TEDS 0.7933 (Phase 1); the full-755 0.301/0.563 is a corpus
  reference, never a subset comparator.

**Harnesses:** `scripts/mineru_crucible_soak.sh` (leak-metric corrected to a
whitelist) and `scripts/crucible_vlm_pipeline.sh` (soak -> M5 enrichment ->
revalidate). Deferred review follow-ups #8/#9/#10 tracked in `[[project_open_issues]]`.

## Prior state (2026-06-06) - MinerU+Qwen-for-code hybrid is the default route, corpus-validated

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
  M5 Qwen) for FULL-QUALITY code-bearing docs. A Qwen transport outage trips the
  circuit breaker (no cross-engine MinerU fallback for that code page) and the
  `extract()` fail-closed ladder then serves the page from tier-2 Docling /
  tier-3 PyMuPDF, PROVENANCE-STAMPED `extraction_degraded_pages > 0` - the doc is
  NOT halted, but its laddered code pages are Docling-quality (stripped indentation)
  and STALE by the Phase 4 re-extraction policy. The mandatory pre-batch smoke
  asserts `degraded == 0` precisely to catch this silent-ladder case before a
  corpus run (charter §4.1). Use `USE_MINERU_ENGINE=1` for MinerU-only
  availability, or `USE_DOCLING_FAST=1` for the offline rollback. See `DECISIONS.md`
  "MinerU+Qwen-for-code hybrid is the default extraction route" -> Operational
  envelope, and "Phase 4 - the MinerU+Qwen hybrid is the production default".

## Test command

```
pytest tests/ -q            # current: 1623 passed / 100 skipped / 0 failures (F6;
                            #  measured 2026-06-11. The 1624/99 headline differs by one
                            #  endpoint-gated skip when the inference servers are
                            #  unreachable. The prior test_v3_vlm_code_form contract
                            #  conflict was adjudicated to the F4 fenced contract, 4f20801)
bash scripts/smoke_production.sh   # -> SMOKE_PRODUCTION_PASS (offline)
```

## Archived history

All v2.10-v2.16 plans, telemetry, calibration reports, quality snapshots,
diagnostics, prior handovers, and dated audit/smoke one-offs are quarantined
under `docs/.archive/`; `.aiignore` blocks agent reads there. Do not reference
archived paths from active docs.
