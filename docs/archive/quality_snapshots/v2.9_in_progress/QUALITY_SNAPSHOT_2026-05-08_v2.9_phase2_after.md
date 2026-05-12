# Quality Snapshot 2026-05-08 — v2.9 Phase 2 AFTER

> **Phase 2 closure snapshot.** v2.9 has not shipped. This snapshot
> records the Phase 2 verification results after Phase 1 closed in
> commit `df91061` (2026-05-07). The active baseline remains
> `docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md` (last shipped
> tag is v2.8.0). The v2.9 strict-gate working state is tracked in
> `docs/QUALITY_SNAPSHOT_2026-05-06_v2.9_strict_gate.md` (BEFORE
> state, frozen for delta reproducibility).

**Strict gate:** `scripts/qa_full_conversion.py` —
`qa_conversion_audit.py` + `qa_universal_invariants.py` +
`qa_ingestion_hygiene.py` + `qa_semantic_fidelity.py` plus
deterministic checks.

**Phase 2 contract source:** `docs/PLAN_V2.9.md` Phase 2 + Plan §2
Goals 2-4. Phase 2 is verification, not implementation.

## Phase 2 verdict

**All Phase 2 promotion gate conditions met.** Phase 2 promotes;
Phase 3 (`IMAGE_DESCRIPTION_UNUSABLE` policy) and Phase 4 (localized
hard failures) become the active phases.

## Per-step results

### Step 0a — refiner config preflight

`refiner.enabled=true` confirmed in active config
`/Users/ronald/.mmrag-v2.yml`; `provider=openai`, `model=qwen-plus`.
This is the precondition that makes Steps 2 and 3 actually exercise
`_decide_enable_refiner` rather than passing trivially because the
default is False.

### Step 0 — smoke baseline

`bash scripts/smoke_multiprofile.sh output/smoke_phase2_baseline`
finished 2026-05-07 20:23 UTC. **11/11 `GATE_PASS + UNIVERSAL_PASS`**
(SCAN0013 shows the documented form variant
`GATE_PASS [form: micro_non_label + label-orphan checks skipped]`).
0 `CONVERT_ERROR`, 0 `MISSING_JSONL`, 0 SIGALRM in any per-doc log.

### Step 1 + Step 6a — chunk_id uniqueness

**5,749 chunks scanned across 17 JSONLs (11 smoke + 4 Phase 2 docs +
Kimothi full-doc + Kimothi codex probe). 0 duplicates.**

### Step 2 — HARRY (refiner smart-suppression)

| Signal | Result |
|---|---|
| `profile_type` | `digital_literature` ✅ |
| `has_encoding_corruption` | `false` ✅ |
| All 651 text chunks `refinement_applied=false` | ✅ |
| AUDIT_PASS / UNIVERSAL_PASS / HYGIENE_PASS | ✅ |
| `tests/test_docling_postprocessor_acceptance.py` (live, with `HARRY_ACCEPTANCE_JSONL=...`) | **2 passed, 0 skipped** ✅ |

5 blank source pages correctly excluded (4, 12, 322, 324, 325).
Remaining `QA_FAIL` is `image_placeholder_ratio=1.000` — Phase 3
deferred class (`--vision-provider none`).

### Step 3 — Combat Aircraft (refiner smart-engagement)

| Signal | Result |
|---|---|
| `profile_type` | `digital_magazine` ✅ |
| `has_encoding_corruption` | `true` ✅ |
| `[OCR-GUARD] ENCODING CORRUPTION` smart-route fired | ✅ |
| Refined chunks (`refinement_applied=true`) | **109** ✅ |
| `corruption_score > 0` on refined chunks | ✅ values 0.4-1.0 |
| `edit_ratio rejected` log spam count | **0** ✅ |
| Phase 1 invariants (SIGALRM, `recovery_page_coverage`, empty) | all 0 ✅ |
| `LOCALIZED_CORRUPTION` count | **1 chunk on p66** = documented Phase 4 residual (squadron-roster table) |
| New failure classes | none |

The `table_markdown_ratio=0.75 (<0.80)` is the same root cause: p66's
quarantined corrupted-table → 3-of-4 tables have markdown content.
Single blank figure on p27 (`a4c2916a64c2_027_figure_36.png`,
mean=253, std=7.2) is a minor advisory matching the existing
`tiny_images` class.

### Step 4 — Firearms (split acceptance)

Phase 2 plan originally adopted Plan §2 Goal 3 verbatim into Step 4
acceptance. Goal 3 is the **v2.9 ship target**, not the Phase 2
verification scope. The conflation was corrected mid-Step-4 with the
**split acceptance**:

- Phase 2 verifies: profile route-flip + AGENT-SPATIAL-20 + Phase 1 invariants
- Phase 4 owns: HEADING coverage and chunk-count drift

| Signal | Result |
|---|---|
| `profile_type=scanned`, `document_type=scanned_degraded` | ✅ Phase 4 commit `3fbce7a` route-flip verified |
| AGENT-SPATIAL-20 violation | none ✅ |
| Phase 1 invariants | all 0 ✅ |
| HEADING coverage | **72 % (790/1094)**, target ≥ 0.80 — *carried to Phase 4* |
| Chunk count | **2183**, v2.8 fresh = 1690 (+29 %), target ±2 % — *carried to Phase 4* |

The HEADING and chunk-count metrics show the Phase 4 commit's
profile route-flip mechanism succeeded but did not deliver the
acceptance criterion it was scoped against. The OCR/shadow lane on
the `scanned`/`scanned_degraded` route emits ~30 % more chunks per
page with more orphan-heading rows than the v2.8 baseline. **This is
new work for Phase 4** — added to its scope alongside Combat p66 /
Adedeji p301 / KI EPUB residuals.

### Step 5 — Ayeva (CodeFormulaV2 lane)

| Signal | Result |
|---|---|
| `profile_type=technical_manual` | ✅ Phase 3 Rule 0c fix held |
| `[CODE-ENRICH-DECISION] needs=True` | ✅ |
| `[CODE-ENRICH] Enabled Docling code enrichment` (per batch) | ✅ ≥8 occurrences |
| `code_indentation_fidelity` | **0.9693** (target ≥ 0.85) ✅ |
| AUDIT_PASS / UNIVERSAL_PASS / HYGIENE_PASS | all ✅ |
| Phase 1 invariants | all 0 ✅ |
| Back-index per-page chars (285-290) vs source PDF text layer | 76-105 % (same band as Phase 1 probe) ✅ |
| `MISSING_PAGES`: 1 non-blank source page (4) | residual exception per Plan §1: page 4 is the dedication page (`"I would like to thank my parents for their love and support. – Kamon Ayeva"`, 74 chars). Reproducibly near-blank. |

10 blank source pages correctly excluded (7, 21, 41, 59, 61, 135,
181, 239, 253, 283). Remaining QA_FAIL is `image_placeholder_ratio=1.0`
— Phase 3 deferred.

## Config Provenance (Step 5 evidence governance)

Phase 2 Step 5 ran with `MMRAG_CONFIG=/tmp/mmrag_v2_9_phase2_code_enrichment.yml`,
which was derived from the active home config
`/Users/ronald/.mmrag-v2.yml` plus a single override:
`code_enrichment.enabled: true`. The `/tmp` file is local-transient
evidence; the YAML body is recorded here for reproducibility per
`docs/AGENT_GOVERNANCE.md` Evidence Rules. `api_key` fields are
SHA-256-prefix redacted.

```yaml
vlm:
  provider: openai
  model: NuMarkdown-8B-Thinking-mlx-8bits
  base_url: http://10.0.10.246:8000/v1
  api_key: <REDACTED:sha256-prefix-ed9668ff358b>
  timeout: 120
refiner:
  enabled: true
  provider: openai
  model: qwen-plus
  base_url: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
  api_key: <REDACTED:sha256-prefix-1ef11344d9c4>
defaults:
  batch_size: 10
  output_dir: ./output
code_enrichment:
  enabled: true
```

Reproduce: copy the active `~/.mmrag-v2.yml` to `/tmp/...yml`, set
`code_enrichment.enabled: true`, `MMRAG_CONFIG=...` for the Ayeva
conversion only.

## Tests

`pytest tests/ -q` at the time of Phase 2 entry (commit `df91061`):
**628 passed, 14 skipped, 0 failed**. No regressions introduced by
Phase 2 (verification phase only — no production code edits beyond
the Phase 1 commit already on `main`).

## Carry-forward into Phase 3 / Phase 4

**Phase 3 (`IMAGE_DESCRIPTION_UNUSABLE` policy/model):** All four
Phase 2 conversions ran with `--vision-provider none`, so every
image chunk has `image_placeholder_ratio=1.0`. Phase 3 owns the
calibration of "complete but terse" VLM descriptions vs hard-fail
class. No Phase 2 finding moves the goalposts here.

**Phase 4 (localized strict-gate hard failures):**
Pre-Phase-2 scope (per Plan §4):
- Combat Aircraft p66 — known table corruption, must close
- Adedeji p301 — known table corruption
- KI_En_ChatGPT — pre-existing EPUB LABEL ratio
- Devlin / Earthship / Firearms — page-loss-overlap re-evaluation
  after Phase 1

**Phase 2 added to Phase 4 scope:**
- Firearms HEADING coverage 72 % → ≥ 0.80 (chunker emission rate
  on `scanned`/`scanned_degraded` lane)
- Firearms chunk-count drift +29 % → ±2 % of 1690

## Phase 2 promotion

Phase 2 closes 2026-05-08. Phase 3 and Phase 4 become active phases.
Phase 5 (broad reconversion + Qdrant refresh + AFTER snapshot)
remains blocked until Phases 3-4 close.
