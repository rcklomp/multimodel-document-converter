# Plan: v2.9 Phase 3 — `IMAGE_DESCRIPTION_UNUSABLE` resolution

**Status:** v3 — Steps 1-3 implemented 2026-05-09. Step 1 baseline at `docs/QUALITY_SNAPSHOT_2026-05-09_v2.9_phase3_vlm_baseline.md`. Steps 4-5 carried forward to next iteration.
**Master plan:** `docs/PLAN_V2.9.md` Phase 3.
**Predecessors:** Phase 1 closed `df91061` (2026-05-07); Phase 2 closed `29a7242` (2026-05-08).
**Phase nature:** Gate calibration + VLM retry. Real cloud spend (~$15-30 expected).

## Why this plan exists

Every Phase 2 conversion ran with `--vision-provider none`. All image chunks land with `vision_status="pending"` and the strict-gate `image_placeholder_ratio=1.000` failure. Phase 3 is the lane where VLM enrichment actually produces descriptions and the gate decides what's acceptable. Per `docs/PLAN_V2.9.md` Phase 3:

> Decide and implement the strict-gate handling for terse but valid VLM responses such as `Venn diagram.` or `Line chart.` This is gate/model behavior, not extraction.

## Non-negotiable contract

| Contract | Source | Hard requirement |
|---|---|---|
| **0 `vision_status="pending"`** in final corpus | Plan §3 | All images either `complete` or `hard_fallback` |
| **`hard_fallback` chunks exempt from placeholder/blankish checks** | Phase 3 v2 review (option a) | Exemption applies only when `metadata.vision_error` AND `metadata.vision_provider_used` are present and non-empty. The hard-fallback **rate** is still bounded (see next row). The enrichment script keeps its current behavior — does NOT rewrite `content` or `visual_description` for hard_fallback chunks; the gate logic accommodates the unchanged data shape. |
| **Hard-fallback rate bounded by documented v2.9 cloud threshold** | Plan §3 + §5 pre-flight | Bound set during Step 1 from cloud baseline. Above bound → strict FAIL on `VISION_HARD_FALLBACK_RATE` (currently WARN) |
| **Source Sanctity inviolable** | `AGENTS.md` "Modality Boundaries", `docs/DECISIONS.md` "Source Sanctity" | No text transcription. `validate_vlm_response()` rules untouched. `VISUAL_ONLY_PROMPT` not weakened. Phase 3 changes the GATE and adds RETRIES; it does not relax validation. |
| **Short-vs-complex classification is asset-driven, not flat character count** | Plan §3 acceptance | `_is_blankish_visual_description` (currently `len < 20 and "layout" not in lower()`) replaced with a complexity-aware rule. `qa_semantic_fidelity.py:is_placeholder_image_or_table` updated to match. |

## Out of scope

- Local VLM lane (`NuMarkdown-8B-Thinking-mlx-8bits` at `http://10.0.10.246:8000/v1`) — Plan §2 Non-Goal; cloud `qwen3-vl-plus` only.
- Broad 34-doc enrichment — Phase 5.
- Qdrant refresh / drop+recreate — Phase 5.
- Phase 4 work (Combat p66 / Adedeji p301 / KI EPUB / Firearms HEADING) — separate phase.
- `cfg.vlm.*` config surface in `mmrag_v2.config` — informational only; the enrichment lane resolves keys differently (see P-1).

## Pre-flights (run BEFORE Step 0)

### P-1: enrichment lane API key resolves

`scripts/enrich_image_chunks_v29.py` hardcodes `qwen3-vl-plus` against the DashScope international OpenAI-compatible endpoint and resolves keys via `_resolve_api_key()`: `MMRAG_REFINER_API_KEY` → `DASHSCOPE_API_KEY` → `~/.mmrag-v2.yml refiner.api_key`.

```bash
conda run -n mmrag-v2 python -c "
import os, yaml
from pathlib import Path

PROVIDER = 'qwen3-vl-plus (DashScope international, OpenAI-compatible)'

env_key = os.environ.get('MMRAG_REFINER_API_KEY') or os.environ.get('DASHSCOPE_API_KEY')
file_key = None
home = Path.home() / '.mmrag-v2.yml'
if home.exists():
    cfg = yaml.safe_load(home.read_text()) or {}
    file_key = ((cfg.get('refiner') or {}).get('api_key'))

source = (
    'env MMRAG_REFINER_API_KEY' if os.environ.get('MMRAG_REFINER_API_KEY') else
    'env DASHSCOPE_API_KEY'   if os.environ.get('DASHSCOPE_API_KEY') else
    f'{home} refiner.api_key' if file_key else
    None
)
print('enrichment lane provider:', PROVIDER)
print('api_key source           :', source)
assert source is not None, (
    'Phase 3 cannot start: enrichment lane has no resolvable API key. '
    'Set MMRAG_REFINER_API_KEY or refiner.api_key in ~/.mmrag-v2.yml.'
)
print('STEP_P1_OK')
"
```

### P-2: Source Sanctity guard tests pass

```bash
conda run -n mmrag-v2 python -m pytest tests/test_vlm_text_detection.py -v
```

This is the canonical text-reading detection regression suite. Step 4 acceptance includes re-running this command and confirming the test count went **up** (new retry-prompt tests added), not just stayed flat.

**Stop condition:** Phase 3 doesn't start until both pre-flights are green.

---

## Step 0 — Targeted Phase 5b enrichment to produce baseline data

**Why first:** Phase 3's calibration decisions (asset complexity, retry threshold, hard-fallback ceiling) are empirical. Cannot calibrate against zero data.

**Scope:** 3 image-rich docs spanning the corpus shape:
- **PCWorld** (`digital_magazine`, layout-heavy, full conversion needed — smoke only ran 10 pages)
- **Hao** (`technical_manual`, mixed photo+diagram — bridge run output reusable)
- **Adedeji** (`technical_manual`, simpler diagrams — bridge run output reusable)

```bash
# PCWorld needs a fresh full conversion — bridge_phase2plus did not include it.
conda run -n mmrag-v2 python -m mmrag_v2.cli process \
  "data/digital_magazine/PCWorld_July_2025_USA.pdf" \
  --output-dir output/probe_phase3_pcworld \
  --vision-provider none --batch-size 10 --no-cache --verbose

# Enrich all three. Atomic write-back; restartable on crash.
conda run -n mmrag-v2 python scripts/enrich_image_chunks_v29.py \
  output/bridge_phase2plus/pdf_hao/ingestion.jsonl \
  output/bridge_phase2plus/pdf_adedeji/ingestion.jsonl \
  output/probe_phase3_pcworld/ingestion.jsonl \
  > /tmp/phase3_step0_enrich.log 2>&1

# Post-enrichment invariant scan (mandatory).
python /tmp/phase3_verify_enrichment.py \
  output/bridge_phase2plus/pdf_hao/ingestion.jsonl \
  output/bridge_phase2plus/pdf_adedeji/ingestion.jsonl \
  output/probe_phase3_pcworld/ingestion.jsonl
```

The post-enrichment verifier (run after Step 0 and again after Step 4) — single-shot, can live in `/tmp` or be tracked under `scripts/verify_enrichment_invariants.py`:

```python
# Reports per-JSONL: vision_status counts, placeholder_content count,
# placeholder_visual_description count, and hard_fallback chunks missing
# either vision_error or vision_provider_used.
import json, sys, re
from pathlib import Path

PLACEHOLDER_RE = re.compile(r'^\[Figure on page', re.MULTILINE)

def scan(jsonl):
    counts = {'complete': 0, 'hard_fallback': 0, 'pending': 0, 'other': 0}
    total_images = 0
    placeholder_content = 0
    placeholder_vd = 0
    non_hf_placeholder_content = 0
    non_hf_placeholder_vd = 0
    hf_missing_error_or_provider = 0
    for line in Path(jsonl).read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r.get('object_type') == 'ingestion_metadata' or r.get('modality') != 'image':
            continue
        total_images += 1
        meta = r.get('metadata') or {}
        status = meta.get('vision_status') or 'other'
        counts[status] = counts.get(status, 0) + 1
        content = r.get('content') or ''
        vd = meta.get('visual_description') or ''
        content_is_placeholder = bool(PLACEHOLDER_RE.search(content))
        vd_is_placeholder = bool(PLACEHOLDER_RE.search(vd))
        if content_is_placeholder:
            placeholder_content += 1
            if status != 'hard_fallback':
                non_hf_placeholder_content += 1
        if vd_is_placeholder:
            placeholder_vd += 1
            if status != 'hard_fallback':
                non_hf_placeholder_vd += 1
        if status == 'hard_fallback' and (
            not (meta.get('vision_error') or '').strip()
            or not (meta.get('vision_provider_used') or '').strip()
        ):
            hf_missing_error_or_provider += 1
    print(f'  {jsonl}')
    print(f'    total_image_chunks={total_images}  {counts}  '
          f'placeholder_content={placeholder_content}  placeholder_vd={placeholder_vd}  '
          f'non_hf_placeholder_content={non_hf_placeholder_content}  '
          f'non_hf_placeholder_vd={non_hf_placeholder_vd}  '
          f'hf_missing_error_or_provider={hf_missing_error_or_provider}')

for p in sys.argv[1:]:
    scan(p)
```

**Acceptance:**
- All three runs complete with non-zero `enriched` counts.
- `hard_fallback` count recorded per doc (the **starting metric** for Step 3's threshold setting).
- Verifier reports: `pending=0`; `complete + hard_fallback = total_image_chunks`; `hf_missing_error_or_provider=0`.
- `placeholder_content` and `placeholder_vd` may be non-zero **on hard_fallback chunks only**.
- Verifier reports `non_hf_placeholder_content=0` and `non_hf_placeholder_vd=0`.

**Stop condition:** Hard-fallback rate >50 % on any doc → endpoint drift or asset-loading break; stop and triage before continuing.

**Estimated:** $10-20 cloud spend; 60-90 min wall time.

---

## Step 1 — Empirical analysis of actual VLM response shapes

**Why second:** Asset-complexity classification is hand-waving without seeing what `qwen3-vl-plus` actually produces.

Build a one-off analysis script that reads the three Step 0 JSONLs and bins each `metadata.visual_description` (excluding hard_fallback) by:
- length (chars)
- presence of "layout" / "diagram" / "chart" / "graph" / "table" / "photo" keyword
- whether the source asset is "simple" (small bbox, single Docling picture-class label) vs "complex" (full-page, table-with-text, multi-panel)
- whether `validate_vlm_response()` flagged anything

**Output:** `docs/QUALITY_SNAPSHOT_<date>_v2.9_phase3_vlm_baseline.md` with:
- For each (length-bucket × asset-class) cell: count + 3 example descriptions.
- A **proposed asset-complexity classifier** documented (rules: "simple if A; complex if B; text-heavy if C") with the empirical signal each rule reads.
- The proposal's contract: short complete descriptions on simple assets → WARN; short on complex → FAIL → trigger retry.

**Stop condition:** If qwen3-vl-plus systematically returns under-20-char descriptions on >30 % of complex assets, gate calibration alone won't fix it — Step 4 retry harness becomes load-bearing. Document and continue.

**Estimated:** 30-45 min; no cloud spend.

---

## Step 2 — Asset-complexity classifier (production code)

**Where:** `src/mmrag_v2/vision/asset_complexity.py` (new) — production code, NOT `scripts/`. Both the QA gate (Step 3) and the Step 4 retry harness import it. No duplication.

**Inputs the classifier reads (cheap, no model call):**
- `chunk.metadata.spatial.bbox` width × height ratio of page area
- `asset_ref.file_path` size on disk (proxy for visual richness)
- Docling picture-class label if present in metadata (`other`, `icon`, `chart`, `bar_chart`, `pie_chart`, `flow_chart`, etc.)
- Optional: re-classify by reading the asset PNG if needed

**Output:** `asset_complexity ∈ {"simple", "complex", "text_heavy"}` plus a reason string.

**Tests:**
- Unit-shape with synthetic inputs covering every classifier rule (`tests/test_asset_complexity.py`).
- Real-shape (env-gated, `RUN_PHASE3_REAL_SHAPE=1`) on the Step 0 corpus: counts per class and at least 3 hand-verified examples per class.

**Acceptance:**
- Classifier is data-driven; no filename-specific logic.
- Real-shape test runs against the Step 0 enriched JSONLs.

**Estimated:** 1-2 hours.

---

## Step 3 — Gate calibration (TWO surfaces)

The strict-gate's image-placeholder failure has **two independent surfaces**, both must change:

### 3a. `scripts/qa_full_conversion.py`

Replace `_is_blankish_visual_description` with a complexity-aware rule:

```python
def _is_blankish_visual_description(
    text, complexity, vision_status, vision_error, vision_provider_used
):
    # 0. Hard-fallback with recorded reason + provider → not blankish.
    if (
        vision_status == "hard_fallback"
        and (vision_error or "").strip()
        and (vision_provider_used or "").strip()
    ):
        return False
    # 1. Empty or true placeholder → still always blankish.
    t = (text or "").strip()
    if not t:
        return True
    if PLACEHOLDER_VISUAL_RE.search(t):
        return True
    # 2. Short complete description on a simple asset → not blankish (WARN class only).
    # 3. Short complete description on complex/text-heavy asset → blankish (FAIL/retry).
    if len(t) < 20 and "layout" not in t.lower():
        return complexity in ("complex", "text_heavy")
    return False
```

Plus: same exemption for the `IMAGE_CONTENT_PLACEHOLDER` warning emission. Plus: introduce a **hard-fallback ceiling** parameter, defaulted from a constant; documented in `docs/QUALITY_GATES.md`. Above ceiling → `FAIL` on `VISION_HARD_FALLBACK_RATE` (currently `WARN`).

### 3b. `scripts/qa_semantic_fidelity.py`

This script computes `image_placeholder_ratio` from `is_placeholder_image_or_table(content)` independently. Update it to:

```python
def is_placeholder_image_or_table(
    content, vision_status=None, vision_error=None, vision_provider_used=None
):
    # Exempt hard_fallback chunks only when reason + provider are recorded.
    if (
        vision_status == "hard_fallback"
        and (vision_error or "").strip()
        and (vision_provider_used or "").strip()
    ):
        return False
    # ... existing logic ...
```

And update the `image_placeholders` count loop (line 95) to pass status, error, and provider from chunk metadata.

**Tests:**
- Pin every branch of both functions in unit tests (`tests/test_qa_image_gate_calibration.py`).
- Real-shape test on Step 0 corpus: counts of FAIL vs WARN under new rule, with a snapshot stored alongside the gate.

**Acceptance:**
- Step 0 corpus reports **0 FAIL** on `IMAGE_DESCRIPTION_UNUSABLE` after the new rule, OR every FAIL is a complex/text-heavy asset where Step 4 (retry) is expected to fix it.
- Old behavior preserved on the synthetic "all placeholders, no VLM" case so Phase 2 outputs (`--vision-provider none`) still trigger the deferred-class path.

**Estimated:** 2-3 hours including tests.

---

## Step 4 — VLM retry harness for complex assets

**Where:** `scripts/enrich_image_chunks_v29.py`. Calls the cloud provider **directly** with a detail-prompt variant. `VisionManager.enrich_image()` retries on `validate_vlm_response()` failures (text-reading, empty, etc.) — NOT on short-but-valid responses. Adding a callback to `VisionManager` would conflate validation-retry with content-quality-retry; keeping the new retry in the script lane preserves `VisionManager`'s contract.

**Behavior:** When the first VLM call returns a complete-but-short description AND `asset_complexity ∈ {"complex", "text_heavy"}`, re-prompt with a fuller-detail prompt variant that still respects Source Sanctity:

- Current `enrich_image_chunks_v29.py` skips `vision_status="complete"` chunks when the description is non-placeholder. Phase 3 must add a separate `needs_retry` path that reprocesses **already-complete** short descriptions when all of the following are true:
  - `vision_status == "complete"`
  - `visual_description` is short under the calibrated rule
  - `asset_complexity in {"complex", "text_heavy"}`
  - retry has not already been consumed, tracked by either `metadata.vision_detail_retry_attempted != true` or an equivalent `vision_attempts`/retry counter
- Same `VISUAL_ONLY_PROMPT` base.
- Append a phrase like: *"Describe the visual layout, components, and their relationships in detail. Do not transcribe text from the image."*
- Validate the retry response with `validate_vlm_response()` exactly as the first call.
- **Cap retries at 1 per asset** (cost guardrail).
- If retry still short → mark `vision_status="hard_fallback"` with reason `"complex_asset_short_response_after_retry"`.
  - Preserve `metadata.vision_provider_used="qwen3-vl-plus"` and set a retry marker so reruns do not call the VLM again for the same short asset.

**Tests** (added to `tests/test_vlm_text_detection.py` or a new sibling file):
- Mock the VLM call; verify retry path engages only for complex/text-heavy + short-but-complete.
- Verify already-complete non-placeholder short descriptions are eligible for retry only through the explicit `needs_retry` path.
- Verify retry prompt doesn't change Source Sanctity rules (Source Sanctity test count goes UP, not flat).
- Verify cap-at-1 holds.

**Acceptance:**
- Re-run enrichment with retry on the Step 0 corpus.
- Run the post-enrichment verifier again.
- Compare BEFORE vs AFTER retry: complex/text-heavy short-complete count drops.
- Any new hard_fallback additions have `vision_error="complex_asset_short_response_after_retry"` (or equivalent exact reason) and `vision_provider_used="qwen3-vl-plus"`.
- Hard-fallback count may increase if the retry still returns a short complex/text-heavy response; that is acceptable only if the final hard-fallback rate remains under the documented ceiling.
- 0 instances of retry calling the VLM more than once for the same asset (assertable via call-count log).

**Estimated:** $5-10 additional cloud spend (bounded — only retries fire); 3-4 hours including tests.

---

## Step 5 — End-to-end verification

```bash
for d in pcworld hao adedeji; do
  echo "=== $d ==="
  conda run -n mmrag-v2 python scripts/qa_full_conversion.py \
    output/<paths-from-Step-0>/ingestion.jsonl \
    --source-pdf data/<...>.pdf \
    --allow-warnings
    # NOTE: --no-require-image-descriptions is action="store_true" — OMIT it
    # to require descriptions. Do NOT pass =False.
done
```

## Promotion gate (Phase 3 closes when ALL true)

- [ ] P-1: enrichment-lane API key resolves from one of the three documented sources.
- [ ] P-2: `tests/test_vlm_text_detection.py` baseline green BEFORE Step 4; same suite still green AFTER Step 4 with **higher test count** (new retry-prompt tests added).
- [ ] Step 0: 3 docs enriched, verifier reports `pending=0`, `hf_missing_error_or_provider=0`, `non_hf_placeholder_content=0`, and `non_hf_placeholder_vd=0`.
- [ ] Step 1: VLM-baseline snapshot tracked at `docs/QUALITY_SNAPSHOT_<date>_v2.9_phase3_vlm_baseline.md`.
- [ ] Step 2: `src/mmrag_v2/vision/asset_complexity.py` shipped with unit + real-shape tests; classifier is data-driven (no filename-specific logic).
- [ ] Step 3: BOTH gate surfaces (`qa_full_conversion.py:_is_blankish_visual_description` AND `qa_semantic_fidelity.py:is_placeholder_image_or_table`) updated; old synthetic "all placeholders" case still triggers Phase-2-style WARN; new corpus triggers fewer FAILs.
- [ ] Step 4: retry harness shipped; explicit `needs_retry` path covers already-complete short complex/text-heavy assets; cap-at-1 verified; complex/text-heavy short-complete count drops after retry; any new hard_fallback has the retry-exhausted reason and provider recorded.
- [ ] Step 5: All three Step 0 docs report `image_placeholder_ratio` ignoring hard_fallback exemptions (effectively 0 on complete chunks); `IMAGE_DESCRIPTION_UNUSABLE` either PASS or WARN-only on simple-asset class; `VISION_HARD_FALLBACK_RATE` under documented ceiling.
- [ ] `pytest tests/ -q` passes with 0 failures.
- [ ] No filename-specific production logic.

## Risk register

1. **Cloud endpoint drift between runs.** Mitigate via Step 0's recorded hard-fallback baseline; Step 5 re-checks against it.
2. **Asset-complexity classifier disagrees with human judgment.** Mitigate via Step 1's empirical analysis informing classifier rules + 3-example-per-class hand-verification.
3. **Retry harness drifts Source Sanctity.** Mitigate via P-2 pre-flight + Step 4 retry-prompt-tested-by-validation tests + the "test count goes UP" check at Step 4 acceptance.
4. **Per-asset cost balloons** (retries × 5,500 images on broad reconversion). Mitigate via cap-at-1 + Phase 5 will only retry images that pass the complexity filter.
5. **`cfg.vlm.*` confusion.** P-1 documents that the enrichment lane ignores `cfg.vlm`; this plan does not change that. If a future contributor edits `cfg.vlm` thinking it affects enrichment, that's a doc-clarity issue for v2.10.

## What I will NOT do without explicit user sign-off

- Touch `validate_vlm_response()` body or `VISUAL_ONLY_PROMPT` text.
- Add a 2nd or 3rd retry tier.
- Change cloud provider or model.
- Run enrichment on the broad 34-doc corpus (Phase 5 territory).
- Drop or recreate Qdrant collections.
- Lower the existing 20-char threshold globally without an asset-complexity gate.
- Have hard_fallback chunks rewrite their `content` or `visual_description` (data fabrication; the gate handles them via the F4 exemption instead).

## Estimated total wall time + cost

| Step | Wall time | Cloud cost |
|---|---:|---:|
| P-1, P-2 | 2 min | 0 |
| 0 | 60-90 min | $10-20 |
| 1 | 30-45 min | 0 |
| 2 | 1-2 h | 0 |
| 3 | 2-3 h | 0 |
| 4 | 3-4 h | $5-10 |
| 5 | 30 min | 0 |
| **Total** | **8-12 h** | **$15-30** |

The 8-12 h wall time is mostly Step 0 + Steps 2-4 implementation, broken across review checkpoints between steps so course corrections happen early.

---

## Review history

- **v1 (this session, before second-pass review):** Initial draft. Contained five defects: P-1 referenced wrong config surface (`cfg.vlm` vs the enrichment lane's actual key resolution); P-2 referenced non-existent test files; Step 5 used invalid argparse syntax (`--no-require-image-descriptions=False`); hard-fallback contract was internally inconsistent (claimed "0 placeholder descriptions" while allowing hard_fallback to retain placeholders); only `qa_full_conversion.py` was scoped, missing `qa_semantic_fidelity.py`'s independent placeholder check.
- **v2 (this document):** All five defects fixed; three recommended edits incorporated (post-enrichment verifier; asset-complexity classifier in production code; Step 4 retry calls provider directly).
- **v3 (2026-05-08):** Tightened hard-fallback exemption to require both `vision_error` and `vision_provider_used`; expanded the verifier with `total_image_chunks`, non-fallback placeholder counts, and `hf_missing_error_or_provider`; added the explicit `needs_retry` path for already-complete short complex/text-heavy assets; replaced the non-guaranteed "net hard_fallback drops" acceptance with short-complex reduction plus bounded hard-fallback rate.
