# Plan: v2.14 — Carry-Forwards Closeout + Local LLM Integration

**Status:** **Draft v0.2** (2026-05-22). Phase 0 + Phase 4a + Phase 5
landed locally on the same day as the plan draft; remaining phases
(1, 2, 3, 4b, 4c, N) still to do.

## Phase outcomes so far

| Phase | Status | Outcome |
|---|---|---|
| Phase 0 (judge calibration) | **SHIPPED 2026-05-22** | Per-axis verdict: relevance 81.7% (RESTRICTED), **format 90.2% (TRUSTWORTHY)**, faithfulness 76.1% (RESTRICTED). ±1 agreement ~100% on all axes — local LLM has same ordinal scale, slightly more lenient. Report: `docs/CALIBRATION_2026-05-22_v2.14_p0_local_judge.md` |
| Phase 4a (local HyDE) | **SHIPPED 2026-05-22** | `src/mmrag_v2/retrieval/hyde.py` gained `provider="vllm"` knob. Defaults to GX10 at `http://10.0.10.239:8000` with `Qwen/Qwen2.5-14B-Instruct`. 5 unit tests added. End-to-end smoke OK (392-char hypothesis generated locally in ~2s, $0). Default provider remains `dashscope` — no behavior change for existing callers. |
| Phase 5 (disk precheck) | **SHIPPED 2026-05-22** | `_check_disk_headroom()` in `synthetic_soak.py` aborts retrieve/judge stages below 10 GB free. Override via `SOAK_DISK_HEADROOM_FLOOR_GB` env. 5 unit tests added. |
| Phase 1 (format_form judge axis) | pending | |
| Phase 2 (per-doc regression) | pending | |
| Phase 3 (rollback drop) | pending (2026-06-19) | |
| Phase 4b (local judge in soak) | gated on Phase 1 + reduced scope | Per Phase 0 verdict: Format axis can be cloud-replaced fully; Relevance + Faithfulness should stay on qwen-max for cycle-close go/no-go. |
| Phase 4c (local query gen) | pending | Likely safe — generation isn't judging; the Phase 0 leniency bias doesn't apply. |
| Phase N (close-out) | pending | Engine bump to 2.14.0 when enough phases land to justify a tag. |

---

**Original Draft v0.1** (2026-05-22, preserved below as cycle archaeology).
Authored after v2.13.0 SHIPPED (commit `a77758b`, annotated tag staged
for user push) as the planning artifact for the next cycle.

**Predecessor:** [`docs/PLAN_V2.13.md`](PLAN_V2.13.md) — CLOSED
2026-05-22 with `v2.13.0` annotated tag staged on commit `a77758b`,
pending user push to GitHub + Gitea.
**Owner:** ingestion + retrieval + LLM-integration pipeline.

---

## 1. Why this plan exists

### Thesis

v2.13 closed the embedder workstream (local omlx wins 6/6 axes
apples-to-apples vs cloud) and shipped OCR auto-routing (Earthship
+6.2pp Format). The retrieval-quality side is in good shape:
R@1 ~58%, R@5 chunk ~78%, R@5 doc ~95%, Format ~93%.

What's left:

1. **A handful of per-doc regressions** the omlx swap revealed:
   German content (ATZ_Elektronik -12.5pp R@1), code-dense docs
   (Python_Cookbook -12.4pp, IRJET -12.5pp, Hybrid_electric
   -12.6pp, Greenhouse -12.5pp). These are offset by aggregate
   wins but worth investigating.

2. **The CarOK form-class Format penalty** is a documented judge-
   calibration limitation, not a content defect. The soak protocol
   needs a `format_form` axis variant so future form-class docs
   don't drag the headline Format number.

3. **The 30-day rollback window closes on 2026-06-19.** Decision
   point: drop `mmrag_v2_8__qwen3_dashscope` (1024-dim, ~17 GB)
   and `mmrag_v2_8` (legacy llava) if no rollback fired during
   the window.

4. **Local LLM capability now available.** Asus Ascent GX10 (DGX
   Spark clone) at `10.0.10.239:8000` runs vLLM with
   `Qwen/Qwen2.5-14B-Instruct` (32K context). Free local
   experimentation accelerator unlocks soak workloads the
   $25/cycle cap currently caps: bigger eval sets, hyperparameter
   sweeps, prompt iteration, HyDE default-on.

5. **Open carry-forwards from v2.11** (3a VLM swap, 3c UIR
   refactor PAUSED, 3e magazine rendered-region-crop) still live
   if user re-prioritizes them.

### Non-goals

- **No retrieval-side architecture changes.** v2.13's stack
  (hybrid + RRF + ModernBERT rerank + omlx embedder) is the
  production shape; v2.14 doesn't reshuffle that.
- **No new embedder swap.** omlx Qwen3-Embedding-8B is the
  production embedder for the foreseeable future.
- **No silent gate weakening.** Per the contract-violation rule.

---

## 2. Phases (proposed)

Each phase is a small, soak-validated change. Each can run
autonomously if calibration data justifies it. Phase 0 is the
gating prerequisite for Phase 4.

### Phase 0 — Local-LLM judge calibration *(prerequisite for Phase 4)*

**Goal:** measure per-axis agreement between `Qwen/Qwen2.5-14B-Instruct`
(local on GX10) and `qwen-max` (cloud) using the v2.13 P1 soak's
518 already-judged queries as ground truth. Same JUDGE prompt
structure on both sides — only the model differs.

**Method:** `scripts/calibrate_local_judge_vs_qwen_max.py`
(added 2026-05-22). Reuses `JUDGE_SYSTEM` + `JUDGE_USER_TEMPLATE`
from `synthetic_soak.py` to keep prompts identical. Outputs
`docs/CALIBRATION_2026-05-22_v2.14_p0_local_judge.md` with
per-axis agreement %, confusion matrices, and a disposition
verdict per axis.

**Disposition thresholds:**

| Per-axis exact-match | Verdict | Usage |
|---|---|---|
| ≥85% | TRUSTWORTHY | Use local judge for exploration soaks (RRF weight sweeps, top_k sweeps, prompt iteration) |
| 70-85% | RESTRICTED | HyDE-only — weaker semantics still help retrieval, but not enough for go/no-go judging |
| <70% | NOT USABLE | Either pick a stronger local model (Qwen3.6-35B-A3B-FP8 once vLLM supports it) or stay on cloud |

**Output:** SHIP if all three axes ≥85%; otherwise document
the restricted-use envelope and either (a) accept reduced scope
for Phase 4, or (b) gate Phase 4 on a stronger local model.

**Cost:** $0 (local LLM only; reuses existing qwen-max judgments).

### Phase 1 — Form-class `format_form` judge axis

**Goal:** decouple CarOK-style form-class Format scores from the
prose-calibrated Format axis. Either:
- (a) add a `format_form` axis with a content-shape-aware rubric
      (recognizes that row-by-row inventory data is correctly
      "well-formed" for that content type), OR
- (b) carve the `FORM_AUDIT_PASS` lane out of Format scoring
      entirely and report it on its own line.

**Method:** extend `JUDGE_USER_TEMPLATE` in `synthetic_soak.py`
with a content-shape detection step + axis routing. Existing
form-class detection (`docs/QUALITY_GATES.md` §"Form / Invoice
Acceptance Class") provides the routing signal.

**Acceptance:** CarOK_voorraadtelling Format score recovers to
≥85% on the new axis without regressing prose-doc Format scores
on the other 33 corpus docs.

**Cost:** ~$1-2 for a re-judging pass on the v2.13 P1 fixture
with the amended prompt.

### Phase 2 — Per-doc regression investigation (German + code)

**Goal:** understand whether the omlx regressions on German /
code-dense docs are:
- (a) inherent to Qwen3-Embedding-8B's training distribution,
- (b) artifacts of the v2.13 P1 fixture sampling, or
- (c) compoundable across future similar content.

**Method:**
1. Sample 50 NEW queries each from ATZ_Elektronik (German) +
   Python_Cookbook (code-heavy) — generated via qwen-max with
   the same prompt as the v2.13 P1 fixture.
2. Run hybrid+rerank on both omlx and dashscope (apples-to-apples
   same approach as v2.13 P1).
3. Compare per-doc deltas. If the regression confirms on a fresh
   sample, document the language/code envelope and decide
   whether to:
   - Accept and document (just record "omlx underperforms on X")
   - Add a per-doc embedder routing lane (production)
   - Quarantine to dashscope for those doc classes

**Acceptance:** evidence-backed disposition in `docs/DECISIONS.md`.
If routing lane is chosen, ship as Phase 2a; otherwise document
+ close.

**Cost:** ~$2-3 (qwen-max generates 100 queries + judges 100
on each provider = 400 judgments).

### Phase 3 — 30-day rollback drop (decision point: 2026-06-19)

**Goal:** if no v2.13.0 rollback fired during the 30-day window
(2026-05-22 → 2026-06-19), drop the dashscope rollback collection
and reclaim ~17 GB.

**Method:**
```bash
curl -X DELETE http://localhost:6333/collections/mmrag_v2_8__qwen3_dashscope
# Also drop the v2.10 legacy llava collection if still present:
curl -X DELETE http://localhost:6333/collections/mmrag_v2_8
```
Update `scripts/retrieval_regression_v2_12.py` to skip with a
"rollback collection deleted, see docs/DECISIONS.md" message
rather than failing.

**Acceptance:** disk reclaim verified + `docs/PROJECT_STATUS.md`
"Qdrant collections" table updated.

**Cost:** $0.

**Skip condition:** if any production user reported a regression
during the window that suggested swapping back, defer this phase
to v2.15 and document the issue first.

### Phase 4 — Local-LLM exploration accelerator *(GATED ON PHASE 0)*

**Goal:** wire the local LLM into the experimentation loop so
big-fixture soaks (1000+ queries) and hyperparameter sweeps
(RRF weights, top_k, prompt variants) become free.

**Scope** (per-axis, based on Phase 0 verdict):
- If all axes ≥85%: full local-judge soak mode in
  `synthetic_soak.py` via `--judge-provider vllm --judge-url … --judge-model …`
- If only some axes ≥85%: split judge — use local for the
  trustworthy axes, qwen-max for the rest
- If only HyDE-trustworthy: wire local-HyDE only (modify
  `src/mmrag_v2/retrieval/hyde.py` to support a local-LLM backend),
  leave judging on cloud

**Sub-deliverables:**
- a) `scripts/synthetic_soak.py` gains `--judge-provider` and
     `--judge-url` / `--judge-model` flags (default still
     `dashscope` / `qwen-max` so existing soaks don't shift)
- b) `scripts/calibrate_local_judge_vs_qwen_max.py` becomes the
     persistent calibration harness (re-run before any local-judge
     soak)
- c) `src/mmrag_v2/retrieval/hyde.py` gains a `provider` knob
     (default `dashscope`, alternative `vllm`)
- d) A 1500-query exploration soak (3× current size) demonstrates
     the new capability + produces denser per-doc evidence for
     v2.15 scoping

**Acceptance:** 1500-query soak completes in <60 min and matches
the 518-query baseline within noise on the trustworthy axes.

**Cost:** Phase 0 + a) + b) + c) is ~$0-0.50; (d) demonstration
soak is ~$0 if all axes are local, ~$5 if Phase 0 verdict is
RESTRICTED and some axes need cloud.

### Phase 5 — Disk-headroom precheck in `synthetic_soak.py` *(QoL)*

**Goal:** prevent a repeat of the v2.13 P1 disk-full incident
(disk hit 100% mid-judge, crashed Qdrant). Add a precheck that
refuses to start a stage if free disk is below a configurable
threshold (default 10 GB).

**Method:** small helper at the top of `stage_retrieve` and
`stage_judge` — `shutil.disk_usage(repo_root)` + warning/abort
threshold.

**Acceptance:** unit test simulates low-disk + verifies the
precheck aborts cleanly.

**Cost:** $0.

### Phase N — Cycle close-out

- Engine version bump `2.13.0` → `2.14.0`
- v2.14 retrieval-regression fingerprint (only if Phase 2's
  per-doc routing changes the production stack; otherwise the
  v2.13 fingerprint stays canonical and v2.14 doesn't ship a new
  one)
- AFTER snapshot `docs/QUALITY_SNAPSHOT_<date>_v2.14_after.md`
- Layer-0/1 docs sweep
- v2.14.0 annotated tag staged for user push

---

## 3. v2.11+ carry-forwards (still open)

| Item | Source | v2.14 disposition |
|---|---|---|
| 3a NuMarkdown-8B local VLM | v2.11 plan | Re-evaluate if user re-prioritizes; not blocking |
| 3c UIR refactor (PAUSED) | v2.11 plan | Still PAUSED for user signoff |
| 3e Magazine rendered-region-crop | v2.11 plan | Defer with soak-data rationale (image-axis perf is OK without it) |

---

## 4. Phase ordering rationale

```
Phase 0 (calibration)  ←  prerequisite for Phase 4
Phase 1 (format_form)  ←  independent, ships when ready
Phase 2 (regression)   ←  independent, ships when ready
Phase 3 (rollback drop) ← time-gated (2026-06-19)
Phase 4 (local-LLM acc) ← gated on Phase 0
Phase 5 (disk precheck) ← independent quick win
Phase N (close-out)    ← terminal
```

Suggested execution order:

1. Phase 0 + Phase 5 (parallel, fast, cheap)
2. Phase 1 (form-class judge axis) — small but valuable
3. Phase 2 (per-doc regression investigation) — informs whether
   we need routing
4. Phase 4 (local-LLM exploration) — gated on Phase 0 verdict
5. Phase 3 (rollback drop) — wait for the 2026-06-19 date
6. Phase N (close-out) — once all above ship or are documented as
   deferred

---

## 5. Budget

- **Cost cap**: $25/cycle (same as v2.13)
- **Estimated spend**: $5-10 (Phase 1 + Phase 2 + Phase 4(d)
  if forced to cloud-judge)
- **Local LLM usage**: $0 (LAN)

## 6. Risks

| Risk | Mitigation |
|---|---|
| Phase 0 verdict is <70% on all axes | Phase 4 reduces to "wait for stronger local LLM" (Qwen3.6-35B-A3B-FP8); plan a Phase 4 RFC |
| Phase 2 finds a real language/code regression | Build the routing lane or document the envelope; either way ships safely |
| 30-day window had a real rollback need we didn't catch | Phase 3 skip rule + grace period |
| Disk fills again during a soak | Phase 5 precheck prevents recurrence |

## 7. Open questions for the user (when they're back)

- (none required to start) — Phase 0, 1, 2, 5 are all autonomously
  executable. Phase 3 needs your "no rollback fired, drop it" signoff
  on or after 2026-06-19. Phase 4 is gated on Phase 0 verdict.
- Optional: do you want SSH key onboarding for the GX10? Currently
  the only blocker for me to do live diagnostics on the local LLM
  side if it crashes mid-soak.
