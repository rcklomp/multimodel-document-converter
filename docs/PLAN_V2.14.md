# Plan: v2.14 — Carry-Forwards Closeout + Local LLM Integration

**Status:** **Draft v0.5** (2026-05-23). User audit of Draft v0.4
added 10 improvements across all phases: HIGH-priority quantitative
fallback criteria (Phase 1), HIGH-priority GX10-offline resilience
(Phase 4), MEDIUM-priority lang-detection specifics (Phase 2),
existing-test interaction (Phase 6), VLM fallback budget; and
LOW-priority calibration drift detection, rollback snapshot,
Definition of Done gate, plus parallelization note. See "Draft v0.5
revision summary" below.

## Draft v0.5 revision summary (resilience + quantification audit)

The Draft v0.4 plan was thorough but four gaps surfaced under audit:
(a) subjective fallback escalation criteria in Phase 1, (b) no
graceful-degradation path if the GX10 endpoint goes offline, (c)
implementation ambiguity for the Phase 2 language detector, and
(d) no defined "Definition of Done" gate for the cycle close. All
ten audit items applied in this revision; HIGH-priority items are
hard requirements before their phases can ship.

| # | Priority | Audit item | Where applied |
|---|---|---|---|
| 2 | HIGH | Quantitative Phase 1 fallback escalation | Phase 1 Method |
| 8 | HIGH | GX10 offline graceful fallback + health-check probe | Phase 4 new "Resilience" subsection + Risks row |
| 1 | MED | Phase 2 language-detection implementation specifics | Phase 2 Method |
| 4 | MED | Phase 6 interaction with `tests/test_code_chunking.py` (6 existing tests) | Phase 6 Method |
| 7 | MED | Phase 1 budget accounts for VLM fallback path cost | Budget section |
| 3 | LOW | Calibration drift detection trigger (30-day OR model-change) | Phase 4 GX10 guardrails (already covers model-change); §"Calibration freshness" note added |
| 5 | LOW | Rollback drop snapshot retention (90-day cold storage) | Phase 3 Method |
| 6 | LOW | Definition of Done gate for v2.14.0 tag | Phase N Acceptance |
| 9 | LOW | Archive note for Draft v0.1 archaeology | Phase N close-out (archive after tag) |
| 10 | LOW | Phase 6 and Phase 1 can run in parallel | Phase ordering section |

Source: user audit of Draft v0.4, 2026-05-23.

## Draft v0.4 revision summary (GX10 deployment-hazards review)

A proposed GX10 swap to `Qwen/Qwen2.5-72B-Instruct` via
`vllm/vllm-openai:latest` surfaced four deployment hazards and one
calibration trap (model-swap-invalidates-Phase-0). Captured as a
hard-constraint block under Phase 4 ("GX10 deployment guardrails")
plus a new Risk row. No phase outcomes change; the swap was NOT
executed. Also saved as a feedback rule in
`memory/feedback_gx10_deployment_guardrails.md`.

## Draft v0.3 revision summary (user audit response)

The Draft v0.2 plan proposed two band-aids that the user correctly
flagged: a lenient `format_form` judge axis (Phase 1) and a per-doc
embedder routing lane (Phase 2). Both would have masked or
architecturally entrenched real extraction defects. Evidence in the
v2.13 weakest-query rationales (`docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx.md`)
confirms: CarOK chunks have "table truncation + odd whitespace",
Fluent_Python has "truncated code". The judge isn't wrong — the
extraction is. Saved as a feedback rule in
`memory/feedback_fix_extraction_not_judge.md`.

Four revisions:

1. **Phase 1** redefined: `format_form` judge axis → **Form/Table
   layout extraction recovery** (fix the data, not the rubric).
2. **Phase 2** redefined: per-doc embedder routing → **Targeted HyDE
   bridging for code + minority languages** (use the already-shipped
   Phase 4a local HyDE to bridge the semantic gap at query time;
   no new collections, no permanent routing infra).
3. **Phase 4** guardrails tightened: explicit list of permitted
   local-judge uses (Format-axis judging only; prompt A/B; pre-filter
   tie-breaker pattern). Forbidden: tuning RRF weights, top_k,
   sparse/dense balance, or anything that adjusts retrieval breadth —
   the local judge's upward leniency bias on relevance + faithfulness
   would push tuning toward noisier retrieval.
4. **New Phase 6**: **Code-block chunking hygiene** (fix the
   Fluent_Python / ArcGIS_Python_Cookbook truncation root cause).

## Phase outcomes so far

| Phase | Status | Outcome |
|---|---|---|
| Phase 0 (judge calibration, 14B) | **SUPERSEDED 2026-05-23** | First run against retired 14B. Verdict: relevance 81.7% (RESTRICTED), format 90.2% (TRUSTWORTHY), faithfulness 76.1% (RESTRICTED). Report retained as `docs/CALIBRATION_2026-05-22_v2.14_p0_local_judge.md` for historical comparison. |
| Phase 0 (judge calibration, 27B-MTP) | **SHIPPED 2026-05-23** | Live verdict on `Qwen/Qwen3.6-27B-FP8`: relevance 82.0% (RESTRICTED), **format 70.7% (RESTRICTED)**, faithfulness 78.8% (RESTRICTED). All three axes in 70-85% "HyDE-only" band; ±1 agreement still 98.6-99.8% (ordinal scale consistent). **Bias direction flipped vs 14B**: 27B is systematically STRICTER on format (132 cases where qwen-max said `2`, 27B downgraded to `1`). Phase 4 PERMITTED list contracts — see Phase 4 section. n=518, 0 parse failures after the [[feedback-fix-extraction-not-judge]]-cousin harness fix (commit `0c5e818`, disable `chat_template_kwargs.enable_thinking` for vLLM payloads). Report: `docs/CALIBRATION_2026-05-23_v2.14_p0_local_judge_qwen36_27b_mtp.md`. |
| Phase 4a (local HyDE) | **SHIPPED 2026-05-22 + harness re-validated 2026-05-23** | `src/mmrag_v2/retrieval/hyde.py` gained `provider="vllm"` knob (5 unit tests). Default model updated 2026-05-23 (14B → 27B-MTP). Commit `0c5e818` added `chat_template_kwargs={enable_thinking: False}` to the vLLM payload after the Phase 0 debug revealed the 27B was silently dropping content via reasoning-mode routing; 2 bridge tests assert the payload includes the flag for vllm and omits it for dashscope. Live re-smoke: 670-char hypothesis in 8.8s (the pre-fix "392-char in 2s" claim was the pre-thinking-discovery 14B smoke). Default provider remains `dashscope` — no behavior change for existing callers. |
| Phase 5 (disk precheck) | **SHIPPED 2026-05-22** | `_check_disk_headroom()` in `synthetic_soak.py` aborts retrieve/judge stages below 10 GB free. Override via `SOAK_DISK_HEADROOM_FLOOR_GB` env. 5 unit tests added. |
| Phase 6 (code-block chunking hygiene — `_chunk_text_with_overlap` path) | **PARTIAL — committed 2026-05-23, scope clarification REQUIRED** | Commit `d737147` adds block-extension policy + `partial_code` schema field + 3 new tests (9/9 in `tests/test_code_chunking.py`; suite 1036/1036). Fluent_Python re-extracted with the new chunker: chunk count 2150 → 2101 (-2.3%; well within ≤+15% bound). **However: 547/547 code chunks in the re-extracted Fluent_Python have `extraction_method=hybrid_chunker` / `hybrid_chunker_pagesplit` — they DO NOT go through `_chunk_text_with_overlap`.** Docling's `HybridChunker` (enabled by default for `technical_manual` profile per `pdf_plan.py:90`) is the actual production chunker for code-dense docs. The committed change improves the `scanned_book` path and adds universal `partial_code` observability, but does NOT reach the v2.13 P1 "truncated code" defect on Fluent_Python. Needs user sign-off on direction: (a) post-process HybridChunker output to merge severed code chunks, (b) configure HybridChunker with code-awareness (per [[feedback-libraries-first]]), or (c) accept narrower scope and ship as observability-only Phase 6. |
| Phase 1 (Form/Table extraction recovery) | **PENDING — VLM-fallback sign-off required** | Evidence gathered 2026-05-23: CarOK output has **0 table chunks** (`extraction_method=docling_table_markdown` count == 0). The inventory grid was extracted as flat prose (chunks contain "1 AC Delco, merk = X. 1 Behr, ink.ex.BTW Titel = Y. ..."). Docling TSR (currently `do_table_structure=True`, `do_cell_matching=False`) didn't classify the grid as a table at all — so `do_cell_matching=True` wouldn't help (it's downstream of detection). Per the handoff "stop and ask" item #3, escalation to the VLM fallback (qwen-vl-max or Qwen3-VL-8B on omlx, v2.11 carry-forward 3a) needs user sign-off. Also: CarOK is 12 pages + native_digital, doesn't match the existing `QUALITY_GATES.md` form-class detection rule (`total_pages ≤ 5 AND scanned AND heading_coverage < 0.10`), so any Phase 1 routing change must extend the classifier OR introduce a different form-class signal. |
| Phase 2 (Targeted HyDE bridging) | **PENDING — Draft v0.3 redefinition** | See section 2 below |
| Phase 3 (rollback drop) | pending (decision point 2026-06-19) | Unchanged |
| Phase 4b (local judge in soak — was Format-only) | **DOWNGRADED — Format axis no longer trustworthy on 27B** | Format axis dropped to 70.7% (RESTRICTED) on the 27B, so the Draft v0.3 "Format-axis-only" scope evaporates. Remaining viable uses: prompt A/B (relative ranking, same biased judge on both arms) and tie-breaker harness (cloud `qwen-max` is the final word). The 1500-query Format-axis exploration soak (Phase 4e) is NOT viable on this endpoint. |
| Phase 4c (local query gen) | pending | **Safe — verdict unchanged.** Generation isn't judging; bias direction doesn't apply. Wire `--gen-provider vllm` when convenient. |
| Phase 4e (1500-query Format demo) | **DROPPED 2026-05-23** | Predicate failed — Format axis is RESTRICTED on the 27B, not TRUSTWORTHY. Either skip or repurpose as a Phase 4c query-gen demonstration. |
| Phase N (close-out) | pending | Engine bump to 2.14.0 when enough phases land to justify a tag |

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
| <70% | NOT USABLE | Fall back to cloud `qwen3-max` (Dashscope) |

**Output:** SHIP if all three axes ≥85%; otherwise document
the restricted-use envelope and either (a) accept reduced scope
for Phase 4, or (b) gate Phase 4 on a stronger local model.

**Cost:** $0 (local LLM only; reuses existing qwen-max judgments).

### Phase 1 — Form/Table layout extraction recovery (REDEFINED in Draft v0.3)

**Goal:** Fix the actual extraction defect for form-class documents
like `CarOK_voorraadtelling`. The v2.13 weakest-query rationales
explicitly cite "table is slightly truncated", "odd spacing and
truncation", "odd whitespace" — these are real defects, not judge
miscalibration. Lenient judging would mask defects that downstream
generators will still hallucinate against.

**Why this is the right layer:** CarOK's tabular inventory data
needs to land in chunks as well-structured rows (markdown table or
key-value pairs), not as whitespace-collapsed prose. Once the
chunks are clean, the existing prose-calibrated Format judge
scores them correctly — no rubric carve-out needed.

**Method:**
1. Inspect a sample of current CarOK chunks. Confirm the truncation
   + whitespace damage at the chunk-emission step (not just at
   render time).
2. Route form-class documents (per existing
   `docs/QUALITY_GATES.md` §"Form / Invoice Acceptance Class"
   classifier) through a structured-table extraction path:
   - Primary attempt: Docling's table-structure-recognition
     (`do_table_structure=True` in `PdfPipelineOptions`). Emits
     each table as a Markdown table.
   - **Fallback escalation criteria (Draft v0.5 — quantitative):**
     Escalate from Docling TSR to VLM-assisted parse when ANY of:
     (a) Docling resolves fewer columns than the page's visible cell
     count suggests (heuristic: count vertical grid lines or text-
     baseline column-group clusters; ratio `cols_recovered / cols_visible
     < 0.7`), OR (b) `>20%` of resolved cells are empty / `None`
     when the source page has ink in those bounding boxes, OR
     (c) the Markdown table has zero rows after the header row
     (Docling produced a header-only stub). All three thresholds
     are deterministic + measurable from Docling's
     `TableData.num_rows` / `num_cols` / cell-content fields plus
     a PyMuPDF page-level ink/grid probe; no human judgment in the
     loop.
   - VLM-assisted fallback: prefer local Qwen3-VL-8B on the omlx
     server first; only escalate to cloud qwen-vl-max if the local
     VLM also fails one of the three thresholds above. (3a from
     v2.11 carry-forwards — finally justified by a concrete defect,
     not speculation.)
3. Re-extract `CarOK_voorraadtelling` and any other form-class
   doc in the corpus. Re-ingest into both production
   (`mmrag_v2_8__qwen3_local`) and the dashscope rollback collection
   so the apples-to-apples symmetry is maintained.
4. Sample a 30-query mini-soak on the re-extracted CarOK chunks
   to confirm the prose-Format judge no longer cites truncation /
   whitespace defects.

**Acceptance:**
- CarOK Format score recovers ≥85% on the EXISTING prose-Format
  rubric (no judge change required) on a 30-query mini-soak.
- No regression on the other 33 corpus docs' chunk counts or
  strict-gate pass status.
- Evidence + decision in `docs/DECISIONS.md`.

**Cost:** ~$0.50 mini-soak (qwen-max judge) + Docling CPU
re-extraction time. **If VLM fallback is needed** (Phase 1 fallback
escalation triggers): + $0 if local Qwen3-VL-8B suffices, + $2-5
extra if qwen-vl-max cloud is also required for CarOK-class
complexity. Worst case bound: **~$5.50 for Phase 1 with cloud-VLM
fallback path engaged.**

### Phase 2 — Targeted HyDE bridging for code + minority languages (REDEFINED in Draft v0.3)

**Goal:** Address the omlx embedder's per-doc regressions on
German (`ATZ_Elektronik` -12.5pp R@1) and code-dense (`Python_Cookbook`
-12.4pp, `IRJET` -12.5pp, `Hybrid_electric_vehicles` -12.6pp,
`Greenhouse_Design` -12.5pp) content **at query time**, NOT by
routing to multiple embedder collections.

**Why this is the right layer:** Per-doc embedder routing forces
permanent architectural fragmentation — parallel Qdrant collections,
language/content classifiers upstream of retrieval, broken
cross-corpus search for queries that span both classes. Targeted
HyDE bridging reuses the already-shipped Phase 4a local HyDE
infrastructure: when a query is detected as code-intent or
minority-language, generate a synthetic hypothetical answer in
the target style/language, embed THAT, and the omlx embedder's
semantic gap shrinks dramatically.

The 2-second HyDE latency cost only applies to the targeted
lanes (estimated 5-10% of production queries); the default
retrieval path stays at ~1.6s p99.

**Method:**
1. Add a lightweight query-intent classifier as a new module
   `src/mmrag_v2/retrieval/query_intent.py`. **Implementation
   specifics (Draft v0.5):**
   - **Code-intent detection** — deterministic regex over the query
     text: any of `\bdef\s+\w+\s*\(`, `\bclass\s+\w+`, `\bimport\s+\w+`,
     `[\w_]+\([^)]*\)\s*\{`, fenced \`\`\`code, `>>>` REPL prefix,
     `function\s+\w+\s*\(`. Threshold: ≥1 match → `intent=code`.
   - **Minority-language detection** — pure-stdlib heuristic for the
     two known v2.13 P1 cases (German, possibly Dutch later):
     count Unicode characters outside the ASCII printable range
     (`U+0080–U+024F` Latin Extended) plus German indicator words
     (`der|die|das|und|ist|nicht|für|über|hoeveel|kost`); if
     `(non_ascii_chars / total_chars) > 0.02` OR ≥2 indicator
     hits → `intent=minority_language`, with the detected language
     defaulting to `de` (German) for the v2.14 scope. **No new
     dependency** — no `langdetect` / `fasttext` install needed for
     v2.14; if the corpus grows beyond German + Dutch in v2.15,
     revisit and add `pycld3` or similar (single-doc native dep,
     ~3MB, well-maintained).
   - Classifier returns a small dataclass:
     `{"intent": "code" | "minority_language" | "default",
       "lang": str | None}`. Deterministic; covered by unit tests
     (no LLM calls).
2. When the classifier flags `intent=code` or
   `intent=minority_language`, the retrieval path automatically
   sets `use_hyde=True` + `hyde_provider="vllm"` (using the
   Phase 4a infrastructure) with a content-aware HyDE system
   prompt variant ("generate a code snippet answer" / "generate
   the answer in {detected_lang}").
3. Wire `hyde_provider` through `retrieve_hybrid_reranked` so
   the soak harness can A/B test the targeted-HyDE setting.
4. Validate on a focused mini-soak: same 5 affected docs (ATZ,
   Python_Cookbook, IRJET, Hybrid_electric_vehicles,
   Greenhouse_Design), 50 queries total, omlx-with-targeted-HyDE
   vs omlx-no-HyDE. Target: recover the -12.5pp R@1 deficit to
   within -3pp of dashscope on the same queries (so the
   aggregate stays a win without permanent infra).

**Acceptance:**
- Targeted HyDE on the 5 affected docs lifts R@1 by ≥8pp vs
  the no-HyDE baseline (recovering most of the gap without
  re-introducing cloud dependence).
- No regression on the 28 other corpus docs (sample 30 of them
  in the mini-soak to confirm; the default retrieval path
  remains HyDE-off so this is a non-issue, but worth a smoke).
- HyDE is OPT-IN per-query via the intent classifier; default
  production retrieval path unchanged.

**Cost:** ~$1-2 mini-soak.

**Why this defers permanent routing:** if targeted HyDE recovers
the deficit, no parallel embedder collection is needed. If it
doesn't, Phase 2 follow-up can revisit routing — but with
concrete evidence the cheaper bridge didn't suffice.

### Phase 3 — 30-day rollback drop (decision point: 2026-06-19)

**Goal:** if no v2.13.0 rollback fired during the 30-day window
(2026-05-22 → 2026-06-19), drop the dashscope rollback collection
and reclaim ~17 GB.

**Method (Draft v0.5 — snapshot before drop):**

1. **Snapshot the collections to cold storage first.** Qdrant has
   a native snapshot endpoint that produces a tar+segments archive
   without locking the cluster. Run before any DELETE:
   ```bash
   # Snapshot both rollback collections to local disk.
   curl -X POST http://localhost:6333/collections/mmrag_v2_8__qwen3_dashscope/snapshots
   curl -X POST http://localhost:6333/collections/mmrag_v2_8/snapshots
   # The snapshot files land in the Qdrant container's /qdrant/snapshots
   # volume (named volume mmrag_v2_8__qdrant_snapshots; verified live).
   # Compress + move to ~/qdrant_rollback_snapshots_2026-06-19/ on the
   # Mac mini's external drive; expected size ~12 GB compressed for
   # mmrag_v2_8__qwen3_dashscope, ~10 GB for mmrag_v2_8.
   ```
2. **Retain snapshots for 90 days** (through 2026-09-17). Note the
   retention end-date in `docs/DECISIONS.md` Phase 3 entry.
3. **Then drop:**
   ```bash
   curl -X DELETE http://localhost:6333/collections/mmrag_v2_8__qwen3_dashscope
   curl -X DELETE http://localhost:6333/collections/mmrag_v2_8
   ```
4. Update `scripts/retrieval_regression_v2_12.py` to skip with a
   "rollback collection deleted, see `docs/DECISIONS.md`" message
   rather than failing.

**Recovery procedure** (if a regression surfaces within the 90-day
retention window): restore from snapshot via
`POST /collections/{name}/snapshots/upload` → re-create the
collection from the snapshot, then re-deploy v2.12.0 from tag
`5a2ce18` for the dashscope retrieval path. No re-ingestion needed.

**Acceptance:** disk reclaim verified + snapshot file integrity
verified (size matches, can be re-uploaded to a scratch Qdrant
instance), + `docs/PROJECT_STATUS.md` "Qdrant collections" table
updated.

**Cost:** $0. Disk: snapshot archives ~22 GB total on external
storage; the net reclaim on the working drive is still ~17-20 GB.

**Skip condition:** if any production user reported a regression
during the window that suggested swapping back, defer this phase
to v2.15 and document the issue first.

### Phase 4 — Local-LLM exploration accelerator *(SCOPE TIGHTENED in Draft v0.3)*

**Goal:** wire the local LLM into the experimentation loop where
the Phase 0 calibration verdict says it's safe, with explicit
guardrails to prevent the **leniency trap** (a lenient judge biased
upward on relevance/faithfulness will reward configs that retrieve
more noise because it forgives noise; optimization against it →
noisier retrieval).

**Phase 0 verdict (2026-05-22):**
- Format: 90.2% exact → **TRUSTWORTHY**
- Relevance: 81.7% exact → **RESTRICTED** (systematic upward bias)
- Faithfulness: 76.1% exact → **RESTRICTED** (systematic upward bias)

**PERMITTED local-judge uses:**
1. **Format-axis judging on any soak** (the well-calibrated axis).
   `--judge-provider vllm` is allowed for Format scoring.
2. **Prompt A/B testing** — comparing two judge prompts or two
   query-generation prompts. The relative-ranking signal survives
   the leniency bias as long as both arms are judged by the same
   biased judge.
3. **Pre-filter / tie-breaker pattern** — local-judge runs first
   on the full fixture; queries with score deltas above a noise
   threshold (e.g. local rates 2 but a previous-cycle qwen-max
   judgment said 0) get re-judged by cloud `qwen-max` as the
   tie-breaker. Cuts cloud spend dramatically while preserving
   ground-truth integrity.
4. **Free large-fixture query generation** (`gen-provider vllm`)
   — generation isn't judging; leniency bias doesn't apply.

**FORBIDDEN local-judge uses** (the leniency trap):
- Tuning **RRF weights**, **top_k_retrieve**, **top_n_fuse**,
  or any retrieval-breadth parameter
- Tuning **sparse/dense weight balance** in hybrid retrieval
- Selecting **rerank backends** or **reranker hyperparameters**
- Cycle-close go/no-go judging on Relevance + Faithfulness axes
- Anything that adjusts how much / how broadly the retrieval
  stack pulls in candidates

The rule of thumb: if the change could legitimately retrieve more
or fewer chunks per query, the lenient judge will systematically
prefer "more" because it forgives the added noise. Use cloud
`qwen-max` for those decisions.

**GX10 deployment guardrails (Draft v0.4 — hard constraints):**

Any change to the GX10 vLLM endpoint (model swap, container bump,
hardware-tuning flags) MUST satisfy all five before deploying. A
near-miss on 2026-05-22 (proposed `Qwen2.5-72B-Instruct` swap with
`vllm/vllm-openai:latest`) would have violated 1, 2, 3, 4, and 5.

1. **Container architecture must be aarch64.** GX10 is a Grace‑Blackwell
   GB10 box; the host CPU is **ARM (aarch64)**. Docker Hub
   `vllm/vllm-openai:*` is `linux/amd64`-only and will silently fall
   through QEMU (single-digit tok/s) or fail to start. Use NVIDIA's
   NGC Grace‑Blackwell vLLM container or build vLLM from source on
   the box. Sanity-check before pull:
   `docker manifest inspect <image> | jq '.manifests[].platform'`
   — confirm `linux/arm64` is in the list.
2. **No port collision with the running 14B endpoint.** Phase 4a
   shipped with `http://10.0.10.239:8000` as the production HyDE
   backend running `Qwen/Qwen2.5-14B-Instruct`. Any new model goes
   on a different port unless the 14B is **explicitly retired**
   AND `src/mmrag_v2/retrieval/hyde.py` default URL/model are
   updated AND `MEMORY.md` is amended. Docker will refuse a
   second `-p 8000:8000` bind; the failure mode is "old container
   keeps running, new one silently exits."
3. **Memory sizing against unified memory.** GX10 has **128 GB
   LPDDR5X shared CPU+GPU** (not discrete VRAM). Weights + fp8 KV
   cache + activations + CUDA graphs must fit at the chosen
   `--gpu-memory-utilization`. Rough sizing for Qwen2.5: 14B-bf16
   ≈ 28 GB, 32B-fp8 ≈ 32 GB, 72B-fp8 ≈ 72 GB; fp8 KV scales
   linearly with `--max-model-len` (~10–18 GB at 32K). 72B-fp8 at
   32K with `--gpu-memory-utilization 0.8` has no headroom and
   will OOM on the first long prompt. Run a `max-model-len` load
   smoke before declaring success.
4. **Use pre-quantized FP8 checkpoints, not on-the-fly quantization.**
   `--quantization fp8` against a BF16 checkpoint runs runtime
   quantization at load: doubles transient memory and slows startup.
   Prefer `neuralmagic/<model>-FP8` or equivalent pre-quantized
   variants; drop the `--quantization fp8` flag in that case
   (vLLM auto-detects from the model config).
5. **CRITICAL — model swap invalidates Phase 0 calibration.** The
   Phase 0 verdict (Format 90.2% TRUSTWORTHY, Relevance 81.7%
   RESTRICTED, Faithfulness 76.1% RESTRICTED) is measured against
   `Qwen/Qwen2.5-14B-Instruct` specifically. The leniency profile
   is **model-specific**: a "stronger" model is not automatically
   more trustworthy. Any swap MUST re-run
   `scripts/calibrate_local_judge_vs_qwen_max.py` against the new
   model and produce a new `docs/CALIBRATION_<date>_v2.14_p0_local_judge_<model>.md`
   before any Phase 4b/4c/4d/4e usage is permitted. If the new
   calibration drops any axis below the RESTRICTED threshold,
   Phase 4's PERMITTED/FORBIDDEN list must be re-derived from
   the new numbers.

If a swap is contemplated, draft a one-paragraph deployment note
covering each of the five points (with the platform manifest
output pasted in for #1, the port plan for #2, the sizing arithmetic
for #3, the checkpoint URL for #4, and the calibration-rerun ETA
for #5) and pause for user sign-off before running `docker run`.

**GX10 endpoint resilience (Draft v0.5):**

The GX10 vLLM endpoint is now on the production retrieval path
(Phase 4a HyDE — opt-in via `use_hyde=True`, defaulting to dashscope
but flippable to vllm; Phase 2 will auto-enable on intent flags).
A GX10 outage must not break retrieval or the soak harness.
Mandatory behaviors:

1. **Pre-flight health probe.** `mmrag_v2.retrieval.hyde` adds a
   `_check_vllm_health()` helper that does a `GET /v1/models` with
   a 3s timeout when `provider="vllm"`. On failure, log a single
   warning and fall back to `provider="dashscope"` automatically.
   This makes the existing `generate_with_fallback()` semantics
   stronger: a dead endpoint degrades to cloud without a per-query
   timeout penalty (~3s once per process vs ~45s × retries per
   query).
2. **Health probe at soak start.** `synthetic_soak.py` calls the
   same helper at startup when `--judge-provider vllm` or
   `--gen-provider vllm` is set; aborts with a clear message
   ("GX10 at `--judge-url=…` not reachable; pass `--judge-provider
   dashscope` or restart the GX10 vLLM") rather than spamming
   per-query failures into the work file.
3. **Per-query circuit breaker.** If `>5` consecutive vLLM calls
   fail mid-soak (e.g. the GX10 OOMs partway through), the next
   call automatically downgrades to `dashscope` for the remainder
   of the soak run, with a single notice line. This protects long
   judging passes from getting wedged.
4. **Cost cap reminder.** When fallback fires, the soak harness
   logs the estimated cloud cost delta vs the planned $0 local path
   so the user sees the budget impact and can abort.

**Sub-deliverables:**
- 4a) Local HyDE provider — **SHIPPED 2026-05-22** (`src/mmrag_v2/retrieval/hyde.py`
  `provider="vllm"`)
- 4b) `scripts/synthetic_soak.py` gains `--judge-provider {dashscope,vllm}`
  + `--judge-url` / `--judge-model` flags. When `vllm`, the soak
  WARNS at startup that the run is restricted to PERMITTED uses
  per the list above. Default remains `dashscope` / `qwen-max`.
- 4c) `--gen-provider {dashscope,vllm}` for query generation (no
  leniency concern; safe by default once smoke-tested).
- 4d) Tie-breaker harness as a new script (`scripts/local_then_cloud_soak.py`)
  — local judges everything, cloud re-judges the high-delta
  subset. Targets ~80% cost reduction on big-fixture soaks.
- 4e) A 1500-query Format-only exploration soak demonstrates
  the new capability + produces denser Format-axis evidence
  (the only axis the local judge is fully trusted on).

**Acceptance:** 1500-query Format-only soak completes in <60 min
on local and matches the 518-query baseline within ±2pp on Format.

**Cost:** 4b/4c/4d wiring: $0 (code only). 4e demonstration soak:
$0 (Format-only on local).

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

### Phase 6 — Code-block chunking hygiene (NEW in Draft v0.3)

**Goal:** Fix the chunk-boundary policy so fenced code blocks
(```python … ```) and indented code regions are never severed
mid-block. The v2.13 weakest-query rationales explicitly cite
"contains truncated code" for `Fluent_Python` and the same
pattern likely affects `ArcGIS_Python_Cookbook`, `Python_Distilled`,
and `Ayeva_Python_Patterns`.

**Why this is the right layer:** A code snippet sliced at line 17
of 25 is useless for retrieval (the embedder doesn't know the
function signature ends three lines later) and worse for the
downstream generator (it sees an incomplete function and may
hallucinate the rest). The fix is in the chunker, not the
retriever or the judge.

**Method:**
1. Inspect the current chunking logic in
   `src/mmrag_v2/universal/element_processor.py` and
   `src/mmrag_v2/batch_processor.py`. Identify where chunk
   boundaries can fall.
2. Add a code-block-aware policy:
   - Detect fenced code (markdown ```), indented blocks (4+
     spaces leading 3+ consecutive lines), and Docling
     `CodeItem` elements.
   - If a chunk would otherwise close mid-block, extend the
     chunk to end-of-block, up to a safe maximum (e.g., +50%
     over the normal chunk-size limit). Beyond that, accept
     a code-split but mark the chunk with `chunk.metadata.partial_code=True`
     and emit a paired continuation chunk with the rest.
3. **Coexist with existing `tests/test_code_chunking.py`
   (Draft v0.5):** that file already pins 6 invariants
   (`test_code_not_treated_as_noise`,
   `test_long_body_text_not_treated_as_heading_noise`,
   `test_page_number_only_still_treated_as_noise`,
   `test_mixed_prose_and_code_chunking`,
   `test_long_code_splits_on_line_boundaries`,
   `test_english_from_not_misclassified_as_code`). The new
   end-of-block extension policy MUST NOT break any of them.
   Two specific risks: (a) `test_long_code_splits_on_line_boundaries`
   already permits splits — make sure the new "never sever
   mid-block" rule is *additive* (extends boundaries up to safe-max,
   then accepts a line-boundary split with `partial_code=True`,
   not "forbids splits"); (b) `test_mixed_prose_and_code_chunking`
   expects clean prose/code boundaries — the new policy should
   sharpen those, not blur them. Extend the existing test file
   with three new shape-specific cases (fenced, indented, Docling
   `CodeItem`) covering: (a) chunk fits cleanly, (b) chunk
   extends to end-of-block, (c) block exceeds the safe max and
   gets split with `partial_code=True` metadata. Run the full
   suite (`pytest tests/test_code_chunking.py -v`) after each
   policy iteration; require 9/9 PASS (6 existing + 3 new) before
   re-extracting any production doc.
4. Re-extract the four affected Python docs and re-ingest.
5. Mini-soak: 30 queries on the re-extracted Python docs,
   confirm "truncated code" rationales no longer appear in
   the weakest-15.

**Acceptance:**
- Zero "truncated code" rationales in the mini-soak weakest-15
  across the four re-extracted docs.
- No regression in chunk count >+15% on any of them (some
  growth expected from end-of-block extension; >15% suggests
  the safe-max is too generous).
- Existing chunk schema unchanged except for the optional new
  `metadata.partial_code` boolean.

**Cost:** ~$0.50 (mini-soak qwen-max judge cost) + chunker
re-extraction CPU time.

### Phase N — Cycle close-out

**Definition of Done (Draft v0.5 — minimal bar for v2.14.0 tag):**

A v2.14.0 tag MAY ship when ALL of the following are true:

- ✓ **At least Phase 1 AND Phase 6 land** (both fix concrete
  extraction defects from v2.13 P1 evidence — without these the
  cycle has nothing visible to ship).
- ✓ Phase 4b OR Phase 4c lands as a follow-on QoL win (any one of
  them is enough to demonstrate the local-LLM accelerator stack
  end-to-end).
- ✓ All shipped phases pass their own acceptance bars.
- ✓ Full pytest suite green; v2.13 fingerprint still passes (or a
  fresh v2.14 fingerprint captured if Phase 1/6 changed chunk_ids).
- ✓ Strict-gate corpus state unchanged (34/34 PASS) or improved.

Phase 2 (HyDE bridging) is **optional** for the initial v2.14.0
tag — it can ship in v2.14.1 if the targeted-HyDE recovery
acceptance (≥8pp R@1 lift) doesn't clear, without holding the
cycle. Phase 4d / 4e are QoL extras; Phase 3 is time-gated to
2026-06-19 and may land in v2.14.1 if the calibration cycle ran
quickly.

**Close-out activities:**

- Engine version bump `2.13.0` → `2.14.0`
- v2.14 retrieval-regression fingerprint (re-capture if any of
  Phase 1, Phase 2, or Phase 6 changed the production retrieval
  shape or chunk_ids on production docs; otherwise the v2.13
  fingerprint stays canonical)
- AFTER snapshot `docs/QUALITY_SNAPSHOT_<date>_v2.14_after.md`
- Layer-0/1 docs sweep
- **Archive Draft v0.1 archaeology section** (lines roughly 67-126
  of this plan; the "Original Draft v0.1" preserved-history block):
  move to `docs/archive/PLAN_V2.14_drafts.md` once the cycle closes.
  Keeps the active plan file lean for future v2.15 readers.
- **Calibration freshness check (Draft v0.5):** the Phase 0
  calibration verdict is tied to the specific GX10 model in use
  (per the GX10 deployment guardrails). If >30 days have elapsed
  since the last Phase 0 calibration run OR the GX10 model has
  changed since, re-run the calibration before any local-judge soak
  in Phase N (the AFTER soak in particular). Stale calibration
  >30 days → all axes downgrade to RESTRICTED until re-verified.
- v2.14.0 annotated tag staged for user push

---

## 3. v2.11+ carry-forwards (still open)

| Item | Source | v2.14 disposition |
|---|---|---|
| 3a NuMarkdown-8B / Qwen3-VL-8B local VLM | v2.11 plan | **Promoted to Phase 1 fallback** — VLM-assisted table parse is the Phase 1 fallback when Docling TSR doesn't resolve form-class table columns. v2.14 finally justifies this work with a concrete defect. |
| 3c UIR refactor (PAUSED) | v2.11 plan | Still PAUSED for user signoff |
| 3e Magazine rendered-region-crop | v2.11 plan | Defer with soak-data rationale (image-axis perf is OK without it) |

---

## 4. Phase ordering rationale (REVISED in Draft v0.3)

```
Phase 0 (calibration)        ← SHIPPED 2026-05-22
Phase 4a (local HyDE)        ← SHIPPED 2026-05-22 (prereq for Phase 2)
Phase 5 (disk precheck)      ← SHIPPED 2026-05-22

Phase 6 (code chunking)      ← independent; fix the root cause first
Phase 1 (form/table extract) ← independent; same shape as Phase 6 (fix-then-validate)
Phase 2 (HyDE bridging)      ← depends on Phase 4a (shipped); validates targeted-HyDE recovers per-doc deficits
Phase 4b/c/d/e (local LLM)   ← guardrails tightened; ships when Phase 1/2/6 land or are scoped
Phase 3 (rollback drop)      ← time-gated to 2026-06-19
Phase N (close-out)          ← terminal
```

Suggested execution order:

1. **Phase 6 (code chunking hygiene)** AND **Phase 1 (form/table
   extraction)** in **parallel** (Draft v0.5 — these are
   independent extraction-layer fixes touching different
   sub-systems: Phase 6 modifies `element_processor.py` chunk
   boundaries; Phase 1 modifies `pdf_plan.py` / `docling_adapter.py`
   table-recognition path. They can run as parallel branches and
   merge in either order). Both must land before Phase 2 so the
   per-doc regression measurements there are confounder-free.
2. **Phase 2 (targeted HyDE bridging)** — once Python docs are
   re-chunked (Phase 6) and form docs are re-extracted (Phase 1),
   the per-doc regressions can be re-measured cleanly and the
   targeted-HyDE bridge can be A/B tested without confounders.
3. **Phase 4b/4c/4d/4e (local LLM in soak)** — guardrails tightened;
   ship Format-only local judging + tie-breaker harness; gated
   permitted-uses list.
4. **Phase 3 (rollback drop)** — wait for 2026-06-19; snapshot
   first (per Draft v0.5 Phase 3 method update).
5. **Phase N (close-out)** — bump engine, retag, AFTER snapshot;
   archive Draft v0.1 archaeology block. See Phase N Definition
   of Done for the minimal-bar gate.

---

## 5. Budget (REVISED in Draft v0.5)

- **Cost cap**: $25/cycle (same as v2.13)
- **Estimated spend (base path):** $4-6 across Phase 1 mini-soak +
  Phase 2 mini-soak + Phase 6 mini-soak + Phase 4e Format-only demo
  (most of which is cloud-free on the local path); Phase 4b/4c/4d
  are $0 code work
- **Estimated spend (worst case):** **$7-12** if Phase 1 escalates
  to the cloud VLM fallback path (qwen-vl-max for CarOK-class
  complexity, +$2-5) AND a Phase 4 GX10-offline event forces
  partial cloud fallback during a soak (+$1).
- **Local LLM usage:** $0 (LAN)
- **Snapshot storage (Phase 3):** ~22 GB on external/cold storage
  for 90 days; net working-drive reclaim still ~17-20 GB.

## 6. Risks (REVISED in Draft v0.3)

| Risk | Mitigation |
|---|---|
| Phase 1 Docling TSR can't recover CarOK tables cleanly | VLM-assisted parse fallback (already on hand: Qwen3-VL-8B on omlx or qwen-vl-max cloud); 3a promoted |
| Phase 2 targeted HyDE doesn't recover the -12.5pp R@1 deficit | Revisit per-doc embedder routing as Phase 2-bis with concrete "HyDE didn't suffice" evidence (not speculative) |
| Phase 6 code-extension policy explodes chunk count | Acceptance bound: ≤+15% chunk growth per doc; safe-max forces a partial-code split beyond that with metadata flag |
| Local judge leniency trap masking real retrieval regression | Phase 4 guardrails: FORBIDDEN list of tuning uses; cloud `qwen-max` retained for all retrieval-breadth decisions and cycle-close go/no-go |
| GX10 endpoint swap breaks Phase 4a HyDE or invalidates Phase 0 calibration silently | Phase 4 "GX10 deployment guardrails" — 5-point hard checklist (aarch64 image, port plan, unified-memory sizing, pre-quantized FP8, calibration rerun) with mandatory user sign-off before `docker run` |
| **GX10 endpoint goes offline mid-cycle** (network hiccup, OOM, container restart) | Phase 4 "GX10 endpoint resilience" subsection (Draft v0.5) — pre-flight health probe in `hyde.py`; soak harness aborts at startup with clear message; per-query circuit breaker after 5 consecutive failures auto-downgrades to dashscope; cost-delta logged so the user can abort if cloud fallback exceeds the cap |
| Phase 0 calibration goes stale (model drift, time elapsed) | Phase N "Calibration freshness check" (Draft v0.5) — >30 days OR model change → axes downgrade to RESTRICTED until re-run of `scripts/calibrate_local_judge_vs_qwen_max.py` |
| Phase 3 drop and a regression surfaces afterward | Phase 3 "snapshot before drop" (Draft v0.5) — Qdrant native snapshot + 90-day cold-storage retention + documented recovery procedure |
| 30-day rollback window had a real rollback need we didn't catch | Phase 3 skip rule + grace period; collection retained 30 days; snapshot retained 90 days |
| Disk fills again during a soak | Phase 5 precheck (SHIPPED) prevents recurrence |

## 7. Open questions for the user (when back)

- (none required to start) — Phase 6, 1, 2, 4b–e are autonomously
  executable. Phase 3 needs your "no rollback fired, drop it" signoff
  on or after 2026-06-19.
- Optional: SSH key onboarding for the GX10 (still pending — only
  blocker for live diagnostics if the local LLM crashes mid-soak).
- Heads-up: Phase 1 may need the user to OK the choice between
  "Docling TSR alone" vs "Docling TSR + VLM fallback" if Docling's
  default doesn't resolve CarOK cleanly. I'll proceed with TSR
  first and escalate with evidence if it falls short.
