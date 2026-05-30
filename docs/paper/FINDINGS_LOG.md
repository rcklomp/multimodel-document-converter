# MM-RAG V3 — Findings Log (paper source material)

**Purpose.** Append-only, dated capture of paper-grade material: motivation,
method, **verbatim measured data**, conclusions, and **rejected approaches with
reasoning**. This is the raw material for an eventual systems paper / background
article. It complements — does not duplicate — the terse recall memory files,
the churning `PROJECT_STATUS.md`, the point-in-time `docs/V3_OVERNIGHT_REPORT.md`,
and the locked `docs/DECISIONS.md`.

**Convention.** At the end of a substantive session/milestone, append a dated
`##` entry. Preserve raw numbers in tables. Record dead-ends, not just wins.
Tag each entry with the paper section(s) it feeds: `[Motivation] [Architecture]
[Method] [Results] [Lessons]`.

**Working paper title (placeholder):** *"Building a Vision-Native Multimodal RAG
ETL on Heterogeneous Local Hardware: throughput is a memory-bandwidth problem."*

---

## 2026-05-30 — Local VLM extraction throughput: the binding constraint is memory bandwidth  `[Method][Results][Lessons]`

**Context.** V3 is vision-native (a VLM transcribes rendered page images instead
of the Docling DOM cascade). The production blocker is *extraction throughput* on
local hardware. We investigated empirically rather than by spec-sheet.

**F1 — Extraction is decode-bound with a fixed vision-prefill floor.**
Stream-profiled one real page (Form_0013 invoice, rendered 150 DPI = 1236×1752)
through the GX10 (NVIDIA GB10, BF16 Qwen3-VL-8B, vLLM):

| Metric | Value |
|---|---|
| prompt tokens (incl. ~2100 vision tokens) | 2178 |
| completion tokens | 1062 |
| TTFT / prefill | 38.3 s (~57 tok/s) |
| decode | 94.4 s → **11.3 tok/s** |
| total | 132.6 s/page |
| split | **71% decode-bound**, 29% prefill |

**F2 — Decode is memory-bandwidth-bound (not compute-bound).** GB10 unified
LPDDR5X ≈ 273 GB/s. BF16 8B = ~16 GB weights → ~17 tok/s theoretical decode
ceiling (one full weight read per token). Observed 11.3 ≈ 66% of ceiling. This
is the key reframing: the bottleneck is bytes-moved-per-token, not FLOPs.

**F3 — GB10 unified memory causes vLLM OOM crashes (and the fix).** 121 GB is
*unified* CPU+GPU (no discrete VRAM). vLLM `--gpu-memory-utilization 0.85`
walls off ~100 GB of the *shared* pool; the OS + the VLM's transient image
activations then collide with the physical ceiling → repeated
`NVRM: Out of memory [NV_ERR_NO_MEMORY] @ mem_desc.c:1359`, killing the engine
on the *first image request* (model loads fine, crashes on inference). Fix:
`--gpu-memory-utilization ≤ 0.55` (VLM) / 0.60 (text). Empirically validated:
the page that crashed at 0.85 transcribed cleanly at 0.55, 0 NVRM events.

**F4 — REJECTED: vLLM online FP8 corrupts VLM vision.** `--quantization fp8`
gave a real 1.73× decode speedup (11.3→19.6 tok/s) — but on **garbage output**:
the model hallucinated blank pages, describing the Form_0013 invoice as *"a
gear-like pattern, no text."* Naive online quantization wrecks the vision tower.
**Lesson: validate content fidelity, not just speed — a speed-only check would
have shipped a pipeline that silently drops every page.**

**F5 — NVFP4 reconsidered.** vLLM supports `modelopt_fp4`, but no Qwen3-VL-8B
NVFP4 checkpoint exists (only a 235B MoE) → would require self-quantizing via
NVIDIA modelopt with the vision tower excluded, plus quality re-validation.
Expected gain: ~3-4× *decode* but only **~2× overall**, because the ~38 s
vision-prefill floor (F1) doesn't shrink with weight quantization. Marginal ROI
vs effort+risk → deprioritized.

**F6 — Cross-hardware comparison.** Each local box fails differently:
- **Cloud OpenRouter Qwen3-VL:** fast, but hard weekly budget ceiling (403s).
- **Mac Mini (MLX 4-bit):** quality *good* (read Form_0013 table correctly,
  totals reconciled), but RAM-starved → tiered-cache/swap thrash death on dense
  magazine pages (16.9k-token prompts).
- **GX10 / GB10 (BF16):** stable (post-fix), but bandwidth-starved → 11.3 tok/s.

**F7 — The bandwidth thesis, anchored to a real number.** omlx.ai public bench,
Qwen3-VL-8B-Instruct, 8-bit, 32k ctx: **M1 Max (24c, 2021): PP 162.2, TG 15.1
tok/s.** A four-year-old laptop chip beats the 2025 GB10 on *both* axes (GX10:
PP ~57, TG 11.3) at higher precision and longer context — because Apple's
unified memory is bandwidth-rich (M1 Max ~400 GB/s vs GB10 ~273) and its GPU
prefills faster. Extrapolated M5 Max (≥546 GB/s, newer GPU + neural
accelerators, 128 GB): est. **~35–50 s/page (~2.6–3.8× the GX10)**, and the
prefill floor largely dissolves (PP ~250–450 → ~5–9 s). *Unverified — to be
measured with the same probe.*

**Architectural conclusion.** The binding constraint is **memory bandwidth, not
compute**. Optimal division of labor by hardware strength:
**M5 Max → extraction** (bandwidth, MLX 4-/8-bit, quality-verified) ·
**GX10 → judge/soak** (text LLM at FP8, stable) ·
**Mac Mini → embedder + reranker**.

**Meta-lessons (paper "Lessons" section).**
1. Identify the binding constraint *before* optimizing — we chased compute
   (quantization/NVFP4) on a bandwidth-bound workload.
2. Measure, don't extrapolate from spec sheets (a 2021 laptop beat the AI server).
3. For generative quality work, verify *content fidelity*, not just throughput
   (the FP8 trap).
4. Unified-memory accelerators (GB10, Apple Silicon) break the discrete-GPU
   assumptions baked into tools like vLLM (the `gpu-memory-utilization` over-commit).

**Infra hardened this session:** GX10 fully patched + rebooted (kernel 1021,
driver 580, CUDA 13.3); `scripts/gx10/docker-compose.yml` + `scripts/v3_sequential_soak.sh`
(supervised, leak-proof, unified-memory-safe serving) committed `4d33ca1`.

---

## Predecessor lineage: V1 → V2 (the story before V3)  `[Motivation][Architecture]`

Reconstructed 2026-05-30 from git tags/commit messages (all verifiable), README
lineage, memory cards, and the (quarantined) `docs/.archive/`. Repo spans
**332 commits, 2026-01-05 → 2026-05-30** (~5 months).

**V1 / origin (Jan 2026) — *recoverable*.** The repo was seeded `v1.0.0`
(2026-01-05) as a snapshot of an earlier internal line versioned "v18.x":
`v18.1` (01-12) *"Full transition to multimodal async pipeline + English docs"*;
`v18.1.1` (01-13) *"Resolve Bbox-Paradox and implement Full Governance (Cluster
A+B)."* A formal **SRS existed** (`SRS_Multimodal_Ingestion_V2.3.md`, `…V2.4.md`)
plus an early design corpus (`ARCHITECTURE_V2_ANTI_OVERFITTING.md`,
`DECISION_OCR_CASCADE.md`, `RETRIEVAL_OPTIMIZATION_SUMMARY.md`,
`SCANNED_DOCUMENT_IMPROVEMENT_PLAN.md`, a Dutch `PVA` project plan,
`GEMINI_AUDIT_RESPONSE.md`). All retrievable via `git show <commit>:<path>` even
where deleted from the tree. So V1 = the multimodal-async-pipeline genesis with
formal requirements + a "Bbox-Paradox" + governance clusters.

**V2.4.1 "Integrity Milestone" (01-18, `d635341`).** Complete architecture
overhaul + metadata-schema alignment + legacy cleanup — the start of the V2 line
proper. Centralized versioning (SSOT), OCR-bypass for native-digital PDFs,
60-char gap-fill recovery ("The Infiltrator").

**V2.4.2 → 2.9 (extraction plumbing).** `DocumentClassifier` (later **rejected**
in favor of `ProfileClassifier` — a recorded dead-end), structural diagnostic
router (flat-code rescue + encoding-corruption detection, v2.5), Docling
2.66→**2.86** upgrade + shared `PdfConversionPlan`/adapter (single Docling
construction site, v2.7), seven named root-cause classes (v2.9-rc1).

**V2.10 "Chunker baseline" — the extraction-quality certification.** 34-doc
canonical corpus all strict-gate PASS (16 PASS + 18 PASS_WITH_ADVISORIES, no
threshold weakened); 975 tests; synthetic-soak **Format 98.3%**; Qdrant 30,454
pts. The "chunks are well-formed" milestone.

**V2.11 → 2.14 (retrieval maturation).** v2.11 embedder swap (~**10× lift**);
v2.12 hybrid + ModernBERT rerank (**+32 pp Recall@1**); v2.13 local omlx
embedder (Qwen3-Embedding-8B) + OCR auto-routing; v2.14 local-LLM accelerator
stack (HyDE, GX10 Qwen2.5-14B-FP8 judge, qwen-max fallback).

**V2.15 → 2.16 (convergence).** dedup, personal_importance overlay, partial_code
adjacency, VLM-table IoU dedup. **`v2.16.0` = FINAL v2.X, feature-complete**;
corpus 34→38; production stack = omlx Qwen3-Embedding-8B + BM25 + RRF +
ModernBERT rerank (~34k pts). Notably: **ColPali/VisRAG explicitly declared
"v3.0 OUT-OF-SCOPE"** here — the visual-retrieval idea was parked, then
superseded by the V3 *VLM-native extraction* bet.

**The V2→V3 hinge (the paper's turning point).** V2.16 was feature-complete on
*Docling-based* extraction — but the Identity Gate later showed Docling silently
dropped ~80% of spreadsheet rows (CarOK) and collapsed tables/forms. That
extraction-fidelity ceiling, not a retrieval problem, motivated the V3
**vision-native re-charter**. → continues in the V3 entries above.

**Deeper detail still on disk:** `docs/.archive/` (70 files). Mined 2026-05-30
under one-time documentation authorization — distilled into the four subsections
below (the raw files remain the citation source).

### V2 retrieval-quality trajectory (the data)  `[Results]`

> Caveat: soak *fixtures/samples changed across cycles*, so cross-version
> absolute numbers drift. The trustworthy signals are the **within-cycle,
> same-fixture A/B deltas** (bolded). Format gate floor was ≥96%.

| Version | R@1 | R@5(chunk) | R@5(doc) | Relevance | Format | Faithfulness | Controlled delta |
|---|--:|--:|--:|--:|--:|--:|---|
| v2.10 (llava embedder) | 2.1% | 6.8% | 54.2% | 5.9% | 98.3% | 4.7% | chunker baseline; extraction 34/34 strict-gate PASS |
| v2.11 (→ Dashscope embed) | 35.5% | 66.8% | 91.7% | 59.3% | 89.8% | 50.6% | **embedder swap ≈ 16.9× R@1 (+33.4pp), same fixture** |
| v2.12 (+ rerank + hybrid) | 67.8% | 90.2% | 98.6% | 82.1% | 88.4% | 72.6% | **rerank+BM25/RRF +32.3pp R@1, +23.4pp R@5c** |
| v2.13 (→ omlx local embed) | 57.5%* | 78.0%* | 95.2%* | 74.6% | 92.9% | 66.9% | **omlx vs Dashscope same-fixture: 6/6 axes, +2.5pp R@1, +3.7pp Format; latency 2.05→1.05s** |
| v2.14–v2.16 | (retrieval unchanged from v2.13) | | | | | | infra/telemetry/corpus only; pipeline.py unmodified |

\*different soak sample than v2.12. **Two findings dominate the V2 story:** (1) the
embedder, not the chunker, was the bottleneck (≈17× lift); (2) "right doc, wrong
chunk" (R@5-doc 91.7% vs R@5-chunk 66.8% at v2.11) was closed by reranker+hybrid,
not by per-doc-class chunking. HyDE measured at **+0.5pp** → shipped opt-in, later killed.

### V2 dead-ends & falsified experiments  `[Lessons]`

The paper's "what didn't work" section — every one is recorded with data:
- **HyDE bridging (intent-classified): FALSIFIED.** +0.4pp aggregate; **−20pp** on
  the targeted minority-language subset. Cause: query-time intent doesn't map to
  doc content type, *and* HyDE adds hallucination noise that hurts an already-strong
  multilingual embedder (negative synergy). Shipped opt-in, never promoted.
- **Query-rewriting for the omlx −12pp deficit: KILLED.** The deficit was a uniform
  −12.4…−12.6pp across 5 *heterogeneous* docs (German, code, magazine) → multi-factor,
  not single-lever-solvable. A/B validation was also blocked by losing the Dashscope
  baseline collection mid-cycle (infra-loss > hypothesis tests).
- **Dynamic top-k pre-flight: KILLED** (0 PASS baseline → retention undefined).
- **omlx −12pp deficit: ACCEPTED as a known limitation.** Declared the point where
  *pure-text retrieval levers were exhausted* → motivated the visual-retrieval bet.
- **Extraction defects (the V3 motivation):** image-only-page drops (~700 chunks,
  no guaranteed image→chunk path), TOC quarantine over-fire (~62), cross-page-split
  *page-attribution* bug (content present but stamped on the wrong page), OCR-lane
  heading-propagation gaps (Firearms 72%→99.7% after fix). Docling's 1-D text view
  of 2-D pages is the through-line.

### Judge calibration (v2.14)  `[Method]`

Selecting a local LLM judge for $0 exploration soaks; agreement vs the `qwen-max`
ship-gate reference (n≈518). Rule: ≥85% = TRUSTWORTHY, 70–85% = RESTRICTED (HyDE-only),
<70% = unusable.

| Judge | Relevance | Format | Faithfulness | Verdict |
|---|--:|--:|--:|---|
| **Qwen2.5-14B-Instruct (FP8) — chosen** | 81.7% | **90.2%** | 76.1% | format TRUSTWORTHY |
| Qwen3.6-27B-FP8 + MTP | 82.0% | 70.7% | 78.8% | all RESTRICTED |
| Llama-3.1-70B (OpenRouter) | 79.9% | 87.5% | 75.1% | format TRUSTWORTHY (slow) |
| Qwen3-32B (OpenRouter) | 78.9% | 84.1% | 77.3% | RESTRICTED (40% queries failed, thinking-mode) |

Bigger ≠ better (27B-MTP lost 19.5pp Format to the 14B). Cloud `qwen-max` stayed
the ship-gate judge (leniency-trap rule); local judge gates exploration only.

### V3 design evolution & the ColPali → VLM-native pivot  `[Architecture]`

The V3 charter went through **four drafts** (`ARCHITECTURE_V3_DRAFT_0.1–0.4`,
→ live `0.5`), each tightened by an external review:
- **0.1** diagnosed three v2.X limits: spatial→text deficit (−12pp), heuristic-patch
  ceiling (~88–93% Format), chunker-state fragmentation. Proposed a 4-layer stack
  (UIR · LLM-sanitization · ColPali visual retrieval · modality-aware gates).
- **0.2** hardened the UIR contract; shifted the Phase-A gate from byte-identical to
  **semantic-identity**; mandated a ColPali pre-flight spike before any Phase-A code.
- **0.3/0.4** corrected the root-cause diagnosis to actual repo state, split the
  acceptance gate (identity-half ≥95% + explained-delta-half ≤5%), rebudgeted Phase A
  to 24 days, added third-party regression checks.

**The pivotal spike — ColPali visual retrieval FAILED at page granularity.** The
C-spike (20 queries, ATZ_Elektronik) gave: visual top-1 55% (= text), **visual
recovery of text-failures only 44%** (gate ≥60%), **visual harm 36%** (gate ≤10%),
rerank-on-gold-page **47%** (gate ≥60%). Root cause: *page-level* granularity —
the visually-richest page over-pulls on body-text queries, and the reranker can't
discriminate chunks within a page. Verdict: page-level ColPali insufficient →
**redirect to VLM-native parsing.** That redirect *is* current V3: instead of
embedding page images for retrieval, a VLM transcribes them at extraction time.

**Phase A (UIR refactor) spikes PASSED:** A0 mapped ATZ 63→63 chunks at 100%
identity, 0 deltas; the A2 shim projected **3,890 v2.X chunks across 4 docs
losslessly (100% identity, 0 errors)** — proving the UIR contract carries v2.X
content without loss, which de-risked the refactor that this session closed.

**The full arc (paper spine):** V1 multimodal pipeline → V2 Docling extraction +
retrieval maturation (embedder 17× → rerank/hybrid +32pp) → **text levers
exhausted** (HyDE/query-rewrite/omlx-deficit) → bet on **visual retrieval (ColPali)**
→ **ColPali C-spike fails at page granularity** → pivot to **VLM-native extraction**
(current V3) → **throughput wall** (2026-05-30 entry: bandwidth-bound; M5 Max thesis).

---

## Backfill backlog (remaining threads — to expand when drafting)

*Covered above as of 2026-05-30:* V1→V2 lineage · V2 metrics trajectory · V2
dead-ends · judge calibration · V3 draft evolution + ColPali pivot + Phase A
spikes. Still to do:

- **V3 vision-native vs V2.16 Docling — retrieval quality result.** Head-to-head
  synthetic soak (11 canonical docs in both, GX10 Qwen2.5-14B-FP8 judge):
  Recall@1 +22.8 pp, **Recall@5(chunk) +31.9 pp**, **Faithfulness +22.7 pp**,
  Relevance +11.4 pp. The core "why V3 works" result — pull the full per-doc
  tables from `docs/V3_OVERNIGHT_REPORT.md`.
- **Identity Gate finding** — V2.16 Docling silently dropped ~80% of spreadsheet
  rows (CarOK); V3 VLM-native rebaselined to full extraction. Structural Form_0013
  comparison (logo description vs placeholder; table as row-level chunks).
- **V1 primary sources** — extract `SRS_Multimodal_Ingestion_V2.3/V2.4.md` and the
  early design docs from git history (`git show`) into `docs/paper/archive_extracts/`
  before they're harder to retrieve.
- **M5 Max measurement** — fills/corrects the F7 extrapolation once unboxed.
- **(legacy backlog note)** Judge calibration — GX10 Qwen2.5-14B-FP8: format TRUSTWORTHY,
  rel/faith RESTRICTED; treat as directional. Source: memory `feedback_v2_14_gx10_14b_fp8_swap`.
