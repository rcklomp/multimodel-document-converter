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

## 2026-05-29 — V3 vs V2.16: the core retrieval result (why V3 works)  `[Results]`

Head-to-head synthetic soak, the **11 canonical docs sampled in both** runs, same
embedder/reranker/seed; judge = GX10 Qwen2.5-14B-FP8. (Per-doc full tables in the
committed `V3_OVERNIGHT_REPORT.md`.)

| Axis | V3 | V2.16 | Δ |
|---|--:|--:|--:|
| Recall@1 (chunk) | **77.3%** | 54.5% | **+22.8 pp** |
| Recall@5 (chunk) | **95.5%** | 63.6% | **+31.9 pp** |
| Recall@5 (doc) | **100.0%** | 86.4% | +13.6 pp |
| Relevance | **84.1%** | 72.7% | +11.4 pp |
| Format (judge-TRUSTWORTHY axis) | **97.7%** | 95.5% | +2.2 pp |
| Faithfulness | **84.1%** | 61.4% | **+22.7 pp** |

V3 wins every axis. Headline: **R@5(chunk) +31.9 pp** and **Faithfulness +22.7 pp**
— V3 chunks are both easier to retrieve *and* more answer-providing. Decisive
per-doc wins where V2.16's Docling extraction collapsed: Combat_Aircraft (magazine),
PCWorld, Hybrid_electric_vehicles, HarryPotter (prose), and Kimothi_RAG_Guide
(V2.16 missed the doc entirely). This is the result that justified the VLM-native bet.

**Honest caveats (must accompany the numbers in the paper):** (1) the GX10
14B-FP8 judge is calibration-TRUSTWORTHY only on the **Format** axis — read
relevance/faithfulness as *directional*; (2) narrow sample (V3 covered 12 docs /
24 queries vs V2.16's 36 / 72, because the budget-limited extraction run couldn't
cover the long tail); (3) V3 chunk counts differ widely from V2.16 (±50% on some
docs) — "fewer chunks" was often *denser/better* (forms), not worse. The result is
strong and directionally robust, but not a large-N significance claim.

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

## 2026-05-31 — Inference-server choice for VLM extraction: cache architecture must match the workload  `[Method][Lessons]`

Survey prompted by the Mac Mini's oMLX instability on large-context (≈17k-token)
magazine pages: throughput collapsed 57→110→212→331 s/page with "SSD cache write
queue full / Failed to save block to tiered cache".

**Finding: it was an architectural mismatch, not a bug.** Every MLX VLM server
sits on the *same* upstream kernels — Blaizzy's `mlx-vlm` (oMLX serves "other
mlx-vlm models"; LM Studio uses "mlx-vlm vision add-ons"). What differs is the
serving layer. **oMLX's headline feature — a two-tier hot-RAM/cold-SSD KV cache —
is tuned for coding agents:** long, growing, *reused* prefixes, where restoring
cached blocks turns 30–90 s TTFT into 1–3 s. Our extraction workload is the
inverse: single-shot pages, one large image, **no shared prefix between pages**.
The KV cache never hits → the SSD tier delivers zero benefit and, under RAM
pressure, becomes an active liability (the write-queue thrash). VLM is bolted onto
a text-first stack; the cache that *would* help — encoded vision-feature reuse —
is mlx-vlm's, not oMLX's.

**Decision:** for the extraction node, **mlx-vlm's own server is primary**
(VLM-native upstream, Qwen3-VL lands there first, vision-feature cache fits the
workload, continuous batching, no SSD-tier failure mode); **oMLX kept as A-B
comparand** (128 GB on the M5 may neutralise the thrash — a hypothesis to probe,
not a default). **LM Studio** is the better long-term *text-judge* host (mature,
headless `llmster`, Apple-tuned for M5) but has **not** migrated Qwen3-VL to its
unified vision engine (only Gemma 3 + Pixtral) — not a vision option today.
Marketing-heavy "vllm-mlx/Rapid-MLX" repos (duplicate clones under different
usernames, "400+ tok/s") down-weighted on credibility.

**Method lesson (generalises):** pick a serving stack by whether its *caching
strategy matches the access pattern*, not by headline tok/s. A coding-agent
server and a batch-extraction server want opposite caches. And — per the FP8
fraud — **let the probe arbitrate on real hardware**: run the same
`vlm_profile_probe.py` against both servers on the page that actually thrashed,
score on s/page **and** fidelity. Runbook updated: `scripts/m5_max_setup.md` §3/§5.

## 2026-05-31 — M5 Max measured: the F7 bandwidth extrapolation confirmed, and beaten  `[Results][Method][Lessons]`

**Context.** F7 (2026-05-30 entry) anchored the bandwidth thesis to the omlx.ai
M1 Max bench and *extrapolated* the unboxed M5 Max at **~35–50 s/page** (PP
~250–450, decode ~21–26 8-bit / ~30–40 4-bit), explicitly flagged *"Unverified —
to be measured with the same probe."* This entry is that measurement. **The
extrapolation held directionally and the M5 beat it by ~2×.**

**Setup.** Apple M5 Max, 128 GB, macOS 26.5. Primary server = `mlx-vlm 0.5.0`
(mlx 0.31.2 / mlx-metal 0.31.2), Qwen3-VL-8B served on `:8000`, per the
2026-05-31 server-choice decision. Same `vlm_profile_probe.py`, same 150 DPI
render, same Form_0013 invoice as the GX10 F1 baseline — apples-to-apples.

| Probe | prompt tok (incl. vision) | completion tok | prefill / TTFT | decode | s/page | fidelity |
|---|--:|--:|--:|--:|--:|---|
| **8-bit** — Form_0013 invoice p0 | 2177 | 843 | 4.1 s (~**536** tok/s) | **58.9** tok/s | **18.4** | **4/4 OK** ✅ |
| **4-bit** — IRJET academic p3 | 2177 | 1738 | 2.9 s (~752 tok/s) | **92.6** tok/s | 21.7 | not checked (no `--expect`) |

**F8 — The M5 Max makes local VLM extraction decisively viable.** 8-bit decode
**58.9 tok/s** is **5.2× the GX10's 11.3** (F1) and **3.9× the M1 Max's 15.1**
(F7 anchor). 18.4 s/page on the invoice (843-tok output) vs the GX10's 132.6 s
on the *same page* (1062-tok output) ≈ **7× the throughput**. Fidelity holds at
8-bit — all four expected substrings present, full German invoice transcribed
(line items, `1.949,60` total, 19% VAT reconciled). The extrapolation
*understated* the chip: measured prefill 536 tok/s > predicted 250–450; measured
18.4 s/page < predicted 35–50.

**F9 — Decode efficiency confirms the bandwidth model.** Against the F7-estimated
M5 bandwidth (≥546 GB/s): 8-bit 8B ≈ 8 GB weights → ~68 tok/s decode ceiling;
observed 58.9 ≈ **87% of ceiling** (vs the GB10's 66% in F2). 4-bit ≈ 4 GB →
~137 tok/s ceiling; observed 92.6 ≈ 68%. Both sit where a bandwidth-bound,
one-weight-read-per-token model predicts — the F2 thesis (bytes-moved, not FLOPs)
reproduces on a third, faster unified-memory architecture. *(Bandwidth figure is
the F7 estimate, not independently measured — read the ratios as directional.)*

**Corpus extrapolation.** Decode rate is the stable cost driver; s/page scales
with output length. At ~1,200 output tok/page typical, 8-bit ≈ `4 + 1200/58.9`
≈ **~24 s/page** → ~720 VLM-routed pages ≈ **~4.8 h** — a comfortable overnight
job, and *under* the runbook's earlier 7–10 h guess. **8-bit is the default:**
fidelity-safe (the FP8 trap does not recur at MLX 8-bit) and fast enough that
4-bit's speed edge isn't needed for the corpus.

**Still open (honest scope).** (1) The decisive *stability* test per §5 — the
large-context (~17k-token) **magazine page that thrashed the Mac Mini's oMLX** —
has **not** yet been run on the M5; the invoice is the throughput/fidelity test,
not the thrash test. (2) The **oMLX A-B comparand** (does 128 GB neutralise the
SSD-tier thrash?) is unrun. (3) 4-bit fidelity was not substring-checked. (4)
Env caveat: installing `mlx-vlm` into `mmrag-v2` pulled `numpy 2.2.6`, violating
the project's `numpy<2.0.0` pin — harmless for serving + the fitz-only probe, but
the docling conversion pipeline would need numpy pinned back in this env.

**Lesson (paper).** Measure-don't-extrapolate (F7→F8) cuts *both* ways: the spec
sheet was *pessimistic* here. The 2021-laptop-beats-2025-server surprise (F7) and
the M5-beats-its-own-extrapolation surprise (F8) are the same lesson — unified
memory bandwidth is the axis that matters and it is not yet priced into either
intuition or vendor positioning.

---

## 2026-05-31 / 06-01 - M5 production smoke exposed the sandbox/shipping split; PLAN_V3.1 reconverged it  `[Results][Method][Lessons]`

**Context.** The first end-to-end smoke of the V3 *production CLI* (`mmrag-v2
process`) against the self-hosted M5 VLM endpoint (not OpenRouter). Until now every
V3 quality number came from `scripts/v3_batch_ingest.py`, which used the retired
`v3_execution_root` sandbox chunker + a permissive schema. The smoke ran the path
that actually ships - and it immediately broke, three different ways. That kicked
off a five-phase reconvergence (PLAN_V3.1), all merged to a clean linear `main`.

**F10 - The headline result was measured on a path that does not ship.** The
V3_OVERNIGHT "V3 beats V2.16 on every axis" numbers (Recall@1 +22.8pp, Faith
+22.7pp) came from the sandbox chunker. The production CLI emits *different*
(fewer, denser) chunks. The result may still hold, but it was not reproducible
through the shipping path. **Lesson (paper): an evidence path that diverges from
the shipping path is a latent integrity failure - the prettier the numbers, the
more dangerous, because nobody re-checks a win.**

**F11 - Three production bugs were invisible because the sandbox bypassed the
gates.** First real M5 CLI run surfaced, in order: (a) the V3 provider hardcoded
`response_format=json_object`, which OpenRouter accepts but self-hosted mlx-vlm
**400s** - so the path only ever worked on the cloud endpoint; (b) vision-native
IMAGE/TABLE chunks reached `from_uir` with no `asset_ref` -> QA-CHECK-05 failed the
whole batch to 0 chunks (the sandbox wrote a permissive schema that never checked);
(c) image chunks had empty `visual_description` (VLM text went only to `content`).
All three were one-commit fixes - but they had been masked for the entire V3 cycle.

**F12 - REJECTED (the crucible's centrepiece): spatial boundary-repair merging is
an anti-pattern on VLM-native extraction.** A legacy heuristic
(`_apply_final_boundary_repairs`: hungry-operator / trailing-heading / mid-sentence
merges), built for OCR/Docling physically splitting sentences across bounding
boxes, was found orphaned by Phase A and re-wired back in - then falsified by a
*deterministic* A/B. Method mattered: a random-sample judge soak would have buried
the ~3-merges-per-doc signal under VLM run-to-run variance (two live extractions of
IRJET were only **84% identical** chunk-for-chunk). Instead, applying repairs
in-process to ONE fixed extraction (same input, repairs on vs off) gave a clean
delta: IRJET 104 -> 97 chunks. An omlx targeted retrieval probe on the 7 merge
boundaries: only ~1 was a clean split-sentence rejoin; the rest **over-merged**
distinct concepts (equations, references, conclusion+bibliography) into oversized
blobs, and the focused fragment **out-retrieved** the merged chunk on 2 of 4 probed
queries (M2 equation cos 0.786 vs 0.766; M6 reference 0.736 vs 0.716), the rest
neutral. **Lesson (paper): geometric merging overrides the VLM's semantic chunk
boundaries. The VLM already reads where a paragraph ends and an equation begins;
layering a spatial merger on top lets dumb geometry override smart semantics. The
Docling-era reading-order/boundary heuristics do not transfer to a VLM substrate -
they are not just unnecessary, they are net-negative for retrieval.** This
generalizes F-series: V3 isn't "V2 + a better extractor," it requires *removing*
the geometric scaffolding V2 needed.

**F13 - Untested invariants drift; the fix is to make them executable.**
`AGENT-SPATIAL-20` (the single 20-unit vertical merge threshold) was a hard
invariant in prose only - a bare magic literal `if 0 <= v_gap <= 20` at 6+ live
call sites, zero tests. Converted to a mutation-verified guard: flipping 20->25
failed 2 assertions; restoring greened all 7. **Lesson: a prose invariant guarding
live code is a liability, not a contract; only an executable test that demonstrably
fails on drift counts.**

**Governance dead-letter finding.** The post-V3 control docs had split into
load-bearing rules (honored) and dead-letter rules (unachievable/violated, so
ignored): a Definition-of-Done citing an Identity-Gate script that was never built,
"all heuristics permanently deferred" to a Phase B whose hypothesis was already
falsified, "the single governance file" when seven exist. **Lesson: a rule the
project visibly ignores trains agents to ignore all rules, eroding the load-bearing
ones. Dead-letter strict rules are worse than no rule.** Repaired: achievable DoD,
dispositioned deferrals (restore / delete-by-decision / defer-with-trigger), and
the deferral registry regenerated from the real 6 skips (it had listed ~90, mostly
passing tests).

**Infra.** GX10 judge made a standing `unless-stopped` service (was orchestrator-
ephemeral, so "GX10 down" was its normal idle state); survived the session. M5 VLM
remains boot-non-persistent by design (`vlm_serve.sh`), confirmed down->up twice -
the production-smoke's FULL mode now preflights it with 3 consecutive probes and
fails loudly rather than hanging on per-page VLM timeouts.

**What shipped (PLAN_V3.1, all on `main`).** One canonical extraction path (sandbox
tooling repointed); UIR-native TOC heading propagation (qa HEADING coverage 68% ->
100%); the gate wall restored (chunker-entry contract test + AGENT-SPATIAL-20
executable + deferred-test registry reconciled); the F12 heuristic reverted; and
`scripts/smoke_production.sh` - a mandatory pre-merge gate that runs one doc per
routing lane and asserts batch integrity, IMAGE/TABLE `asset_ref` + on-disk asset,
V3-path routing, and `QA_PASS`. That gate would have caught all three F11 bugs.
Open: P6 (retire the non-batch `V2DocumentProcessor`/Docling lane), gated on V3
gaining non-PDF extraction.

---

## 2026-06-01 - V3 extraction hardening: the crucible 0/18 was a schema bug, not the outage  `[Results][Method][Lessons]`

The "Grand Crucible" VLM soak (18 docs, ~600 pages, local M5 Qwen3-VL-8B) was
reported as a node-outage catastrophe (M5 dropped -> silent Docling fallback ->
"hours of junk data"). The disk said otherwise: docs 1-13 completed while M5 was
HEALTHY and ALL failed `IngestionChunk.from_uir` with `ValidationError` BEFORE
any JSONL was written. 0/18 usable baselines. Two schema defects, both in the
VLM-native -> from_uir path:

- 11/13: `QA-CHECK-05 VIOLATION: modality=image/table requires asset_ref`. The
  VLM describes a region but emits no on-disk binary; `from_uir` left
  `asset_ref=None`. Fix: a shared `asset_materializer.materialize_visual_assets`
  crops the bbox region to PNG and sets `asset_ref` - used by BOTH the batch
  path and the soak harness (the soak had silently bypassed the batch path's
  existing crop step; that divergence WAS the bug).
- 1/13: `visual_description` > 400 chars. `from_uir` mirrored full VLM content
  into a 400-capped field. Fix: producer-side truncation; full text stays in
  `content`. Cap not raised, QA-CHECK-05 not weakened.

The M5 outage was SECONDARY and touched only doc 14 (PCWorld), which produced
nothing. Fixing the from_uir crash UNMASKED the next defect: the VLM emits
`type:'code'`, which crashed `ElementType('code')` (enum has only
text/image/table) and dropped the whole page to Docling (strips code
indentation - the original v2 defect). Resolved without widening the legacy
`ElementType` (Charter §7.1): smuggle code/form through as TEXT + a
`promoted_modality` tag, promote to `Modality.CODE`/`FORM` in the chunker.

**M5 throughput (data point for the bandwidth thread):**

| node state | s / VLM page | source |
|---|---:|---|
| degraded (right before the drop) | 150-275 | original crucible per-doc elapsed |
| healthy (this session) | ~49 | single-doc `0013` smoke |

The 6x spread is the difference between a ~24h and a ~4-7h crucible. The
"degraded" timing was a dying node, not the model's real rate.

**Router blind spot, quantified:** `HybridEngine._classify_page` routes on
OBJECT presence (`get_images`/`find_tables`/`get_drawings>10`), no visual-intent
or code/text-complexity signal. Measured on AIOS: 35 pages -> 25 VLM, 10 to the
Docling whitespace-stripping lane. Code-as-text on those 10 pages loses
indentation before the VLM sees it. Deferred (measure-via-smoke before building
a heuristic).

**Lessons.** (1) A passing gate is not correct data: QA_PASS checks asset_ref
PRESENCE, not crop CORRECTNESS, so a hallucinated VLM bbox produces a garbage
PNG that still passes - hence the crop-audit (edge-clamp = clamped-overflow
fingerprint; low-information = whitespace-landing fingerprint; cannot catch
interior misplacement). (2) Smoke the HARDEST representative case: validating the
schema fix on a 1-page FORM (no code) gave false confidence; the first code-heavy
doc broke in 30s. (3) Verify premises under "just do it" pressure - this session
caught a "purge" that would have deleted the only forensic evidence, a wrong ETA,
and a "widen ElementType" directive that violated a locked contract.

---

## 2026-06-02 - Why V3 still failed despite being built to fix V2: a semantic engine wrapped around the geometric one it was meant to replace  `[Lessons][Architecture][Results]`

The paradox that frames this whole reconvergence: on the 2026-05-31 overnight
soak, V3 (VLM-native) beat the V2.16 baseline on EVERY measured axis (Recall@1
+22.8pp, Faithfulness +22.7pp, etc.). The engine is not wrong - it reads better
than Docling. And yet V3 failed, repeatedly, all session: 0/18 crucible
baselines, code pages crashing, a silent infra corruption, a routing blind spot,
provenance loss, a qdrant ingest crash, and a markdown that still needs a
post-hoc repair script. None of those failures were IN the VLM. Every one was at
a SEAM between the new engine and the old pipeline it was dropped into.

### Catalog: every V3 failure this cycle was a boundary mismatch, not an engine fault

| Seam | V2-era contract assumed | VLM / router reality | Failure -> fix |
|---|---|---|---|
| asset_ref (QA-CHECK-05) | Docling extracts a binary image file per picture | VLM describes the region inline, emits no binary | from_uir rejected every image/table chunk = 0/18 baselines -> shared crop materializer (c6c2105) |
| visual_description cap | OCR-era ~400-char captions | VLM writes paragraph-length descriptions | from_uir rejected long text -> producer-side truncation, full text kept in content (c6c2105) |
| ElementType (3 values) | Docling vocabulary: text/image/table | VLM emits 'code' / 'form' | ElementType('code') CRASHED -> whole page dropped to Docling, which strips indentation = the exact original V2 defect -> smuggle-and-promote to Modality.CODE/FORM (c6c2105) |
| Resilience model | Docling is local CPU; a network failure class does not exist | VLM is a remote endpoint that drops mid-run | breaker-less soak silently fell back to Docling and fabricated hours of junk -> VlmInfraError + resilient pause-poll breaker (c6c2105) |
| Router pre-flight | cheap geometric object-counts pick the engine | a GEOMETRIC gate decides the fate of SEMANTIC content | code-styled TEXT (no image/table tags) routed to Docling, indentation stripped - 10/35 pages on AIOS -> monospace-ratio heuristic (2a60a99) |
| Type provenance | one type in, one type out | the rich VLM type is squeezed through the 3-value ElementType | the original type was silently lost downstream -> original_vlm_type marker threaded adapter -> chunk -> JSONL (3d5d9e5) |
| Empty-content asset chunk | text chunks always carry content | a VLM asset chunk can legitimately have empty content | qdrant ingest crashed -> empty-content guard (b44724b) |
| The Docling lane itself | "V3 replaces Docling" | the router STILL sends prose pages to DoclingFast to save VLM cost | Docling's per-page layout inconsistency persists (abbrev pairs classed TEXT on p1 / TABLE on p2; placeholder images; mid-sentence page breaks) -> postprocess_markdown.py band-aid (untracked) |

### Two structural causes (the seams were never going to hold)

**(A) V3 is additive, not a replacement.** `HybridEngine` routes visually-complex
pages to the VLM and "simple/prose" pages to `DoclingFastEngine` for cost. So
Docling - and every V2 caveat it carries (stripped code indentation, per-page
layout inconsistency, placeholder images, mangled reading order) - is STILL the
engine for a large fraction of pages. V2's problems were never eliminated; they
were CONFINED to the Docling lane and made conditional on a router guess. The
untracked `postprocess_markdown.py` is the smoking gun: a V2-era markdown repair
living inside a "V3" pipeline, whose own docstring names "Docling's per-page
layout model inconsistency" as the root cause. Worse, the gate that decides which
content suffers V2's caveats is itself a GEOMETRIC heuristic (object counts, now a
monospace ratio) - precisely the geometric reasoning V3 was supposed to
transcend.

**(B) Richer, more dynamic output meeting fail-closed contracts.** The VLM emits
inline descriptions, long text, new modalities, network failures, and empty asset
content. The surrounding V2-era schema, vocabulary, and resilience layer were
built to FAIL CLOSED - crash, reject, strip, or silently drop - rather than fail
open (generate, fit, carry-through, degrade). Every fix in the catalog is the
same move: convert one fail-closed boundary into a fail-open one.

### Why they surfaced one at a time

- **Masking.** `from_uir` crashed FIRST, hiding every downstream seam behind it.
  Fixing it peeled the onion to the next (code crash), then the next (router),
  then provenance, then empty-content, then the Docling-lane band-aid. You cannot
  see boundary N+1 until boundary N stops crashing - hence commit after commit.
- **Easy-case validation.** Each fix was smoked on the easiest available doc (a
  1-page form, no code), so the next content class hit a FRESH un-migrated seam in
  production rather than in the smoke. (See the dated entry above and
  `feedback_smoke_hardest_case`.)

### The deeper cause: an unfinished migration

Charter §7.1 defined the correct end-state - a one-way migration from the 3-value
`ElementType` to the 5-value `Modality`, retiring the old vocabulary. That
migration was STARTED (Modality widened, VLM engine added) but never COMPLETED:
`ElementType` stayed 3-value, the schema validators / field caps / router /
resilience model were never migrated to V3's assumptions, and the Docling lane was
never retired or fully hardened. So V3 ran as a hybrid - emitting V3-shaped data
into V2-shaped contracts while still leaning on the very engine it was meant to
replace.

### Conclusion

V3 did not fail because the VLM is wrong; it fails because "overcome V2's caveats"
was implemented as "add a better engine" instead of "complete the migration and
retire or repair the old one." That yields the worst of both worlds: V2's caveats
SURVIVE in the still-active Docling lane (now band-aided post-hoc), AND a new
class of integration failures appears at every seam where the richer VLM output
meets the narrower, fail-closed V2 contracts. "Done" is therefore not when the
VLM extracts well - it already does. "Done" is when (1) every contract between
extraction and disk speaks `Modality`, not `ElementType`; (2) the router stops
being a geometric gate on semantic content, or the Docling lane is retired; and
(3) the boundaries fail open, not closed. Until those three hold, each new hard
document will keep finding the next un-migrated seam - which is exactly what it
did, commit after commit, across this cycle.

---

## 2026-06-02 - Grand Soak halted: VLM JSON validity collapses on dense pages, bbox crops drift 40-50%  `[Results][Lessons]`

The first real Grand Soak (M5 Qwen3-VL-8B, `--max-pages 200`) was stopped by the
operator at doc 9/17 (~2.4h in). It is the empirical proof of the
"additive-hybrid degrades on the hardest docs" thesis. Per-doc:

| # | doc | pages | vlm | docling | json-fail fallback | crop-audit | time |
|---|---|---:|---:|---:|---:|---|---:|
| 1 | AIOS (agent paper, born-digital) | 35 | 25 | 10 | 0 | PASS 0/32 | 745s |
| 2 | hybrid-electric review | 31 | 6 | 24 | 1 | PASS 0/45 | 319s |
| 3 | Hybrid EV challenges | 16 | 7 | 7 | 2 | WARN 3/18 (17%) | 641s |
| 4 | IRJET solar | 7 | 6 | 0 | 1 | PASS 2/16 (12%) | 331s |
| 5 | Recent Transport | 5 | 4 | 1 | 0 | WARN 2/5 (40%) | 134s |
| 6 | Form_0013 | 1 | 1 | 0 | 0 | PASS 0/2 | 55s |
| 7 | betwistingsformulier | 1 | 1 | 0 | 0 | WARN 1/2 (50%) | 41s |
| 8 | CarOK voorraadtelling | 12 | 7 | 0 | 5 | WARN 6/12 (50%) | 704s |
| 9 | Combat Aircraft (magazine) | 43+ | - | - | ~25 | stopped mid-doc | - |

**A. VLM emits invalid JSON on dense pages -> mass Docling fallback.** 34
page-level `json.loads` failures in `_parse_strict_json`: 27 "Unterminated
string" (truncation, consistently mid-`content`-value = output hit the token
cap) + 7 structural ("Expecting value / ',' delimiter / property name"). Each is
a *semantic* failure -> per-page Docling fallback (breaker working as designed).
Concentrated on the densest doc: Combat Aircraft, ~25 of 43 pages -> Docling, the
layout-mangling path V3 exists to replace, on the one magazine the soak reached.
Root cause: an 8B VLM cannot reliably emit a large strictly-valid whole-page
JSON; the "one strict-JSON per page" design has a density ceiling. There is no
`finish_reason=length` detection today, so a truncated response is silently a
parse failure. Levers: raise/handle `max_completion_tokens`, guided/constrained
JSON decoding, or per-region extraction.

**B. VLM bbox crop drift 40-50% on forms/scans/tables.** `QA_WARN_CROP_DRIFT`
fired on 5 of 8 completed docs (above). The charter §3.3 residual risk (interior
misplacement / edge-clamp) is now MEASURED: roughly half the image/table crops on
visually-busy docs are garbage. The crop-audit catches it correctly; the
extraction does not produce correct coordinates.

**C. Throughput + coverage.** ~20-60 s/VLM-page; dense pages with retries ran
2.5-5 min each. `--max-pages 200` excluded ~20 of the largest books (Fluent
Python 766p, Zephyr 689p, ...), so the "Grand" soak only attempted 17
small/medium docs and could not finish them.

**Quality eval (15-doc subset, GX10 judge):** R@1 70.0% / R@5 chunk 76.7% / R@5
doc 96.7% / Relevance 78.3% / Format 98.3% / Faithfulness 73.3% - directional,
below the (sampled, single-seed) V3_OVERNIGHT head-to-head, with cross-doc
confusion in the weakest cases (an invoice query retrieving the wrong form; a
Python-prerequisites query retrieving Fluent Python over the RAG guide).

**Verdict.** The operator stopped the soak because the pipeline does not meet
requirements on the documents V3 targets. AIOS-class clean born-digital docs work
well; dense magazines and forms/scans do not - invalid JSON -> Docling fallback,
and 40-50% crop drift. The remedy is extraction-layer (VLM JSON validity + bbox
fidelity), not another soak run. This sharpens the 2026-06-02 "why V3 still
failed" analysis above with measured rates.

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
(current V3) → **throughput wall** (2026-05-30 entry: bandwidth-bound; M5 Max thesis)
-> **sandbox/shipping reconvergence** (2026-05-31/06-01, F10-F13: the shipping path
had never been tested; fixing it falsified the carried-over Docling boundary
heuristics and proved V3 requires *removing* geometric scaffolding, not adding to
it; closed with an executable pre-merge gate).

---

## 2026-06-03 - Blocker remediation shipped (A1-A4, B1-B2) + the json_schema latency trap  `[Results][Method][Lessons]`

The two 2026-06-02 Grand-Soak blockers (Charter §9.1) were remediated cheap ->
structural and validated against a dense magazine through M5. Seven atomic
commits, each gated on the full suite + repo-integrity + `smoke_production.sh`
(`SMOKE_PRODUCTION_PASS`), pushed to origin.

**Blocker A (VLM invalid JSON on dense pages).** Four composing fixes:
- **A1** typed truncation: `vlm_provider.describe` reads `finish_reason` on every
  200, escalates `max_tokens` once on `length`, then raises
  `VlmTruncationError(partial_content=...)` instead of silently returning a
  truncated body that `json.loads` rejects. (Also fixed a latent fall-through
  where a 200-retry hit the status else-raise via `if` not `elif`.)
- **A2** adaptive budget: floor 4096 -> 8192 (the known-good overnight value),
  scaled per page by `estimate_output_budget` (text chars x2.5 / 3.5), capped at
  16384. Wired at both VLM call sites.
- **A4** bounded repair: `repair_truncated_json` walks the `elements` array with a
  strict `JSONDecoder`, keeps the N complete elements, drops the cut trailing one
  - partial VLM extraction beats full Docling fallback.
- **A3** json_schema/guided_json constrained-decode capability + a fail-open 400
  strip-and-retry.

**Blocker B (bbox crop drift 40-50%).** B1: crop from a deterministic geometric
bbox (`get_image_info`/`find_tables`) when the page has a detectable object;
trust VLM coords only when none exists. B2: a drift-flagged VLM crop is
re-rendered to the full page before persisting (detection fingerprints preserved
for the gate; `reextracted` flag added) - a garbage crop is never written.

**Live numbers (M5, Combat Aircraft dense magazine, 2026-06-03):**
- Run 1 (batch=10, shipped json_schema-on default): one ultra-dense page exceeded
  the **180s read timeout** x3 -> `VlmInfraError` -> breaker tripped (correct B4)
  -> under batch=10 the whole 10-page batch fell to the text recovery path (37
  recovery chunks, 0 assets, 0 real VLM pages).
- Run 2 (batch=1, `VLM_NATIVE_STRUCTURED_OUTPUT=off`): **10 chunks via
  `uir_native_chunker`, 5 image assets, 1 B2 re-extraction fired, 0 truncation, 0
  Docling fallback, 0 breaker trips - 4 dense pages in 352s (~88 s/page).**
- Run 3 (PCWorld, the 2nd named dense doc, batch=1, shipped defaults =
  structured output off): **27 chunks all via `uir_native_chunker`, 12 image
  assets, 0 truncation, 0 Docling fallback, 0 breaker trips, 0 crop-drift
  re-extractions - 6 dense pages in 135s (~23 s/page).** Second confirmation of
  the §9.1 acceptance criteria (fallback ~0, crop drift < 15%) on a different
  dense-magazine class.

**Lesson - the json_schema latency trap.** mlx-vlm ACCEPTS the json_schema
`response_format` (no 400) but its grammar-constrained decode is pathologically
slow on dense pages, blowing the 180s client timeout. The charter "verified
available on both endpoints" was true for *acceptance*, false for *throughput* -
"supported" is not "fast." Net: A3's self-hosted default was flipped to OFF
(prompt-only, known-good); json_schema/guided_json stay opt-in via env for vLLM +
xgrammar. The capability ships; the default is conservative on live evidence.

**A1-A4 verdict: the JSON-validity blocker is cleared.** On the pages that
complete, dense-page fallback went from the soak's ~58% (Combat Aircraft ~25/43)
to **0**. So **A5 (per-region extraction) was NOT built** - the mandate gates it
on "A1-A4 do not clear the fallback rate," and they did.

**Residual, flagged (out of §9.1 scope - throughput / Charter §5+§8):** an
occasional ultra-dense page still exceeds the 180s read timeout; under batch=10
the breaker sinks the batch. The correct B4 fail-fast is intact; the fix is
operator/throughput-side - batch=1 for dense docs, a `VLM_NATIVE_TIMEOUT` env,
distinguishing ReadTimeout (slow page) from ConnectTimeout (node down) in the
breaker, or A5 to cut per-call latency. Not changed this cycle.

---

## 2026-06-04 - The dense-page "timeout" was a budget x decode-speed mismatch, not a hang  `[Results][Method][Lessons]`

Follow-up to the 2026-06-03 residual. Measured Combat Aircraft INTERIOR pages
(24-36) per-page through the V3 router on M5 (`scripts/measure_vlm_page_latency.py`),
which corrected an over-optimistic read: the earlier "json_schema-off cleared it"
was a LEADING-page artifact (covers/TOC are light, 23-88s). Interior feature
pages at the 180s default: median 265s, **9/13 over 180s, 5/13 fully timed out
(547s = 3x180s retries) producing nothing** - i.e. a ~46% silent page loss that
the leading-page runs hid.

**The hypothesis we were about to build was wrong.** Image density does NOT
predict latency: the slow pages span 1-4 images; a 1-image / 153-char page took
287s and failed while a 3-image page took 70s and passed. So "adaptive batch
size by image density" would have shrunk batches for the wrong pages. The real
driver is vision prefill + generation length, invisible to the cheap pre-flight
signals.

**Root cause: a budget x decode-speed x timeout mismatch.** Re-measuring the same
13 pages at `VLM_NATIVE_TIMEOUT=600` (+ read-timeout retries capped at 1): **13/13
ok, zero hangs**, and the previously-failed pages all completed at **~248s**. That
number is the tell - 8192 tokens (the A2 budget floor) at M5's ~33 tok/s is ~248s,
which physically cannot finish inside 180s. The old timeout wasn't catching a
pathology; it was guillotining normal dense-page generation.

**Resolution (shipped):** `VLM_NATIVE_TIMEOUT` wired into `VlmProviderConfig.from_env`
and the hardcoded default raised 180 -> 600s (a ceiling, not a target - cloud
endpoints still return in seconds). Read timeouts now get a dedicated 1-attempt
cap (a heavy page would just blow the same timeout again; connect faults keep the
full `max_retries`; all terminal cases still raise `VlmInfraError`, B4 intact).

**What this buys / costs:** dense-doc interior page loss ~46% -> 0. A5 (per-region),
a render-DPI cut, and density-keyed batch sizing are all UNNECESSARY for
correctness - nothing hangs, so there is nothing to bound or insure against.
Cost: dense pages are ~250s each, so a full magazine is multi-hour (already known)
- but now with zero page loss. **Correction to the 2026-06-03 entry: Blocker A's
JSON-VALIDITY half was cleared by A1-A4, but a separate dense-page TIMEOUT failure
dominated interior pages; that is what this entry resolves.**

**Lesson:** measure the hard case fully before designing for it. A plausible,
intuitive knob (batch size by image density) was killed by 13 pages of data, and
the actual fix was a one-line coupling (budget/decode-speed/timeout) that no
amount of density-classification would have found.

---

## Backfill backlog (remaining threads — to expand when drafting)

*Covered above as of 2026-05-30:* V1→V2 lineage · V2 metrics trajectory · V2
dead-ends · judge calibration · V3 draft evolution + ColPali pivot + Phase A
spikes · **V3-vs-V2.16 core result** (2026-05-29 entry; per-doc tables in the
committed `V3_OVERNIGHT_REPORT.md`). Still to do:
- **Identity Gate finding** — V2.16 Docling silently dropped ~80% of spreadsheet
  rows (CarOK); V3 VLM-native rebaselined to full extraction. Structural Form_0013
  comparison (logo description vs placeholder; table as row-level chunks).
- **V1 primary sources** — extract `SRS_Multimodal_Ingestion_V2.3/V2.4.md` and the
  early design docs from git history (`git show`) into `docs/paper/archive_extracts/`
  before they're harder to retrieve.
- ~~**M5 Max measurement** — fills/corrects the F7 extrapolation once unboxed.~~
  **DONE** (2026-05-31 entry, F8/F9): extrapolation beaten ~2×, 8-bit 58.9 tok/s
  / 18.4 s/page, fidelity OK. Remaining: large-magazine thrash test + oMLX A-B.
- **(legacy backlog note)** Judge calibration — GX10 Qwen2.5-14B-FP8: format TRUSTWORTHY,
  rel/faith RESTRICTED; treat as directional. Source: memory `feedback_v2_14_gx10_14b_fp8_swap`.
