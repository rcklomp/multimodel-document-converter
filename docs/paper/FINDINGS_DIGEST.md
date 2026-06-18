# Findings Digest — START HERE

The to-the-point history for a cold-start session. For the full blow-by-blow with data
tables, see `FINDINGS_LOG.md` (1500+ lines) - this digest is the index to it. For current
task state see `docs/PROJECT_STATUS.md`; for locked decisions `docs/DECISIONS.md`.

---

## Where we are now (2026-06-15)

A PDF→JSONL multimodal RAG pipeline (V3, branch `feat/omnidocbench-phase0`, not pushed).
The extraction engine is **settled** (MinerU+Qwen hybrid). The last cycle reframed the goal
from "perfect PDF conversion" to "the RAG works well", measured the RAG end-to-end, and
found the real answer-quality lever is trivial (**feed the LLM top-10 chunks, not top-5**).
The one open retrieval problem is the ~6% of queries that never retrieve the right document.

## SETTLED — load-bearing, do NOT re-litigate

- **Engine = MinerU+Qwen-for-code hybrid.** Phase 4 shadow window: 0/16 QA_FAIL vs the
  offline-floor's 4/16. Pure-VLM and pure-pipeline both refuted. MinerU does tables/layout,
  Qwen does code (MinerU mangles dense code; Qwen empties dense tables).
- **Render = cap1600** longest-side. dpi200 oversizes pages → VLM repetition loops → worse
  fidelity (proven on the dense academic class: cap1600 0.05 vs dpi200 0.37 text-ED).
- **Reliability = fail-closed 3-tier ladder** (selected engine → offline docling → PyMuPDF),
  provenance-stamped. Not the primary story; a safety net.
- **Code indentation is SOLVED** on the production path (R3 0.947): it was a transcription-
  PROMPT property, not a model limit. Do not "fix code indentation".
- **Conversion is NOT the RAG bottleneck** (measured twice). Doc-level retrieval is strong
  (R@5-doc 91.6%).
- **The answer-quality lever is FEED TOP-10 chunks** (+4.9pp answer correctness, German-safe).

## DEAD ENDS — measured and rejected, do NOT re-propose

- **OCR / `do_ocr=True`** — was the v2 default for years; its 0.301/0.563 ceiling DROVE the
  V3 pivot. MinerU OCRs scans better (0.221). (`[[project_ocr_already_tried]]`)
- **Filtering empty image chunks** to help retrieval — measured **+0.0pp**. (They're an
  enrichment-quality question, not a retrieval one.)
- **Sorting reranker output by `rerank_score`** — measured **-10pp** (the score field is
  incompletely populated; the server's native order is correct).
- **Hybrid/BM25 as the retrieval default for THIS corpus** — no gain over plain top-10, and
  it regresses the German docs. (The BM25-index persistence fix was still a real bug fix -
  hybrid was silently broken - but it's not needed for the win.)
- **code-repo-diff oracle with ABSOLUTE indentation** — confounded by books de-nesting class
  methods to column 0; not a fidelity signal.
- **chunker-contiguity fix for Chaubal code** — correct but fires 0x; wrong lever.
- **engine-swap reflex / more scaffolding around a single general VLM** — caused the churn;
  the answer was a complementary hybrid + measurement, not another model.

## OPEN — the actual remaining work

- **~6% of queries never retrieve the right document** even at top-100 (embedder /
  query-expansion / HyDE problem). THE NEXT TASK.
- **Chaubal-type code residual** (REPL/notebook transcripts + engine token corruption):
  fullwidth scrub shipped; de-LaTeX `\(\equiv\)` + CJK strip DEFERRED (need a trustworthy
  code-fidelity measure first).
- **Code books mostly NOT ingested**, so code retrieval is essentially unmeasured. The R3
  code-router gap is diagnosed (LOG 2026-06-11): the fix is a post-extraction quality-flag
  re-extraction lane (degraded code -> re-do that page via the Qwen lane), NOT a router tweak;
  not yet built. The doc-level `ProfileClassifier` profile is also dropped at the
  `mmrag_v3.extract(path)` seam, so the engine routes code-vs-table blind.
- **No omission-sensitive labelled GT for the internal classes** (German/Dutch/automotive) -
  OmniDocBench is EN-only; the deep, deferred measurement gap.

## One-line history (the arc)

- **~Jan–May 2026 (V1→V2):** Docling PDF→JSONL ETL; ~5 months; retrieval/format gates ~88–93%.
- **2026-05-29 V3 core result:** VLM-native V3 beats V2.16 on retrieval (R@5 +31.9pp).
- **2026-05-30/31:** throughput is memory-bandwidth-bound (M5 Max thesis); serving topology.
- **2026-06-01/03:** sandbox-vs-shipping reconvergence; fail-closed extraction ladder.
- **2026-06-04/05:** Qwen3-VL table/repetition failures → MinerU2.5 chosen → MinerU+Qwen hybrid.
- **2026-06-08:** 16-doc crucible clean after 4 systemic fixes; multimodal image policy.
- **2026-06-09:** OmniDocBench ground-truth benchmark (EN baseline text-ED 0.301 / TEDS 0.563).
- **2026-06-10:** render cap (cap1600); seeded-fault validation (text-ED is blind to indentation).
- **2026-06-11:** two-corpus bake-off INCONCLUSIVE (code-free benchmark); hybrid = validated default (Phase 4).
- **2026-06-13/14:** root-cause of the multi-class struggle = acceptance measures PRESENCE not
  FIDELITY on failing classes; residual burndown (green-gate integrity + named residuals); WS3 cap1600 proven.
- **2026-06-15:** REFRAME to a retrieval bar; measured the RAG → conversion isn't the bottleneck;
  answer lever = feed top-10; three clever levers rejected by measurement; 6% doc-recall is next.

---

## How to maintain this doc (the contract behind it)

This digest is the project's **anti-circle cold-start index** — the one page an
agent reads first to avoid re-litigating settled decisions and re-proposing
measured-and-rejected approaches. Its function lives entirely in the three
load-bearing sections above: **SETTLED**, **DEAD ENDS**, **OPEN**.

- **It is not a Layer-0 contract that wins conflicts.** The conflict-resolution
  authority is `docs/V3_EXECUTION_MANDATE.md`; this doc does not override it. It
  is *required-present* knowledge (G2-enforced tracked) referenced by
  `AGENT-PRECEDENT-01` in `AGENTS.md` and Read-First #11 in `CLAUDE.md`.
- **The three section headers + the back-references to `FINDINGS_LOG.md`,
  `docs/PROJECT_STATUS.md`, and `docs/DECISIONS.md` are mechanically enforced**
  by `tests/test_repo_integrity.py::test_g7_findings_digest_anti_circle_structure_intact`
  (guard G7). A "condensing edit" that folds them into prose will fail the suite
  instead of silently neutering the anti-circle function.
- **Update trigger.** When an approach is measured-and-rejected OR a decision is
  settled: add it here (`## DEAD ENDS` / `## SETTLED`), append the full data to
  `docs/paper/FINDINGS_LOG.md`, and add the load-bearing entry to the "Settled
  Precedents" index at the top of `docs/DECISIONS.md`. Re-opening a SETTLED or
  DEAD-END item requires **new evidence** (a different corpus / model / a measured
  reversal of the prior result) — per `AGENT-PRECEDENT-01`, restating the old
  argument is a defect, not a proposal.
- **Keep it a digest.** If an item's full rationale does not fit one line, the
  line points at the DECISIONS heading or the FINDINGS_LOG date; the detail lives
  there, not here. This file is the index; the others are the source.
