# V3 EXECUTION MANDATE

This is the **conflict-resolution authority** for V3 work: where it conflicts
with another doc, this wins. It is NOT the only governance doc - the governance
set is the Layer-0 contract list in `docs/README.md`. Plans, audits, and
execution docs are not governance docs and may be added freely.

## 1. THE ARCHITECTURAL CONTRACT
* **Engines** (e.g. `src/mmrag_v3/engines/docling_fast.py`) MUST parse source files and return a single `UniversalDocument`. Engines are forbidden from chunking.
* **Extraction entry** (`mmrag_v3.extract`) MUST be engine-agnostic (zero Docling imports); it returns the `UniversalDocument`.
* **Chunker** (`src/mmrag_v2/chunking/uir_chunker.py`) MUST accept a `UniversalDocument` and emit `UIRChunk` objects. Zero Docling imports.
* **Orchestrator** (`src/mmrag_v2/batch_processor.py`) is limited to engine-agnostic orchestration: batching, routing, dedup, quality-filter finalize, emission-side asset/visual finalization (e.g. rendering region crops so IMAGE/TABLE chunks satisfy QA-CHECK-05), and JSONL writing. It MUST construct and import no Docling (AST firewall + AGENTS.md boundary note) and MUST NOT perform source extraction.

## 2. DEFINITION OF DONE (achievable + mechanically checkable)
An architectural phase is complete only when ALL hold:
1. `pytest tests/test_v3_security.py` returns exit 0 (the AST firewall).
2. `pytest tests/` returns exit 0. New skips are allowed ONLY as a registered §3 deferral (owner + un-defer trigger, listed in `docs/V3_DEFERRED_TESTS.md`); an unregistered skip is a failure (enforced by `tests/test_repo_integrity.py` G6).
3. The production-CLI smoke passes on at least one document per routing lane (VLM / mixed / prose): no zero-chunk batch; every IMAGE/TABLE chunk carries `asset_ref` (and, in FULL/VLM mode, a non-empty `visual_description`; offline/no-VLM ships a documented ID-only image per `QUALITY_GATES.md` `IMAGE_NO_VLM`); routing matches the lane; `qa_full_conversion` reports QA_PASS or a documented QA_WARN. Harness `scripts/smoke_production.sh` (PLAN_V3.1 P5) is SHIPPED and is the mandatory pre-merge gate for any extraction-path change; it must print `SMOKE_PRODUCTION_PASS` (exit 0) in offline mode.

RETIRED criterion: the prior Identity-Gate step (`scripts/run_identity_gate.py`, NOT YET BUILT, "< 5% delta") is dropped - that script was never built, and a single sub-5% delta against the v2.16 baseline is impossible by design once the V3 chunker changes chunk shape (§3). Identity is now an explained-delta review (identity-half >= 95% AND explained-delta <= 5%), not one number against v2.16.

ADVISORY criterion (fidelity outcome gate, F5, AGENT-GATE-PROGRESSION): an extraction-path change SHOULD report the OmniDocBench fidelity delta (text edit distance, table TEDS) versus the recorded hybrid regression baseline (158-page fixed set: text-ED **0.2212** / TEDS **0.7933**, `PLAN_EXTRACTION_FIDELITY_V1` Phase 1) via `scripts/omnidocbench_adapter.py`. This is an OFFLINE selection/regression gate, ADVISORY for now (no ground truth at conversion time); it is promoted to a hard regression gate only per `docs/QUALITY_GATES.md` and Section 6 of that plan. It is NOT a per-conversion production gate.

## 3. SCOPE + DEFERRAL DISCIPLINE
* The V3 chunker fundamentally alters chunk shape; chunk COUNT/CONTENT parity vs v2.16 is NOT a smoke requirement (use the explained-delta review).
* Deferrals are DISPOSITIONED, never "permanent." Every deferred v2.16 heuristic or skipped test is exactly one of:
  - (a) RESTORED - re-implemented UIR-native and its test un-skipped; or
  - (b) DELETED - removed with a `docs/DECISIONS.md` entry recording the dropped behavior and rationale (this is the sanctioned exception to AGENT-TEST-01); or
  - (c) DEFERRED - with an explicit OWNER + UN-DEFER TRIGGER + date, listed in `docs/V3_DEFERRED_TESTS.md`.
  A deferral that is none of (a)-(c) is a defect, not a state (AGENT-STATUS-01).
* The "deferred until the Phase B LLM-sanitization layer subsumes them" rationale is RETIRED: that layer's hypothesis was falsified (see `docs/V3_DEFERRED_TESTS.md`). Heuristics that still earn their place are adopted under (a); the rest are restored or deleted. Heading/breadcrumb propagation is explicitly UN-BLOCKED for repair - it is TOC-driven, not sanitization-dependent (`docs/PLAN_V3.1_PIPELINE_RECONVERGENCE.md` P2).

## 4. STATUS ENFORCEMENT
There is no "in-progress," "rebooked," or "implemented but not validated." A phase either passes the §2 gates or it has failed. A dispositioned deferral (§3c) is an explicit, owned, triggered exception - not open-ended limbo.
