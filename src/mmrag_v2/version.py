"""
Centralized version definitions for MM-RAG V2.

Single source of truth for schema and engine versions to avoid hardcoded
scattering across the codebase.
"""

# Schema version for ingestion output (chunk-shape contract).
# v2.8 made no changes to the IngestionChunk JSON shape — the
# behavioral changes (keyword-aware control-char replacement,
# CodeFormulaV2 enrichment, form-aware audit gate, adapter
# invocation guard) all preserve the existing schema. v2.9 likewise
# preserves the JSON shape; the chunk_id *value* changes for the 427
# previously-colliding chunks (per-doc position is hashed in) but no
# chunk field is added or removed. Downstream consumers that key on
# chunk_id for cross-version mapping must rebuild from v2.9 outputs.
__schema_version__ = "2.7.0"

# Engine/runtime version. v2.16.0 is the **convergence release** —
# MM-Converter-V2 is feature-complete as of this tag. Every open
# carry-forward item from v2.11 → v2.15 received a binary disposition
# (SHIP / KILL / OUT-OF-SCOPE for v3.0); see `docs/DECISIONS.md`
# "v2.16 …" entries.
#
# Production retrieval stack: omlx Qwen3-Embedding-8B-mxfp8 + BM25 +
# RRF + ModernBERT rerank against `mmrag_v2_8__qwen3_local` —
# unchanged from v2.15.0 baseline shape. Phase 3 (partial_code
# adjacency fetch) adds a bounded post-rerank stitch that is
# mechanism-correct but currently inert on the production corpus (no
# chunks carry `partial_code=True` from the HybridChunker path). Phase
# 4 (VLM-table IoU dedup at 0.85) lands as an ingestion-layer change
# in `BatchProcessor._apply_vlm_table_iou_dedup`. Phase 5 (dynamic
# top-k) KILL'd by pre-flight; Phase 6 (query rewriting) KILL'd by
# Phase 2 multi-factor verdict; Phase 7 (image re-read) KILL'd by
# default (no user opt-in).
#
# v2.16 Phase 1 added the `personal_importance` decision-mechanism
# overlay on the documented-limitation registry — HIGH forces Option
# A regardless of telemetry. The cycle-open checklist gained a
# 2-minute review item (`docs/CYCLE_OPEN_CHECKLIST.md` §5).
#
# Post-v2.16.0 governance: only bug-fix patches (v2.16.x) accepted.
# New features = re-charter as v3.0. v2.17 fires only on §7 trigger
# (e.g., Item #9 reopens for HybridChunker partial_code coverage,
# which Phase 3 acceptance depends on). Audit history: 8 external
# rounds + 1 self-audit; v2.15 §9 stopping rule fired at Round 8;
# full archaeology at `docs/archive/plans/PLAN_V2.16_0.10.md`.
#
# Phases shipped under Option F (this cycle):
#   Phase 3 [F] (telemetry) Documented-limitation registry + telemetry
#                           primitives (compute_document_class_hits +
#                           build_telemetry_record) + dated TELEMETRY
#                           _REPORT analyzer + soak-harness write
#                           hook + USER_ISSUES.md append-only table +
#                           CYCLE_OPEN_CHECKLIST.md + Phase N DoD
#                           gate (verify_phase2_teardown.py;
#                           vacuously satisfied under F). DECISIONS
#                           .md "v2.15 Documented-Limitation
#                           Telemetry Threshold" entry transitioned
#                           from PRE-CYCLE PROPOSAL to ACTIVE RULE
#                           concurrent with the code landing.
#   Phase 6 [U] (cal-fresh) FP8-14B Phase 0 calibration verified
#                           fresh through 2026-06-22; no re-cal
#                           needed. T-72h pre-tag checkpoint armed
#                           via CYCLE_OPEN_CHECKLIST.md cycle_slip.log.
#   Phase N (close-out)     Engine bump 2.14.0 → 2.15.0; AFTER
#                           snapshot; tag.
#
# Phase 1 [U] (HyDE bridging) — CODE SHIPPED, SOAK EXECUTION
# DEFERRED-WITH-EVIDENCE per the v0.9 DoD silent-default clause:
#   - scripts/sample_phase1_narrow_fixture.py ready (5-doc fixture
#     spec: 100 ATZ_Elektronik + 4×20 code-dense per R7F1
#     statistical-defensibility bar)
#   - Intent classifier + targeted-HyDE infra already shipped in
#     v2.14 Phase 2 (commit 156dfa7); re-targets to this narrower
#     soak rather than the broad-corpus null v2.14 falsified
#   - SOAK EXECUTION BLOCKED on MLX_API_KEY availability for the
#     omlx embedder + omlx reranker. Rerun when key available:
#       python scripts/sample_phase1_narrow_fixture.py
#       python scripts/synthetic_soak.py --stage generate --work ...
#       python scripts/synthetic_soak.py --stage retrieve --work ... [+/-use-hyde]
#       python scripts/synthetic_soak.py --stage judge --work ...
#     Compound acceptance gate per PLAN_V2.15.md §Phase 1: aggregate
#     ≥6pp R@1 + German ≥+10pp on n=100 + ≥3/4 code-dense positive
#     + format no-regression + faith ≤-1pp.
#
# Skipped phases (Option F deferral):
#   Phase 2 [A] (pdfplumber lane)         — v2.16 contingent on
#                                          Phase 3 telemetry
#   Phase 4 [A] (Docling config tuning)   — carry-fwd 6.1 re-eval
#                                          trigger (Docling ≥2.87
#                                          OR 90d) in CYCLE_OPEN
#                                          _CHECKLIST.md
#   Phase 5 [E] (retrieval investments)   — v2.16 contingent on
#                                          F→E telemetry escalation
#
# Audit history: 8 rounds (Gemini round 1 + claude rounds 2-8)
# closed ~40 concrete failure modes in PLAN_V2.15.md before any
# phase executed; stopping rule fired at rounds 7+8 both 0 HIGH.
# Full plan + Appendix A archaeology: docs/PLAN_V2.15.md.
# Predecessor: v2.14.0 (2026-05-23, tag 122a62e; PUSHED to origin).
__engine_version__ = "2.16.0"
