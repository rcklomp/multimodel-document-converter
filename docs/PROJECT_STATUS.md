# Project Status

Last updated: 2026-05-03

Purpose: fast orientation for a new coding session. Read this before deeper project docs.

## Current Objective

The project is in a quality-stabilization phase before broad reconversion and Qdrant re-ingestion.

Immediate goal: make conversion quality measurable and reproducible across document categories, then fix the highest-risk quality issues without overfitting to one document or one VLM.

## Active Baseline

The current quality reference point is:

- **2026-05-03 work** (post-Docling sanity pass + `digital_literature` profile): see `CHANGELOG.md` entry "Post-Docling Sanity Pass (2026-05-03)" and `docs/PLAN_DOCLING_POSTPROCESSOR.md`. No quality snapshot file yet; `tests/test_docling_postprocessor_acceptance.py` (HARRY pages 13-30 reading-order fixture) is the new binding regression test.
- `docs/QUALITY_SNAPSHOT_2026-05-01.md` (current — Milestone 1 closure, RAG Guide unblock, Ayeva re-conversion)
- `docs/QUALITY_SNAPSHOT_2026-04-30.md` (Vision-Aided Front Matter, Shared PDF Plan, Coordinate Audit, Domain-Specific Search Priority completion evidence)
- `docs/QUALITY_SNAPSHOT_2026-04-29.md` (pre-Milestone-1 corpus baseline; rows for Ayeva and Harry Potter are now stale and superseded by the entries above)

Use the latest snapshot as the before-state for future comparisons.

## Active Model/Endpoint State

Do not print or commit API keys.

Current local VLM setting:

- provider: OpenAI-compatible
- model: `NuMarkdown-8B-Thinking-mlx-8bits`
- base URL: `http://10.0.10.246:8000/v1`

Cloud comparison tested:

- provider: OpenAI-compatible DashScope endpoint
- model: `qwen3-vl-plus`

Observed behavior:

- local NuMarkdown is faster in the PCWorld harness after retry flow but needs many hard fallbacks
- Qwen3-VL-Plus gives richer visual descriptions and fewer hard fallbacks
- both models still read visible text, so model-agnostic enforcement is required

## Current Quality Summary

### Non-Magazine PDFs

Existing non-magazine outputs: 14 of 17 pass `scripts/qa_conversion_audit.py`.

Known failures:

- `A_comprehensive_review_on_hybrid_electri`: one control-character text artifact
- `Chaubal_PyTorch_Projects`: code indentation fidelity (re-probe pending after Milestone 1 fixes)

Recently resolved:

- `Ayeva_Python_Patterns`: re-converted to `output/ayeva_qa_20260501/`; AUDIT_PASS + GATE_PASS + UNIVERSAL_PASS, `indentation_fidelity=0.93` (was 0.22).
- `A Simple Guide to RAG (Kimothi)`: previously hung at 7200s on batch 25; now passes via SIGALRM 120s fallback. `output/probe_rag_guide_guard/`, AUDIT_PASS + GATE_PASS + UNIVERSAL_PASS, 680 chunks.

Known caveat:

- Some documents pass audit while routing through questionable profiles. Classifier correctness remains a separate workstream.

Workstream B direction:

- Code-block fidelity must use a selective Docling-native enrichment lane, not more regex/reflow heuristics.
- Docling 2.86.0 exposes `CodeItem` and `do_code_enrichment`; targeted Ayeva evidence shows CodeFormulaV2 can restore multiline/indented code, but local CPU runtime is not viable for broad conversion.
- Do **not** trigger CodeFormulaV2 from `has_encoding_corruption` alone. Encoding corruption also appears in Combat Aircraft and would pull Workstream C into expensive code inference.
- Preferred architecture: cheap code-evidence pass first (`CodeItem` count/code density/sampled code candidates), then `needs_code_enrichment=True` with reason/counts, then remote-capable CodeFormulaV2 on a stronger local-network host or cloud endpoint.
- Client-side MLX/transformers acceleration is diagnostic/fallback only, not the main production strategy.
- Shared PDF extraction refactor: `complete` (2026-04-30). `batch_processor.py`, `processor.py`, and `engines/pdf_engine.py` consume a shared `PdfConversionPlan` and `DoclingPdfAdapter`; static guards reject new production `PdfPipelineOptions` / `DocumentConverter` construction outside the adapter.
- Canonical target flow remains diagnostics/config -> `PdfConversionPlan` -> Docling adapter -> `UniversalDocument` -> `ElementProcessor` -> chunks; legacy direct Docling-item glue was not expanded.
- The current fenced-flat detector work is provisional fallback evidence, not a completed Workstream B fix.

### Magazines

`Combat_Aircraft_full_promptfix_v2`:

- full 100-page conversion completed
- image side improved substantially
- Combat-style firearm/bolt/exploded-view hallucinations reduced to zero in the measured patterns
- still fails text audit because of encoding/high-corruption artifacts

`PCWorld_July_2025`:

- full existing output passes audit but uses placeholder image descriptions
- use it as an asset source, not as proof of VLM quality

`PCWorld_promptfix_pages1_20`:

- partial production smoke passes audit with VLM descriptions
- some residual text leaks were found before the latest sanitizer additions

## Active Engineering Direction

VLM Source Sanctity enforcement status: **validated-cloud / local-pending** (2026-04-29).

1. Visual-only prompt enforced for all VLM calls.
2. Every response validated for text-reading behavior.
3. Retry with corrective instructions on first failure.
4. Sanitizer salvages text-reading responses (77.4% success rate on PCWorld).
5. Hard fallback to neutral description when unsalvageable.
6. Structured `vision_validation_issues` emitted in production JSONL.
7. Machine-readable quality summary via `scripts/vlm_quality_summary.py`.

Cloud-tested with `qwen3-vl-plus`. PCWorld raw text-reading detections: 36.5% → 22.2%. Zero measured Combat-style hallucinations. Blind-set run: 87.5% final-valid by the current validator.

Known limitations:

- Local NuMarkdown/local Qwen comparison is pending because the local inference server is unavailable off-network.
- Blind-set evidence is a reproducible fixture: manifest tracked at `tests/fixtures/blind_set_manifest.json`; generated assets belong in ignored `output/` and must be regenerated locally before re-running the harness.
- `scripts/vlm_quality_summary.py` harness mode now reports `clean` and `sanitized` as separate classes. Production mode classifies images as `clean_or_sanitized` (combined) because raw and issues fields are not available there.

Next engineering focus: code-block fidelity (Workstream B) and Combat Aircraft text corruption (Workstream C).

Latest validation:

- Contextual Retrieval (Anthropic approach, Combined Plan #4 / Plan Feature 6): `complete` (2026-05-01). Embed-time `build_contextualized_text(...)` builder added under `src/mmrag_v2/chunking/contextual_retrieval.py` with AGENT-CONTEXTUAL-01..07 invariants and AST-level drift guard. Optional `IngestionChunk.contextualized_text` schema field added (never read by QA). `scripts/ingest_to_qdrant.py` wires text+table modalities through the builder with a `--no-contextual` byte-stable rollback flag. Static guards `2 passed`; focused contextual suite `32 passed`; focused boundary suite `93 passed`; full unit suite `512 passed, 1 skipped, 0 failed`; probe `output/probe_contextual_retrieval_rag_guide/` AUDIT_PASS + UNIVERSAL_PASS with byte-identical 680 chunks (text=559, image=99, table=22) `indentation_fidelity=0.91` matching the Boundary Closeout baseline; smoke `output/smoke_multiprofile_20260501_153101/` 10/10 GATE_PASS + UNIVERSAL_PASS including Greenhouse blind-test. See `docs/QUALITY_SNAPSHOT_2026-05-01.md` "Contextual Retrieval (Anthropic approach)".
- Refactor Boundary Closeout (Plan Section 5, steps 6+7): `complete` (2026-05-01). Removed `BatchProcessor.set_intelligence_metadata` deprecated `[V2.8-COMPAT]` API + dead CLI fallback + 5 deprecated-path tests; added `tests/test_pdf_conversion_plan.py::test_all_typed_policy_fields_round_trip_full_chain` as drift insurance. Static guards `2 passed`; focused boundary suite `93 passed`; full unit suite `480 passed, 1 skipped, 0 failed`; probe `output/probe_boundary_closeout_rag_guide/` AUDIT_PASS + UNIVERSAL_PASS (`indentation_fidelity=0.91`, 680 chunks); smoke `output/smoke_multiprofile_20260501_134909/` 10/10 GATE_PASS + UNIVERSAL_PASS. See `docs/QUALITY_SNAPSHOT_2026-05-01.md` "Refactor Boundary Closeout".
- Milestone 2 — Plan Control Plane: `complete` (2026-05-01). `PdfConversionPlan` promoted to typed policy object (`extraction_route` vocabulary, `hybrid_chunker_enabled`, `allow_page_level_visuals`, `asset_validation_policy`, `corruption_recovery_policy`); legacy bools preserved as derived `@property` bridges. Full unit suite `484 passed, 1 skipped, 0 failed`; focused suite `64 passed`; static guards `2 passed`; smoke `output/smoke_multiprofile_20260501_120514/` 10/10 GATE_PASS + UNIVERSAL_PASS; probe `output/probe_milestone2_rag_guide/` AUDIT_PASS + GATE_PASS + UNIVERSAL_PASS. See `docs/QUALITY_SNAPSHOT_2026-05-01.md`.
- Milestone 1 — Stabilize Extraction: `complete` (2026-05-01). RAG Guide unblocked (was 7200s timeout); per-element chunker guard added; Ayeva re-converted to `output/ayeva_qa_20260501/` with `indentation_fidelity=0.93`. See `docs/QUALITY_SNAPSHOT_2026-05-01.md`.
- Vision-Aided Front Matter Detection: `complete` (2026-04-30). See `docs/QUALITY_SNAPSHOT_2026-04-30.md`.
- Domain-Specific Search Priority: `complete` (2026-04-30). See `docs/QUALITY_SNAPSHOT_2026-04-30.md`.
- Coordinate Normalization Audit: `complete` (2026-04-30). See `docs/QUALITY_SNAPSHOT_2026-04-30.md`.
- Dependency metadata: installed Docling is `2.86.0` (now exact-pinned in `pyproject.toml`); `pip check` is blocked by pre-existing `torch 2.10.0 is not supported on this platform`.

## Immediate Next Work

Follow `docs/PROGRESS_CHECKLIST.md`.

Recommended sequence:

1. Fix Workstream A evidence durability and metric labeling.
2. Run local VLM comparison when the local inference server is reachable.
3. Fix Combat Aircraft text corruption.
4. Address classifier correctness.
5. Re-run broad conversion and update the quality snapshot.

## Must-Respect Constraints

- Python 3.10 only.
- Batch size must stay at or below 10 pages.
- Do not use `--profile-override` for acceptance runs.
- Do not add filename-specific or document-specific quality rules.
- OCR handles text; VLMs describe visuals only.
- BBoxes must remain normalized integer `[0,1000]`.
- Acceptance requires `GATE_PASS` plus `UNIVERSAL_PASS` across the smoke matrix.
