# Phase 5 Devlin Heading Diagnostic

Status scope: `validated-local` (2026-05-13). Smoke completed
`11/11 GATE_PASS` + `11/11 UNIVERSAL_PASS` in an independent audit
re-run after the initial environment-specific stall. See
`docs/PROJECT_STATUS.md` Phase 5 entry and `docs/PLAN_V2.10.md`
§Phase 5 "2026-05-13 close — `validated-local`".

## Summary

Phase 5 confirmed Candidate D: the Devlin failure was a combined
HybridChunker/export-shape problem plus heading-quality guard gap.

The first Phase 5 implementation raised HEADING coverage to 100%, but a later
audit showed it did so by propagating corrupted Docling section headers such as
`Type Type TypeTypeTypeType`, `GRAPH DATA: {`, and the generic bookmark
`Start`. That was rejected as a quality regression.

The corrected fix:

- keeps propagation at one production site, the export boundary, after the
  final chunk set is known;
- removes the export write-loop dict mutation that bypassed the
  `IngestionChunk` model;
- tightens the shared `is_valid_heading` guard against repeated-token artifacts
  and code/JSON payload shapes;
- prevents generic `Start` / `Front Matter` labels from seeding carry-forward
  state;
- keeps existing b429cb5 propagation behavior for valid headings.

## Probe Evidence

Manual probe command:

```bash
conda run -n mmrag-v2 python tests/manual/inspect_devlin_heading_propagation.py
```

Initial rc1 evidence showed 249 null-heading text chunks, concentrated in
HybridChunker page-split rows:

```text
0002: ... total_pages=365 batch_size=10 metadata_chunk_count=970 text_chunks=903
0146: !!NONE!! page=052 ... method='hybrid_chunker_pagesplit' ...
0995: batch=13 none_parent=027/033 ratio=0.818
1102: total_none_parent_rows=249
```

After the rejected coverage-only build, audit found that 428/790 chunks were
attributed to four garbage/generic headings. The corrective probe now shows
only seven null headings, all on front-matter pages before the first real
section signal:

```text
0006-0012: !!NONE!! pages=003-007 front-matter rows
0870: batch=01 none_parent=007/014 ratio=0.500
0871-0906: batches=02-37 none_parent=000/... ratio=0.000
0956: total_none_parent_rows=7
```

The boundary-window problem is closed across all later batches:

```text
0910-0946: batch=01..37 boundary_none_parent=000/... ratio=0.000
```

The corrected top-heading distribution no longer has one garbage heading
dominating the document:

```text
0958: ========== TOP PARENT HEADING DISTRIBUTION ==========
0959: heading_count=024 parent='6. Automating Evaluation'
0960: heading_count=021 parent='B. Parallelization and Scalability'
0961: heading_count=020 parent='Generation:'
0962: heading_count=017 parent='Each message'
0963-0978: remaining top headings are each <=16 chunks
```

The specific rejected audit headings are absent:

```text
0980: ========== REJECTED QUALITY HEADING CHECK ==========
0981: rejected_quality_heading_rows=0
```

## Fix Contract

The minimal corrected contract:

- Valid HybridChunker headings may carry forward across page-split siblings and
  batch boundaries.
- Explicit valid headings are not overwritten by a prior carry heading.
- Repeated-token artifacts and code/JSON-shaped section headers must not enter
  carry-forward state.
- Rejected heading chunks are not backfilled as ordinary empty siblings during
  page-context fill.
- Generic `Start` and `Front Matter` labels may remain local metadata where
  produced, but they must not seed forward propagation.
- Final JSONL hierarchy must be produced through `IngestionChunk` metadata, not
  by mutating serialized dicts in the write loop.

## Validation Evidence

Devlin reconvert:

```text
conda run -n mmrag-v2 python scripts/convert_books_v29.py --only Devlin_LLM_Agents --force --timeout 5400
2026-05-13 20:23:55 START Devlin_LLM_Agents
2026-05-13 20:27:55 DONE Devlin_LLM_Agents (859 lines)
```

Devlin strict QA after the corrected fix:

```text
conda run -n mmrag-v2 python scripts/qa_full_conversion.py \
  output/Devlin_LLM_Agents/ingestion.jsonl \
  --source-pdf "data/technical_manual/Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf" \
  --allow-warnings

HEADING: PASS
coverage: 783/790 (99%) [PASS]
null_headings: 7
qa_conversion_audit.py: PASS
qa_universal_invariants.py: UNIVERSAL_PASS
qa_ingestion_hygiene.py: PASS
```

The full wrapper still exits `QA_FAIL` on no-VLM image findings:

```text
VISION_PENDING: 68 image chunk(s) still pending VLM.
IMAGE_DESCRIPTION_UNUSABLE: 68/68 image chunks lack useful visual_description.
QA_FAIL: failures=2 warnings=2
```

That remains outside Phase 5's heading-propagation scope, but because the
completion standard requires `QA_PASS` or `QA_PASS_WITH_ADVISORIES`, this is
not sufficient by itself to close the phase.

Pytest evidence:

```text
conda run -n mmrag-v2 pytest tests/test_hybrid_chunker_heading_propagation.py -q
7 passed, 5 warnings

conda run -n mmrag-v2 pytest tests/test_cross_page_split_page_attribution.py -q
27 passed, 5 warnings

conda run -n mmrag-v2 pytest tests/test_vision_aided_front_matter.py -q
8 passed, 5 warnings

conda run -n mmrag-v2 pytest tests/ -x --ignore=tests/manual -q
860 passed, 14 skipped, 19 warnings
```

Smoke evidence is blocked in this run:

```text
bash scripts/smoke_multiprofile.sh output/smoke_phase5_quality_guard_20260513_stream
```

The smoke harness stalled on the first AIOS conversion row after emitting
TorchInductor lock debug output and had to be terminated. A standalone direct
conversion of that same first row completed in 42.9s, so this is recorded as a
smoke-run blocker rather than a Phase 5 pass. Phase 5 must remain unclosed
until the canonical smoke command completes.
