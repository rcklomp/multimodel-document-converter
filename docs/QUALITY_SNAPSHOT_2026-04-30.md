# Quality Snapshot 2026-04-30

## Vision-Aided Front Matter Detection

Status: `complete`

Evidence:
- Class: tracked + local-run
- Command: `conda run -n mmrag-v2 pytest tests/test_code_enrichment_decision.py tests/test_vision_aided_front_matter.py tests/test_cross_chunk_semantic_stitching.py -q`
- Input: tracked unit and bridge tests
- Output: terminal result
- Result: `41 passed`
- Tracked: yes
- Limitations: unit/bridge coverage only; acceptance requires smoke evidence below.

Evidence:
- Class: tracked + local-run
- Command: `conda run -n mmrag-v2 pytest -q`
- Input: tracked unit suite
- Output: terminal result
- Result: `356 passed, 1 skipped`
- Tracked: yes
- Limitations: does not prove corpus-level conversion quality by itself.

Evidence:
- Class: local-run
- Command: `conda run -n mmrag-v2 bash scripts/smoke_multiprofile.sh output/smoke_multiprofile_20260430_frontmatter_complete`
- Input: smoke matrix in `scripts/smoke_multiprofile.sh`
- Output: `output/smoke_multiprofile_20260430_frontmatter_complete/_summary.txt`
- Result: 10 non-empty rows, `min_chunks=8`, all rows `GATE_PASS` + `UNIVERSAL_PASS`; Greenhouse blind-test document included.
- Tracked: no
- Limitations: output artifacts are ignored; rerun the command above to reproduce.

Additional check:
- Command: `rg -n "CONVERT_ERROR|MISSING_JSONL|Completed with Errors|unexpected keyword|Error|Traceback|Exception" output/smoke_multiprofile_20260430_frontmatter_complete -S`
- Result: no matches.

## Shared PDF Extraction Plan

Status: `complete`

Evidence:
- Class: tracked + local-run
- Command: `conda run -n mmrag-v2 python -m pytest tests/test_pdf_conversion_plan.py tests/test_code_enrichment_decision.py tests/test_universal_pipeline.py -q`
- Input: tracked unit, negative, bridge, adapter, static guard, and UIR tests
- Output: terminal result
- Result: `73 passed`
- Tracked: yes
- Limitations: focused tests do not prove the full test suite or conversion smoke by themselves.

Evidence:
- Class: tracked + local-run
- Command: `conda run -n mmrag-v2 python -m pytest -q`
- Input: tracked full unit suite
- Output: terminal result
- Result: `412 passed, 1 skipped`
- Tracked: yes
- Limitations: unit suite does not prove corpus-level conversion behavior by itself.

Evidence:
- Class: local-run
- Command: `conda run -n mmrag-v2 bash scripts/smoke_multiprofile.sh`
- Input: smoke matrix in `scripts/smoke_multiprofile.sh`
- Output: `output/smoke_multiprofile_20260430_083922/_summary.txt`
- Result: all 10 rows `GATE_PASS` + `UNIVERSAL_PASS`; Greenhouse blind-test document included.
- Tracked: no
- Limitations: generated output artifacts are ignored; rerun the command above to reproduce.
