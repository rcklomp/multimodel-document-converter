# Claude Prompt: Contextual Retrieval (Combined Plan #4)

**Task:** Implement Contextual Retrieval per Anthropic's research
(https://www.anthropic.com/news/contextual-retrieval) — embed each chunk with
a short hierarchical + neighbor prefix so isolated chunks retain document-level
meaning at retrieval time. This is **Combined Plan point #4**, the dependent
follow-on to the now-shipped Combined Plan points #1–#3 (Stabilize Extraction,
Plan Control Plane, Refactor Boundary Closeout) and to `PLAN_V2.7` Feature 6
("Contextual Retrieval (Anthropic approach)").

You are a Senior Python ETL Architect on the MMRAG V2 project at
`/Users/ronald/Projects/MM-Converter-V2.4.1`. The PDF extraction stack is
stable and locked in; you will not touch it. Your scope is **embed-time
only** — a tiny new builder, one new optional schema field, the ingestor wired
to use the builder, negative tests, and a drift guard that fails the day
someone bypasses the boundary.

---

## 1. Read First (in this order)

1. `docs/PROJECT_STATUS.md` — current objective and active baseline.
2. `docs/QUALITY_SNAPSHOT_2026-05-01.md` — closure evidence for Milestones 1+2
   and Refactor Boundary Closeout.
3. `docs/PROGRESS_CHECKLIST.md` — Document Understanding rows + Workstream B
   status.
4. `docs/PLAN_V2.7_DOCUMENT_UNDERSTANDING.md` — **Feature 6 ("Contextual
   Retrieval (Anthropic approach)")** is the canonical design rationale.
5. `docs/DECISIONS.md` — must contain the new "Contextual Retrieval" entry
   when this task closes; check first whether one already exists.
6. `AGENTS.md` — Level 0 invariants (`AGENT-VAL-01`, `AGENT-EVIDENCE-01`,
   `AGENT-STATUS-01`, `AGENT-DOCS-01`, `AGENT-TEST-01`).
7. `CLAUDE.md` — "Engineering Principles", "Test Contract Integrity", and the
   "Workstream B Code Enrichment Guardrail" (the contextualization runs
   *after* code-enrichment; do not perturb code-block content).
8. `docs/SRS_Multimodal_Ingestion_V2.5.md` — schema/QA contracts that
   constrain what `content` may and may not contain.

If any of those contradict this prompt, follow the docs and stop to flag the
contradiction.

---

## 2. Already Shipped — DO NOT REIMPLEMENT (verify first)

Combined Plan points #1–#3 are closed. Confirm each via the indicated artifact
**before touching anything**. If something listed below is not in fact shipped,
stop and flag it; do not silently extend scope.

| Item | Evidence to confirm |
|---|---|
| `PdfConversionPlan` typed policy object (route, chunker guards, asset/corruption policies, `__post_init__` validation) | `src/mmrag_v2/engines/pdf_plan.py` |
| `DoclingPdfAdapter` is the only construction site for `PdfPipelineOptions` / `DocumentConverter` | `src/mmrag_v2/engines/docling_adapter.py` + `tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter`, `::test_no_production_docling_imports_outside_adapter` |
| HybridChunker total-text + per-element guards | `src/mmrag_v2/processor.py` ~lines 2110-2160 + `tests/test_chunker_guard.py` |
| Bridge tests at every typed-policy boundary, plus consolidated round-trip drift insurance | `tests/test_pdf_conversion_plan.py::test_all_typed_policy_fields_round_trip_full_chain` and surrounding bridge tests |
| Latest smoke evidence: 10/10 `GATE_PASS` + `UNIVERSAL_PASS`, blind-test `Greenhouse Design and Control` included | `output/smoke_multiprofile_20260501_134909/_summary.txt` |
| `IngestionChunk.semantic_context` already carries `prev_text_snippet`, `next_text_snippet`, `parent_heading`, `breadcrumb_path` | `src/mmrag_v2/schema/ingestion_schema.py` ~line 320 |

If any of those is broken, the right fix is to restore the missing piece — not
to re-derive the whole refactor and not to absorb that scope into this task.

---

## 3. Design Invariants (Non-Negotiable)

These are the contract this work must preserve. Promote them to `DECISIONS.md`
under a single new "Contextual Retrieval (Anthropic approach)" entry; do not
add a parallel governance doc.

1. **AGENT-CONTEXTUAL-01 — Content immutability.** The canonical
   `IngestionChunk.content` is *never* mutated. The prefixes live in a
   separate, optional embedding-time field (`contextualized_text`) that is
   never read by QA, source-text validation, refiner threshold logic, or
   any chunk creator.
2. **AGENT-CONTEXTUAL-02 — Single embed-time builder.** The only function
   allowed to assemble contextualized text is
   `mmrag_v2.chunking.contextual_retrieval.build_contextualized_text`.
   Importers are: the embedding lane in `scripts/ingest_to_qdrant.py`, tests
   in `tests/test_contextual_retrieval.py`, and (optionally) a future RAG
   adapter — nothing else.
3. **AGENT-CONTEXTUAL-03 — QA isolation.** Markers `[Context: …]`,
   `[Heading: …]`, `[Previous: …]`, `[Next: …]`, and `[Modality: …]` MUST
   NOT appear in `IngestionChunk.content`, `metadata.refined_content`, the
   payload `text` field that goes into Qdrant, or any artifact that is fed
   back into `qa_conversion_audit.py` / `qa_universal_invariants.py` /
   `token_validator.py`.
4. **AGENT-CONTEXTUAL-04 — Length budget.** Per Anthropic, target ~50–100
   tokens (~200–400 chars) of context. Cap each `prev_text_snippet` and
   `next_text_snippet` to `MAX_CONTEXT_CHARS = 300`. Truncate; do not reflow.
   UTF-8 safety: truncate on a code-point boundary; never emit a bare
   continuation byte.
5. **AGENT-CONTEXTUAL-05 — Image lane untouched.** Image chunks already embed
   via `embed_image()` with the visual description as fallback. The
   contextualization path is for `modality in {"text", "table"}` only. Do not
   contextualize image-chunk content.
6. **AGENT-CONTEXTUAL-06 — Refiner ordering.** `PLAN_V2.7` Feature 6 calls out
   that the refiner must run *before* contextualization. The builder reads
   `metadata.refined_content` first, falls back to `chunk["content"]`. Do not
   run the refiner on contextualized text — that path is forbidden.
7. **AGENT-CONTEXTUAL-07 — Cache key safety.** If/when an embedding cache is
   keyed on text, it must key on the contextualized string, not the raw
   content (otherwise toggling `--no-contextual` would return stale vectors).
   Document this in the inline doc of `build_contextualized_text`; do not
   silently change cache shape.

---

## 4. What You Must Deliver

### Step 1 — State Audit (no code changes)

Produce a numbered audit (in your response, not a new doc) covering:

1. Run the static guards:
   ```bash
   conda run -n mmrag-v2 python -m pytest \
     tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter \
     tests/test_pdf_conversion_plan.py::test_no_production_docling_imports_outside_adapter -q
   ```
2. Confirm the schema already has, or does not have:
   - `IngestionChunk.contextualized_text: Optional[str]`
   - `SemanticContext.prev_text_snippet`, `next_text_snippet`,
     `parent_heading`, `breadcrumb_path` (used as the data source).
3. Confirm whether the following already exist:
   - `src/mmrag_v2/chunking/contextual_retrieval.py` exporting
     `build_contextualized_text` and `MAX_CONTEXT_CHARS`.
   - `tests/test_contextual_retrieval.py`.
   - `--no-contextual` CLI flag on `scripts/ingest_to_qdrant.py`.
   - `build_contextualized_text` import + use site in
     `scripts/ingest_to_qdrant.py` for text/table modalities.
4. Inventory potential leak sites — places that could accidentally write a
   contextualized string into a content field. Search and report:
   - Any production code that writes `[Context:` / `[Heading:` / `[Previous:`
     / `[Next:` / `[Modality:` (must be exactly zero outside the new module
     and tests).
   - Any production code that calls `build_contextualized_text` outside the
     allowed callers in AGENT-CONTEXTUAL-02.
   - Any chunk-creation path (`schema/ingestion_schema.create_text_chunk`,
     `create_table_chunk`, `create_image_chunk`) that reads
     `contextualized_text` (must be zero — that field is set only by the
     ingestor, if at all).
5. Confirm the embedding cache (if present) keys on the text actually sent
   to the embedder.

End the audit with a verdict per item: **DONE / NEEDS-CLEANUP / NEEDS-IMPL /
NEEDS-BRIDGE-TEST / OUT-OF-SCOPE**. Only then proceed.

### Step 2 — Implement / Reconcile

Drive each delta from the audit verdict. Do not redo work already shipped.

**2a. The builder — `src/mmrag_v2/chunking/contextual_retrieval.py`**

- Module docstring lists AGENT-CONTEXTUAL-01..07 verbatim and links to
  `https://www.anthropic.com/news/contextual-retrieval`.
- Public API: `MAX_CONTEXT_CHARS: int = 300`,
  `build_contextualized_text(content, *, breadcrumb_path=None,
  parent_heading=None, prev_text_snippet=None, next_text_snippet=None,
  modality="text") -> str`.
- Pure function. No I/O, no logging, no global state, no exceptions raised
  for empty inputs. Whitespace-only fields are skipped silently.
- Order of assembled prefixes (top → bottom), each on its own line:
  1. `[Context: A > B > C]` from non-empty `breadcrumb_path` joined with
     ` > ` after `.strip()`-ing each level.
  2. `[Heading: <parent_heading.strip()>]`.
  3. `[Previous: <prev_text_snippet.strip()[:MAX_CONTEXT_CHARS]>]` — UTF-8
     safe truncation (use `str` slicing on the already-decoded string; do
     not encode-then-slice).
  4. `[Next: <next_text_snippet.strip()[:MAX_CONTEXT_CHARS]>]`.
  5. `[Modality: <modality>]` only when `modality not in {"", "text"}`.
  6. Canonical `content` appended verbatim as the **final** line.
- Empty `content` is allowed; the function returns whatever prefixes are
  present plus a trailing empty line (or just the prefixes if nothing
  applies). It does not raise.
- Return value is `"\n".join(parts)`. No trailing newline.

**2b. Schema field — `src/mmrag_v2/schema/ingestion_schema.py`**

- On `IngestionChunk`, add `contextualized_text: Optional[str] = Field(
  default=None, description="Embedding-time contextualized text. NOT used
  for QA / source-text validation. Set only by the ingestor.")`.
- Do not add a validator. Do not add a `@computed_field`. Do not write to
  this field from any chunk-creation helper. The field exists so a
  pre-computed value can ride along in JSONL when desirable; the ingestor
  does not require it (it builds at embed time).
- Bump `SCHEMA_VERSION` only if other field changes warrant it; this single
  optional addition does not, by itself.
- Update the schema regression test (whichever asserts the field set;
  search for it) to include the new field.

**2c. Ingestor wiring — `scripts/ingest_to_qdrant.py`**

- Add `--no-contextual` argparse flag (boolean, default `False`).
- For `modality in {"text", "table"}`:
  - When `--no-contextual` is set, fall back to the pre-existing
    breadcrumb-only path: `f"{breadcrumb}\n{content}" if breadcrumb else
    content`. This is the v2.7.0 behavior — preserve it byte-for-byte so
    `--no-contextual` is a strict regression toggle.
  - Otherwise, build the embedding text via `build_contextualized_text(...)`
    using `metadata.hierarchy.breadcrumb_path`,
    `metadata.hierarchy.parent_heading`,
    `chunk.semantic_context.prev_text_snippet`, and
    `chunk.semantic_context.next_text_snippet`.
- For `modality == "image"`, do not touch the existing `embed_image(...)`
  flow.
- Read `content` exactly as today: `metadata.get("refined_content") or
  chunk.get("content", "")`. Refiner ordering (AGENT-CONTEXTUAL-06) is
  preserved by this read order.
- Build the Qdrant payload via `build_qdrant_payload(...)` unchanged. The
  payload `text` (or whatever the field is called) reads
  `chunk["content"]` / `metadata["refined_content"]` — never
  `text_to_embed`. Confirm with a grep.

### Step 3 — Tests (`tests/test_contextual_retrieval.py`)

These are executable requirements (`AGENT-TEST-01`). Add **at least these
classes** with **at least these named cases**. Do not weaken or rewrite;
add more cases if a class invites them.

- `TestContentImmutability`
  - `test_content_unchanged_after_call` — input `content` is byte-identical
    after the call; appears verbatim in the output.
  - `test_content_appears_verbatim_at_end` — output `endswith(content)`.
  - `test_content_with_special_chars` — `$`, `%`, `[`, `]`, parentheses
    survive untouched.
- `TestContextualPrefixSeparation`
  - `test_contextual_text_differs_from_content` when context is provided.
  - `test_each_marker_appears_only_in_prefix_lines` — assert
    `result.split("\n")[-1] == content` and no marker in `content`.
- `TestMissingContextHandling`
  - `test_no_context_returns_content_verbatim` — every kwarg `None` ⇒
    `result == content`.
  - `test_partial_context_skips_missing_lines` — only present fields render.
  - `test_whitespace_only_fields_are_skipped` — empty/whitespace breadcrumbs,
    headings, snippets do not produce empty marker lines.
- `TestContextLengthBounds`
  - `test_prev_snippet_truncated_to_MAX_CONTEXT_CHARS`.
  - `test_next_snippet_truncated_to_MAX_CONTEXT_CHARS`.
  - `test_long_breadcrumb_levels_are_not_truncated` (truncation is
    intentionally only on `prev`/`next` snippets — document this).
  - `test_utf8_truncation_does_not_split_codepoint` — non-ASCII (e.g.
    Japanese / accented Latin) snippet at exactly the boundary stays
    decodable.
- `TestQAValidationIntegrity`
  - `test_qa_audit_only_reads_content` — fixture `IngestionChunk` with
    `content="X"` and a fake `contextualized_text="[Context: foo]\nX"`;
    invoke the audit's content-reading helper (or a stand-in) and confirm
    it returns `"X"`, not the contextualized string.
  - `test_universal_invariants_only_reads_content` — same shape, against
    `qa_universal_invariants` reading helper.
  - `test_token_validator_only_reads_content` — same shape, against
    `validators/token_validator.py`.
- `TestContextualForImageChunks`
  - `test_modality_image_marker_added_when_called` (sanity for callers
    that *choose* to contextualize an image; does not imply the ingestor
    does).
  - `test_modality_text_emits_no_modality_marker`.
  - `test_modality_empty_string_emits_no_modality_marker`.
- `TestIntegrationSemanticContext`
  - `test_full_semantic_context_round_trip` — feed a real
    `SemanticContext` into `build_contextualized_text` (via the ingestor's
    actual extraction logic, called as a function or via a small helper)
    and assert the resulting string contains exactly the expected lines
    in the expected order.
- `TestIngestorBoundary` (new — drift insurance, mirror the round-trip
  test from the boundary closeout)
  - `test_ingest_no_contextual_flag_falls_back_to_breadcrumb_only` — load a
    minimal JSONL fixture, monkeypatch `embed_text`/`embed_image` to capture
    the input string, run with `--no-contextual`, assert the captured
    string is `f"{breadcrumb}\n{content}"` (or `content` when breadcrumb
    is empty), with **no** `[Context:` / `[Heading:` / `[Previous:` /
    `[Next:` / `[Modality:` markers.
  - `test_ingest_default_uses_contextualized_text` — same fixture without
    `--no-contextual`; assert captured string starts with `[Context:` and
    ends with `content`, and that the **payload** text fields contain
    only `content`.
  - `test_ingest_image_chunks_unaffected` — image chunk in fixture; assert
    `embed_image` is called and the captured fallback text is the visual
    description, not a contextualized string.
  - `test_no_marker_strings_in_payload_for_any_modality` — single
    consolidated guard: every payload built in the run must have zero
    occurrences of `[Context: `, `[Heading: `, `[Previous: `, `[Next: `,
    or `[Modality: `.

Pin a **target count** in the docstring of the module (e.g. "23 cases").
The acceptance command in Step 4 must show that exact number.

### Step 4 — Static Drift Guard (in `tests/test_contextual_retrieval.py`)

Add **one** AST-level guard test, modeled on
`tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter`:

```
test_no_contextual_marker_strings_in_production_code
```

- Walk every `*.py` under `src/mmrag_v2/` and every script under `scripts/`
  except the allowlist:
  `src/mmrag_v2/chunking/contextual_retrieval.py`,
  `scripts/ingest_to_qdrant.py`.
- Assert no production file contains the literal substrings `[Context: `,
  `[Heading: `, `[Previous: `, `[Next: `, `[Modality: `.
- Assert no production file (outside the allowlist + tests) calls
  `build_contextualized_text(`.
- Failure message must list the offending file + line number.

This is the single guard that fails loudly the day someone writes a
contextualized string into chunk content, into a refiner output, or into
a chunk-creation helper.

### Step 5 — Documentation

Per `AGENTS.md` §4 and `AGENT-DOCS-01` — extend existing docs, do not add
new governance docs.

1. `docs/DECISIONS.md` — add a "Contextual Retrieval (Anthropic approach)"
   entry: scope, AGENT-CONTEXTUAL-01..07 invariants, file locations,
   refiner-ordering rationale, `--no-contextual` rollback flag, embedding
   cache key consequence.
2. `docs/QUALITY_SNAPSHOT_<today>.md` — append a "Contextual Retrieval"
   section using the existing `Class / Command / Input / Output / Result /
   Tracked / Limitations` evidence-block format. Include four blocks
   (static guards, focused contextual suite, full unit suite, smoke +
   probe + ingest dry-run). Today's date format is the existing
   `YYYY-MM-DD`. If a snapshot for today already exists, extend it.
3. `docs/PROGRESS_CHECKLIST.md` — under the "Document Understanding Plan
   Items" block, mark Contextual Retrieval `[x]` with a one-line evidence
   reference.
4. `docs/PROJECT_STATUS.md` — extend the "Latest validation" list with one
   line for this closeout.
5. `docs/PLAN_V2.7_DOCUMENT_UNDERSTANDING.md` — set the Feature 6 block
   status to `complete` and add a one-line link to today's snapshot. Do
   not rewrite the design rationale; it stands.
6. `CHANGELOG.md` — append a `## v2.7.1 — Contextual Retrieval` entry
   summarizing the new field, the builder, the `--no-contextual` flag,
   and the AGENT-CONTEXTUAL invariants.

### Step 6 — Regression Evidence (per `AGENT-EVIDENCE-01`)

Run, in order, and capture the exact terminal results.

```bash
# Static guards (must remain green)
conda run -n mmrag-v2 python -m pytest \
  tests/test_pdf_conversion_plan.py::test_no_pipeline_options_construction_outside_adapter \
  tests/test_pdf_conversion_plan.py::test_no_production_docling_imports_outside_adapter -q

# Focused contextual suite + drift guard
conda run -n mmrag-v2 python -m pytest tests/test_contextual_retrieval.py -q

# Focused boundary suite (must not regress)
conda run -n mmrag-v2 python -m pytest \
  tests/test_pdf_conversion_plan.py tests/test_chunker_guard.py \
  tests/test_corruption_quarantine.py tests/test_blank_asset_quarantine.py \
  tests/test_finalization_bridge.py -q

# Full unit suite (must remain green; baseline 480 passed, 1 skipped)
conda run -n mmrag-v2 python -m pytest -q

# Targeted probe — same regression target as Milestones 1, 2 and the
# Boundary Closeout, so quality is comparable line-by-line.
conda run -n mmrag-v2 python -m mmrag_v2.cli process \
  "data/technical_manual/A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf" \
  --output-dir output/probe_contextual_retrieval_rag_guide \
  --batch-size 10 --vision-provider none --no-refiner --no-cache
conda run -n mmrag-v2 python scripts/qa_conversion_audit.py output/probe_contextual_retrieval_rag_guide/ingestion.jsonl
conda run -n mmrag-v2 python scripts/qa_universal_invariants.py output/probe_contextual_retrieval_rag_guide/ingestion.jsonl

# Ingestor dry-run — confirm contextual text reaches the embedder and not the
# payload. Must be a no-Qdrant variant; if no `--dry-run` exists, monkeypatch
# in a small harness script under tests/. The point is: prove no marker
# string lands in a payload field.
conda run -n mmrag-v2 python -m pytest \
  tests/test_contextual_retrieval.py::TestIngestorBoundary -q

# Acceptance smoke matrix (AGENT-VAL-01)
conda run -n mmrag-v2 bash scripts/smoke_multiprofile.sh
```

Acceptance:
- Static guards: `2 passed`.
- Focused contextual suite: every named case passes; the test count matches
  the number stated in the test module docstring.
- Focused boundary suite: pass with no skips other than the one already
  tracked.
- Full unit suite: at least the post-Boundary-Closeout baseline
  (`480 passed, 1 skipped, 0 failed`) **plus** the new contextual cases.
- Probe: `AUDIT_PASS` + `UNIVERSAL_PASS`. Chunk count and
  `infix_strict=0` must match the Boundary Closeout baseline
  `output/probe_boundary_closeout_rag_guide/` (680 chunks; `text=559`,
  `image=99`, `table=22`; `indentation_fidelity=0.91`). Contextualization
  is embed-time only and must not change conversion output.
- Ingestor boundary tests: prove no marker leaks into payload for any
  modality.
- Smoke matrix: 10/10 rows `GATE_PASS` + `UNIVERSAL_PASS`, including
  `Greenhouse Design and Control by Pedro Ponce` (`AGENT-VAL-01`).

If any row fails, **fix the implementation, never the test or the gate.**

---

## 5. Hard Constraints (non-negotiable)

1. **No mutation of `content` ever.** Rejection criterion: any write to
   `chunk.content`, `metadata.refined_content`, or any
   `create_*_chunk(content=...)` callsite that adds a `[Context:` /
   `[Heading:` / `[Previous:` / `[Next:` / `[Modality:` marker is a P0
   defect.
2. **No new Docling construction.** The static guards must stay green.
3. **No UIR rewrite, no element-mapping refactor.** This is an embed-time
   feature.
4. **No new typed `PdfConversionPlan` field.** Contextualization is not a
   PDF extraction policy.
5. **No new CLI flags on `mmrag-v2 process` / `mmrag-v2 batch`.** Only
   `scripts/ingest_to_qdrant.py` gains `--no-contextual`.
6. **No test weakening** (`AGENT-TEST-01`). If a pre-existing test
   conflicts with the implementation, the implementation is wrong; stop
   and document the proposed contract change.
7. **`--profile-override` is for debugging only** — never in acceptance
   evidence runs.
8. **Python 3.10 only.** Apple Silicon target. `docling==2.86.0` exact-pin.
9. **Batch size ≤10 pages** for any probe.
10. **No filename- or document-specific rules.** Zero hardcoded titles,
    word lists, or domain-specific overrides.
11. **No new governance doc.** Extend `DECISIONS.md`; do not create a new
    `*_GOVERNANCE.md` or parallel plan.
12. **No embedding-time I/O outside the embedder.** The builder is pure;
    if logging is desired, do it in the ingestor, not the builder.
13. **Refiner runs first.** Reading
    `metadata.refined_content or chunk["content"]` is the only allowed
    ordering. Never re-run the refiner on a contextualized string.
14. **Surgical scope only.** Touch only what this work demands.

---

## 6. Self-Audit Checklist (before reporting complete)

Code:
- [ ] `build_contextualized_text` is pure, has no I/O, has no global state,
      and raises no exceptions for empty/missing inputs.
- [ ] `MAX_CONTEXT_CHARS = 300` and is used for both `prev_text_snippet`
      and `next_text_snippet`.
- [ ] UTF-8 truncation is code-point-safe.
- [ ] `IngestionChunk.contextualized_text` exists and defaults to `None`.
      No chunk-creation helper writes it.
- [ ] `scripts/ingest_to_qdrant.py` accepts `--no-contextual` and routes
      text/table chunks accordingly; image lane unchanged.
- [ ] No production file outside the allowlist contains the literal
      substring `[Context: `, `[Heading: `, `[Previous: `, `[Next: `, or
      `[Modality: `.
- [ ] No production file outside the allowlist calls
      `build_contextualized_text(`.
- [ ] Qdrant payload `text` and `content` fields read from
      `chunk["content"]` / `metadata["refined_content"]` only.
- [ ] No new public API beyond `build_contextualized_text` and
      `MAX_CONTEXT_CHARS`. No new CLI flag on process/batch.

Tests:
- [ ] Every named case in `TestContentImmutability`,
      `TestContextualPrefixSeparation`, `TestMissingContextHandling`,
      `TestContextLengthBounds`, `TestQAValidationIntegrity`,
      `TestContextualForImageChunks`, `TestIntegrationSemanticContext`,
      and `TestIngestorBoundary` is present and passing.
- [ ] `test_no_contextual_marker_strings_in_production_code` static guard
      passes.
- [ ] All pre-existing bridge / static-guard / chunker-guard tests still
      pass unmodified.
- [ ] No test was rewritten to fit the implementation.
- [ ] Test count in module docstring matches actual count.

Evidence (`AGENT-EVIDENCE-01`):
- [ ] Static guards: `2 passed` (terminal output captured).
- [ ] Focused contextual suite: pass output captured, count matches
      docstring.
- [ ] Focused boundary suite: pass output captured.
- [ ] Full unit suite: pass output captured; total ≥ baseline + new cases.
- [ ] RAG Guide probe: `AUDIT_PASS` + `UNIVERSAL_PASS`, chunk counts
      identical to the Boundary Closeout baseline, output path recorded.
- [ ] Ingestor boundary tests pass: no marker leaks for any modality.
- [ ] Smoke matrix: 10/10 `GATE_PASS` + `UNIVERSAL_PASS`, summary path
      recorded, Greenhouse blind-test included.

Docs:
- [ ] `docs/DECISIONS.md` Contextual Retrieval entry added (or extended
      if pre-existing); AGENT-CONTEXTUAL-01..07 invariants stated.
- [ ] `docs/QUALITY_SNAPSHOT_<today>.md` updated/added with the four
      evidence blocks.
- [ ] `docs/PROGRESS_CHECKLIST.md` Document Understanding row updated.
- [ ] `docs/PROJECT_STATUS.md` "Latest validation" extended by one line.
- [ ] `docs/PLAN_V2.7_DOCUMENT_UNDERSTANDING.md` Feature 6 status set to
      `complete`.
- [ ] `CHANGELOG.md` v2.7.1 entry appended.
- [ ] No new governance doc.

Status word (`AGENT-STATUS-01`):
- [ ] Use `complete` only if local validation, durable evidence, the
      smoke matrix, AND the ingestor boundary tests all passed in this
      run. Otherwise use `implemented` plus a one-line blocker.

---

## 7. Edge Cases You Must Handle

These are the ways this feature can silently go wrong; cover them in code
and tests, not just in your head.

1. **Empty content.** Builder must not raise. Result is allowed to be
   prefixes plus a trailing empty line.
2. **Content already containing a `[Context:` or `[Heading:` substring.**
   This is a legitimate document occurrence (e.g. a programming book
   showing example log output). The builder must not strip or rewrite it
   in `content`; the static drift guard must explicitly allow this in
   `content` while still failing if it appears in any *other* production
   file. Add a positive test for this.
3. **Multilingual content.** Test with at least one non-ASCII case
   (Japanese or accented Latin) hitting `MAX_CONTEXT_CHARS` exactly.
4. **Markdown table content.** `modality == "table"` chunks have markdown
   pipes. The builder must not add a `[Modality: table]` marker
   *inside* the markdown — the marker is its own line, before `content`.
   Add a test using a representative markdown table.
5. **Very long heading.** A 500-char `parent_heading` is rare but
   possible from poor PDFs. The builder does not truncate headings;
   document this explicitly so reviewers don't add a hidden truncation
   later.
6. **Whitespace-only breadcrumb levels.** `breadcrumb_path = ["", "  ",
   "Sec 1"]` must render as `[Context: Sec 1]`, not `[Context:  >   >
   Sec 1]`.
7. **`semantic_context is None`.** Real chunks may not have a semantic
   context block. The ingestor reads with `chunk.get("semantic_context")
   or {}` and passes `None` for missing fields. Test this.
8. **`metadata.hierarchy is None`.** Same defensive read in the ingestor.
9. **`--no-contextual` toggle is byte-stable.** With the flag set, the
   embedded text must equal the v2.7.0 string exactly. Capture and
   diff in the test, do not just check substring.
10. **Embedding cache.** If a cache exists, toggling `--no-contextual`
    must not return v2.7.1 vectors for v2.7.0 queries (and vice versa).
    Either key on the actual embedded string, or invalidate the cache
    explicitly. The audit must report which one applies in this repo.

---

## 8. Execution Order

1. Read the docs in §1.
2. Audit (Step 1) — produce the verdict table.
3. Drive Step 2 deltas only from NEEDS-IMPL / NEEDS-CLEANUP verdicts.
4. Add the test cases (Step 3).
5. Add the static drift guard (Step 4).
6. Run all evidence commands (Step 6).
7. If anything fails, fix the implementation and rerun from the failing
   point.
8. Update docs (Step 5).
9. Report: short summary + verdict table + evidence paths + checklist
   state. State word per `AGENT-STATUS-01`.

If the audit shows the feature is already shipped and only the drift guard
+ docs are missing, that is a valid outcome — make those changes, capture
evidence, and close out. Do not invent work.

---

## 9. Out of Scope (Do Not Touch)

- The contextualization prompt itself (we are using a deterministic
  template, not an LLM-generated context per chunk; that is a future
  optimization out-of-scope for this task).
- BM25 / sparse retrieval. Anthropic's full Contextual Retrieval combines
  contextual embedding + contextual BM25; the BM25 lane is a separate
  workstream gated on a sparse-index decision in `DECISIONS.md`.
- The refiner pipeline. It runs as today; only the read order in the
  ingestor (`refined_content` → `content`) is preserved.
- The chunker. HybridChunker total-text and per-element guards are locked
  in by the Boundary Closeout and must not be touched.
- Any change to `IngestionChunk.content`, `metadata.refined_content`,
  `IngestionChunk.semantic_context`, the chunker, the OCR lane, or the
  VLM lane.
- Combined Plan point #5 (broad corpus reconversion). That happens *after*
  this work and only when smoke remains green.
