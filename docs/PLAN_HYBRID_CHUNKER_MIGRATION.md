# Plan: Migrate to Docling HybridChunker

## Why

The custom 30+ pass post-processing pipeline has reached its limits. Each fix for one document causes regressions on others. The root cause: we process Docling elements one-by-one and build chunks manually, ignoring Docling's document structure model.

Docling's `HybridChunker` (already installed) solves the three core problems:
1. **Mid-sentence splits** — uses `semchunk` for sentence-boundary-aware splitting
2. **Heading hierarchy** — uses document structure, not heuristic inference
3. **Heading appended to paragraph** — processes the `DoclingDocument` model, not raw text

## What Changes

| Component | Current | After Migration |
|-----------|---------|----------------|
| Text chunking | Custom element-by-element (processor.py:1939-2800) | `HybridChunker` from `docling-core` |
| Heading hierarchy | Heuristic inference from all-caps lines | Document structure from `DoclingDocument` |
| Sentence boundaries | Regex-based `_chunk_by_sentences()` | `semchunk` (transformer-based) |
| Chunk sizing | Custom `_apply_oversize_breaker()` | `HybridChunker(max_tokens=512)` |
| Chunk merging | Custom `_merge_mid_sentence_chunks()` | `HybridChunker(merge_peers=True)` |

## What Stays (no changes)

- Docling 2.66.0 for PDF extraction
- VLM page transcription for scanned documents
- VLM image descriptions (Qwen VL Max)
- Refiner for OCR cleanup (qwen-plus)
- Code detection and reflow (`_apply_code_hygiene`)
- Token validation (QA-CHECK-01)
- Perceptual image dedup (pHash)
- FULL-PAGE EDITORIAL image filter
- Intra-chunk text dedup (cover page repetition)
- Qdrant ingestion and search tools
- Config file support (~/.mmrag-v2.yml)
- All CLI options and commands

## Implementation Phases

### Phase 1: HybridChunker integration (core change)

**Goal:** Replace custom text chunking with HybridChunker while keeping all enrichment layers.

**Steps:**
1. In `processor.py`, after `self._converter.convert(file_path)`, pass the `DoclingDocument` to `HybridChunker` instead of iterating elements manually
2. Map `DocChunk` objects to `IngestionChunk` schema (chunk_id, modality, content, metadata, hierarchy, spatial)
3. Extract heading hierarchy from `DocChunk.meta.headings` instead of our `ContextStateV2` breadcrumb tracker
4. Keep image/table elements processed separately (not through the chunker — they need VLM/asset handling)

**Risk:** Medium. The chunk structure changes — downstream code that assumes our chunk format may need adjustment.

**Validation:** Re-run smoke test (10 docs × 10 pages). All must pass GATE + UNIVERSAL.

### Phase 2: Enrichment on HybridChunker output

**Goal:** Apply VLM, refiner, and code detection to HybridChunker chunks.

**Steps:**
1. VLM page transcription: runs BEFORE chunking (on scanned pages). Feed transcribed text into the `DoclingDocument` so HybridChunker sees it
2. Refiner: runs AFTER chunking on each text chunk. No change to refiner logic
3. Code detection: runs AFTER chunking. Reclassify paragraph chunks that contain code
4. Image descriptions: handled separately (not text chunks). Keep current VLM flow

**Risk:** Low. These are post-processing steps that operate on individual chunks.

**Validation:** Convert AIOS + Firearms + Kimothi. Compare quality metrics against current output.

### Phase 3: Remove deprecated post-processing passes

**Goal:** Clean up code that the HybridChunker makes unnecessary.

**Remove:**
- `_chunk_by_sentences()` — replaced by semchunk
- `_chunk_text_with_overlap()` — replaced by HybridChunker overlap
- `_apply_oversize_breaker()` — replaced by max_tokens
- `_merge_mid_sentence_chunks()` — replaced by merge_peers
- `_strip_trailing_headings()` — fixed by proper document structure
- `_deduplicate_chunk_overlap()` — replaced by HybridChunker's overlap control
- `_remove_near_duplicate_chunks()` — evaluate if still needed
- Heading inference (`_infer_headings_from_text()`) — replaced by DocChunk.meta.headings

**Keep:**
- `_apply_code_hygiene()` — code detection is domain-specific, not in HybridChunker
- `_dedup_intra_chunk_repeats()` — VLM transcription repeats are our problem
- `_normalize_chunk_text()` — PUA chars, spaced headings, de-hyphenation
- `_repair_cross_chunk_hyphenation()` — evaluate if HybridChunker handles this
- `_filter_no_visual_images()` — image filtering is our domain logic

**Risk:** Low. Removing code that's no longer called.

**Validation:** Full regression test. 300 unit tests must pass.

### Phase 4: Contextual retrieval (bonus)

**Goal:** Prepend chunk-specific context for better embedding, per Anthropic's research.

**Steps:**
1. Use `HybridChunker.contextualize()` to prepend heading context to each chunk
2. Or implement Anthropic's approach: use an LLM to generate a 50-100 token context prefix per chunk

**Risk:** Low. Additive improvement, no existing code changes.

**Validation:** Compare search quality before/after on Qdrant validation tool.

## Success Criteria

| Metric | Current (custom pipeline) | Target (HybridChunker) |
|--------|--------------------------|------------------------|
| Mid-sentence endings (Kimothi) | 157 / 693 (23%) | < 30 / ~400 (< 8%) |
| Bad headings (credit lines) | Requires pattern filter | 0 (structural) |
| Heading hierarchy depth | 2 levels (flat) | 3+ levels (from document) |
| Smoke test (10 docs) | 10/10 GATE_PASS | 10/10 GATE_PASS |
| Unit tests | 300 pass | 300 pass |

## Timeline

- Phase 1: 1 day (core integration + validation)
- Phase 2: 0.5 day (enrichment wiring)
- Phase 3: 0.5 day (cleanup)
- Phase 4: Optional (when needed for RAG quality)

## References

- [Docling Chunking Concepts](https://docling-project.github.io/docling/concepts/chunking/)
- [Docling HybridChunker Example](https://docling-project.github.io/docling/examples/hybrid_chunking/)
- [Docling Issue #2245: Header at end of page](https://github.com/docling-project/docling/issues/2245)
- [Docling Issue #2032: Mid-sentence splits](https://github.com/docling-project/docling/issues/2032)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [FloTorch 2026 Chunking Benchmark](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/)
