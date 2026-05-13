"""Regression tests for Phase 2 — TextIntegrityScout per-batch trigger.

See `docs/PLAN_V2.10.md` §"Phase 2" and
`src/mmrag_v2/validators/text_integrity_scout_trigger.py`. The trigger
exists because doc-level token-balance variance averages out on large
documents (e.g. Fluent_Python: 770 pages, 6 missing pages,
doc-variance ~0.2 %) and the scout never runs. These tests pin the
universal page-shape rule so it cannot regress to the old doc-only check.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from mmrag_v2.validators.text_integrity_scout_trigger import (
    BatchShortfall,
    any_batch_fires,
    classify_batches,
    MIN_MISSING_PAGES,
    MIN_PAGE_CHUNK_CHARS,
    MIN_PAGE_SOURCE_CHARS,
    MIN_SOURCE_CHARS,
    VARIANCE_PCT,
)


def _build_batches(total_pages: int, batch_size: int = 10) -> List[Tuple[int, int, int]]:
    """Bucket pages into (batch_index, start_page, end_page) triples."""
    out: List[Tuple[int, int, int]] = []
    bi = 0
    for start in range(1, total_pages + 1, batch_size):
        end = min(start + batch_size - 1, total_pages)
        out.append((bi, start, end))
        bi += 1
    return out


def test_per_batch_variance_fires_on_localized_drop() -> None:
    """Synthetic 8-page batch where 6 pages emit 0 chunks while source
    has ~500 chars/page. The scout must fire on that batch."""
    batches = [(0, 1, 8)]
    source = {p: 500 for p in range(1, 9)}
    # Pages 1, 2 emit normal content; pages 3-8 emit nothing.
    chunks = {1: 480, 2: 470, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

    fires, shapes = any_batch_fires(
        batches=batches,
        source_chars_per_page=source,
        chunk_chars_per_page=chunks,
    )

    assert fires is True
    assert len(shapes) == 1
    s = shapes[0]
    assert s.source_chars == 8 * 500
    assert s.chunk_chars == 480 + 470
    # 6 of 8 pages are >=MIN_PAGE_SOURCE_CHARS with <MIN_PAGE_CHUNK_CHARS chunk chars
    assert len(s.missing_pages) == 6
    assert s.fires()


def test_doc_level_variance_under_threshold_does_not_suppress_per_batch_trigger() -> None:
    """Simulate Fluent's shape: 770 pages, doc-variance ~0%, but one
    batch holds 6 zero-chunk pages. Negative regression — the trigger
    MUST NOT regress to the old doc-only check.
    """
    total_pages = 770
    batch_size = 10
    batches = _build_batches(total_pages, batch_size=batch_size)

    # Healthy baseline: every page has source==chunks (variance 0%).
    source: Dict[int, int] = {p: 2000 for p in range(1, total_pages + 1)}
    chunks: Dict[int, int] = {p: 2000 for p in range(1, total_pages + 1)}

    # Knock out 6 pages in one batch: pages 121-130 hold 5 normal pages
    # and pages 122, 124, 125, 126, 128, 129 emit 0 chunks.
    knocked_out = [122, 124, 125, 126, 128, 129]
    assert all(121 <= p <= 130 for p in knocked_out)
    for p in knocked_out:
        chunks[p] = 0

    # Doc-level variance: 6 of 770 pages missing 2000 chars each.
    total_src = sum(source.values())
    total_chk = sum(chunks.values())
    doc_variance = (total_src - total_chk) / total_src
    assert doc_variance < 0.05, (
        f"Test setup invariant: doc-level variance must be small (got {doc_variance:.4f}) "
        "so this scenario reproduces the missed-trigger bug Phase 2 closes."
    )

    fires, shapes = any_batch_fires(
        batches=batches,
        source_chars_per_page=source,
        chunk_chars_per_page=chunks,
    )

    assert fires is True, (
        "Per-batch trigger must fire on localized 6-page shortfall even "
        "though doc-level variance is well within tolerance."
    )

    firing = [s for s in shapes if s.fires()]
    assert len(firing) == 1
    target = firing[0]
    assert target.start_page == 121 and target.end_page == 130
    assert set(target.missing_pages) == set(knocked_out)


def test_clean_doc_does_not_trigger_scout() -> None:
    """Healthy multi-batch documents must not fire the per-batch trigger.

    Five representative shapes (one per profile category) — every batch
    has matching source/chunk char counts.
    """
    profile_shapes = {
        # (total_pages, batch_size, src_per_page, chunk_per_page)
        "technical_manual_book": (350, 10, 2200, 2100),
        "academic_whitepaper": (24, 10, 3500, 3400),
        "digital_magazine": (110, 10, 1800, 1750),
        "scanned_short": (8, 10, 1200, 1180),
        "tech_report": (60, 10, 2400, 2300),
    }
    for label, (n_pages, bs, src, chk) in profile_shapes.items():
        batches = _build_batches(n_pages, batch_size=bs)
        source = {p: src for p in range(1, n_pages + 1)}
        chunks = {p: chk for p in range(1, n_pages + 1)}
        fires, shapes = any_batch_fires(
            batches=batches,
            source_chars_per_page=source,
            chunk_chars_per_page=chunks,
        )
        assert fires is False, (
            f"Healthy {label} must not trigger scout (got firing batches: "
            f"{[s for s in shapes if s.fires()]})"
        )


def test_low_source_floor_suppresses_noisy_batches() -> None:
    """A batch with very little source text (e.g. cover sheet) must not
    trigger even if variance is high — this is the MIN_SOURCE_CHARS floor.
    """
    batches = [(0, 1, 10)]
    # Tiny source (total < MIN_SOURCE_CHARS) but a "missing" page.
    source = {1: 40, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    chunks = {p: 0 for p in source}

    fires, shapes = any_batch_fires(
        batches=batches,
        source_chars_per_page=source,
        chunk_chars_per_page=chunks,
    )
    assert fires is False
    assert shapes[0].source_chars < MIN_SOURCE_CHARS


def test_missing_pages_rule_alone_fires_below_variance_threshold() -> None:
    """A batch where the missing-page count satisfies rule (B) must
    fire even when the variance ratio is below the 30% threshold."""
    # Batch pp121-130: 8 healthy pages, 2 zero-chunk pages.
    # Make per-page source totals so that overall variance < 30%.
    batches = [(12, 121, 130)]
    source = {p: 2000 for p in range(121, 131)}
    chunks = {p: 2000 for p in range(121, 131)}
    chunks[125] = 0
    chunks[126] = 0
    total_src = sum(source.values())
    total_chk = sum(chunks.values())
    batch_variance = (total_src - total_chk) / total_src
    assert batch_variance < VARIANCE_PCT, (
        f"Test setup invariant: batch variance must be < {VARIANCE_PCT} "
        f"(got {batch_variance:.4f}) so rule (B) is the load-bearing trigger."
    )

    fires, shapes = any_batch_fires(
        batches=batches,
        source_chars_per_page=source,
        chunk_chars_per_page=chunks,
    )
    assert fires is True
    assert len(shapes[0].missing_pages) == 2


def test_classify_batches_returns_one_shape_per_batch() -> None:
    """classify_batches preserves input order and shape count."""
    batches = _build_batches(35, batch_size=10)
    source = {p: 1000 for p in range(1, 36)}
    chunks = {p: 1000 for p in range(1, 36)}

    shapes = classify_batches(
        batches=batches,
        source_chars_per_page=source,
        chunk_chars_per_page=chunks,
    )
    assert len(shapes) == len(batches) == 4
    for (bi, s, e), shape in zip(batches, shapes):
        assert shape.batch_index == bi
        assert shape.start_page == s
        assert shape.end_page == e


def test_thresholds_match_corpus_probed_defaults() -> None:
    """Defaults must match the values defended by
    `scripts/probe_phase2_scout_threshold.py`. Tightening these without
    re-running the probe is a regression."""
    assert VARIANCE_PCT == 0.30
    assert MIN_SOURCE_CHARS == 500
    assert MIN_MISSING_PAGES == 2
    assert MIN_PAGE_SOURCE_CHARS == 100
    assert MIN_PAGE_CHUNK_CHARS == 50


def test_batch_processor_helper_fires_on_localized_pdf(tmp_path) -> None:
    """End-to-end: `BatchProcessor._per_batch_shortfall_fires` opens a real
    PDF, computes per-page source chars, and reports True when one batch
    has localized shortfall while the others are healthy."""
    import fitz
    from mmrag_v2.batch_processor import BatchProcessor
    from mmrag_v2.schema.ingestion_schema import (
        FileType,
        HierarchyMetadata,
        create_text_chunk,
    )
    from mmrag_v2.utils.pdf_splitter import BatchInfo

    # Build a 20-page PDF: every page has substantial text.
    pdf_path = tmp_path / "synthetic_localized_shortfall.pdf"
    doc = fitz.open()
    body = (
        "Recovery acceptance synthesizes structural invariants. "
        "Each clause expands the page beyond the per-page floor. " * 10
    )
    for _ in range(20):
        page = doc.new_page()
        page.insert_text((72, 72), body)
    doc.save(str(pdf_path))
    doc.close()

    processor = BatchProcessor(
        output_dir=str(tmp_path / "out"),
        batch_size=10,
        vision_provider="none",
        enable_ocr=False,
    )
    processor._processed_pages = set(range(1, 21))

    # Emit chunks only for pages in batch 1 (pp1-10) — batch 2 starves.
    chunks = []
    for page_no in range(1, 11):
        chunks.append(
            create_text_chunk(
                doc_id="doc_synth",
                content=body,
                source_file=pdf_path.name,
                file_type=FileType.PDF,
                page_number=page_no,
                hierarchy=HierarchyMetadata(
                    parent_heading=None,
                    breadcrumb_path=["doc_synth", f"Page {page_no}"],
                    level=2,
                ),
            )
        )

    batches = [
        BatchInfo(
            batch_index=0,
            batch_path=pdf_path,
            start_page=1,
            end_page=10,
            page_count=10,
            page_offset=0,
        ),
        BatchInfo(
            batch_index=1,
            batch_path=pdf_path,
            start_page=11,
            end_page=20,
            page_count=10,
            page_offset=10,
        ),
    ]

    fires = processor._per_batch_shortfall_fires(
        chunks=chunks,
        batches=batches,
        pdf_path=pdf_path,
    )
    assert fires is True

    # Inverse: emit chunks for both batches; trigger must NOT fire.
    chunks_all = []
    for page_no in range(1, 21):
        chunks_all.append(
            create_text_chunk(
                doc_id="doc_synth",
                content=body,
                source_file=pdf_path.name,
                file_type=FileType.PDF,
                page_number=page_no,
                hierarchy=HierarchyMetadata(
                    parent_heading=None,
                    breadcrumb_path=["doc_synth", f"Page {page_no}"],
                    level=2,
                ),
            )
        )
    fires_clean = processor._per_batch_shortfall_fires(
        chunks=chunks_all,
        batches=batches,
        pdf_path=pdf_path,
    )
    assert fires_clean is False


def test_quarantine_keeps_legitimate_low_ratio_hex_escapes(tmp_path) -> None:
    """Phase 2 (2026-05-12): the corruption quarantine must not drop chunks
    that contain a small number of legitimate ``\\xHH`` hex escapes inside
    otherwise-clean prose. Fluent Python's chapter on encodings has Python
    REPL output like ``bytearray(b'caf\\xc3\\xa9')`` where ``\\xc3`` /
    ``\\xa9`` are the *subject of the prose*, not a CIDFont leak. With the
    single-match detector that previously gated the quarantine, every such
    chunk was being dropped post-scout and the strict gate registered
    MISSING_PAGES on pp125/126/136. Switching to the ratio-based
    `is_irreparably_corrupt` detector (artifact ratio + em-dash / C/S
    OCR-failure runs) keeps the chunk while still dropping genuine OCR
    garbage.
    """
    from mmrag_v2.batch_processor import BatchProcessor
    from mmrag_v2.schema.ingestion_schema import (
        FileType,
        HierarchyMetadata,
        create_text_chunk,
    )

    processor = BatchProcessor(
        output_dir=str(tmp_path),
        batch_size=10,
        vision_provider="none",
        enable_ocr=False,
    )

    fluent_p126_shape = (
        ">>> cafe_arr[-1:]\n"
        "bytearray(b'caf\\xc3\\xa9')\n"
        + "bytes can be built from a str, given an encoding. "
        "Each item is an integer in range(256). Slices of bytes are also "
        "bytes — even slices of a single byte. There is no literal syntax "
        "for bytearray: they are shown as bytearray() with a bytes literal "
        "as argument. " * 6
    )
    chunk = create_text_chunk(
        doc_id="doc_fluent_like",
        content=fluent_p126_shape,
        source_file="byteliteral.pdf",
        file_type=FileType.PDF,
        page_number=126,
        hierarchy=HierarchyMetadata(
            parent_heading=None,
            breadcrumb_path=["byteliteral", "Page 126"],
            level=2,
        ),
    )

    kept = processor._quarantine_corrupted_text_chunks([chunk])
    assert len(kept) == 1, (
        "Low-ratio hex-escape literals embedded in legitimate prose must "
        "not be quarantined; the surrounding text is the load-bearing signal."
    )
    assert kept[0].content == fluent_p126_shape


def test_batch_shortfall_records_missing_pages_for_observability() -> None:
    """The BatchShortfall must surface which pages drove the trigger so
    BatchProcessor logging can be specific (no opaque flags)."""
    batches = [(0, 1, 10)]
    source = {p: 1500 for p in range(1, 11)}
    chunks = {p: 1500 for p in range(1, 11)}
    chunks[3] = 0
    chunks[5] = 0
    chunks[9] = 0

    shapes = classify_batches(
        batches=batches,
        source_chars_per_page=source,
        chunk_chars_per_page=chunks,
    )
    s = shapes[0]
    assert isinstance(s, BatchShortfall)
    assert s.missing_pages == (3, 5, 9)
    assert s.fires()
