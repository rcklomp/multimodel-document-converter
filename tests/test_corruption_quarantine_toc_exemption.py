"""Phase B1 — TOC-page corruption-quarantine exemption (v2.9 plan).

`_quarantine_corrupted_text_chunks` at `batch_processor.py:829` drops
TEXT chunks whose content matches the strict-gate corruption regex
(CIDFont placeholders, em-dash runs, replacement chars, etc.). Phase
4 Step 3 added the OCR-failure patterns; Phase B1 of the strict-gate
recovery plan (`docs/PLAN_V2.9.md` §3 Phase B1) discovered that the
quarantine was over-firing on TOC chunks emitted by the Phase 1
dense-index router. Those chunks legitimately contain U+FFFD
replacement characters in dotted-leader regions because the source
publisher template's `.` glyph lacks a ToUnicode mapping. Quarantining
them discards entire TOC entries (Cronin 24 pages, Nagasubramanian 16,
Sekar 13, Chaubal 9 — verified via probe 2026-05-11).

This test pins the surgical exemption: TEXT chunks whose
`metadata.extraction_method` starts with `hybrid_chunker_pageskip`
(the Phase 1 router's output marker) are exempt from quarantine, even
when their content matches the corruption regex.

The corresponding negative case (Combat p66-style body corruption)
must still be quarantined — those chunks come from regular
`hybrid_chunker` extraction and contain the same regex matches but
are not from the trusted Phase 1 router.
"""
from __future__ import annotations

from pathlib import Path

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.processor import _sanitize_toc_index_text
from mmrag_v2.schema.ingestion_schema import FileType, create_text_chunk


def _bp() -> BatchProcessor:
    bp = BatchProcessor.__new__(BatchProcessor)
    bp._quarantine_corrupted_chunks = True
    return bp


# Real Cronin p5 TOC content shape: dotted-leader entries with embedded
# U+FFFD glyphs from the missing-ToUnicode `.` font. The replacement
# char ratio is well above the corruption regex threshold.
CRONIN_TOC_CONTENT = (
    "About the Author"
    + "�" * 60  # the dotted-leader rendered as replacement chars
    + "xxxiii\n"
    "Acknowledgments"
    + "�" * 60
    + "xxxv"
)


# Real Combat p66 corruption shape: long em-dash runs in a body text
# chunk (squadron-roster table where OCR couldn't decode typography).
COMBAT_P66_BODY_CORRUPTION = (
    "Squadron Note ——————|PuebloMemorialAirport,Colorado————"
    "CCCCCCCCCC SSSSSSSSSS"
)


def test_pageskip_toc_chunk_with_replacement_chars_is_preserved() -> None:
    chunk = create_text_chunk(
        content=CRONIN_TOC_CONTENT,
        doc_id="cronin_test_doc",
        source_file="Cronin.pdf",
        file_type=FileType.PDF,
        page_number=5,
        extraction_method="hybrid_chunker_pageskip",
        position=0,
    )
    bp = _bp()
    kept = bp._quarantine_corrupted_text_chunks([chunk])
    assert len(kept) == 1, "Phase 1 dense-index TOC chunk must survive quarantine"
    assert kept[0].chunk_id == chunk.chunk_id


def test_pageskip_source_pdf_variant_also_preserved() -> None:
    chunk = create_text_chunk(
        content=CRONIN_TOC_CONTENT,
        doc_id="adedeji_test_doc",
        source_file="Adedeji.pdf",
        file_type=FileType.PDF,
        page_number=300,
        extraction_method="hybrid_chunker_pageskip_source_pdf",
        position=0,
    )
    bp = _bp()
    kept = bp._quarantine_corrupted_text_chunks([chunk])
    assert len(kept) == 1, (
        "Phase 4 Step 2 back-index source-PDF fallback chunks "
        "must also survive quarantine"
    )


def test_body_corruption_from_regular_hybrid_chunker_is_dropped() -> None:
    chunk = create_text_chunk(
        content=COMBAT_P66_BODY_CORRUPTION,
        doc_id="combat_test_doc",
        source_file="Combat.pdf",
        file_type=FileType.PDF,
        page_number=66,
        extraction_method="hybrid_chunker",
        position=0,
    )
    bp = _bp()
    kept = bp._quarantine_corrupted_text_chunks([chunk])
    assert len(kept) == 0, (
        "Combat p66 body-text corruption from regular hybrid_chunker "
        "must continue to be dropped"
    )


def test_body_corruption_with_non_ffd_signature_still_dropped() -> None:
    # Plan v2.9 Phase B1 extension (2026-05-11): the U+FFFD-only
    # signature is now cleaned at chunk creation via the universal
    # `_collapse_replacement_chars` validator, so a chunk with
    # ONLY replacement chars no longer reaches the BatchProcessor's
    # corruption-quarantine — it arrives clean. The quarantine
    # still drops chunks whose corruption signature is something
    # else (CIDFont placeholders, em-dash runs, C/S filler runs);
    # this test pins that behavior.
    chunk = create_text_chunk(
        content=COMBAT_P66_BODY_CORRUPTION,  # em-dash + C/S runs
        doc_id="body_test_doc",
        source_file="Test.pdf",
        file_type=FileType.PDF,
        page_number=200,
        extraction_method="hybrid_chunker",
        position=0,
    )
    bp = _bp()
    kept = bp._quarantine_corrupted_text_chunks([chunk])
    assert len(kept) == 0, (
        "Non-pageskip chunk with em-dash/C-S filler corruption must "
        "still be quarantined; the universal U+FFFD collapse does NOT "
        "silence those signatures"
    )


def test_clean_pageskip_chunk_still_preserved() -> None:
    # Negative-control: a Phase 1 router chunk with no corruption
    # signature must still survive (no regression).
    chunk = create_text_chunk(
        content="Chapter 1 Introduction 1 What is RAG? 1 Why It Matters 5",
        doc_id="clean_test_doc",
        source_file="Clean.pdf",
        file_type=FileType.PDF,
        page_number=5,
        extraction_method="hybrid_chunker_pageskip",
        position=0,
    )
    bp = _bp()
    kept = bp._quarantine_corrupted_text_chunks([chunk])
    assert len(kept) == 1


def test_pageskip_toc_chunk_survives_finalize_drop_path() -> None:
    # The parallel-site audit (Phase A diagnostic §6) identified a second
    # corruption-drop path at `_drop_corrupted_chunks_before_metadata`
    # (the finalize-stage filter). The TOC exemption must apply there
    # too, otherwise Cronin TOC chunks pass the first filter but die at
    # the second one.
    chunk = create_text_chunk(
        content=CRONIN_TOC_CONTENT,
        doc_id="cronin_test_finalize",
        source_file="Cronin.pdf",
        file_type=FileType.PDF,
        page_number=5,
        extraction_method="hybrid_chunker_pageskip",
        position=0,
    )
    bp = _bp()
    kept = bp._drop_corrupted_chunks_before_metadata([chunk])
    assert len(kept) == 1, (
        "Phase 1 pageskip TOC chunks must survive the finalize-stage "
        "_drop_corrupted_chunks_before_metadata filter too"
    )


def test_body_corruption_still_dropped_at_finalize() -> None:
    chunk = create_text_chunk(
        content=COMBAT_P66_BODY_CORRUPTION,
        doc_id="combat_finalize",
        source_file="Combat.pdf",
        file_type=FileType.PDF,
        page_number=66,
        extraction_method="hybrid_chunker",
        position=0,
    )
    bp = _bp()
    kept = bp._drop_corrupted_chunks_before_metadata([chunk])
    assert len(kept) == 0, (
        "Combat p66 body corruption must continue to drop at the finalize "
        "stage; the pageskip exemption is extraction-method-scoped, not "
        "content-scoped"
    )


def test_sanitizer_collapses_replacement_char_runs() -> None:
    # The Phase 1 dense-index router calls _sanitize_toc_index_text
    # on the line-joined TOC text before chunk emission. The strict
    # gate's qa_conversion_audit / qa_universal_invariants scripts
    # have their own corruption regexes that fire on raw U+FFFD; the
    # sanitizer must collapse those runs so the chunk lands clean.
    raw = (
        "About the Author"
        + "�" * 60
        + "xxix\n"
        + "Vision and Image Generation"
        + " "
        + "�" * 40
        + " 16"
    )
    cleaned = _sanitize_toc_index_text(raw)
    assert "�" not in cleaned, (
        "U+FFFD runs must be normalized at the producer site so "
        "downstream gates do not flag the chunk as corrupt"
    )
    assert "About the Author xxix" in cleaned, (
        "Sanitizer must preserve the TOC entry's title + page number"
    )
    assert "Vision and Image Generation 16" in cleaned


def test_sanitizer_leaves_normal_toc_text_untouched() -> None:
    # Negative-control: a clean TOC entry (the Phase 1 baseline) must
    # pass through unchanged.
    raw = "Chapter 1 Introduction 1\nWhat is RAG? 5\nWhy It Matters 9"
    cleaned = _sanitize_toc_index_text(raw)
    assert "Chapter 1 Introduction 1" in cleaned
    assert "Why It Matters 9" in cleaned


def test_universal_sanitizer_applies_at_chunk_creation_regardless_of_extraction_method() -> None:
    # Plan v2.9 Phase B1 extension (2026-05-11): the U+FFFD collapse
    # also runs as a universal field-validator on `content` at chunk
    # creation. Any text chunk — regardless of extraction_method or
    # producer site — emits clean content. This pins the wider scope.
    body_text_with_replacement = (
        "The Bücher" + "�" * 30 + "und newspaper reports 99 percent"
    )
    chunk = create_text_chunk(
        content=body_text_with_replacement,
        doc_id="universal_test",
        source_file="Body.pdf",
        file_type=FileType.PDF,
        page_number=42,
        extraction_method="hybrid_chunker",  # NOT pageskip
        position=0,
    )
    assert "�" not in chunk.content
    assert "Bücher" in chunk.content
    assert "newspaper" in chunk.content


def test_universal_sanitizer_does_not_touch_clean_content() -> None:
    clean = "The quick brown fox jumps over the lazy dog."
    chunk = create_text_chunk(
        content=clean,
        doc_id="clean_test",
        source_file="Clean.pdf",
        file_type=FileType.PDF,
        page_number=1,
        extraction_method="hybrid_chunker",
        position=0,
    )
    assert chunk.content == clean


def test_sanitizer_does_not_strip_legitimate_text_content() -> None:
    # Negative-control: a body-text fragment with a single
    # legitimate U+FFFD (e.g. a foreign-language passage where one
    # character failed to encode) is collapsed, but surrounding
    # words survive.
    raw = "The Bücher�und newspaper reports a result of 99 percent"
    cleaned = _sanitize_toc_index_text(raw)
    assert "Bücher" in cleaned
    assert "und newspaper" in cleaned or "und" in cleaned
    assert "�" not in cleaned


def test_quarantine_disabled_returns_all_chunks() -> None:
    # The quarantine has a kill-switch via _quarantine_corrupted_chunks.
    # When disabled, even corruption matches survive.
    chunk = create_text_chunk(
        content=COMBAT_P66_BODY_CORRUPTION,
        doc_id="killswitch_test_doc",
        source_file="Test.pdf",
        file_type=FileType.PDF,
        page_number=1,
        extraction_method="hybrid_chunker",
        position=0,
    )
    bp = _bp()
    bp._quarantine_corrupted_chunks = False
    kept = bp._quarantine_corrupted_text_chunks([chunk])
    assert len(kept) == 1
