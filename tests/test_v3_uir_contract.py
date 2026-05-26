"""Unit tests for the V3.0 UIR contract types.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.2.

These tests pin the additive UIR contract introduced in the foundation
session. They protect against accidental contract drift while the Phase A
refactor is in flight.

Co-existence with v2.X types is tested explicitly: the v2.16 production
types (ElementType, Element, UniversalPage) remain importable and the
v3.0 additions do not collide.
"""

from __future__ import annotations

import pytest

from mmrag_v2.universal.intermediate import (
    # v2.X types — must still be importable
    Element,
    ElementType,
    UniversalDocument,
    UniversalPage,
    # v3.0 additions
    ConfidenceBreakdown,
    CoordinateFrame,
    ExtractionWarning,
    Locator,
    LocatorType,
    Modality,
    StructuralFlag,
    UIRChunk,
)


class TestModality:
    def test_five_v3_values(self):
        names = {m.name for m in Modality}
        assert names == {"TEXT", "IMAGE", "TABLE", "CODE", "FORM"}

    def test_modality_distinct_from_elementtype(self):
        # ElementType has three values; Modality has five.
        # The Phase A migration (Charter §7.1) widens TEXT into TEXT/CODE/FORM
        # and the foundation session keeps both vocabularies alive.
        assert {e.name for e in ElementType} == {"TEXT", "IMAGE", "TABLE"}
        assert "CODE" in {m.name for m in Modality}
        assert "FORM" in {m.name for m in Modality}


class TestStructuralFlag:
    def test_closed_vocabulary_includes_partial_code_variants(self):
        # Charter §3.2: in-block exists in v2.16; cross-page is the v3.0
        # activation of the inert v2.16 case.
        names = {f.name for f in StructuralFlag}
        assert "PARTIAL_CODE_IN_BLOCK" in names
        assert "PARTIAL_CODE_CROSS_PAGE" in names
        # Reserved for v3.1+ but present in the enum so flag-additive gate
        # has the value to diff against:
        assert "PARTIAL_TABLE_CROSS_PAGE" in names

    def test_legacy_heuristic_flags_present(self):
        # v2.16 heuristics get a typed-enum home (Charter §3.2):
        names = {f.name for f in StructuralFlag}
        for name in (
            "DROP_CAP_REPAIRED",
            "READING_ORDER_REPAIRED",
            "OCR_FORCED",
            "OCR_FALLBACK",
            "BBOX_IOU_DEDUPED",
            "TRAILING_PREPOSITION_HEALED",
            "PICTURE_CLASSIFICATION_LABEL",
        ):
            assert name in names, f"StructuralFlag missing {name}"


class TestLocator:
    def test_bbox_locator_happy_path(self):
        loc = Locator(
            type=LocatorType.BBOX,
            bbox=[10, 20, 100, 200],
            page_number=1,
            coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
        )
        assert loc.bbox == [10, 20, 100, 200]
        assert loc.coordinate_frame == CoordinateFrame.PDF_PAGE_PORTRAIT

    def test_bbox_locator_requires_bbox(self):
        with pytest.raises(ValueError, match="requires bbox"):
            Locator(type=LocatorType.BBOX)

    def test_bbox_must_be_four_ints(self):
        with pytest.raises(ValueError, match="must be 4 ints"):
            Locator(type=LocatorType.BBOX, bbox=[10, 20, 30])

    def test_bbox_must_be_normalized(self):
        # AGENTS.md REQ-COORD-01: coords in [0, 1000]
        with pytest.raises(ValueError, match=r"out of range \[0,1000\]"):
            Locator(type=LocatorType.BBOX, bbox=[10, 20, 100, 1500])

    def test_bbox_must_be_integers(self):
        with pytest.raises(TypeError, match="must be int"):
            Locator(type=LocatorType.BBOX, bbox=[10.0, 20.0, 100.0, 200.0])  # type: ignore[list-item]

    def test_flow_offset_locator_requires_path(self):
        with pytest.raises(ValueError, match="requires path"):
            Locator(type=LocatorType.FLOW_OFFSET)
        # Happy path: path supplied
        loc = Locator(type=LocatorType.FLOW_OFFSET, path="cfi:/2/4/6")
        assert loc.path == "cfi:/2/4/6"

    def test_dom_path_locator_requires_path(self):
        with pytest.raises(ValueError, match="requires path"):
            Locator(type=LocatorType.DOM_PATH)


class TestConfidenceBreakdown:
    def test_default_all_none(self):
        c = ConfidenceBreakdown()
        assert c.layout_confidence is None
        assert c.text_extraction_confidence is None
        assert c.ocr_confidence is None
        assert c.classification_confidence is None
        assert c.applicable == set()

    def test_applicable_rejects_unknown_field_name(self):
        with pytest.raises(ValueError, match="unknown field name"):
            ConfidenceBreakdown(applicable={"made_up_score"})

    def test_value_out_of_range_rejected(self):
        with pytest.raises(ValueError, match=r"out of range \[0,1\]"):
            ConfidenceBreakdown(layout_confidence=1.5)
        with pytest.raises(ValueError, match=r"out of range \[0,1\]"):
            ConfidenceBreakdown(ocr_confidence=-0.1)

    def test_single_sentinel_semantics(self):
        # Charter §3.2: a field in `applicable` with value None means
        # "applicable but unavailable" — request a re-extraction or
        # treat as missing data. Test that combination is constructible.
        c = ConfidenceBreakdown(
            layout_confidence=None,
            applicable={"layout_confidence"},
        )
        assert c.layout_confidence is None
        assert "layout_confidence" in c.applicable


class TestUIRChunk:
    def _make_chunk(self, **overrides):
        defaults = dict(
            modality=Modality.TEXT,
            content="hello",
            locator=Locator(
                type=LocatorType.BBOX,
                bbox=[0, 0, 100, 100],
                page_number=1,
                coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
            ),
            confidence=ConfidenceBreakdown(),
            extraction_method="docling_direct",
            extraction_engine_version="docling-2.86.0",
        )
        defaults.update(overrides)
        return UIRChunk(**defaults)

    def test_minimal_construction(self):
        chunk = self._make_chunk()
        assert chunk.modality == Modality.TEXT
        assert chunk.content == "hello"
        assert chunk.uir_version == "3.0"
        assert chunk.sanitization_status == "not_applied"
        assert chunk.continuation_group_id is None
        assert chunk.structural_flags == set()
        assert chunk.extraction_warnings == []

    def test_sanitization_status_accepted(self):
        chunk = self._make_chunk(
            content="cleaned",
            content_original="raw",
            content_sanitized="cleaned",
            sanitizer_model_id="qwen2.5-14b-fp8",
            sanitizer_prompt_version="abc123",
            sanitization_status="accepted",
        )
        assert chunk.sanitization_status == "accepted"
        assert chunk.content == "cleaned"
        assert chunk.content_original == "raw"

    def test_sanitization_status_rejected_with_guard_name(self):
        chunk = self._make_chunk(
            sanitization_status="rejected:edit_distance",
        )
        assert chunk.sanitization_status == "rejected:edit_distance"

    def test_sanitization_status_skipped_unreachable(self):
        chunk = self._make_chunk(
            sanitization_status="skipped:endpoint_unreachable",
        )
        assert chunk.sanitization_status == "skipped:endpoint_unreachable"

    def test_invalid_sanitization_status_rejected(self):
        with pytest.raises(ValueError, match="sanitization_status"):
            self._make_chunk(sanitization_status="garbage")

    def test_continuation_group_id_uuid(self):
        # Charter §3.2 (Draft 0.5): cross-page sibling joins require a
        # shared UUID across the half-chunks.
        cg = UIRChunk.new_continuation_group_id()
        # UUIDv4 → 36 chars including hyphens
        assert len(cg) == 36
        assert cg.count("-") == 4
        c1 = self._make_chunk(continuation_group_id=cg)
        c2 = self._make_chunk(continuation_group_id=cg)
        assert c1.continuation_group_id == c2.continuation_group_id

    def test_structural_flags_as_set(self):
        chunk = self._make_chunk(
            structural_flags={
                StructuralFlag.PARTIAL_CODE_CROSS_PAGE,
                StructuralFlag.CROSS_PAGE_SPLIT,
            }
        )
        assert len(chunk.structural_flags) == 2
        # Flags are additive: adding more does not remove existing.
        chunk.structural_flags.add(StructuralFlag.READING_ORDER_REPAIRED)
        assert len(chunk.structural_flags) == 3


class TestExtractionWarning:
    def test_severities(self):
        # Literal["info", "warn", "error"]; runtime is permissive (dataclass
        # doesn't enforce Literal), but the contract is documented.
        for sev in ("info", "warn", "error"):
            w = ExtractionWarning(code="X", severity=sev, message="m")  # type: ignore[arg-type]
            assert w.severity == sev


class TestV2xCoexistence:
    """The Phase A migration deletes ElementType at end of Phase A
    (Charter §7.1). During the foundation session and Phase A development
    both vocabularies must live in the module without colliding."""

    def test_v2x_types_still_importable(self):
        # If any of these imports failed, the module-level test_import
        # at the top of this file would have already raised.
        assert ElementType.TEXT.value == "text"
        # The v2.X Element does NOT carry the new fields:
        assert not hasattr(Element, "modality")
        assert not hasattr(Element, "structural_flags")
        # v3.0 UIRChunk does NOT carry the v2.X Element fields:
        assert "raw_image" not in UIRChunk.__dataclass_fields__
        assert "extraction_method" in UIRChunk.__dataclass_fields__

    def test_v2x_element_and_v3_uirchunk_are_distinct_types(self):
        assert Element is not UIRChunk
        assert ElementType is not Modality

    def test_v2x_universaldocument_still_constructible(self):
        # The v2.X production type chain is unaffected.
        page = UniversalPage(
            page_number=1,
            elements=[],
            classification=UniversalPage.classify_page(150),
            dimensions=(800, 1000),
        )
        # UniversalDocument default-constructible (doc_id is required):
        doc = UniversalDocument(
            doc_id="abc123",
            source_file="x.pdf",
            file_type="pdf",
            pages=[page],
        )
        assert doc.doc_id == "abc123"
        assert len(doc.pages) == 1
