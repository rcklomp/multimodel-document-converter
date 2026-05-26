"""Unit tests for V3 identity-half gate + fusion helpers.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.2, §3.4, §7.8, §8.2.
"""

from __future__ import annotations

import math

import pytest

from mmrag_v2.retrieval.fusion_v3 import (
    DEFAULT_TOP_N_CHUNKS_PER_PAGE,
    PROSE_DEFAULT_VISUAL_WEIGHT,
    RetrievalDebugPayload,
    bounded_page_chunk_join,
    renormalize_on_leg_skip,
)
from mmrag_v2.v3_identity_gate import (
    IdentityGateReport,
    IdentityHashableChunk,
    compare_for_identity,
    normalize_chunk_for_identity,
)


# ---------------------------------------------------------------------------
# Identity-half gate
# ---------------------------------------------------------------------------


class TestNormalizeChunkForIdentity:
    def test_metadata_fields_dropped(self):
        chunk = {
            "doc_id": "abc",
            "chunk_id": "abc_001_text_deadbeef",  # metadata-only per §8.2
            "page_number": 1,
            "content": "hello",
            "schema_version": "2.7.0",  # metadata-only per §8.2
            "pipeline_version": "v2.16.0",  # metadata-only per §8.2
            "modality": "text",
        }
        proj = normalize_chunk_for_identity(chunk).payload
        assert "chunk_id" not in proj
        assert "schema_version" not in proj
        assert "pipeline_version" not in proj
        assert proj["content"] == "hello"
        assert proj["page_number"] == 1
        assert proj["modality"] == "text"

    def test_confidence_rounded_to_two_decimals(self):
        chunk = {
            "content": "x",
            "page_number": 1,
            "confidence_breakdown": {
                "layout_confidence": 0.123456789,
                "text_extraction_confidence": 0.987654321,
            },
        }
        proj = normalize_chunk_for_identity(chunk).payload
        assert proj["confidence_breakdown"]["layout_confidence"] == 0.12
        assert proj["confidence_breakdown"]["text_extraction_confidence"] == 0.99

    def test_confidence_rounding_handles_none(self):
        chunk = {
            "content": "x",
            "page_number": 1,
            "confidence_breakdown": {"ocr_confidence": None},
        }
        proj = normalize_chunk_for_identity(chunk).payload
        assert proj["confidence_breakdown"]["ocr_confidence"] is None

    def test_structural_flags_v2x_dict_form(self):
        # v2.X: Dict[str, bool] of flag → fired
        chunk = {
            "content": "x",
            "page_number": 1,
            "structural_flags": {
                "ocr_forced": True,
                "drop_cap_repaired": True,
                "bbox_iou_deduped": False,  # didn't fire → excluded
            },
        }
        proj = normalize_chunk_for_identity(chunk).payload
        # Only fired flags survive, sorted.
        assert proj["structural_flags"] == ["drop_cap_repaired", "ocr_forced"]

    def test_structural_flags_v3_set_form(self):
        # v3.0: Set[StructuralFlag] serializes to list
        chunk = {
            "content": "x",
            "page_number": 1,
            "structural_flags": ["ocr_forced", "drop_cap_repaired"],
        }
        proj = normalize_chunk_for_identity(chunk).payload
        # Sorted output is deterministic regardless of input order.
        assert proj["structural_flags"] == ["drop_cap_repaired", "ocr_forced"]

    def test_identity_hash_deterministic(self):
        chunk = {"content": "hello", "page_number": 1, "modality": "text"}
        h1 = normalize_chunk_for_identity(chunk).identity_hash()
        h2 = normalize_chunk_for_identity(chunk).identity_hash()
        assert h1 == h2

    def test_identity_hash_field_order_independent(self):
        # JSON sort_keys=True (§8.2 #1) → order doesn't matter
        c1 = {"content": "x", "page_number": 1, "modality": "text"}
        c2 = {"modality": "text", "page_number": 1, "content": "x"}
        h1 = normalize_chunk_for_identity(c1).identity_hash()
        h2 = normalize_chunk_for_identity(c2).identity_hash()
        assert h1 == h2


class TestCompareForIdentity:
    def _chunk(self, **fields):
        defaults = {
            "doc_id": "doc1",
            "page_number": 1,
            "content": "default content",
            "modality": "text",
            "chunk_type": "PARAGRAPH",
            "schema_version": "2.7.0",
        }
        defaults.update(fields)
        return defaults

    def test_all_match_perfect_identity_ratio(self):
        baseline = [self._chunk(content=f"chunk{i}") for i in range(10)]
        candidate = [self._chunk(content=f"chunk{i}") for i in range(10)]
        report = compare_for_identity(
            baseline_chunks=baseline, candidate_chunks=candidate
        )
        assert report.matched == 10
        assert report.identity_ratio == 1.0
        assert report.passes(0.95) is True

    def test_metadata_change_does_not_break_identity(self):
        # Per §8.2, chunk_id is metadata-only — changing it must not
        # break identity if content/modality/page/etc. match.
        baseline = [self._chunk(content="hello", chunk_id="OLD_ID")]
        candidate = [self._chunk(content="hello", chunk_id="NEW_ID")]
        report = compare_for_identity(
            baseline_chunks=baseline, candidate_chunks=candidate
        )
        assert report.identity_ratio == 1.0

    def test_below_threshold_fails(self):
        baseline = [self._chunk(content=f"chunk{i}") for i in range(10)]
        candidate = [
            self._chunk(content=f"chunk{i}")
            for i in range(5)
        ] + [
            self._chunk(content=f"DIFFERENT_chunk{i}")
            for i in range(5, 10)
        ]
        report = compare_for_identity(
            baseline_chunks=baseline, candidate_chunks=candidate
        )
        # 5 of 10 match (different content → different stable identity key
        # → counted as missing not differing)
        assert report.matched == 5
        assert report.identity_ratio == 0.5
        assert report.passes(0.95) is False

    def test_missing_chunks_tracked(self):
        baseline = [self._chunk(content=f"chunk{i}") for i in range(5)]
        candidate = [self._chunk(content=f"chunk{i}") for i in range(3)]
        report = compare_for_identity(
            baseline_chunks=baseline, candidate_chunks=candidate
        )
        assert len(report.missing_baseline_ids) == 2

    def test_new_candidate_chunks_tracked(self):
        baseline = [self._chunk(content=f"chunk{i}") for i in range(3)]
        candidate = [self._chunk(content=f"chunk{i}") for i in range(5)]
        report = compare_for_identity(
            baseline_chunks=baseline, candidate_chunks=candidate
        )
        assert len(report.new_candidate_ids) == 2

    def test_differing_when_keys_match_but_normalized_proj_differs(self):
        # Same stable key (same content) but different identity-relevant
        # field (different modality) → "differing" not "missing".
        baseline = [self._chunk(content="hello", modality="text")]
        candidate = [self._chunk(content="hello", modality="code")]
        report = compare_for_identity(
            baseline_chunks=baseline, candidate_chunks=candidate
        )
        assert report.matched == 0
        assert len(report.differing_baseline_ids) == 1

    def test_empty_baseline_passes_with_empty_candidate(self):
        report = compare_for_identity(baseline_chunks=[], candidate_chunks=[])
        assert report.identity_ratio == 1.0
        assert report.passes(0.95) is True

    def test_returns_report_type(self):
        report = compare_for_identity(baseline_chunks=[], candidate_chunks=[])
        assert isinstance(report, IdentityGateReport)


class TestUnicodeAndWhitespaceNormalization:
    """Charter §3.2 identity-half matching policy: content equality
    "modulo trailing whitespace", with NFC + CRLF→LF applied at the
    projection level (platform-noise-resistant) but internal whitespace
    preserved (chunker-output-strict)."""

    def test_nfc_normalization(self):
        # NFC vs NFD content with bytes that truly differ.
        nfc = b"caf\xc3\xa9".decode("utf-8")    # NFC single-codepoint U+00E9
        nfd = b"cafe\xcc\x81".decode("utf-8")   # NFD: e + combining acute U+0301
        assert nfc != nfd  # confirm bytes truly differ before normalization
        c1 = {"doc_id": "d", "page_number": 1, "content": nfc, "modality": "text"}
        c2 = {"doc_id": "d", "page_number": 1, "content": nfd, "modality": "text"}
        report = compare_for_identity(baseline_chunks=[c1], candidate_chunks=[c2])
        # NFC normalization at the projection level -> matched.
        assert report.matched == 1

    def test_crlf_normalized_to_lf(self):
        # Platform-level line-ending difference must not show as regression.
        c1 = {"doc_id": "d", "page_number": 1, "content": "a\r\nb", "modality": "text"}
        c2 = {"doc_id": "d", "page_number": 1, "content": "a\nb", "modality": "text"}
        report = compare_for_identity(baseline_chunks=[c1], candidate_chunks=[c2])
        assert report.matched == 1

    def test_trailing_whitespace_stripped(self):
        # Charter §3.2 "modulo trailing whitespace".
        c1 = {"doc_id": "d", "page_number": 1, "content": "hello world   ", "modality": "text"}
        c2 = {"doc_id": "d", "page_number": 1, "content": "hello world", "modality": "text"}
        report = compare_for_identity(baseline_chunks=[c1], candidate_chunks=[c2])
        assert report.matched == 1

    def test_internal_whitespace_difference_is_a_real_delta(self):
        # Charter §3.2 matching policy requires content equality (modulo
        # trailing whitespace only). Internal whitespace difference is a real
        # chunker-output change and must land in the explained-delta half.
        c1 = {"doc_id": "d", "page_number": 1, "content": "a    b", "modality": "text"}
        c2 = {"doc_id": "d", "page_number": 1, "content": "a b", "modality": "text"}
        report = compare_for_identity(baseline_chunks=[c1], candidate_chunks=[c2])
        # Keys collide (whitespace-collapsed); projections differ -> differing.
        assert report.matched == 0
        assert len(report.differing_baseline_ids) == 1


# ---------------------------------------------------------------------------
# Fusion re-normalization
# ---------------------------------------------------------------------------


class TestRenormalizeOnLegSkip:
    def test_no_skip_returns_weights_unchanged(self):
        weights = {"dense": 1.0, "sparse": 1.0, "visual": 0.1}
        out = renormalize_on_leg_skip(weights, skipped_legs=[])
        # No-skip case: all weights still present, L2-normalized.
        assert set(out.keys()) == {"dense", "sparse", "visual"}
        norm = math.sqrt(sum(v * v for v in out.values()))
        assert norm == pytest.approx(1.0)

    def test_skip_visual_drops_visual_key(self):
        weights = {"dense": 1.0, "sparse": 1.0, "visual": 0.1}
        out = renormalize_on_leg_skip(weights, skipped_legs=["visual"])
        assert "visual" not in out
        assert set(out.keys()) == {"dense", "sparse"}

    def test_skip_visual_produces_equal_text_weights(self):
        # Charter §3.4 #6 example: PROSE (1.0, 1.0, 0.1) with visual
        # skipped → (1.0, 1.0) L2-normalized → (0.707, 0.707).
        weights = {"dense": 1.0, "sparse": 1.0, "visual": 0.1}
        out = renormalize_on_leg_skip(weights, skipped_legs=["visual"])
        assert out["dense"] == pytest.approx(math.sqrt(0.5))
        assert out["sparse"] == pytest.approx(math.sqrt(0.5))

    def test_diagram_with_visual_skipped_equals_prose_with_visual_skipped(self):
        # Charter §3.4 #6 footer: "All profiles converge to equal text-
        # leg weights on leg skip — the profiles differ only in the
        # presence of the visual leg."
        prose = renormalize_on_leg_skip(
            {"dense": 1.0, "sparse": 1.0, "visual": 0.1},
            skipped_legs=["visual"],
        )
        diagram = renormalize_on_leg_skip(
            {"dense": 1.0, "sparse": 1.0, "visual": 0.4},
            skipped_legs=["visual"],
        )
        assert prose == pytest.approx(diagram)

    def test_all_legs_skipped_returns_empty(self):
        weights = {"dense": 1.0, "sparse": 1.0}
        out = renormalize_on_leg_skip(weights, skipped_legs=["dense", "sparse"])
        assert out == {}

    def test_prose_default_visual_weight_constant(self):
        # Charter §3.4 #5 table: PROSE visual weight 0.1.
        assert PROSE_DEFAULT_VISUAL_WEIGHT == 0.1


# ---------------------------------------------------------------------------
# Bounded page → chunk join
# ---------------------------------------------------------------------------


class TestBoundedPageChunkJoin:
    def test_default_top_n_is_three(self):
        # Charter §3.4 #4: top-N default = 3.
        assert DEFAULT_TOP_N_CHUNKS_PER_PAGE == 3

    def test_top_n_per_page_respected(self):
        # One visually-ranked page with 5 chunks → top 3 selected.
        visual_ranks = [("page_1", 1)]
        chunks = {
            "page_1": [
                ("c1", 0.1),
                ("c2", 0.5),  # top-3
                ("c3", 0.3),  # top-3
                ("c4", 0.9),  # top-3 (highest)
                ("c5", 0.2),
            ],
        }
        result = bounded_page_chunk_join(
            visual_page_ranks=visual_ranks, chunks_by_page=chunks
        )
        assert len(result) == 3
        chunk_ids = {chunk_id for chunk_id, _ in result}
        assert chunk_ids == {"c4", "c2", "c3"}

    def test_chunks_inherit_page_visual_rank(self):
        visual_ranks = [("page_2", 5), ("page_1", 1)]
        chunks = {
            "page_1": [("c_p1", 0.8)],
            "page_2": [("c_p2", 0.6)],
        }
        result = bounded_page_chunk_join(
            visual_page_ranks=visual_ranks, chunks_by_page=chunks
        )
        # c_p2 should inherit rank 5; c_p1 should inherit rank 1.
        as_dict = dict(result)
        assert as_dict["c_p1"] == 1
        assert as_dict["c_p2"] == 5

    def test_empty_page_skipped(self):
        # A visually-ranked page with no chunks contributes nothing.
        visual_ranks = [("page_1", 1)]
        chunks: dict = {"page_1": []}
        result = bounded_page_chunk_join(
            visual_page_ranks=visual_ranks, chunks_by_page=chunks
        )
        assert result == []

    def test_replaces_v0_3_broadcast_behavior(self):
        # Charter §3.4 #4 footer: "Draft 0.3's 'page scores propagated
        # to ALL chunks on that page' is corrected." This test confirms
        # we do NOT broadcast.
        visual_ranks = [("page_dense", 1)]
        chunks = {
            "page_dense": [("c1", 0.5), ("c2", 0.5), ("c3", 0.5), ("c4", 0.5), ("c5", 0.5)],
        }
        result = bounded_page_chunk_join(
            visual_page_ranks=visual_ranks, chunks_by_page=chunks,
            top_n=3,
        )
        # If broadcasting, all 5 would have rank 1. Bounded join → only 3.
        assert len(result) == 3


# ---------------------------------------------------------------------------
# RetrievalDebugPayload
# ---------------------------------------------------------------------------


class TestRetrievalDebugPayload:
    def test_default_construction(self):
        payload = RetrievalDebugPayload()
        assert payload.leg_scores == {}
        assert payload.weights_applied == {}
        assert payload.legs_skipped == []
        assert payload.fusion_input == []
        assert payload.bounded_join_decisions == []
        assert payload.rerank_input == []
        assert payload.rerank_output == []
        assert payload.fusion_vs_rerank_flips == []
        assert payload.profile_used is None
        assert payload.visual_collection_pin == ""
        assert payload.timing_ms == {}

    def test_populated_fields(self):
        payload = RetrievalDebugPayload(
            leg_scores={"dense": {"c1": 0.8}, "visual": {"p1": 0.6}},
            weights_applied={"dense": 0.707, "sparse": 0.707},
            legs_skipped=["visual"],
            profile_used="PROSE",
            visual_collection_pin="vidore/colqwen2.5-v0.2 @ render_dpi=200",
        )
        assert payload.legs_skipped == ["visual"]
        assert "visual" in payload.leg_scores  # raw scores still preserved
        assert payload.profile_used == "PROSE"
