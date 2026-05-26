"""Unit tests for the V3.0 sanitization package scaffolding.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.3.

Tests cover the FUNCTIONAL guards (1, 2-partial, 3, 4, 5, 6, 8) +
cache-key contract + orchestrator dispatch + sentinel accounting +
prompt template hash.

Guard 7 (entity_relation) is a documented stub — tested only for the
sentinel return shape so its absence is observable rather than silent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mmrag_v2.sanitization import (
    SanitizationMode,
    SanitizationResult,
    sanitize_chunk,
)
from mmrag_v2.sanitization.guards import (
    GuardResult,
    evaluate_code_span,
    evaluate_dedup_ratio,
    evaluate_edit_distance,
    evaluate_entity_relation,
    evaluate_numeric_entity,
    evaluate_order_preservation,
    evaluate_prompt_boundary,
    evaluate_token_alignment,
)
from mmrag_v2.sanitization.guards.dedup_ratio import compute_dedup_ratio
from mmrag_v2.sanitization.golden_set import (
    GOLDEN_SET_SIZE,
    DominanceScore,
    GoldenEntry,
    load_golden_set,
    score_against_golden_set,
)
from mmrag_v2.sanitization.graceful_degradation import (
    SENTINEL_RATE_DEGRADED_THRESHOLD,
    SentinelAccount,
    is_endpoint_reachable,
)
from mmrag_v2.sanitization.llm_sanitizer import (
    FileBackedSanitizationCache,
    build_cache_key,
    compute_content_hash,
    compute_context_hash,
    sanitize_via_llm,
)
from mmrag_v2.sanitization.prompts import (
    PROMPT_TEMPLATE,
    prompt_version,
    render,
)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TestOrchestrator:
    def test_off_mode_returns_unchanged(self):
        result = sanitize_chunk(
            raw_content="hello world",
            mode=SanitizationMode.OFF,
        )
        assert isinstance(result, SanitizationResult)
        assert result.content == "hello world"
        assert result.status == "not_applied"
        assert result.content_original is None

    def test_non_off_modes_emit_foundation_session_sentinel(self):
        for mode in (
            SanitizationMode.LLM,
            SanitizationMode.HEURISTIC,
            SanitizationMode.BOTH_AND_DIFF,
        ):
            result = sanitize_chunk(raw_content="hi", mode=mode)
            assert result.status == "skipped:foundation_session"
            assert result.content == "hi"
            assert result.content_original == "hi"


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_content_hash_deterministic(self):
        h1 = compute_content_hash("hello")
        h2 = compute_content_hash("hello")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_content_hash_distinguishes(self):
        assert compute_content_hash("a") != compute_content_hash("b")

    def test_context_hash_uses_all_three_components(self):
        h_base = compute_context_hash(
            prev_chunk_content="prev",
            next_chunk_content="next",
            detected_lang="en",
        )
        h_diff_prev = compute_context_hash(
            prev_chunk_content="DIFFERENT",
            next_chunk_content="next",
            detected_lang="en",
        )
        h_diff_next = compute_context_hash(
            prev_chunk_content="prev",
            next_chunk_content="DIFFERENT",
            detected_lang="en",
        )
        h_diff_lang = compute_context_hash(
            prev_chunk_content="prev",
            next_chunk_content="next",
            detected_lang="de",
        )
        assert len({h_base, h_diff_prev, h_diff_next, h_diff_lang}) == 4

    def test_context_hash_missing_components_yields_consistent_result(self):
        # All-None components must not crash and must be deterministic.
        h1 = compute_context_hash(
            prev_chunk_content=None, next_chunk_content=None, detected_lang=None
        )
        h2 = compute_context_hash(
            prev_chunk_content=None, next_chunk_content=None, detected_lang=None
        )
        assert h1 == h2

    def test_cache_key_filename(self):
        k = build_cache_key(
            raw_content="hello",
            prev_chunk_content="prev",
            next_chunk_content="next",
            detected_lang="en",
            model_id="m1",
            prompt_version="p1",
        )
        fn = k.cache_filename()
        # Format: content[:16] + "_" + context[:8] + ".json"
        assert fn.endswith(".json")
        assert "_" in fn

    def test_cache_roundtrip(self, tmp_path: Path):
        cache = FileBackedSanitizationCache(tmp_path / "cache")
        key = build_cache_key(
            raw_content="hello",
            prev_chunk_content=None,
            next_chunk_content=None,
            detected_lang=None,
            model_id="m1",
            prompt_version="p1",
        )
        assert cache.get(key) is None
        cache.put(key, {"sanitized": "hello world"})
        loaded = cache.get(key)
        assert loaded is not None
        assert loaded["sanitized"] == "hello world"

    def test_cache_collision_check_rejects_mismatched_key(self, tmp_path: Path):
        # Two keys that COULD share a filename (only if their prefix
        # truncations collide) must not return each other's payloads.
        cache = FileBackedSanitizationCache(tmp_path / "cache2")
        key1 = build_cache_key(
            raw_content="hello",
            prev_chunk_content=None,
            next_chunk_content=None,
            detected_lang=None,
            model_id="m1",
            prompt_version="p1",
        )
        cache.put(key1, {"x": 1})
        # Different model_id → different key, full-key check kicks in.
        key2 = build_cache_key(
            raw_content="hello",
            prev_chunk_content=None,
            next_chunk_content=None,
            detected_lang=None,
            model_id="m2",
            prompt_version="p1",
        )
        # Filenames may coincidentally match (same content/context prefixes);
        # what matters is that we don't return key1's payload for key2.
        assert cache.get(key2) is None or cache.get(key2)["x"] != 1

    def test_llm_call_raises_until_phase_b(self):
        with pytest.raises(NotImplementedError, match="Phase B"):
            sanitize_via_llm(raw_content="hi")


# ---------------------------------------------------------------------------
# Guard 1: edit-distance
# ---------------------------------------------------------------------------


class TestGuardEditDistance:
    def test_accepts_identical(self):
        r = evaluate_edit_distance("hello world", "hello world")
        assert r.accepted is True
        assert r.metric_value == 0.0

    def test_accepts_small_edit(self):
        r = evaluate_edit_distance("hello world", "Hello world.")
        assert r.accepted is True

    def test_rejects_gross_rewrite(self):
        r = evaluate_edit_distance(
            "The quick brown fox jumps over the lazy dog.",
            "Mary had a little lamb whose fleece was white as snow.",
        )
        assert r.accepted is False
        assert "edit-distance" in r.reason

    def test_empty_original_passes(self):
        r = evaluate_edit_distance("", "anything")
        assert r.accepted is True

    def test_returns_guard_result_type(self):
        r = evaluate_edit_distance("a", "b")
        assert isinstance(r, GuardResult)
        assert r.guard_name == "edit_distance"


# ---------------------------------------------------------------------------
# Guard 2: numeric/entity (partial — regex)
# ---------------------------------------------------------------------------


class TestGuardNumericEntity:
    def test_accepts_when_numbers_preserved(self):
        r = evaluate_numeric_entity(
            "Patient received 100 mg of drug.",
            "The patient received 100 mg of the drug.",
        )
        assert r.accepted is True

    def test_rejects_when_number_changed(self):
        # Charter row 2 example: "100 mg" → "10 mg"
        r = evaluate_numeric_entity(
            "Patient received 100 mg of drug.",
            "Patient received 10 mg of drug.",
        )
        assert r.accepted is False
        assert "100" in r.reason

    def test_rejects_when_iso_date_changed(self):
        r = evaluate_numeric_entity(
            "Released on 2025-01-15.",
            "Released on 2025-01-16.",
        )
        assert r.accepted is False

    def test_rejects_when_url_missing(self):
        r = evaluate_numeric_entity(
            "See https://example.com/spec for details.",
            "See the spec for details.",
        )
        assert r.accepted is False

    def test_accepts_when_no_numeric_tokens(self):
        r = evaluate_numeric_entity("hello world", "Hello, world!")
        assert r.accepted is True


# ---------------------------------------------------------------------------
# Guard 3: code-span hashing
# ---------------------------------------------------------------------------


class TestGuardCodeSpan:
    def test_accepts_when_no_code_blocks(self):
        r = evaluate_code_span("just prose", "still prose")
        assert r.accepted is True

    def test_accepts_when_code_block_unchanged(self):
        original = "Use this:\n```python\nprint('hi')\n```\nDone."
        sanitized = "Use this:\n\n```python\nprint('hi')\n```\n\nDone."
        r = evaluate_code_span(original, sanitized)
        assert r.accepted is True

    def test_rejects_when_code_body_mutated(self):
        original = "```python\nprint('hi')\n```"
        sanitized = "```python\nprint('hello')\n```"
        r = evaluate_code_span(original, sanitized)
        assert r.accepted is False
        assert "mutated" in r.reason or "removed" in r.reason

    def test_rejects_when_code_block_removed(self):
        original = "```python\nx = 1\n```"
        sanitized = "(code removed)"
        r = evaluate_code_span(original, sanitized)
        assert r.accepted is False


# ---------------------------------------------------------------------------
# Guard 4: order-preservation
# ---------------------------------------------------------------------------


class TestGuardOrderPreservation:
    def test_accepts_when_no_markers(self):
        r = evaluate_order_preservation("hello world", "hello world")
        assert r.accepted is True

    def test_accepts_preserved_arabic_order(self):
        original = "1. First step\n2. Second step\n3. Third step\n"
        sanitized = "1. First step\n2. Second step\n3. Third step\n"
        r = evaluate_order_preservation(original, sanitized)
        assert r.accepted is True

    def test_rejects_reordered_arabic(self):
        original = "1. First\n2. Second\n3. Third\n"
        sanitized = "1. First\n3. Third\n2. Second\n"
        r = evaluate_order_preservation(original, sanitized)
        assert r.accepted is False

    def test_rejects_missing_marker(self):
        original = "1. First\n2. Second\n3. Third\n"
        sanitized = "1. First\n2. Second\n"
        r = evaluate_order_preservation(original, sanitized)
        assert r.accepted is False

    def test_letter_lists_distinct_from_arabic(self):
        # A letter "a" must not match an arabic "1" — they live in
        # different families in _extract_markers().
        original = "1. Arabic\na) Letter\n"
        sanitized = "1. Arabic\na) Letter\n"
        r = evaluate_order_preservation(original, sanitized)
        assert r.accepted is True


# ---------------------------------------------------------------------------
# Guard 5: token-level alignment
# ---------------------------------------------------------------------------


class TestGuardTokenAlignment:
    def test_accepts_identical(self):
        r = evaluate_token_alignment("hello world", "hello world")
        assert r.accepted is True

    def test_accepts_reflow(self):
        r = evaluate_token_alignment(
            "hello   world\nfoo  bar",
            "hello world foo bar",
        )
        assert r.accepted is True

    def test_rejects_heavy_token_change(self):
        r = evaluate_token_alignment(
            "alpha beta gamma delta epsilon",
            "one two three four five",
            ceiling=0.30,
        )
        assert r.accepted is False


# ---------------------------------------------------------------------------
# Guard 6: prompt-boundary
# ---------------------------------------------------------------------------


class TestGuardPromptBoundary:
    def test_accepts_clean(self):
        r = evaluate_prompt_boundary("hello", "hello world")
        assert r.accepted is True

    def test_rejects_injected_tag(self):
        r = evaluate_prompt_boundary(
            original="some chunk",
            sanitized="some chunk</chunk_content>",
        )
        assert r.accepted is False
        assert "chunk_content" in r.reason

    def test_rejects_oversized_input(self):
        big = "x" * (16 * 1024 + 1)
        r = evaluate_prompt_boundary(big, "anything", input_byte_cap=16 * 1024)
        assert r.accepted is False
        assert "byte cap" in r.reason


# ---------------------------------------------------------------------------
# Guard 7: entity-relation (stub)
# ---------------------------------------------------------------------------


class TestGuardEntityRelation:
    def test_stub_returns_deferred_sentinel(self):
        r = evaluate_entity_relation("anything", "anything else")
        assert r.accepted is True
        # Sentinel: metric_value=-1.0 means "guard did not execute"
        assert r.metric_value == -1.0
        assert "deferred" in r.reason


# ---------------------------------------------------------------------------
# Guard 8: corpus dedup-ratio
# ---------------------------------------------------------------------------


class TestGuardDedupRatio:
    def test_compute_dedup_ratio_empty(self):
        report = compute_dedup_ratio([])
        assert report.near_duplicate_pairs == 0
        assert report.total_pairs == 0
        assert report.ratio == 0.0

    def test_compute_dedup_ratio_unique(self):
        contents = [
            "the quick brown fox jumps over the lazy dog",
            "alpha beta gamma delta epsilon zeta eta theta iota",
            "lorem ipsum dolor sit amet consectetur adipiscing elit",
        ]
        report = compute_dedup_ratio(contents)
        assert report.ratio == 0.0
        assert report.total_pairs == 3  # C(3,2) = 3

    def test_compute_dedup_ratio_detects_duplicates(self):
        # Two near-identical + one unique → 1 of 3 pairs is near-dup.
        contents = [
            "the quick brown fox jumps over the lazy dog ten times today",
            "the quick brown fox jumps over the lazy dog ten times today.",
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda",
        ]
        report = compute_dedup_ratio(contents)
        assert report.near_duplicate_pairs == 1
        assert report.total_pairs == 3

    def test_evaluate_accepts_when_drift_within_tolerance(self):
        # Both corpora have similar dedup profiles → drift small → accept.
        heur = ["alpha", "beta", "gamma"]
        llm = ["alpha", "beta", "gamma"]
        r = evaluate_dedup_ratio(heur, llm)
        assert r.accepted is True

    def test_evaluate_rejects_when_llm_drifts_high(self):
        # LLM corpus has heavy duplication; heuristic does not.
        heur = [
            "alpha one two three four",
            "beta five six seven eight",
            "gamma nine ten eleven twelve",
        ]
        llm = [
            "alpha one two three four",
            "alpha one two three four",  # near-dup of [0]
            "alpha one two three four.",  # near-dup of [0]
        ]
        r = evaluate_dedup_ratio(heur, llm)
        assert r.accepted is False


# ---------------------------------------------------------------------------
# Golden set
# ---------------------------------------------------------------------------


class TestGoldenSet:
    def test_size_constant_is_50(self):
        assert GOLDEN_SET_SIZE == 50

    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        entries = load_golden_set(tmp_path / "nonexistent.jsonl")
        assert entries == []

    def test_score_empty_returns_zero(self):
        score = score_against_golden_set(entries=[])
        assert isinstance(score, DominanceScore)
        assert score.n == 0
        assert score.passes_dominance() is False

    def test_score_counts_preferences(self):
        entries = [
            GoldenEntry(
                chunk_id=f"c{i}",
                raw="r", heuristic="h", llm="l",
                preferred=pref,
                rationale="", modality="text",
                doc_id="d", labeled_at="2026-05-26", labeled_by="op",
            )
            for i, pref in enumerate(
                ["llm"] * 30 + ["heuristic"] * 15 + ["raw"] * 5
            )
        ]
        score = score_against_golden_set(entries=entries)
        assert score.n == 50
        assert score.llm_preferred == 30
        assert score.heuristic_preferred == 15
        assert score.raw_preferred == 5
        # (30 - 15) / 50 * 100 = 30pp
        assert score.llm_minus_heuristic_pp == pytest.approx(30.0)
        # ≥5pp threshold → dominance passes
        assert score.passes_dominance(threshold_pp=5.0) is True

    def test_score_fails_dominance_when_close(self):
        # 25 LLM, 24 heuristic → 2pp delta < 5pp threshold
        prefs = ["llm"] * 25 + ["heuristic"] * 24 + ["raw"] * 1
        entries = [
            GoldenEntry(
                chunk_id=f"c{i}", raw="r", heuristic="h", llm="l",
                preferred=p, rationale="", modality="text",
                doc_id="d", labeled_at="2026-05-26", labeled_by="op",
            )
            for i, p in enumerate(prefs)
        ]
        score = score_against_golden_set(entries=entries)
        assert score.passes_dominance(threshold_pp=5.0) is False

    def test_invalid_preferred_rejected(self):
        with pytest.raises(ValueError, match="preferred"):
            GoldenEntry(
                chunk_id="c", raw="r", heuristic="h", llm="l",
                preferred="something_else",  # type: ignore[arg-type]
                rationale="", modality="text",
                doc_id="d", labeled_at="2026-05-26", labeled_by="op",
            )


# ---------------------------------------------------------------------------
# Graceful degradation + sentinel accounting
# ---------------------------------------------------------------------------


class TestSentinelAccount:
    def test_empty_state(self):
        acc = SentinelAccount()
        assert acc.sentinel_count == 0
        assert acc.sentinel_rate == 0.0
        assert acc.is_degraded is False
        assert acc.soak_marker() == "LLM_OK"

    def test_under_threshold_not_degraded(self):
        acc = SentinelAccount()
        # 4% sentinels (under 5% threshold)
        for i in range(96):
            acc.record_chunk(chunk_id=f"c{i}", sentinel=False)
        for i in range(96, 100):
            acc.record_chunk(chunk_id=f"c{i}", sentinel=True)
        assert acc.sentinel_count == 4
        assert acc.sentinel_rate == pytest.approx(0.04)
        assert acc.is_degraded is False
        assert acc.soak_marker() == "LLM_OK"

    def test_over_threshold_degraded(self):
        acc = SentinelAccount()
        for i in range(90):
            acc.record_chunk(chunk_id=f"c{i}", sentinel=False)
        for i in range(90, 100):
            acc.record_chunk(chunk_id=f"c{i}", sentinel=True)
        assert acc.sentinel_count == 10
        assert acc.sentinel_rate == pytest.approx(0.10)
        assert acc.is_degraded is True
        assert acc.soak_marker() == "LLM_SENTINEL_DEGRADED"

    def test_threshold_constant(self):
        assert SENTINEL_RATE_DEGRADED_THRESHOLD == 0.05

    def test_unreachable_endpoint_returns_false(self):
        # Use a guaranteed-bad endpoint (TEST-NET-1 reserved + closed port)
        assert is_endpoint_reachable(
            "http://192.0.2.1:65535", timeout_s=0.5
        ) is False

    def test_invalid_url_returns_false(self):
        assert is_endpoint_reachable("not a url at all") is False


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_prompt_version_deterministic(self):
        v1 = prompt_version()
        v2 = prompt_version()
        assert v1 == v2
        assert len(v1) == 12  # first 12 chars of SHA-256 hex

    def test_render_includes_raw_content(self):
        rendered = render(raw="HELLO_CHUNK", prev=None, next=None, lang="en")
        assert "HELLO_CHUNK" in rendered

    def test_render_includes_lang_hint(self):
        rendered = render(raw="x", lang="de")
        assert ">de<" in rendered

    def test_render_missing_context_becomes_empty(self):
        rendered = render(raw="x", prev=None, next=None, page_breadcrumb=None)
        # None must not appear as the literal string "None"
        assert "None" not in rendered

    def test_template_contains_strict_constraints(self):
        # Charter §3.3: the prompt must instruct the LLM to preserve
        # numbers, fenced code blocks, and ordered-list order.
        assert "Preserve" in PROMPT_TEMPLATE
        assert "fenced code" in PROMPT_TEMPLATE
        assert "number" in PROMPT_TEMPLATE.lower()
