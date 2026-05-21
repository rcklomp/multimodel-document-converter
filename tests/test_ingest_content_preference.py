"""v2.12 Phase 0 — content vs metadata.refined_content preference.

Pins the post-2026-05-21 preference: `ingest_to_qdrant.py` uses the
top-level `content` field as canonical, falling back to
`metadata.refined_content` only when `content` is missing/empty.

The v2.11 soak revealed Format dips on three docs
(`CarOK_voorraadtelling`, `Earthship_Vol1`,
`IRJET_Modeling_of_Solar_PV`) traceable to staleness in
`refined_content`: post-refinement normalization passes (whitespace
collapse, page-header strip, hyphenation fixes) updated the top-level
`content` but not the stored `refined_content`. The pre-Phase-0
ingest preferred `refined_content`, so the Qdrant payload carried
the older, dirtier version — and the LLM-as-judge soak correctly
graded it down.

The pre-Phase-0 preference (`refined_content` first) was a design
intent from earlier in the cycle when `refined_content` was the
*newer* of the two fields. The semantics inverted as the chunker /
normalization passes evolved. The fix is a one-line preference swap;
this test exists so the swap can't silently regress.

Two test cases mirror the two call sites in `ingest_to_qdrant.py`:
`build_qdrant_payload()` (line ~352 post-fix) and the main ingest
loop (line ~484 post-fix). Both must prefer `content`.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load_ingest_module():
    """Load `scripts/ingest_to_qdrant.py` as a module (not on sys.path)."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "ingest_to_qdrant", SCRIPTS / "ingest_to_qdrant.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ingest_to_qdrant", module)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_build_qdrant_payload_prefers_top_level_content() -> None:
    """When both `content` and `metadata.refined_content` are present,
    `build_qdrant_payload()` must store the top-level `content` in
    the payload."""
    mod = _load_ingest_module()
    chunk = {
        "chunk_id": "abc_001_text_xxxx",
        "doc_id": "abc",
        "modality": "text",
        "content": "Clean canonical content with single spaces.",
        "metadata": {
            "refined_content": "Stale  refined  content  with  excess   whitespace  artifacts.",
            "page_number": 1,
            "hierarchy": {},
        },
    }
    payload = mod.build_qdrant_payload(
        chunk, source_file="test.pdf", document_domain="general",
    )
    assert payload["content"] == "Clean canonical content with single spaces.", (
        "Payload must use the top-level `content` field, not "
        "`metadata.refined_content`. If this test fails, the preference "
        "in build_qdrant_payload() reverted to the pre-Phase-0 ordering."
    )


def test_build_qdrant_payload_falls_back_to_refined_when_content_empty() -> None:
    """If top-level `content` is missing or empty (rare edge case for
    text chunks; possible for image chunks where content is derived),
    fall back to `metadata.refined_content`."""
    mod = _load_ingest_module()
    chunk_empty = {
        "chunk_id": "abc_002_text_xxxx",
        "doc_id": "abc",
        "modality": "text",
        "content": "",
        "metadata": {
            "refined_content": "Fallback content from refined_content",
            "page_number": 1,
            "hierarchy": {},
        },
    }
    payload = mod.build_qdrant_payload(
        chunk_empty, source_file="test.pdf", document_domain="general",
    )
    assert payload["content"] == "Fallback content from refined_content"

    chunk_missing = {
        "chunk_id": "abc_003_text_xxxx",
        "doc_id": "abc",
        "modality": "text",
        "metadata": {
            "refined_content": "Fallback content from refined_content",
            "page_number": 1,
            "hierarchy": {},
        },
    }
    payload = mod.build_qdrant_payload(
        chunk_missing, source_file="test.pdf", document_domain="general",
    )
    assert payload["content"] == "Fallback content from refined_content"


def test_build_qdrant_payload_handles_both_missing() -> None:
    """If both fields are missing, the payload `content` is the empty
    string (legacy behavior; ingest may filter or warn elsewhere)."""
    mod = _load_ingest_module()
    chunk = {
        "chunk_id": "abc_004_text_xxxx",
        "doc_id": "abc",
        "modality": "text",
        "metadata": {"page_number": 1, "hierarchy": {}},
    }
    payload = mod.build_qdrant_payload(
        chunk, source_file="test.pdf", document_domain="general",
    )
    assert payload["content"] == ""


def test_main_ingest_loop_prefers_content_over_refined_via_source_grep() -> None:
    """The main ingest loop (around line ~484) has a parallel content
    selection. Source-grep this rather than running the whole loop;
    the loop also makes Qdrant + Dashscope calls which would require a
    live stack. Pinning the source string is sufficient to catch
    accidental reverts."""
    src = (SCRIPTS / "ingest_to_qdrant.py").read_text(encoding="utf-8")
    # Both call sites must read `chunk.get("content")` BEFORE
    # `metadata.get("refined_content")`. Two occurrences (build_qdrant_payload
    # + the main loop) are required.
    occurrences = src.count(
        'content = chunk.get("content") or metadata.get("refined_content"'
    )
    assert occurrences >= 2, (
        f"Expected 2 occurrences of `chunk.get(\"content\") or "
        f"metadata.get(\"refined_content\", ...)` in ingest_to_qdrant.py; "
        f"found {occurrences}. The pre-Phase-0 preference (refined_content "
        f"first) caused the v2.11 soak Format dips on the three known "
        f"scanned/form docs; do not silently revert this."
    )
    # And the OLD pattern must NOT appear.
    old_pattern_count = src.count(
        'content = metadata.get("refined_content") or chunk.get("content"'
    )
    assert old_pattern_count == 0, (
        f"The pre-Phase-0 ordering `refined_content or content` appears "
        f"in {old_pattern_count} place(s) in ingest_to_qdrant.py. Revert."
    )
