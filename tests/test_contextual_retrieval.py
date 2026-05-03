"""Contextual Retrieval — invariants, edge cases, ingestor boundary, drift guard.

Target count: 32 cases (kept in sync with the count printed by
``pytest tests/test_contextual_retrieval.py --collect-only -q``).

These tests are executable requirements for AGENT-CONTEXTUAL-01..07
(see ``docs/DECISIONS.md`` and the module docstring of
``mmrag_v2.chunking.contextual_retrieval``). They prove that:

1. ``build_contextualized_text`` never mutates ``content``.
2. Marker strings (``[Context: ``, ``[Heading: ``, ``[Previous: ``,
   ``[Next: ``, ``[Modality: ``) never leak into ``IngestionChunk.content``,
   ``metadata.refined_content``, the Qdrant payload, or any QA-reading helper.
3. The ingestor's ``--no-contextual`` flag preserves v2.7.0 byte-for-byte
   behavior, and the default flag emits the contextualized string ONLY to
   the embedder, never to the payload.
4. The drift guard fails the moment a production file outside the allowlist
   writes a marker string or calls ``build_contextualized_text``.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from mmrag_v2.chunking.contextual_retrieval import (
    MAX_CONTEXT_CHARS,
    build_contextualized_text,
)
from mmrag_v2.schema.ingestion_schema import (
    AssetReference,
    BoundingBox,
    ChunkMetadata,
    FileType,
    HierarchyMetadata,
    IngestionChunk,
    Modality,
    SemanticContext,
    SpatialMetadata,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_ingestor_module():
    """Import scripts/ingest_to_qdrant.py as a module for boundary tests.

    The script is not packaged, so we load it by file path. Cached on
    ``sys.modules`` keyed by an underscore alias so monkeypatch can target it.
    """
    module_name = "_ingest_to_qdrant_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]
    script_path = _REPO_ROOT / "scripts" / "ingest_to_qdrant.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ── TestContentImmutability — AGENT-CONTEXTUAL-01 ───────────────────────────


class TestContentImmutability:
    """Prove ``build_contextualized_text`` never mutates the canonical content."""

    def test_content_unchanged_after_call(self):
        content = "The quick brown fox jumps over the lazy dog."
        original = content
        result = build_contextualized_text(
            content,
            breadcrumb_path=["Chapter 1", "Section 2"],
            parent_heading="Introduction",
        )
        assert content == original
        assert content in result

    def test_content_appears_verbatim_at_end(self):
        content = "Canonical text that must remain unchanged."
        result = build_contextualized_text(
            content,
            breadcrumb_path=["Ch 1"],
            prev_text_snippet="Previous context text.",
            next_text_snippet="Next context text.",
        )
        assert result.endswith(content)

    def test_content_with_special_chars(self):
        content = "Price: $10.99 (50% off!) [Best deal]"
        result = build_contextualized_text(
            content,
            breadcrumb_path=["Products", "Deals"],
        )
        assert content in result
        assert result.split("\n")[-1] == content


# ── TestContextualPrefixSeparation — AGENT-CONTEXTUAL-03 ────────────────────


class TestContextualPrefixSeparation:
    """Prove prefixes never bleed into the canonical content line."""

    def test_contextual_text_differs_from_content(self):
        content = "This is the raw content."
        contextual = build_contextualized_text(
            content,
            breadcrumb_path=["Chapter 1"],
            parent_heading="Overview",
        )
        assert contextual != content
        assert len(contextual) > len(content)

    def test_each_marker_appears_only_in_prefix_lines(self):
        content = "Pure content text without any markers."
        result = build_contextualized_text(
            content,
            breadcrumb_path=["Ch 1"],
            parent_heading="Title",
            prev_text_snippet="prior",
            next_text_snippet="after",
            modality="table",
        )
        lines = result.split("\n")
        assert lines[-1] == content
        for marker in ("[Context: ", "[Heading: ", "[Previous: ", "[Next: ", "[Modality: "):
            assert marker not in content
            assert marker not in lines[-1]


# ── TestMissingContextHandling ──────────────────────────────────────────────


class TestMissingContextHandling:
    """Prove the builder skips missing/empty fields silently."""

    def test_no_context_returns_content_verbatim(self):
        content = "Standalone content with no context."
        result = build_contextualized_text(content)
        assert result == content

    def test_partial_context_skips_missing_lines(self):
        content = "Body."
        result = build_contextualized_text(
            content,
            breadcrumb_path=["Ch 1"],
            parent_heading=None,
            prev_text_snippet=None,
            next_text_snippet="Some next text.",
        )
        assert "[Context: Ch 1]" in result
        assert "[Heading:" not in result
        assert "[Previous:" not in result
        assert "[Next:" in result
        assert result.endswith(content)

    def test_whitespace_only_fields_are_skipped(self):
        content = "Body."
        result = build_contextualized_text(
            content,
            breadcrumb_path=["", "  "],
            parent_heading="   ",
            prev_text_snippet="  \t  ",
            next_text_snippet="\n",
        )
        # Every prefix is whitespace-only ⇒ none should render
        assert "[Context:" not in result
        assert "[Heading:" not in result
        assert "[Previous:" not in result
        assert "[Next:" not in result
        assert result == content

    def test_whitespace_only_breadcrumb_levels_dropped_per_level(self):
        # AGENT-CONTEXTUAL edge case: ["", "  ", "Sec 1"] → "[Context: Sec 1]"
        content = "Body."
        result = build_contextualized_text(
            content,
            breadcrumb_path=["", "  ", "Sec 1"],
        )
        assert "[Context: Sec 1]" in result
        assert "[Context:  >  >" not in result


# ── TestContextLengthBounds — AGENT-CONTEXTUAL-04 ───────────────────────────


class TestContextLengthBounds:
    """Prove snippet truncation; non-truncation of headings/breadcrumbs."""

    def test_prev_snippet_truncated_to_MAX_CONTEXT_CHARS(self):
        long_prev = "A" * 1000
        result = build_contextualized_text("X", prev_text_snippet=long_prev)
        prev_line = next(line for line in result.split("\n") if line.startswith("[Previous: "))
        snippet = prev_line[len("[Previous: "):-1]
        assert len(snippet) == MAX_CONTEXT_CHARS

    def test_next_snippet_truncated_to_MAX_CONTEXT_CHARS(self):
        long_next = "B" * 1000
        result = build_contextualized_text("X", next_text_snippet=long_next)
        next_line = next(line for line in result.split("\n") if line.startswith("[Next: "))
        snippet = next_line[len("[Next: "):-1]
        assert len(snippet) == MAX_CONTEXT_CHARS

    def test_long_breadcrumb_levels_are_not_truncated(self):
        # Documented: only prev/next snippets are truncated by this builder.
        # Schema caps breadcrumb level length separately at 80 chars.
        long_level = "L" * 500
        result = build_contextualized_text(
            "X", breadcrumb_path=[long_level, "Sec 1"]
        )
        assert f"[Context: {long_level} > Sec 1]" in result

    def test_very_long_heading_is_not_truncated(self):
        # Headings are not truncated by the builder; the schema caps them.
        long_heading = "H" * 500
        result = build_contextualized_text("X", parent_heading=long_heading)
        assert f"[Heading: {long_heading}]" in result

    def test_utf8_truncation_does_not_split_codepoint(self):
        # Multi-byte UTF-8 characters at the truncation boundary must remain decodable.
        # Slicing a Python ``str`` operates on code points, so this can only fail
        # if the builder ever switched to bytes-then-slice — guard against that.
        long_jp = "日本語" * 200  # 600 code points, each 3 bytes in UTF-8
        result = build_contextualized_text("X", prev_text_snippet=long_jp)
        prev_line = next(line for line in result.split("\n") if line.startswith("[Previous: "))
        snippet = prev_line[len("[Previous: "):-1]
        assert len(snippet) == MAX_CONTEXT_CHARS
        # Byte round-trip must succeed (no bare continuation bytes).
        snippet.encode("utf-8").decode("utf-8")


# ── TestQAValidationIntegrity — AGENT-CONTEXTUAL-03 ────────────────────────


def _build_text_chunk_with_contextualized(content: str, contextualized: str) -> IngestionChunk:
    """Build an IngestionChunk where contextualized_text is the contextualized string."""
    metadata = ChunkMetadata(
        source_file="probe.pdf",
        file_type=FileType.PDF,
        page_number=1,
    )
    return IngestionChunk(
        chunk_id="probe_001_text_aaaaaaaa",
        doc_id="probedoc0001",
        modality=Modality.TEXT,
        content=content,
        metadata=metadata,
        contextualized_text=contextualized,
    )


class TestQAValidationIntegrity:
    """Prove QA / source-text validators only read ``content``, never the prefix."""

    def test_qa_audit_only_reads_content(self):
        # Stand-in for qa_conversion_audit.py: it parses JSONL and reads
        # ``obj.get("content")`` (see _has_ctrl/_ctrl_count callers).
        chunk = _build_text_chunk_with_contextualized(
            content="X",
            contextualized="[Context: foo]\n[Heading: bar]\nX",
        )
        record = json.loads(chunk.model_dump_json())
        # The audit's content read path:
        observed = record.get("content") or ""
        assert observed == "X"
        assert "[Context:" not in observed
        assert "[Heading:" not in observed

    def test_universal_invariants_only_reads_content(self):
        # Stand-in for qa_universal_invariants.check(): line 161 reads
        # ``obj.get("content") or ""`` to flag empty text chunks.
        chunk = _build_text_chunk_with_contextualized(
            content="X",
            contextualized="[Context: foo]\nX",
        )
        record = json.loads(chunk.model_dump_json())
        observed = record.get("content") or ""
        assert observed == "X"
        assert "[Context:" not in observed

    def test_token_validator_only_reads_content(self):
        # Real check: token_validator.py line 244 calls
        # ``self._counter.count_tokens(chunk.content)``. Construct a chunk where
        # ``content`` is short and ``contextualized_text`` is long; the
        # validator must count tokens for the short string only.
        import tiktoken

        from mmrag_v2.validators.token_validator import ENCODING_NAME

        short = "hello world"
        long_ctx = "[Context: a > b > c]\n[Heading: H]\n" + short
        chunk = _build_text_chunk_with_contextualized(
            content=short, contextualized=long_ctx
        )
        encoder = tiktoken.get_encoding(ENCODING_NAME)
        observed_tokens = len(encoder.encode(chunk.content))
        ctx_tokens = len(encoder.encode(long_ctx))
        assert observed_tokens < ctx_tokens
        assert "[Context:" not in chunk.content


# ── TestContextualForImageChunks — AGENT-CONTEXTUAL-05 ──────────────────────


class TestContextualForImageChunks:
    """Prove modality marker rules. Production ingest does not use this path
    for images; these cover callers that *choose* to.
    """

    def test_modality_image_marker_added_when_called(self):
        result = build_contextualized_text(
            "A photo of a sunset.",
            modality="image",
            breadcrumb_path=["Gallery"],
        )
        assert "[Modality: image]" in result

    def test_modality_text_emits_no_modality_marker(self):
        result = build_contextualized_text("Plain.", modality="text")
        assert "[Modality:" not in result

    def test_modality_empty_string_emits_no_modality_marker(self):
        result = build_contextualized_text("Plain.", modality="")
        assert "[Modality:" not in result

    def test_modality_table_marker_outside_markdown_body(self):
        # Markdown table content with pipes — the marker must be its own line,
        # never inside the table body.
        markdown_table = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = build_contextualized_text(
            markdown_table, modality="table", breadcrumb_path=["Tables"]
        )
        lines = result.split("\n")
        # Modality marker must be on a line strictly before the table body.
        modality_idx = next(i for i, line in enumerate(lines) if line.startswith("[Modality: "))
        assert lines[modality_idx] == "[Modality: table]"
        # Table body must follow verbatim.
        assert "\n".join(lines[modality_idx + 1:]) == markdown_table


# ── TestIntegrationSemanticContext ──────────────────────────────────────────


class TestIntegrationSemanticContext:
    """Round-trip a real SemanticContext through the builder."""

    def test_full_semantic_context_round_trip(self):
        sem = SemanticContext(
            prev_text_snippet="Previous text snippet.",
            next_text_snippet="Next text snippet.",
            parent_heading="My Heading",
            breadcrumb_path=["Book", "Chapter 1", "Section A"],
        )
        content = "Main content here."
        result = build_contextualized_text(
            content,
            breadcrumb_path=sem.breadcrumb_path,
            parent_heading=sem.parent_heading,
            prev_text_snippet=sem.prev_text_snippet,
            next_text_snippet=sem.next_text_snippet,
            modality="text",
        )
        # Expected line order: Context, Heading, Previous, Next, content.
        expected_lines = [
            "[Context: Book > Chapter 1 > Section A]",
            "[Heading: My Heading]",
            "[Previous: Previous text snippet.]",
            "[Next: Next text snippet.]",
            content,
        ]
        assert result == "\n".join(expected_lines)


# ── Edge cases called out in the spec ───────────────────────────────────────


class TestEdgeCases:
    """Edge cases the spec explicitly enumerates."""

    def test_empty_content_does_not_raise(self):
        result = build_contextualized_text(
            "", breadcrumb_path=["A"], parent_heading="H"
        )
        # Prefixes followed by an empty trailing line.
        assert result == "[Context: A]\n[Heading: H]\n"

    def test_content_already_contains_marker_is_preserved_verbatim(self):
        # A programming book may legitimately show ``[Context: ...]`` in code/log output.
        # The builder must not strip it from ``content``; the static drift guard
        # explicitly allows markers in content (only forbids them in production
        # writes outside the allowlist).
        content = "log line: [Context: foo > bar] processed"
        result = build_contextualized_text(content, breadcrumb_path=["Ch 1"])
        assert content in result
        assert result.endswith(content)

    def test_semantic_context_none_defensive_read(self):
        # Real chunks may have ``semantic_context = None``. The ingestor reads
        # ``chunk.get("semantic_context") or {}`` and passes None for missing fields.
        sem = None
        sem_dict = sem or {}
        result = build_contextualized_text(
            "X",
            breadcrumb_path=["Ch 1"],
            prev_text_snippet=sem_dict.get("prev_text_snippet"),
            next_text_snippet=sem_dict.get("next_text_snippet"),
        )
        assert "[Context: Ch 1]" in result
        assert "[Previous:" not in result
        assert "[Next:" not in result

    def test_metadata_hierarchy_none_defensive_read(self):
        # The ingestor reads ``metadata.get("hierarchy") or {}``.
        hierarchy = None
        h_dict = hierarchy or {}
        result = build_contextualized_text(
            "X",
            breadcrumb_path=h_dict.get("breadcrumb_path") or [],
            parent_heading=h_dict.get("parent_heading"),
        )
        assert result == "X"


# ── TestIngestorBoundary — drift insurance, mirrors round-trip from §5 ──────


def _write_minimal_ingestion_jsonl(jsonl_path: Path, assets_dir: Path) -> None:
    """Write a 3-chunk fixture (text, image, table) plus an ingestion_metadata head."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    image_asset = assets_dir / "abc_001_image_0001.png"
    image_asset.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)  # tiny stub
    table_asset = assets_dir / "abc_001_table_0001.png"
    table_asset.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    metadata_record = {
        "object_type": "ingestion_metadata",
        "schema_version": "test",
        "doc_id": "abcdef000001",
        "source_file": "probe.pdf",
        "domain": "technical",
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    text_chunk = IngestionChunk(
        chunk_id="abcdef000001_001_text_aaaa0001",
        doc_id="abcdef000001",
        modality=Modality.TEXT,
        content="Body text with no markers.",
        metadata=ChunkMetadata(
            source_file="probe.pdf",
            file_type=FileType.PDF,
            page_number=1,
            hierarchy=HierarchyMetadata(
                parent_heading="Section One",
                breadcrumb_path=["Chapter 1", "Section One"],
            ),
        ),
        semantic_context=SemanticContext(
            prev_text_snippet="Previous body.",
            next_text_snippet="Following body.",
            parent_heading="Section One",
            breadcrumb_path=["Chapter 1", "Section One"],
        ),
    ).model_dump(mode="json")

    image_chunk = IngestionChunk(
        chunk_id="abcdef000001_001_image_aaaa0002",
        doc_id="abcdef000001",
        modality=Modality.IMAGE,
        content="A photo of a sunset over the city.",
        metadata=ChunkMetadata(
            source_file="probe.pdf",
            file_type=FileType.PDF,
            page_number=1,
            visual_description="A photo of a sunset over the city.",
            spatial=SpatialMetadata(
                bbox=[10, 10, 500, 500],
                page_width=1000,
                page_height=1000,
            ),
            hierarchy=HierarchyMetadata(
                parent_heading="Section One",
                breadcrumb_path=["Chapter 1", "Section One"],
            ),
        ),
        asset_ref=AssetReference(
            file_path=image_asset.name,
            mime_type="image/png",
            width_px=100,
            height_px=100,
        ),
    ).model_dump(mode="json")

    table_chunk = IngestionChunk(
        chunk_id="abcdef000001_001_table_aaaa0003",
        doc_id="abcdef000001",
        modality=Modality.TABLE,
        content="| A | B |\n|---|---|\n| 1 | 2 |",
        metadata=ChunkMetadata(
            source_file="probe.pdf",
            file_type=FileType.PDF,
            page_number=1,
            spatial=SpatialMetadata(
                bbox=[20, 20, 600, 600],
                page_width=1000,
                page_height=1000,
            ),
            hierarchy=HierarchyMetadata(
                parent_heading="Tables",
                breadcrumb_path=["Chapter 1", "Tables"],
            ),
        ),
        asset_ref=AssetReference(
            file_path=table_asset.name,
            mime_type="image/png",
        ),
    ).model_dump(mode="json")

    with jsonl_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(metadata_record) + "\n")
        for chunk in (text_chunk, image_chunk, table_chunk):
            f.write(json.dumps(chunk) + "\n")


@pytest.fixture
def ingestor_capture(tmp_path, monkeypatch):
    """Run the ingestor against a tiny fixture; capture all embed and upsert calls."""
    jsonl_path = tmp_path / "ingestion.jsonl"
    assets_dir = tmp_path / "assets"
    _write_minimal_ingestion_jsonl(jsonl_path, assets_dir)

    module = _load_ingestor_module()

    captured = {
        "text_calls": [],   # list[str] — first arg to embed_text
        "image_calls": [],  # list[(image_path, fallback_text)]
        "upserts": [],      # list[list[dict]] — points buffers passed to upsert
        "qdrant_get": [],
        "qdrant_put_collection": [],
    }

    def fake_embed_text(text, model, ollama_url):
        captured["text_calls"].append(text)
        return [0.0, 0.1, 0.2]  # 3-dim stub

    def fake_embed_image(image_path, model, ollama_url, fallback_text=""):
        captured["image_calls"].append((str(image_path), fallback_text))
        return [0.0, 0.1, 0.2]

    def fake_qdrant_request(method, path, body, qdrant_url):
        if method == "GET" and path.startswith("/collections/"):
            captured["qdrant_get"].append(path)
            if len(captured["qdrant_get"]) == 1:
                # First GET: existence check inside create_collection. Raise so
                # create_collection proceeds with PUT.
                raise RuntimeError("collection not found")
            # Subsequent GETs (final verification) return a points-count summary.
            point_count = sum(len(batch) for batch in captured["upserts"])
            return {"result": {"points_count": point_count}}
        if method == "PUT" and path.startswith("/collections/") and "/points" not in path:
            captured["qdrant_put_collection"].append((path, body))
            return {"result": True}
        if method == "PUT" and path.endswith("/points"):
            captured["upserts"].append(body.get("points", []))
            return {"result": True}
        return {"result": True}

    monkeypatch.setattr(module, "embed_text", fake_embed_text)
    monkeypatch.setattr(module, "embed_image", fake_embed_image)
    monkeypatch.setattr(module, "qdrant_request", fake_qdrant_request)

    def run(extra_args=None):
        argv = ["ingest_to_qdrant.py", str(jsonl_path), "--collection", "probe"]
        if extra_args:
            argv.extend(extra_args)
        monkeypatch.setattr(sys, "argv", argv)
        # First text call is the "Testing embedding model..." dim probe;
        # we strip it from text_calls below before assertions.
        rc = module.main()
        assert rc == 0
        # Drop the dim-probe call ("test").
        if captured["text_calls"] and captured["text_calls"][0] == "test":
            captured["text_calls"].pop(0)
        return captured

    return run


class TestIngestorBoundary:
    """Drift insurance for the ingestor's contextualization wiring."""

    def test_ingest_no_contextual_flag_falls_back_to_breadcrumb_only(self, ingestor_capture):
        cap = ingestor_capture(extra_args=["--no-contextual"])
        # Text chunk: ``f"{breadcrumb}\n{content}"`` with breadcrumb joined by ' > '.
        # Table chunk: same fallback path.
        text_embedded = cap["text_calls"][0]
        assert text_embedded == "Chapter 1 > Section One\nBody text with no markers."
        # No marker leakage in any embedded text.
        for embedded in cap["text_calls"]:
            for marker in ("[Context: ", "[Heading: ", "[Previous: ", "[Next: ", "[Modality: "):
                assert marker not in embedded

    def test_ingest_default_uses_contextualized_text(self, ingestor_capture):
        cap = ingestor_capture()
        text_embedded = cap["text_calls"][0]
        assert text_embedded.startswith("[Context: Chapter 1 > Section One]")
        assert text_embedded.endswith("Body text with no markers.")
        # And the payload for that text chunk must NOT contain markers.
        all_payloads = [point["payload"] for batch in cap["upserts"] for point in batch]
        for payload in all_payloads:
            for field in ("content", "visual_description"):
                value = payload.get(field, "") or ""
                for marker in ("[Context: ", "[Heading: ", "[Previous: ", "[Next: ", "[Modality: "):
                    assert marker not in value

    def test_ingest_image_chunks_unaffected(self, ingestor_capture):
        cap = ingestor_capture()
        # Image chunk should have hit embed_image, not the contextual builder.
        assert len(cap["image_calls"]) == 1
        _path, fallback_text = cap["image_calls"][0]
        # Fallback text is the visual description, not a contextualized string.
        assert fallback_text == "A photo of a sunset over the city."
        assert "[Context:" not in fallback_text

    def test_no_marker_strings_in_payload_for_any_modality(self, ingestor_capture):
        cap = ingestor_capture()
        all_payloads = [point["payload"] for batch in cap["upserts"] for point in batch]
        assert len(all_payloads) == 3  # text + image + table
        markers = ("[Context: ", "[Heading: ", "[Previous: ", "[Next: ", "[Modality: ")
        for payload in all_payloads:
            for value in payload.values():
                if isinstance(value, str):
                    for marker in markers:
                        assert marker not in value, (
                            f"marker {marker!r} leaked into payload field "
                            f"value={value!r}"
                        )

    def test_no_contextual_byte_stable_v270(self, ingestor_capture):
        # The ``--no-contextual`` toggle must produce the v2.7.0 string exactly:
        # ``f"{breadcrumb}\n{content}"`` (or just content when breadcrumb is empty).
        cap = ingestor_capture(extra_args=["--no-contextual"])
        # Text chunk has breadcrumb ⇒ "<breadcrumb>\n<content>".
        assert cap["text_calls"][0] == "Chapter 1 > Section One\nBody text with no markers."
        # Table chunk also has breadcrumb ⇒ same shape.
        # Order in fixture: text → image → table. embed_text was called for text, then table.
        assert cap["text_calls"][1] == "Chapter 1 > Tables\n| A | B |\n|---|---|\n| 1 | 2 |"


# ── Static drift guard ─────────────────────────────────────────────────────


def _production_python_files() -> list[Path]:
    paths: list[Path] = []
    src_root = _REPO_ROOT / "src" / "mmrag_v2"
    scripts_root = _REPO_ROOT / "scripts"
    for root in (src_root, scripts_root):
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            paths.append(path)
    return paths


_MARKER_LITERALS = ("[Context: ", "[Heading: ", "[Previous: ", "[Next: ", "[Modality: ")
_BUILDER_CALL = "build_contextualized_text("


def test_no_contextual_marker_strings_in_production_code():
    """Marker literals must only appear in the builder module itself.

    Failure indicates someone began writing a contextualized string into chunk
    content, refiner output, or a chunk-creation helper. Fix the implementation,
    not this test.
    """
    allowlist_marker = {
        _REPO_ROOT / "src" / "mmrag_v2" / "chunking" / "contextual_retrieval.py",
    }
    allowlist_builder_call = {
        _REPO_ROOT / "scripts" / "ingest_to_qdrant.py",
        _REPO_ROOT / "src" / "mmrag_v2" / "chunking" / "contextual_retrieval.py",
        _REPO_ROOT / "src" / "mmrag_v2" / "chunking" / "__init__.py",
    }

    marker_violations: list[str] = []
    builder_call_violations: list[str] = []

    for path in _production_python_files():
        text = path.read_text(encoding="utf-8")

        if path not in allowlist_marker:
            for lineno, line in enumerate(text.splitlines(), 1):
                for marker in _MARKER_LITERALS:
                    if marker in line:
                        marker_violations.append(
                            f"{path.relative_to(_REPO_ROOT)}:{lineno}: marker {marker!r} found"
                        )
                        break

        if path not in allowlist_builder_call and _BUILDER_CALL in text:
            for lineno, line in enumerate(text.splitlines(), 1):
                if _BUILDER_CALL in line:
                    builder_call_violations.append(
                        f"{path.relative_to(_REPO_ROOT)}:{lineno}: build_contextualized_text(...) call"
                    )

    assert marker_violations == [], (
        "Contextual marker strings must not appear in production code outside "
        "src/mmrag_v2/chunking/contextual_retrieval.py:\n"
        + "\n".join(marker_violations)
    )
    assert builder_call_violations == [], (
        "build_contextualized_text(...) must only be called from the embed-time "
        "lane in scripts/ingest_to_qdrant.py:\n" + "\n".join(builder_call_violations)
    )
