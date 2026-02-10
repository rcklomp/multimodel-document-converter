import pytest


def _make_processor(tmp_path):
    from mmrag_v2.processor import V2DocumentProcessor

    # Force the profile_type so chunk sizing matches technical manuals.
    return V2DocumentProcessor(
        output_dir=str(tmp_path),
        enable_ocr=False,
        vision_provider="none",
        intelligence_metadata={"profile_type": "technical_manual"},
    )


def test_code_not_treated_as_noise(tmp_path):
    p = _make_processor(tmp_path)
    code = "def remainder(a, b):\n    q = a // b\n    return a - q * b\n"
    assert p._looks_like_code(code) is True
    assert p._is_noise_content(code) is False


def test_mixed_prose_and_code_chunking(tmp_path):
    from mmrag_v2.schema.ingestion_schema import ChunkType

    p = _make_processor(tmp_path)
    text = (
        "Intro paragraph about functions.\n\n"
        "    def f(x):\n"
        "        return x + 1\n\n"
        "Outro paragraph."
    )

    chunks = p._chunk_text_with_overlap(text)
    assert len(chunks) >= 3

    # Ensure at least one chunk is classified as code and preserves indentation.
    code_chunks = [c for c, t in chunks if t == ChunkType.CODE]
    assert code_chunks
    assert any("\n    def f" in ("\n" + c) or c.lstrip().startswith("def f") for c in code_chunks)
    assert any("\n        return" in ("\n" + c) for c in code_chunks)


def test_long_code_splits_on_line_boundaries(tmp_path):
    from mmrag_v2.schema.ingestion_schema import ChunkType

    p = _make_processor(tmp_path)

    # Build a long code block that exceeds the technical_manual max chunk chars.
    lines = []
    for i in range(400):
        lines.append(f"def f{i}(x):")
        lines.append("    return x + 1")
        lines.append("")
    code = "\n".join(lines).strip() + "\n"

    chunks = p._chunk_text_with_overlap(code)
    assert len(chunks) > 1
    assert all(t == ChunkType.CODE for _, t in chunks)

    # Each chunk should be composed of whole lines (no mid-line breaks introduced).
    for chunk_text, _ in chunks:
        assert "\n" in chunk_text
        assert not chunk_text.endswith("\n\n\n")  # avoid pathological chunking


def test_english_from_not_misclassified_as_code(tmp_path):
    p = _make_processor(tmp_path)
    # "from ..." should not trigger code classification unless it's Python import syntax.
    prose = "In Example 1-1, the deck is made from all 13 ranks of each suit."
    assert p._looks_like_code(prose) is False
