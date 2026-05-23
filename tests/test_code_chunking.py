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


def test_long_body_text_not_treated_as_heading_noise(tmp_path):
    p = _make_processor(tmp_path)
    text = (
        "Index entries and table of contents rows can be much longer than a "
        "valid heading, especially after OCR fallback on dense Docling cells."
    )

    assert len(text) > 80
    assert p._is_noise_content(text) is False


def test_page_number_only_still_treated_as_noise(tmp_path):
    p = _make_processor(tmp_path)
    assert p._is_noise_content("285") is True
    assert p._is_noise_content("Page 285") is True


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
    code_chunks = [c for c, t, _ in chunks if t == ChunkType.CODE]
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
    assert all(t == ChunkType.CODE for _, t, _ in chunks)

    # Each chunk should be composed of whole lines (no mid-line breaks introduced).
    for chunk_text, _, _ in chunks:
        assert "\n" in chunk_text
        assert not chunk_text.endswith("\n\n\n")  # avoid pathological chunking


# ── v2.14 Phase 6: code-block chunking hygiene (3 new shape-specific cases) ──


def test_code_unit_extends_to_end_of_block_within_safe_max(tmp_path):
    """A function whose end-of-unit fits within safe_max (1.5 × max_chars)
    must NOT be severed — the chunker extends past max_chars to finish the
    unit. No `partial_code` flag set."""
    from mmrag_v2.schema.ingestion_schema import ChunkType

    p = _make_processor(tmp_path)
    # technical_manual max=1200, safe_max=1800. Build a prose preamble of
    # ~1150 chars (just under max) followed by a 200-char code function so
    # the natural chunker close would land mid-function. Block extension
    # must keep the function whole.
    preamble = " ".join(["intro"] * 230)
    function = (
        "def calculate_total_with_tax(price, tax_rate):\n"
        "    subtotal = price\n"
        "    tax = subtotal * tax_rate\n"
        "    return subtotal + tax\n"
    )
    text = preamble + "\n\n" + function

    chunks = p._chunk_text_with_overlap(text)
    # Find the chunk containing the function — it must contain the whole def.
    fn_chunks = [
        c for c, t, _ in chunks
        if t == ChunkType.CODE and "calculate_total_with_tax" in c
    ]
    assert fn_chunks, "the code unit must appear in some chunk"
    assert all("return subtotal + tax" in c for c in fn_chunks), (
        "the chunk containing the def must also contain the return — "
        "block-extension policy must not sever the function"
    )
    # All chunks containing this function are NOT partial (extension fit safe_max).
    for c, _t, partial in chunks:
        if "calculate_total_with_tax" in c:
            assert partial is False, "function fit within safe_max — no partial flag"


def test_oversized_unit_splits_with_partial_code_flag(tmp_path):
    """A single fenced code block bigger than safe_max (1800 chars) must split
    at line boundaries, and EVERY chunk produced from that oversized unit
    must carry `partial_code=True`."""
    from mmrag_v2.schema.ingestion_schema import ChunkType

    p = _make_processor(tmp_path)
    # Build one fenced block of ~2400 chars — exceeds safe_max=1800.
    body_lines = []
    for i in range(120):
        body_lines.append(f"    item_{i:03d} = compute({i})  # line {i}")
    code = "```python\n" + "\n".join(body_lines) + "\n```\n"
    assert len(code) > 1800, "test fixture must exceed safe_max"

    chunks = p._chunk_text_with_overlap(code)
    code_entries = [(c, p_flag) for c, t, p_flag in chunks if t == ChunkType.CODE]
    assert len(code_entries) >= 2, "oversized unit must be split"
    assert all(p_flag is True for _c, p_flag in code_entries), (
        "every chunk that came from the oversized unit must carry "
        "partial_code=True so downstream consumers know it was severed"
    )


def test_indented_block_extends_when_fits_safe_max(tmp_path):
    """A long indented (non-fenced) Python block fits one chunk via the
    same end-of-unit extension policy when its total length stays within
    safe_max. No `partial_code` flag."""
    from mmrag_v2.schema.ingestion_schema import ChunkType

    p = _make_processor(tmp_path)
    # Build prose just under max_chars=1200, then an indented function of
    # ~400 chars. Total 1600 < safe_max=1800 → extend, single chunk.
    preamble = " ".join(["lead"] * 230)
    indented = (
        "    def process(records):\n"
        "        result = []\n"
        "        for record in records:\n"
        "            if record.valid:\n"
        "                result.append(record.value)\n"
        "        return result\n"
    )
    text = preamble + "\n\n" + indented

    chunks = p._chunk_text_with_overlap(text)
    fn_chunks = [
        (c, partial) for c, t, partial in chunks
        if t == ChunkType.CODE and "def process" in c
    ]
    assert fn_chunks, "indented block must be present in a chunk"
    for c, partial in fn_chunks:
        assert "return result" in c, (
            "indented block extension must keep the function whole — "
            "found chunk severed before `return result`"
        )
        assert partial is False, "indented block fit safe_max — no partial flag"


def test_english_from_not_misclassified_as_code(tmp_path):
    p = _make_processor(tmp_path)
    # "from ..." should not trigger code classification unless it's Python import syntax.
    prose = "In Example 1-1, the deck is made from all 13 ranks of each suit."
    assert p._looks_like_code(prose) is False
