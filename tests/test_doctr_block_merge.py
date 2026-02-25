from types import SimpleNamespace

from mmrag_v2.ocr.enhanced_ocr_engine import EnhancedOCREngine


def _make_line(text: str, x0: float, y0: float, x1: float, y1: float):
    word = SimpleNamespace(value=text, confidence=0.99, geometry=((x0, y0), (x1, y1)))
    return SimpleNamespace(words=[word], geometry=((x0, y0), (x1, y1)))


def _make_block(line_specs):
    lines = [_make_line(*spec) for spec in line_specs]
    x0 = min(spec[1] for spec in line_specs)
    y0 = min(spec[2] for spec in line_specs)
    x1 = max(spec[3] for spec in line_specs)
    y1 = max(spec[4] for spec in line_specs)
    return SimpleNamespace(lines=lines, geometry=((x0, y0), (x1, y1)))


def test_merge_fragmented_doctr_blocks_merges_adjacent_single_line_blocks():
    engine = EnhancedOCREngine(enable_tesseract=False, enable_doctr=False)

    # Three fragmented lines that belong to one code block.
    page = SimpleNamespace(
        blocks=[
            _make_block([("def f(x):", 0.10, 0.10, 0.28, 0.12)]),
            _make_block([("return x + 1", 0.14, 0.13, 0.33, 0.15)]),
            _make_block([("print(f(3))", 0.10, 0.16, 0.30, 0.18)]),
        ]
    )

    confidences = []
    blocks, _ = engine._collect_doctr_page_blocks(page, confidences)
    merged = engine._merge_fragmented_doctr_blocks(blocks)

    assert len(blocks) == 3
    assert len(merged) == 1
    merged_lines = merged[0]["lines"]
    assert len(merged_lines) == 3

    block_left = min(lx for _, lx in merged_lines if lx is not None)
    second_offset = merged_lines[1][1] - block_left
    assert second_offset > engine._doctr_indent_threshold
    assert engine._spaces_from_doctr_offset(second_offset) > 0


def test_merge_fragmented_doctr_blocks_keeps_separate_columns_apart():
    engine = EnhancedOCREngine(enable_tesseract=False, enable_doctr=False)

    # Two nearby vertical blocks but in different columns.
    page = SimpleNamespace(
        blocks=[
            _make_block([("left column text", 0.08, 0.10, 0.30, 0.13)]),
            _make_block([("right column text", 0.62, 0.14, 0.88, 0.17)]),
        ]
    )

    confidences = []
    blocks, _ = engine._collect_doctr_page_blocks(page, confidences)
    merged = engine._merge_fragmented_doctr_blocks(blocks)

    assert len(blocks) == 2
    assert len(merged) == 2
