"""Build expected_reading_order.txt for Harry Potter pages 1-30.

Walks the PDF with pymupdf, detects paragraph boundaries by indent
(a line whose x0 sits 5-30pt right of the body's left margin starts
a new paragraph), and emits one anchor per paragraph in top-to-bottom
order. Anchors are the first 60 ASCII-normalized characters of each
paragraph and are designed to be unique enough to locate unambiguously
inside a converted JSONL chunk's body text.

Lines whose x0 is far right of the body left (drop-cap displaced) are
treated as paragraph continuations, not new paragraphs. Single-character
blocks (orphan drop-cap glyphs) and running heads/footers (top y < 80
or bottom y > 580) are skipped.

The resulting file is the binding ground truth: chunks produced by
the pipeline must reference these anchors in the same relative order.

Run from repo root:
    python tests/fixtures/harry_potter_pages_1_to_30/build_fixture.py

Deterministic: re-running on the same PDF produces a byte-identical
expected_reading_order.txt.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pymupdf

REPO_ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = REPO_ROOT / "data" / "digital_literature" / "HarryPotter_and_the_Sorcerers_Stone.pdf"
OUT_PATH = Path(__file__).resolve().parent / "expected_reading_order.txt"

PAGE_RANGE = range(1, 31)
# Body-text starts at page 13 (chapter 1 "THE BOY WHO LIVED"). Pages 1-12
# are front matter (cover photos, title spreads, ALSO BY list, copyright,
# TOC, half-title) - those pages are display-typography fragments where
# the chunker's behavior is necessarily lossy and where short anchors
# like "Harry Potter" collide with multiple occurrences. The post-Docling
# reading-order pass is scoped to body text per
# `docs/PLAN_DOCLING_POSTPROCESSOR.md`; the fixture should match that scope.
BODY_PAGE_START = 13
ANCHOR_LEN = 60
INDENT_MIN = 5.0
INDENT_MAX = 30.0
HEADER_Y_MAX = 80.0
FOOTER_Y_MIN = 575.0


def _line_text(line) -> str:
    return "".join(span["text"] for span in line.get("spans", [])).strip()


def _ascii_normalize(text: str) -> str:
    return (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", "-")
        .replace("–", "-")
        .replace(" ", " ")
        .replace("…", "...")
    )


def _body_left(blocks) -> float:
    xs: List[float] = []
    for block in blocks:
        for line in block.get("lines", []):
            if _line_text(line):
                xs.append(line["bbox"][0])
    if not xs:
        return 0.0
    xs.sort()
    return xs[len(xs) // 2]


def extract_anchors(page: pymupdf.Page) -> List[str]:
    raw = page.get_text("dict")
    blocks = [
        b for b in raw.get("blocks", [])
        if b.get("type", 0) == 0 and any(_line_text(line) for line in b.get("lines", []))
    ]
    if not blocks:
        return []

    body_left = _body_left(blocks)

    def _is_header_or_footer(block) -> bool:
        y0, y1 = block["bbox"][1], block["bbox"][3]
        if y1 < HEADER_Y_MAX:
            return True
        if y0 > FOOTER_Y_MIN:
            return True
        return False

    def _is_dropcap_block(block) -> bool:
        text = " ".join(
            _line_text(line) for line in block.get("lines", []) if _line_text(line)
        ).strip()
        return len(text) == 1 and text.isupper()

    body_blocks = [
        block for block in blocks
        if not _is_header_or_footer(block) and not _is_dropcap_block(block)
    ]
    body_blocks.sort(key=lambda b: b["bbox"][1])

    anchors: List[str] = []
    for block in body_blocks:
        lines = [line for line in block.get("lines", []) if _line_text(line)]
        para_lines: List[str] = []

        def flush():
            if not para_lines:
                return
            joined = " ".join(" ".join(para_lines).split())
            if not joined:
                return
            anchors.append(_ascii_normalize(joined)[:ANCHOR_LEN])

        for line in lines:
            x0 = line["bbox"][0]
            text = _line_text(line)
            indent = x0 - body_left
            is_paragraph_indent = INDENT_MIN <= indent <= INDENT_MAX
            if para_lines and is_paragraph_indent:
                flush()
                para_lines = [text]
            else:
                para_lines.append(text)
        flush()
    return anchors


def main() -> None:
    doc = pymupdf.open(PDF_PATH)
    rows: List[Tuple[int, str]] = []
    for page_no in PAGE_RANGE:
        if page_no < BODY_PAGE_START:
            continue
        anchors = extract_anchors(doc[page_no - 1])
        for anchor in anchors:
            rows.append((page_no, anchor))
    doc.close()

    lines = [
        "# Harry Potter Pages 1-30 - expected paragraph reading order.",
        "# Generated from PDF y-coordinates by build_fixture.py.",
        "# Format: <page>\\t<anchor>",
        f"# Scope: body text only (pages {BODY_PAGE_START}-30). Pages 1-12 are",
        "# display-typography front matter (covers, title spreads, ALSO BY,",
        "# copyright, TOC) where the post-Docling reading-order pass does",
        "# not apply.",
        "# Anchors are the first 60 ASCII-normalized chars of each paragraph",
        "# in PDF top-to-bottom reading order. Edit only by re-running the",
        "# builder against the source PDF.",
        "",
    ]
    for page, anchor in rows:
        lines.append(f"{page}\t{anchor}")
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} anchors -> {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
