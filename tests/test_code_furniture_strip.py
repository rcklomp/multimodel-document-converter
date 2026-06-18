"""Code-chunk furniture stripping (uir_chunker._strip_code_furniture, 2026-06-18).

The VLM intermittently transcribes a page running-header / page-number / publisher
caption INTO a code element (it also emits it as its own heading element, so removal
is LOSSLESS). These tests pin: furniture lines are dropped, real code (incl. shell
and indentation) is kept verbatim, and a code line that merely mentions a furniture
word is never stripped.
"""
from __future__ import annotations

from mmrag_v2.chunking.uir_chunker import _strip_code_furniture as strip


def test_running_header_and_pagenumber_stripped_keeps_code():
    # The exact Hao p71 shape: page number + running header inlined above the code.
    raw = ("50\n"
           "                          Chapter 3  Building applications on Kubernetes\n"
           "        docker run -it -p 8081:8083 hello-joker:v1")
    out = strip(raw)
    assert "Chapter 3" not in out
    assert "50" not in out.split("\n")  # bare page-number line gone
    assert "        docker run -it -p 8081:8083 hello-joker:v1" in out  # code + indent intact


def test_listing_and_figure_captions_stripped():
    raw = ("Listing 3.2  An example Dockerfile for Python applications\n"
           "FROM python:3.10-slim-buster\n"
           "Figure 3.3 Process of building the image")
    out = strip(raw)
    assert "Listing 3.2" not in out
    assert "Figure 3.3" not in out
    assert "FROM python:3.10-slim-buster" in out


def test_real_code_unchanged():
    raw = "def spades_high(card):\n    rank = ranks.index(card.rank)\n    return rank * 4"
    assert strip(raw) == raw


def test_comment_mentioning_chapter_is_kept():
    # A comment that references a chapter must NOT be stripped (starts with #, and it
    # is not a running-header line).
    raw = "# See Chapter 3 for the full example\nx = compute()"
    assert strip(raw) == raw


def test_interior_bare_number_not_adjacent_to_furniture_is_kept():
    # A lone integer between real code lines could be data; only strip page numbers
    # at block edges or next to furniture.
    raw = "x = [\n    1,\n    42,\n    3,\n]"
    assert "42" in strip(raw)


def test_all_furniture_collapses_to_empty():
    raw = "66\n                               Chapter 3  Building applications on Kubernetes"
    assert strip(raw).strip() == ""
