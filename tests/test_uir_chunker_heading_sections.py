"""PLAN_QUALITY_REMEDIATION_V1 WP-B1: headings are chunk boundaries.

Defect (measured on the corpus: median 22% of chunks, IRJET 12 of 30 wrong parents): the UIR
chunker joined ALL text elements between two visual elements, split the joined string by
character window and stamped the LAST heading of the group on every part, so a heading sat in the
middle or at the end of a chunk whose parent_heading it became.

Contract restored here (it is the ordered-attribution contract already tested for the OCR lane in
tests/test_ocr_path_heading_propagation.py): a heading element opens a new section; the section's
heading is the parent of every part of that section; elements before the first heading of the
group have no in-page heading (carry-forward / TOC fill them later, or they stay honestly null).
Consecutive heading elements with no body between them form ONE section (a split title stays
together and no micro heading-only chunk appears); its parent is the LAST heading of the run.
"""

from __future__ import annotations

import random

import pytest

from mmrag_v2.chunking.uir_chunker import chunk_universal_document
from mmrag_v2.universal.intermediate import (
    BoundingBox,
    Element,
    ElementType,
    ExtractionMethod,
    PageClassification,
    UniversalDocument,
    UniversalPage,
)


def el(idx, content, label="paragraph", etype=ElementType.TEXT):
    y0 = 10 + idx * 12
    return Element(
        type=etype,
        content=content,
        bbox=BoundingBox(50, y0, 950, y0 + 8),
        confidence=0.9,
        extraction_method=ExtractionMethod.VLM,
        element_index=idx,
        source_label=label,
    )


def doc_of(elements, page=1):
    return UniversalDocument(
        doc_id="d",
        source_file="s.pdf",
        file_type="pdf",
        pages=[
            UniversalPage(
                page_number=page,
                elements=elements,
                classification=PageClassification.DIGITAL,
                dimensions=(1132, 1600),
            )
        ],
        total_pages=1,
    )


def text_chunks(doc, **kw):
    from mmrag_v2.universal.intermediate import Modality

    return [c for c in chunk_universal_document(doc, **kw) if c.modality == Modality.TEXT]


HEAD = "heading"


# ----------------------------------------------------------------- the IRJET shape
def test_heading_in_the_middle_starts_its_own_chunk_and_owns_only_what_follows():
    # [caption, paragraph of section 2, heading III, paragraph of section III]
    elems = [
        el(0, "Fig. 3 Interfacing model to physical ports"),
        el(1, "Fig. 4 shows the simulation of the array and its groups."),
        el(2, "III. PROPOSED MPPT METHOD", HEAD),
        el(3, "To track the maximum power from the output curve, PSO is used."),
    ]
    chunks = text_chunks(doc_of(elems))
    head_chunk = next(c for c in chunks if "III. PROPOSED MPPT METHOD" in c.content)
    assert head_chunk.content.splitlines()[0] == "III. PROPOSED MPPT METHOD"
    assert head_chunk.parent_heading == "III. PROPOSED MPPT METHOD"
    assert "To track the maximum power" in head_chunk.content
    lead = next(c for c in chunks if "Fig. 4 shows" in c.content)
    assert "III. PROPOSED" not in lead.content
    assert lead.parent_heading is None  # honest null: no heading precedes it in this group


def test_leading_section_is_filled_by_carry_in_heading_not_by_the_later_heading():
    elems = [
        el(0, "Body text belonging to the previous chapter."),
        el(1, "2. NEW SECTION", HEAD),
        el(2, "Body of the new section."),
    ]
    chunks = text_chunks(
        doc_of(elems),
        carry_in_heading="1. PREVIOUS",
        carry_in_breadcrumb=["Document", "1. PREVIOUS"],
    )
    lead = next(c for c in chunks if "previous chapter" in c.content)
    new = next(c for c in chunks if "Body of the new section" in c.content)
    assert lead.parent_heading == "1. PREVIOUS"
    assert new.parent_heading == "2. NEW SECTION"


def test_consecutive_headings_form_one_section_with_the_last_as_parent():
    elems = [
        el(0, "CHAPTER ONE", HEAD),
        el(1, "1.1 Background of the study", HEAD),
        el(2, "Body paragraph text here."),
    ]
    chunks = text_chunks(doc_of(elems))
    assert len(chunks) == 1
    assert chunks[0].content.splitlines()[0] == "CHAPTER ONE"
    assert chunks[0].parent_heading == "1.1 Background of the study"


def test_heading_directly_before_a_visual_element_is_still_emitted():
    from mmrag_v2.universal.intermediate import Modality

    elems = [el(0, "4. RESULTS", HEAD), el(1, "chart", "figure", ElementType.IMAGE)]
    chunks = chunk_universal_document(doc_of(elems))
    assert [c.modality for c in chunks] == [Modality.TEXT, Modality.IMAGE]
    assert chunks[0].content == "4. RESULTS" and chunks[0].parent_heading == "4. RESULTS"


def test_over_budget_section_parts_all_inherit_the_section_heading_and_only_the_first_leads_with_it():
    body = " ".join(f"Sentence number {i} of the long section body." for i in range(80))
    elems = [
        el(0, "Intro paragraph before the heading."),
        el(1, "5. LONG SECTION", HEAD),
        el(2, body),
    ]
    chunks = text_chunks(doc_of(elems), max_chars=400)
    parts = [c for c in chunks if c.parent_heading == "5. LONG SECTION"]
    assert len(parts) >= 3
    assert parts[0].content.splitlines()[0] == "5. LONG SECTION"
    assert all("5. LONG SECTION" not in p.content for p in parts[1:])
    intro = next(c for c in chunks if "Intro paragraph" in c.content)
    assert intro.parent_heading is None


def test_bbox_of_a_section_chunk_covers_only_its_own_elements():
    elems = [
        el(0, "Alpha paragraph before."),
        el(1, "H. SECOND", HEAD),
        el(2, "Beta paragraph after."),
    ]
    chunks = text_chunks(doc_of(elems))
    lead = next(c for c in chunks if "Alpha" in c.content)
    sec = next(c for c in chunks if "Beta" in c.content)
    lead_bb, sec_bb = list(lead.locator.bbox), list(sec.locator.bbox)
    assert lead_bb != sec_bb
    assert (
        lead_bb[3] < sec_bb[1] + 1
    )  # the earlier section ends before the later one starts (page order)


# ----------------------------------------------------------------- properties (seeded)
def _gen(seed):
    rng = random.Random(seed)
    elems, kinds = [], []
    n = rng.randint(4, 14)
    for i in range(n):
        is_head = rng.random() < 0.25 and (i == 0 or kinds[-1] != "H")  # no stacked headings here
        if is_head:
            elems.append(el(i, f"H{i} SECTION TITLE", HEAD))
            kinds.append("H")
        else:
            words = rng.choice([8, 30, 90, 200])
            elems.append(
                el(i, f"S{i}. " + " ".join(f"w{i}x{j}" for j in range(words)), "paragraph")
            )
            kinds.append("B")
    return elems, kinds


SEEDS = list(range(40))


@pytest.mark.parametrize("seed", SEEDS)
def test_property_a_heading_never_follows_body_text_inside_a_chunk(seed):
    elems, kinds = _gen(seed)
    heads = {e.content for e, k in zip(elems, kinds) if k == "H"}
    for c in text_chunks(doc_of(elems), max_chars=600):
        lines = c.content.splitlines()
        seen_body = False
        for ln in lines:
            if ln in heads:
                assert not seen_body, f"seed {seed}: heading {ln!r} after body inside one chunk"
            elif ln.strip():
                seen_body = True


@pytest.mark.parametrize("seed", SEEDS)
def test_property_b_parent_is_the_nearest_preceding_heading_or_none(seed):
    elems, kinds = _gen(seed)
    order = {e.content: i for i, e in enumerate(elems)}
    for c in text_chunks(doc_of(elems), max_chars=600):
        first_idx = (
            min(order[k] for k in order if k in c.content)
            if any(k in c.content for k in order)
            else None
        )
        # index of the first ELEMENT whose text appears in the chunk (headings are unique strings;
        # body elements start with a unique "S<i>." token)
        idxs = [
            i
            for i, e in enumerate(elems)
            if (e.content in c.content)
            or (e.content.split(" ")[0] in c.content and kinds[i] == "B")
        ]
        if not idxs:
            continue
        start = min(idxs)
        expected = None
        for j in range(start, -1, -1):
            if kinds[j] == "H":
                expected = elems[j].content
                break
        # a part that starts mid-section (over-budget continuation) still belongs to its section
        assert c.parent_heading == expected, f"seed {seed}: chunk starting at element {start}"


@pytest.mark.parametrize("seed", SEEDS)
def test_property_c_every_element_text_survives_in_order(seed):
    elems, _ = _gen(seed)
    joined = "\n".join(c.content for c in text_chunks(doc_of(elems), max_chars=600))
    pos = -1
    for e in elems:
        marker = e.content.split(" ")[0] if e.source_label != HEAD else e.content
        nxt = joined.find(marker, pos + 1)
        assert nxt > pos, f"seed {seed}: element {marker!r} lost or reordered"
        pos = nxt


def test_headings_that_are_not_labelled_as_headings_do_not_open_sections():
    # engine-labelled only: a line that merely LOOKS like a heading inside a paragraph element is body
    elems = [el(0, "Body one."), el(1, "IV. SIMULATION", "paragraph"), el(2, "Body two.")]
    chunks = text_chunks(doc_of(elems))
    assert len(chunks) == 1 and chunks[0].parent_heading is None
