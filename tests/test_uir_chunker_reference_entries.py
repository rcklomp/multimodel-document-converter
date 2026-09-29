"""PLAN_QUALITY_REMEDIATION_V1 WP-C2: reference-list entries are not cut mid-citation.

Defect (IRJET): the chunker splits joined text at the LAST sentence-end character before the
1400-char budget, which inside a bibliography is the ". " of "vol. 4, no. 8" - so entries are cut
mid-citation ("no." | "8, August 2013") and a chunk boundary lands inside a record. On the fresh
cloud run the right-column references were ONE element with INLINE labels (no newline before
"[11]"), so a rule that only recognises newline-separated entries does not fix the observed shape.

Contract: inside a references-class section (its heading is References/Bibliography/... or the
text carries >= 3 bracketed labels) the splitter prefers a boundary immediately BEFORE an entry
label. Everywhere else the splitter is unchanged (a numbered manual list, a body sentence
"Reference [10] presented a model" are never treated as entries).
"""

from __future__ import annotations

import re

from mmrag_v2.chunking.uir_chunker import _split_at_sentence_boundaries, chunk_universal_document
from mmrag_v2.universal.intermediate import (
    BoundingBox,
    Element,
    ElementType,
    ExtractionMethod,
    Modality,
    PageClassification,
    UniversalDocument,
    UniversalPage,
)

AUTHORS = [
    "Tsai",
    "Said",
    "Patel",
    "Ramaprabha",
    "Esram",
    "Ahmed",
    "Ishaque",
    "Miyatake",
    "Ngan",
    "Mandour",
    "Verma",
    "Kumar",
    "Singh",
    "Rao",
    "Das",
    "Gupta",
    "Nair",
]


def entries(n=17):
    out = []
    for i in range(1, n + 1):
        a = AUTHORS[(i - 1) % len(AUTHORS)]
        out.append(
            f"[{i}] {a}, X. and Y. Coauthor, “A study of photovoltaic tracking number {i}”, "
            f"Journal of Applied Energy Research, vol. {i}, no. {i % 9 + 1}, pp. {i}00-{i}09, 20{10 + i % 10}."
        )
    return out


def elem(idx, content, label="paragraph"):
    y0 = 10 + idx * 12
    return Element(
        type=ElementType.TEXT,
        content=content,
        bbox=BoundingBox(50, y0, 950, y0 + 8),
        confidence=0.9,
        extraction_method=ExtractionMethod.VLM,
        element_index=idx,
        source_label=label,
    )


def doc_of(elements):
    return UniversalDocument(
        doc_id="d",
        source_file="s.pdf",
        file_type="pdf",
        pages=[
            UniversalPage(
                page_number=1,
                elements=elements,
                classification=PageClassification.DIGITAL,
                dimensions=(1132, 1600),
            )
        ],
        total_pages=1,
    )


def chunks_of(elements):
    return [c for c in chunk_universal_document(doc_of(elements)) if c.modality == Modality.TEXT]


def intact(entry, chunks):
    return any(re.sub(r"\s+", " ", entry) in re.sub(r"\s+", " ", c.content) for c in chunks)


def test_newline_separated_entries_stay_intact():
    ents = entries()
    chunks = chunks_of(
        [elem(0, "REFERENCES", "heading")] + [elem(i + 1, e) for i, e in enumerate(ents)]
    )
    assert len(chunks) >= 2
    assert all(intact(e, chunks) for e in ents), [e[:8] for e in ents if not intact(e, chunks)]


def test_inline_entries_in_one_element_stay_intact():
    # The shape of the fresh cloud run: one element, labels inline after a sentence end.
    ents = entries()
    chunks = chunks_of([elem(0, "REFERENCES", "heading"), elem(1, " ".join(ents))])
    assert len(chunks) >= 2
    assert all(intact(e, chunks) for e in ents), [e[:8] for e in ents if not intact(e, chunks)]


def test_references_are_detected_by_label_density_without_a_references_heading():
    ents = entries()
    chunks = chunks_of([elem(0, " ".join(ents))])
    assert all(intact(e, chunks) for e in ents)


def test_a_chunk_never_ends_inside_an_entry_and_starts_with_a_label_after_the_first():
    ents = entries()
    chunks = chunks_of([elem(0, "REFERENCES", "heading"), elem(1, " ".join(ents))])
    for c in chunks[1:]:
        assert re.match(r"\[\d+\]\s", c.content), c.content[:40]


def test_body_text_with_inline_citations_is_split_exactly_as_before():
    body = " ".join(
        f"Reference [{i}] proposed a model number {i} for tracking the maximum power point of an array under shading."
        for i in range(1, 30)
    )
    with_flag = _split_at_sentence_boundaries(body, 1400, 20, entry_labels=False)
    legacy = _split_at_sentence_boundaries(body, 1400, 20)
    assert with_flag == legacy
    chunks = chunks_of([elem(0, "1. INTRODUCTION", "heading"), elem(1, body)])
    # not references-class: no boundary is moved to a bracketed number, so no chunk starts with one
    assert not any(re.match(r"\[\d+\]", c.content) for c in chunks)


def test_numbered_manual_steps_are_not_treated_as_reference_entries():
    # Outside a references-class section the splitter is byte-for-byte the legacy one. (The legacy
    # rule itself still cuts after a list number, "5." | "Remove ...": that is the weak sentence-end
    # family tracked as WP-C3 and is deliberately NOT changed here.)
    steps = "\n".join(
        f"{i}. Remove the cover and check the fastener number {i} before continuing with the next step."
        for i in range(1, 40)
    )
    chunks = chunks_of([elem(0, "INSTALLATION", "heading"), elem(1, steps)])
    legacy = [p.strip() for p in _split_at_sentence_boundaries("INSTALLATION\n" + steps, 1400, 20)]
    assert [c.content for c in chunks] == legacy


def test_author_year_bibliography_falls_back_to_the_existing_rule_without_loss():
    ents = [
        f"Author{i}, A. ({2000 + i}). Title of paper {i}. Journal, {i}({i % 4}), {i}-{i + 9}."
        for i in range(1, 40)
    ]
    chunks = chunks_of([elem(0, "References", "heading"), elem(1, "\n".join(ents))])
    joined = " ".join(c.content for c in chunks)
    assert all(
        e.split(".")[0] in joined for e in ents
    )  # nothing lost; entries may still be cut (documented blind spot)
