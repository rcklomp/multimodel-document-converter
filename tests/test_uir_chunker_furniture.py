"""PLAN_QUALITY_REMEDIATION_V1 WP-B2: running furniture is removed at ELEMENT level.

Defect (IRJET): the chunk-level F1 filter runs last and only sees chunks of <= 70 characters, so the
151-character journal header, the footer and the page folio were glued into body chunks before it ran
(11 of 30 text chunks). Contract: the chunker drops furniture ELEMENTS before chunking (engine label
first, then a rank-based repetition rule), never touches headings, captions, code, tables or a page's
last content, reports what it removed, and does not mutate its input.
"""

from __future__ import annotations

import copy
import random

import pytest

from mmrag_v2.chunking.furniture import find_running_furniture, normalize_signature
from mmrag_v2.chunking.uir_chunker import chunk_universal_document
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

HEADER = (
    "International Research Journal of Engineering and Technology (IRJET) e-ISSN: 2395-0056 "
    "Volume: 04 Issue: 02 | Feb-2017 www.irjet.net p-ISSN: 2395-0072"
)


def footer(n):
    return (
        f"© 2017, IRJET | Impact Factor value: 5.181 | ISO 9001:2008 Certified Journal | Page {n}"
    )


def el(idx, content, label="paragraph", etype=ElementType.TEXT, metadata=None):
    return Element(
        type=etype,
        content=content,
        bbox=BoundingBox(50, 10 + idx * 12, 950, 18 + idx * 12),
        confidence=0.9,
        extraction_method=ExtractionMethod.VLM,
        element_index=idx,
        source_label=label,
        metadata=metadata or {},
    )


def make_page(n, body, *, header=True, foot=True, folio=True):
    texts = ([(HEADER, "paragraph")] if header else []) + [(b, "paragraph") for b in body]
    if foot:
        texts.append((footer(n), "paragraph"))
    if folio:
        texts.append((str(n), "paragraph"))
    return UniversalPage(
        page_number=n,
        elements=[el(i, t, lab) for i, (t, lab) in enumerate(texts)],
        classification=PageClassification.DIGITAL,
        dimensions=(1132, 1600),
    )


def make_doc(pages):
    return UniversalDocument(
        doc_id="d", source_file="s.pdf", file_type="pdf", pages=pages, total_pages=len(pages)
    )


_SUBJECTS = [
    "photovoltaic array",
    "boost converter",
    "shading pattern",
    "particle swarm",
    "battery bank",
    "inverter stage",
    "irradiance sensor",
    "thermal model",
]
_VERBS = [
    "explains",
    "compares",
    "simulates",
    "derives",
    "measures",
    "validates",
    "optimises",
    "bounds",
]


def body_for(n):
    a, b = _SUBJECTS[n % len(_SUBJECTS)], _SUBJECTS[(n * 3 + 1) % len(_SUBJECTS)]
    v = _VERBS[n % len(_VERBS)]
    return [
        f"Section {n} {v} the {a} under partial shading and reports how the {b} responds in each case.",
        f"A second paragraph on page {n} discusses tracking of the {b} with the {a} model.",
    ]


def texts(doc, **kw):
    return [c for c in chunk_universal_document(doc, **kw) if c.modality == Modality.TEXT]


def joined(doc, **kw):
    return "\n".join(c.content for c in texts(doc, **kw))


# ------------------------------------------------------------------ positive: the IRJET shape
def test_running_header_footer_and_folio_are_removed_and_body_survives():
    doc = make_doc([make_page(n, body_for(n)) for n in range(1, 6)])
    report = []
    out = joined(doc, furniture_report=report)
    assert "International Research Journal" not in out
    assert "Impact Factor value" not in out
    assert "e-ISSN" not in out and "p-ISSN" not in out
    for n in range(1, 6):
        assert f"Section {n} " in out and f"A second paragraph on page {n}" in out
    assert len(report) == 15  # 5 headers + 5 footers + 5 folios
    assert {d.rule for d in report} == {"repetition"}


def test_without_the_pass_the_furniture_is_still_glued_into_the_output():
    # The defect this WP removes: guard against the test above being vacuous.
    doc = make_doc([make_page(n, body_for(n)) for n in range(1, 6)])
    assert "International Research Journal" in joined(doc, drop_furniture=False)


def test_the_input_document_is_not_mutated_and_the_pass_is_idempotent():
    doc = make_doc([make_page(n, body_for(n)) for n in range(1, 6)])
    before = copy.deepcopy(doc)
    report = []
    chunk_universal_document(doc, furniture_report=report)
    assert doc == before
    # idempotence: the survivors contain no further furniture
    survivors = make_doc(
        [make_page(n, body_for(n), header=False, foot=False, folio=False) for n in range(1, 6)]
    )
    assert find_running_furniture(survivors.pages) == []


def test_drop_furniture_false_is_byte_identical_to_the_legacy_output():
    doc = make_doc([make_page(n, body_for(n)) for n in range(1, 6)])
    legacy = [c.content for c in texts(doc, drop_furniture=False)]
    assert legacy == [c.content for c in texts(copy.deepcopy(doc), drop_furniture=False)]
    assert legacy != [c.content for c in texts(doc)]


# ------------------------------------------------------------------ engine labels (primary signal)
def test_an_engine_footer_label_removes_a_short_element_without_repetition():
    page = make_page(1, body_for(1), header=False, foot=False, folio=False)
    page.elements.append(el(9, "Page 1 of 7", label="footer"))
    doc = make_doc([page])
    report = []
    assert "Page 1 of 7" not in joined(doc, furniture_report=report)
    assert [d.rule for d in report] == ["label"]


def test_a_vlm_header_type_kept_as_original_vlm_type_counts_as_a_label():
    page = make_page(1, body_for(1), header=False, foot=False, folio=False)
    page.elements.insert(
        0, el(9, "Journal of Solar Energy 2017", metadata={"original_vlm_type": "header"})
    )
    assert "Journal of Solar Energy" not in joined(make_doc([page]))


def test_a_label_on_a_long_element_does_not_delete_prose():
    long_text = "A footnote-like paragraph that the engine mislabelled as a footer. " * 6
    assert len(long_text) > 200
    page = make_page(1, body_for(1), header=False, foot=False, folio=False)
    page.elements.append(el(9, long_text, label="footer"))
    assert "mislabelled as a footer" in joined(make_doc([page]))


# ------------------------------------------------------------------ negatives (what must NOT be removed)
def test_a_string_on_only_two_pages_is_not_furniture():
    pages = [
        make_page(n, body_for(n), header=(n <= 2), foot=False, folio=False) for n in range(1, 6)
    ]
    assert "International Research Journal" in joined(make_doc(pages))


def test_heading_labelled_elements_are_never_removed_even_when_repeated():
    pages = []
    for n in range(1, 6):
        page = make_page(n, body_for(n), header=False, foot=False, folio=False)
        page.elements.insert(0, el(9, "RESULTS AND DISCUSSION", label="section_header"))
        pages.append(page)
    assert joined(make_doc(pages)).count("RESULTS AND DISCUSSION") == 5


def test_a_running_header_equal_to_a_chapter_title_heading_is_kept_and_keeps_its_role():
    # HarryPotter shape: the chapter-opening element is the heading; the SAME string is the running
    # header on later pages. It is the source of the section heading and must stay (decision D-31).
    title = "THE BOY WHO LIVED"
    pages = []
    for n in range(1, 6):
        page = make_page(n, body_for(n), header=False, foot=False, folio=False)
        if n == 1:
            page.elements.insert(0, el(9, title, label="section_header"))
        else:
            page.elements.insert(0, el(9, title, label="paragraph"))
        pages.append(page)
    doc = make_doc(pages)
    report = []
    chunks = texts(doc, furniture_report=report)
    assert report == []
    legacy = texts(doc, drop_furniture=False)
    assert [c.parent_heading for c in chunks] == [c.parent_heading for c in legacy]


def test_a_repeated_figure_sub_caption_is_content_not_furniture():
    pages = []
    for n in range(1, 6):
        page = make_page(n, body_for(n), header=False, foot=False, folio=False)
        page.elements.append(el(9, "(a) Normalized throughput. Higher is better."))
        pages.append(page)
    assert joined(make_doc(pages)).count("(a) Normalized throughput") == 5


def test_table_and_code_elements_are_never_furniture():
    pages = []
    for n in range(1, 6):
        page = make_page(n, body_for(n), header=False, foot=False, folio=False)
        page.elements.insert(
            0, el(8, "| a | b |\n|---|---|\n| 1 | 2 |", "table", ElementType.TABLE)
        )
        page.elements.append(el(9, "```", metadata={"promoted_modality": "code"}))
        pages.append(page)
    report = []
    chunk_universal_document(make_doc(pages), furniture_report=report)
    assert report == []


def test_a_page_never_loses_its_only_content_to_the_pass():
    # a page that consists of nothing but a folio keeps it (no manufactured MISSING_PAGES)
    pages = [make_page(n, body_for(n)) for n in range(1, 5)]
    pages.append(
        UniversalPage(
            page_number=5,
            elements=[el(0, "5")],
            classification=PageClassification.DIGITAL,
            dimensions=(1132, 1600),
        )
    )
    report = []
    chunks = texts(make_doc(pages), furniture_report=report)
    assert all(d.page != 5 for d in report)  # page 5 keeps its folio ...
    assert any(c.content.strip() == "5" for c in chunks)  # ... which reaches the output
    assert len([d for d in report if d.page == 1]) == 3  # while the other pages lose theirs


def test_a_single_margin_citation_is_not_furniture():
    page = make_page(1, body_for(1), header=False, foot=False, folio=False)
    page.elements.append(el(9, "[12] Mandour, M. Solar cells, 2013."))
    doc = make_doc(
        [page] + [make_page(n, body_for(n), header=False, foot=False, folio=False) for n in (2, 3)]
    )
    assert "[12] Mandour" in joined(doc)


# ------------------------------------------------------------------ properties (seeded)
_WORDS = (
    "array shading tracker converter voltage current panel cell module inverter battery thermal "
    "sensor loop control duty cycle ripple filter diode bypass string parallel series peak curve "
    "irradiance temperature efficiency swarm particle velocity position fitness objective model "
    "simulation matlab block scope input output signal gain limit sample period noise offset"
).split()


def _sentence(rng, tag):
    return f"{tag} " + " ".join(rng.choice(_WORDS) for _ in range(9)) + "."


def _gen(seed):
    rng = random.Random(seed)
    npages = rng.randint(4, 8)
    pages = []
    body_texts = []
    for n in range(1, npages + 1):
        body = [_sentence(rng, f"P{n}S{i}") for i in range(rng.randint(1, 4))]
        body_texts.extend(body)
        pages.append(
            make_page(n, body, header=True, foot=rng.random() < 0.8, folio=rng.random() < 0.8)
        )
    return make_doc(pages), body_texts


@pytest.mark.parametrize("seed", range(25))
def test_property_body_text_always_survives_and_the_header_never_does(seed):
    doc, body = _gen(seed)
    out = joined(doc)
    for b in body:
        assert b in out, f"seed {seed}: body text lost: {b}"
    assert "International Research Journal" not in out


@pytest.mark.parametrize("seed", range(25))
def test_property_removed_tokens_are_a_subset_of_the_source_and_idempotent(seed):
    doc, _ = _gen(seed)
    source_tokens = {t for pg in doc.pages for e in pg.elements for t in e.content.split()}
    report = []
    chunk_universal_document(doc, furniture_report=report)
    for d in report:
        assert set(d.text.split()) <= source_tokens
    survivors = make_doc(
        [
            UniversalPage(
                page_number=pg.page_number,
                elements=[
                    e
                    for i, e in enumerate(pg.elements)
                    if (pg.page_number, i) not in {(d.page, d.position) for d in report}
                ],
                classification=pg.classification,
                dimensions=pg.dimensions,
            )
            for pg in doc.pages
        ]
    )
    assert find_running_furniture(survivors.pages) == []


def test_signature_normalizes_digits_case_and_whitespace():
    assert normalize_signature("Page  12 |  Feb 2017") == normalize_signature("page 7 | feb 1999")


def test_a_copyright_footer_starting_with_a_parenthesised_letter_is_still_furniture():
    # "(c) 2017 ..." must not be mistaken for a "(a) sub-caption": the guard needs a letter after it
    pages = []
    for n in range(1, 5):
        page = make_page(n, body_for(n), header=False, foot=False, folio=False)
        page.elements.append(el(9, f"(c) 2017 IRJET | Impact Factor value: 5.181 | Page {n}"))
        pages.append(page)
    out = joined(make_doc(pages))
    assert "Impact Factor value" not in out
