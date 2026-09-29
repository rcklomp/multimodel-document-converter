"""PLAN_QUALITY_REMEDIATION_V1 WP-B2 (call-site bridge): element-level furniture removal reaches the
export AND is accounted for by the QA-CHECK-01 quality-filter tracker as NOISE_PATTERN.

Registration matters because the token-balance check compares source tokens with emitted tokens:
an unregistered removal reads as lost content. The pass runs inside the real ``process_pdf`` path with a
stubbed extractor, not only through ``chunk_universal_document`` in isolation.
"""

from __future__ import annotations

import json

import fitz

from mmrag_v2.validators.quality_filter_tracker import FilterCategory, QualityFilterTracker

HEADER = (
    "International Research Journal of Engineering and Technology (IRJET) e-ISSN: 2395-0056 "
    "Volume: 04 Issue: 02 | Feb-2017 www.irjet.net p-ISSN: 2395-0072"
)
BODY = [
    "The photovoltaic array is modelled with a single diode equivalent circuit and simulated under shading.",
    "A boost converter tracks the maximum power point with a particle swarm search over the duty cycle.",
    "Partial shading creates multiple peaks on the power voltage curve which defeat hill climbing methods.",
    "The bypass diodes conduct in shaded modules and the string current is limited by the weakest cell.",
    "Simulation results in MATLAB confirm faster convergence and lower steady state ripple for the swarm.",
]


def _pdf(path, pages=5):
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 60), HEADER[:60], fontsize=8)
        page.insert_text((72, 120), BODY[i], fontsize=10)
        page.insert_text((72, 800), f"(c) 2017 IRJET Page {i + 1}", fontsize=8)
    doc.save(str(path))
    doc.close()


def _uir(pages=5):
    from mmrag_v2.universal.intermediate import (
        BoundingBox,
        Element,
        ElementType,
        ExtractionMethod,
        PageClassification,
        UniversalDocument,
        UniversalPage,
    )

    def el(i, content, y):
        return Element(
            type=ElementType.TEXT,
            content=content,
            bbox=BoundingBox(60, y, 940, y + 30),
            confidence=0.9,
            extraction_method=ExtractionMethod.VLM,
            element_index=i,
            source_label="paragraph",
        )

    ups = []
    for n in range(1, pages + 1):
        ups.append(
            UniversalPage(
                page_number=n,
                elements=[
                    el(0, HEADER, 20),
                    el(1, BODY[n - 1], 200),
                    el(2, f"(c) 2017, IRJET | Impact Factor value: 5.181 | Page {n}", 930),
                    el(3, str(n), 960),
                ],
                classification=PageClassification.DIGITAL,
                dimensions=(1132, 1600),
            )
        )
    return UniversalDocument(
        doc_id="x", source_file="s.pdf", file_type="pdf", pages=ups, total_pages=pages
    )


def test_furniture_is_absent_from_the_export_and_registered_as_noise_pattern(tmp_path, monkeypatch):
    import mmrag_v3.processor as v3p
    from mmrag_v2.batch_processor import BatchProcessor

    pdf = tmp_path / "s.pdf"
    _pdf(pdf)
    monkeypatch.setattr(v3p, "extract", lambda path: _uir())
    calls = []
    original = QualityFilterTracker.track_filtered_content

    def spy(self, content, page_number, category, chunk_id="unknown"):
        calls.append((content, page_number, category, chunk_id))
        return original(self, content, page_number, category, chunk_id)

    monkeypatch.setattr(QualityFilterTracker, "track_filtered_content", spy)
    out = tmp_path / "out"
    BatchProcessor(output_dir=str(out), batch_size=10, vision_provider="none").process_pdf(pdf)

    text = "\n".join(
        json.loads(ln).get("content") or ""
        for ln in (out / "ingestion.jsonl").read_text("utf-8").splitlines()
    )
    assert "International Research Journal" not in text
    assert "Impact Factor value" not in text
    for sentence in BODY:
        assert sentence in text  # the body survives
    furniture = [c for c in calls if c[3].startswith("furniture_")]
    assert len(furniture) == 15  # 5 headers + 5 footers + 5 folios
    assert {c[2] for c in furniture} == {FilterCategory.NOISE_PATTERN}
    assert sorted({c[1] for c in furniture}) == [1, 2, 3, 4, 5]
    assert sum(1 for c in furniture if c[0] == HEADER) == 5
