"""PLAN_QUALITY_REMEDIATION_V1 WP-C4: honest output provenance.

Defects (all measured on the IRJET runs): the header stamped the SCHEMA version (2.7.0) as
`pipeline_version` while the engine is 2.16.0, `config_hash` was defined but never assigned, two of
the three header writers stamped neither field, and a page demoted to Docling by an expired key or an
exhausted rate limit left the header saying hybrid / degraded=0 / fallback=null, so a run served by
the wrong lane looked valid. `config_hash` here makes a stale output RECORDABLE; the staleness rule
itself stays route-based until owner decision D-1.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest

from mmrag_v2.provenance import compute_config_hash, file_sha256
from mmrag_v2.version import __engine_version__, __schema_version__
from mmrag_v3.processor import _stamp_routing

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ config_hash
def test_config_hash_is_stable_for_equal_options_regardless_of_key_order():
    a = compute_config_hash({"render_cap_px": 1600, "vlm_model": "m"})
    b = compute_config_hash({"vlm_model": "m", "render_cap_px": 1600})
    assert a == b and re.fullmatch(r"[0-9a-f]{64}", a)


def test_config_hash_changes_when_only_the_render_cap_or_the_model_changes():
    base = {"render_cap_px": 1600, "vlm_model": "m"}
    assert compute_config_hash(base) != compute_config_hash({**base, "render_cap_px": 2000})
    assert compute_config_hash(base) != compute_config_hash({**base, "vlm_model": "n"})


@pytest.mark.parametrize(
    "name",
    ["api_key", "VLM_ENDPOINT", "secret_value", "base_url", "auth_token", "db_host", "password"],
)
def test_config_hash_refuses_credential_and_endpoint_names(name):
    with pytest.raises(ValueError):
        compute_config_hash({name: "x"})


def test_file_sha256_matches_hashlib_and_is_none_for_a_missing_file(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    assert file_sha256(f) == hashlib.sha256(b"hello").hexdigest()
    assert file_sha256(tmp_path / "missing") is None


# ------------------------------------------------------------------ routing stamp
def _doc():
    return SimpleNamespace(metadata=SimpleNamespace(extra={}))


def _engine(decisions, model="qwen3-vl-flash"):
    provider = SimpleNamespace(config=SimpleNamespace(model=model))
    return SimpleNamespace(
        last_routing_decisions=decisions, vlm_engine=SimpleNamespace(_provider=provider)
    )


def test_routing_stamp_counts_vlm_served_and_demoted_pages_and_the_model():
    decisions = [
        (1, "vlm", "images=1"),
        (2, "docling_fallback", "vlm_failed: 401"),
        (3, "docling", "prose"),
        (4, "vlm", "tables=1"),
    ]
    d = _doc()
    _stamp_routing(d, _engine(decisions))
    assert d.metadata.extra["extraction_vlm_served_pages"] == 2
    assert d.metadata.extra["extraction_demoted_pages"] == 1
    assert d.metadata.extra["extraction_vlm_model"] == "qwen3-vl-flash"


def test_routing_stamp_recognises_the_mineru_qwen_vocabulary():
    decisions = [
        (1, "qwen_code", "mono"),
        (2, "qwen_code_block", "block"),
        (3, "mineru", "x"),
        (4, "mineru_fallback", "qwen_failed"),
    ]
    d = _doc()
    _stamp_routing(d, _engine(decisions))
    assert d.metadata.extra["extraction_vlm_served_pages"] == 2
    assert d.metadata.extra["extraction_demoted_pages"] == 1


def test_an_engine_without_a_routing_log_stamps_nothing():
    d = _doc()
    _stamp_routing(d, SimpleNamespace())
    assert d.metadata.extra == {}


def test_a_provider_that_is_not_built_yet_still_stamps_counts_without_a_model():
    d = _doc()
    _stamp_routing(
        d,
        SimpleNamespace(
            last_routing_decisions=[(1, "vlm", "x")], vlm_engine=SimpleNamespace(_provider=None)
        ),
    )
    assert d.metadata.extra["extraction_vlm_served_pages"] == 1
    assert "extraction_vlm_model" not in d.metadata.extra


# ------------------------------------------------------------------ audit constants
def _audit_module():
    spec = importlib.util.spec_from_file_location(
        "qa_conversion_audit_wp_c4", REPO / "scripts" / "qa_conversion_audit.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qa_conversion_audit_wp_c4"] = mod  # dataclasses resolve their module here
    spec.loader.exec_module(mod)
    return mod


def _audit_issues(tmp_path, pipeline_version, schema_version):
    header = {
        "object_type": "ingestion_metadata",
        "schema_version": schema_version,
        "pipeline_version": pipeline_version,
        "source_file": "d.pdf",
        "source_file_hash": "abc",
        "profile_type": "academic_whitepaper",
        "total_pages": 1,
    }
    chunk = {
        "chunk_id": "c1",
        "doc_id": "d",
        "modality": "text",
        "content": "Some body text that is long enough to count as content here.",
        "schema_version": schema_version,
        "metadata": {
            "page_number": 1,
            "chunk_type": "paragraph",
            "hierarchy": {"parent_heading": "Intro", "breadcrumb_path": [], "level": 1},
        },
    }
    p = tmp_path / "ingestion.jsonl"
    p.write_text(json.dumps(header) + "\n" + json.dumps(chunk) + "\n", encoding="utf-8")
    r = _audit_module().audit(p)
    return [i for i in getattr(r, "issues", []) if "PROVENANCE" in str(i)]


def test_audit_accepts_the_engine_version_as_pipeline_version_and_the_schema_version_separately(
    tmp_path,
):
    assert _audit_issues(tmp_path, __engine_version__, __schema_version__) == []


def test_audit_flags_a_pipeline_version_that_is_really_the_schema_version(tmp_path):
    issues = _audit_issues(tmp_path, __schema_version__, __schema_version__)
    assert issues and any("pipeline_version" in str(i) for i in issues)


# ------------------------------------------------------------------ end to end through process_pdf (synthetic PDF, stubbed extractor)
def _synthetic_pdf(path):
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"Synthetic page {i + 1} heading", fontsize=18)
        page.insert_text(
            (72, 140),
            "A body paragraph long enough to be a chunk of text on this page. " * 3,
            fontsize=10,
        )
    doc.save(str(path))
    doc.close()


def _uir(pages=2):
    from mmrag_v2.universal.intermediate import (
        BoundingBox,
        DocumentMetadata,
        Element,
        ElementType,
        ExtractionMethod,
        PageClassification,
        UniversalDocument,
        UniversalPage,
    )

    def el(i, content, label="paragraph"):
        return Element(
            type=ElementType.TEXT,
            content=content,
            bbox=BoundingBox(60, 100 + i * 60, 900, 150 + i * 60),
            confidence=0.9,
            extraction_method=ExtractionMethod.VLM,
            element_index=i,
            source_label=label,
        )

    ups = [
        UniversalPage(
            page_number=n,
            elements=[
                el(0, f"Synthetic page {n} heading", "heading"),
                el(1, "A body paragraph long enough to be a chunk of text on this page. " * 3),
            ],
            classification=PageClassification.DIGITAL,
            dimensions=(1132, 1600),
        )
        for n in range(1, pages + 1)
    ]
    extra = {
        "extraction_engine": "hybrid",
        "extraction_fallback": None,
        "extraction_degraded_pages": 0,
        "extraction_recovered_pages": 0,
        "extraction_vlm_model": "test-model",
        "extraction_vlm_served_pages": 1,
        "extraction_demoted_pages": 1,
    }
    return UniversalDocument(
        doc_id="x",
        source_file="s.pdf",
        file_type="pdf",
        pages=ups,
        metadata=DocumentMetadata(extra=extra),
        total_pages=pages,
    )


def test_process_pdf_header_carries_engine_version_config_hash_source_hash_and_routing(
    tmp_path, monkeypatch
):
    import mmrag_v3.processor as v3p
    from mmrag_v2.batch_processor import BatchProcessor

    pdf = tmp_path / "s.pdf"
    _synthetic_pdf(pdf)
    monkeypatch.setattr(v3p, "extract", lambda path: _uir())
    out = tmp_path / "out"
    bp = BatchProcessor(output_dir=str(out), batch_size=10, vision_provider="none")
    bp.process_pdf(pdf)
    lines = (out / "ingestion.jsonl").read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["pipeline_version"] == __engine_version__ != __schema_version__
    assert header["schema_version"] == __schema_version__
    assert re.fullmatch(r"[0-9a-f]{64}", header["config_hash"])
    assert header["source_file_hash"] == hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert header["extraction_vlm_model"] == "test-model"
    assert header["extraction_vlm_served_pages"] == 1 and header["extraction_demoted_pages"] == 1


def test_config_hash_in_the_header_follows_the_render_cap(tmp_path, monkeypatch):
    import mmrag_v3.processor as v3p
    from mmrag_v2.batch_processor import BatchProcessor

    pdf = tmp_path / "s.pdf"
    _synthetic_pdf(pdf)
    monkeypatch.setattr(v3p, "extract", lambda path: _uir())
    hashes = []
    for cap in ("1600", "2000"):
        monkeypatch.setenv("VLM_RENDER_MAX_PX", cap)
        out = tmp_path / f"out{cap}"
        BatchProcessor(output_dir=str(out), batch_size=10, vision_provider="none").process_pdf(pdf)
        hashes.append(
            json.loads((out / "ingestion.jsonl").read_text(encoding="utf-8").splitlines()[0])[
                "config_hash"
            ]
        )
    assert hashes[0] != hashes[1]
