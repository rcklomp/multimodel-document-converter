"""Section 5.4 action-on-flag consumers: extraction-provenance observability.

PLAN_EXTRACTION_FIDELITY_V1 Section 5.4 (advisory phase, AGENT-GATE-PROGRESSION):
the ``extraction_*`` stamps that ``mmrag_v3.extract`` records on each batch's
UniversalDocument are aggregated to doc level on the IngestionMetadata header and
SURFACED by two live consumers - the ``qa_full_conversion`` advisory block and the
``smoke_production`` NOTES line. These tests pin:

  * the batch aggregation (first engine, MOST-severe ladder tier, summed counts);
  * the IngestionMetadata schema carries the four fields and defaults them to None
    on legacy outputs;
  * the qa advisory printer surfaces engine + ladder-served fraction and is pure
    observability (it raises nothing, returns nothing, changes no verdict).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from mmrag_v2.batch_processor import BatchProcessor
from mmrag_v2.schema.ingestion_schema import IngestionMetadata

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from qa_full_conversion import _print_extraction_provenance  # noqa: E402


# --------------------------------------------------------------------------- #
# batch aggregation
# --------------------------------------------------------------------------- #
class _Acc:
    """Borrows the real aggregation method + severity map without constructing
    the heavy BatchProcessor (the method only touches these two attributes)."""

    _FALLBACK_SEVERITY = BatchProcessor._FALLBACK_SEVERITY
    _accumulate_extraction_provenance = BatchProcessor._accumulate_extraction_provenance

    def __init__(self):
        self._extraction_provenance = {
            "engine": None,
            "fallback": None,
            "degraded": 0,
            "recovered": 0,
            "quality_risk": 0,
            "code_repaired": 0,
        }


def _doc(extra):
    return SimpleNamespace(metadata=SimpleNamespace(extra=extra))


def test_single_healthy_batch_records_engine_no_fallback():
    acc = _Acc()
    acc._accumulate_extraction_provenance(
        _doc(
            {
                "extraction_engine": "mineru",
                "extraction_fallback": None,
                "extraction_degraded_pages": 0,
                "extraction_recovered_pages": 0,
            }
        )
    )
    assert acc._extraction_provenance == {
        "engine": "mineru",
        "fallback": None,
        "degraded": 0,
        "recovered": 0,
        "quality_risk": 0,
        "code_repaired": 0,
    }


def test_counts_sum_and_first_engine_wins_across_batches():
    acc = _Acc()
    acc._accumulate_extraction_provenance(
        _doc(
            {
                "extraction_engine": "mineru",
                "extraction_fallback": None,
                "extraction_degraded_pages": 0,
                "extraction_recovered_pages": 0,
            }
        )
    )
    acc._accumulate_extraction_provenance(
        _doc(
            {
                "extraction_engine": "mineru",
                "extraction_fallback": "docling_fast",
                "extraction_degraded_pages": 2,
                "extraction_recovered_pages": 1,
                "extraction_quality_risk_pages": 5,
                "extraction_code_repaired_pages": 4,
            }
        )
    )
    prov = acc._extraction_provenance
    assert prov["engine"] == "mineru"
    assert prov["fallback"] == "docling_fast"
    assert prov["degraded"] == 2
    assert prov["recovered"] == 1
    # §5.4 consumer 1 counts also sum across batches.
    assert prov["quality_risk"] == 5
    assert prov["code_repaired"] == 4


def test_most_severe_ladder_tier_is_kept():
    acc = _Acc()
    acc._accumulate_extraction_provenance(_doc({"extraction_fallback": "docling_fast"}))
    acc._accumulate_extraction_provenance(_doc({"extraction_fallback": "pymupdf_terminal"}))
    # A later, LESS-severe tier must not overwrite the worse one.
    acc._accumulate_extraction_provenance(_doc({"extraction_fallback": "docling_fast"}))
    assert acc._extraction_provenance["fallback"] == "pymupdf_terminal"


def test_missing_extra_keys_preserve_defaults():
    acc = _Acc()
    acc._accumulate_extraction_provenance(_doc({}))
    acc._accumulate_extraction_provenance(_doc(None))  # metadata.extra can be falsy
    assert acc._extraction_provenance == {
        "engine": None,
        "fallback": None,
        "degraded": 0,
        "recovered": 0,
        "quality_risk": 0,
        "code_repaired": 0,
    }


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
def test_schema_carries_extraction_provenance_fields():
    m = IngestionMetadata(
        schema_version="2.7.0",
        doc_id="d",
        source_file="x.pdf",
        extraction_engine="mineru",
        extraction_fallback="docling_fast",
        extraction_degraded_pages=3,
        extraction_recovered_pages=2,
        extraction_quality_risk_pages=5,
        extraction_code_repaired_pages=4,
    )
    dumped = m.model_dump(mode="json")
    assert dumped["extraction_engine"] == "mineru"
    assert dumped["extraction_fallback"] == "docling_fast"
    assert dumped["extraction_degraded_pages"] == 3
    assert dumped["extraction_recovered_pages"] == 2
    assert dumped["extraction_quality_risk_pages"] == 5
    assert dumped["extraction_code_repaired_pages"] == 4


def test_schema_defaults_to_none_on_legacy_output():
    m = IngestionMetadata(schema_version="2.7.0", doc_id="d", source_file="x.pdf")
    dumped = m.model_dump(mode="json")
    assert dumped["extraction_engine"] is None
    assert dumped["extraction_fallback"] is None
    assert dumped["extraction_degraded_pages"] is None
    assert dumped["extraction_recovered_pages"] is None
    assert dumped["extraction_quality_risk_pages"] is None
    assert dumped["extraction_code_repaired_pages"] is None


# --------------------------------------------------------------------------- #
# qa_full_conversion advisory printer (pure observability)
# --------------------------------------------------------------------------- #
def test_qa_advisory_surfaces_engine_and_ladder_fraction(capsys):
    out = _print_extraction_provenance(
        {
            "extraction_engine": "mineru",
            "extraction_fallback": "docling_fast",
            "extraction_degraded_pages": 2,
            "extraction_recovered_pages": 1,
            "extraction_quality_risk_pages": 5,
            "extraction_code_repaired_pages": 4,
            "total_pages": 10,
        }
    )
    captured = capsys.readouterr().out
    assert out is None  # pure side-effecting observability, no verdict value
    assert "Extraction Provenance (advisory)" in captured
    assert "engine: mineru" in captured
    assert "docling_fast" in captured
    assert "2/10 (20.0%)" in captured  # ladder-served fraction
    assert "ladder-recovered pages: 1/10" in captured
    assert "5/10 (50.0%)" in captured  # §5.4 code-degraded flagged fraction
    assert "VLM-repaired: 4/5" in captured


def test_qa_advisory_handles_legacy_output_without_stamps(capsys):
    _print_extraction_provenance({"total_pages": 5})  # no extraction_* keys
    captured = capsys.readouterr().out
    assert "No extraction_* provenance stamps" in captured


def test_qa_advisory_tolerates_missing_total_pages(capsys):
    # No total_pages -> report bare counts, never divide by zero.
    _print_extraction_provenance(
        {
            "extraction_engine": "docling_fast",
            "extraction_degraded_pages": 4,
        }
    )
    captured = capsys.readouterr().out
    assert "engine: docling_fast" in captured
    assert "ladder-served pages (primary could not serve): 4" in captured
