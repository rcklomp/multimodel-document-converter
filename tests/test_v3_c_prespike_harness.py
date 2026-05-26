"""Smoke tests for the V3 Phase C pre-spike harness.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §4.2 step 1.

Tests cover the pure-Python pieces (config validation, MaxSim numerics,
result formatting). The PDF rendering + ColPali dispatch land in
operator-execution; not unit-tested here because they require either
a real PDF render path (tested implicitly via the dry-run script
invocation) or a real ColPali model (deferred per the harness fence).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


# Load the script module directly — it lives under scripts/, not under
# the installed package, so we import it via its file path.
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "v3_c_prespike.py"

_spec = importlib.util.spec_from_file_location("v3_c_prespike", _SCRIPT_PATH)
v3_c_prespike = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["v3_c_prespike"] = v3_c_prespike
_spec.loader.exec_module(v3_c_prespike)  # type: ignore[union-attr]


class TestMaxSim:
    def test_dry_run_returns_nan(self):
        result = v3_c_prespike.maxsim_score(None, None)
        assert np.isnan(result)

    def test_identical_query_and_page_maximal_score(self):
        # Single token query, single patch page, identical vectors → 1.0
        query = np.array([[1.0, 0.0, 0.0]])
        page = np.array([[1.0, 0.0, 0.0]])
        assert v3_c_prespike.maxsim_score(query, page) == pytest.approx(1.0)

    def test_orthogonal_vectors_zero_score(self):
        query = np.array([[1.0, 0.0]])
        page = np.array([[0.0, 1.0]])
        assert v3_c_prespike.maxsim_score(query, page) == pytest.approx(0.0)

    def test_sums_per_query_token_max(self):
        # 3 query tokens × 2 patches; per-token max sums to 3.0 when
        # each query token has at least one patch it perfectly matches.
        query = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        page = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                # Note: third query token (0,0,1) does NOT have a
                # perfect-match patch here, so the per-token max is the
                # highest available similarity = 0.0 (orthogonal to both).
            ]
        )
        # Per-token max: q0→p0=1.0, q1→p1=1.0, q2→max=0.0
        # Sum = 2.0
        assert v3_c_prespike.maxsim_score(query, page) == pytest.approx(2.0)


class TestConfig:
    def test_pre_spike_config_construction(self):
        cfg = v3_c_prespike.PreSpikeConfig(
            pdf_path=Path("/x.pdf"),
            gold_page=4,
            distractor_pages=[1, 2, 5],
            query="test",
            render_dpi=200,
            colpali_mode="dry-run",
        )
        assert cfg.render_dpi == 200
        assert cfg.gold_page == 4
        assert cfg.distractor_pages == [1, 2, 5]


class TestResultFormatting:
    def test_pass_verdict_string(self):
        result = v3_c_prespike.PreSpikeResult(
            pages_ranked=[(4, 0.95), (1, 0.5), (2, 0.4), (5, 0.3)],
            gold_page=4,
            gold_rank=1,
            passed=True,
        )
        assert result.verdict_str == "PASS"
        text = v3_c_prespike._format_result(result)
        assert "PASS" in text
        assert "← gold" in text

    def test_fail_verdict_string(self):
        result = v3_c_prespike.PreSpikeResult(
            pages_ranked=[(1, 0.9), (4, 0.6), (2, 0.4), (5, 0.3)],
            gold_page=4,
            gold_rank=2,
            passed=False,
        )
        assert result.verdict_str == "FAIL"


class TestDryRunHarness:
    """The dry-run mode exercises the rendering + ranking pipeline
    without requiring ColPali. Validates that the harness wiring is
    correct independent of the model."""

    def test_dry_run_full_path(self, tmp_path):
        # Use the actual ATZ PDF the Charter pre-spike targets.
        pdf = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "technical_report"
            / "ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters.pdf"
        )
        if not pdf.exists():
            pytest.skip(f"Pre-spike target PDF not present at {pdf}")
        cfg = v3_c_prespike.PreSpikeConfig(
            pdf_path=pdf,
            gold_page=4,
            distractor_pages=[1, 2, 5],
            query="Schaltbild",
            render_dpi=200,
            colpali_mode="dry-run",
            output_dir=tmp_path,
        )
        result = v3_c_prespike.run_prespike(cfg)
        # Dry-run is NOT a PASS by design — operator must run live for
        # a real verdict. The result.passed should be False.
        assert result.passed is False
        # But the harness should still place gold first in the dry-run
        # synthetic ranking.
        assert result.gold_rank == 1
        # And the renders should be on disk.
        png_files = list(tmp_path.glob("page_*_dpi200.png"))
        assert len(png_files) == 4
