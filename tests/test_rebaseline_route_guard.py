"""WP-4 [E3]: rebaseline_v3 route guard + target resolution.

The rebaseline script must REFUSE to promote docling/offline extraction to an
authoritative baseline. These tests pin the env-var precedence that decides the
effective route and the allow/refuse decision, plus the --pdf/--baseline target
resolution. No live inference: only resolve_route() (pure env reads) and the
guard's early-return are exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import rebaseline_v3 as rb  # noqa: E402

# Every env var that participates in route resolution.
_ROUTE_ENV = (
    "USE_MINERU_ENGINE",
    "USE_VLM_ENGINE",
    "USE_DOCLING_FAST",
    "USE_HYBRID_ENGINE",
    "USE_MINERU_QWEN_HYBRID",
    "MINERU_ENDPOINT",
)


@pytest.fixture
def clean_route_env(monkeypatch):
    for var in _ROUTE_ENV:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.mark.parametrize(
    "env,expected_route",
    [
        ({"USE_MINERU_ENGINE": "1"}, "mineru"),
        ({"USE_VLM_ENGINE": "1"}, "vlm_native"),
        ({"USE_DOCLING_FAST": "1"}, "docling_fast"),
        ({"USE_HYBRID_ENGINE": "1"}, "hybrid"),
        ({"USE_MINERU_QWEN_HYBRID": "1"}, "mineru_qwen_hybrid"),
        ({"MINERU_ENDPOINT": "http://127.0.0.1:18001"}, "mineru_qwen_hybrid"),
        ({}, "hybrid"),
    ],
)
def test_resolve_route_matrix(clean_route_env, env, expected_route):
    for k, v in env.items():
        clean_route_env.setenv(k, v)
    assert rb.resolve_route() == expected_route


def test_resolve_route_precedence_mineru_over_vlm(clean_route_env):
    # USE_MINERU_ENGINE precedes USE_VLM_ENGINE in _select_engine order.
    clean_route_env.setenv("USE_MINERU_ENGINE", "1")
    clean_route_env.setenv("USE_VLM_ENGINE", "1")
    assert rb.resolve_route() == "mineru"


def test_explicit_docling_overrides_endpoint_default(clean_route_env):
    # An explicit USE_DOCLING_FAST beats the MINERU_ENDPOINT default -> refused.
    clean_route_env.setenv("USE_DOCLING_FAST", "1")
    clean_route_env.setenv("MINERU_ENDPOINT", "http://127.0.0.1:18001")
    assert rb.resolve_route() == "docling_fast"
    assert rb.resolve_route() not in rb.ALLOWED_ROUTES


@pytest.mark.parametrize(
    "route,allowed",
    [
        ("mineru_qwen_hybrid", True),
        ("vlm_native", True),
        ("mineru", False),
        ("hybrid", False),
        ("docling_fast", False),
    ],
)
def test_allowed_routes(route, allowed):
    assert (route in rb.ALLOWED_ROUTES) is allowed


@pytest.mark.parametrize(
    "env",
    [
        {"USE_DOCLING_FAST": "1"},
        {"USE_HYBRID_ENGINE": "1"},
        {"USE_MINERU_ENGINE": "1"},
        {},  # default -> hybrid
    ],
)
def test_rebaseline_refuses_non_vlm_route_before_inference(clean_route_env, env, tmp_path):
    for k, v in env.items():
        clean_route_env.setenv(k, v)
    # A nonexistent PDF proves the guard returns BEFORE the pdf check / extraction.
    missing_pdf = tmp_path / "does_not_exist.pdf"
    baseline = tmp_path / "ingestion.jsonl"
    assert rb.rebaseline(missing_pdf, baseline) == 2
    assert not baseline.exists()


def test_resolve_targets_pdf_default_baseline():
    args = type("A", (), {"pdf": "/data/My Doc.pdf", "baseline": None, "doc": None})()
    pdf, baseline = rb.resolve_targets(args)
    assert pdf == Path("/data/My Doc.pdf")
    assert baseline == _REPO_ROOT / "output" / "My Doc" / "ingestion.jsonl"


def test_resolve_targets_pdf_explicit_baseline():
    args = type(
        "A", (), {"pdf": "/data/doc.pdf", "baseline": "/out/custom.jsonl", "doc": None}
    )()
    pdf, baseline = rb.resolve_targets(args)
    assert pdf == Path("/data/doc.pdf")
    assert baseline == Path("/out/custom.jsonl")


def test_resolve_targets_named_shortcut():
    args = type("A", (), {"pdf": None, "baseline": None, "doc": "CarOK_voorraadtelling"})()
    pdf, baseline = rb.resolve_targets(args)
    assert pdf == rb.REF_DOCS["CarOK_voorraadtelling"]["pdf"]
    assert baseline == rb.REF_DOCS["CarOK_voorraadtelling"]["baseline"]


def test_resolve_targets_unknown_shortcut_returns_none():
    args = type("A", (), {"pdf": None, "baseline": None, "doc": "no_such_doc"})()
    assert rb.resolve_targets(args) is None
