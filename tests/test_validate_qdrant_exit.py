"""PLAN_QUALITY_REMEDIATION_V1 WP-F2: a crashed validation must not exit 0.

`validate_qdrant.main` printed the per-collection exception and still returned 0, so "validated" could
be read from an exit code when nothing had been validated (found on the laptop-travel lineage as
MMC/D3 and confirmed present here).
"""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "validate_qdrant_wp_f2", REPO / "scripts" / "validate_qdrant.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(mod, monkeypatch, collections, validate):
    monkeypatch.setattr(mod, "get_collections", lambda: collections)
    monkeypatch.setattr(mod, "validate_collection", validate)
    monkeypatch.setattr(mod, "print_report", lambda name, v: None)
    monkeypatch.setattr(sys, "argv", ["validate_qdrant.py"])
    return mod.main()


def test_exit_is_zero_when_every_collection_validates(monkeypatch):
    mod = _load()
    assert _run(mod, monkeypatch, [{"name": "a"}, {"name": "b"}], lambda n: {}) == 0


def test_exit_is_nonzero_when_any_collection_crashes_but_the_rest_still_run(monkeypatch):
    mod = _load()
    seen = []

    def validate(name):
        seen.append(name)
        if name == "bad":
            raise RuntimeError("connection refused")
        return {}

    assert _run(mod, monkeypatch, [{"name": "bad"}, {"name": "good"}], validate) == 1
    assert seen == ["bad", "good"]


def test_unknown_collection_still_exits_nonzero(monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "get_collections", lambda: [{"name": "a"}])
    monkeypatch.setattr(sys, "argv", ["validate_qdrant.py", "-c", "nope"])
    assert mod.main() == 1
