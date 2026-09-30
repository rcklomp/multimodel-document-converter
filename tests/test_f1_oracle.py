"""PLAN_F1 WP-3: the Section 6 independent fidelity oracle logic.

Pins the oracle's verdict math (ast.parse pass-rate on repair-touched judgeable
Python chunks >= floor AND strictly above the pre-lane rate) on synthetic
pre/post JSONL - no live extraction.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("f1_oracle", _REPO / "scripts" / "f1_oracle.py")
f1_oracle = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(_REPO / "scripts"))
_spec.loader.exec_module(f1_oracle)

_GOOD = "def f(x):\n    if x:\n        return x\n    return 0"
_MANGLED = "def f(x): if x: return x"  # judgeable (collapsed nested), fails ast.parse


def _chunk(cid, content, *, page, touched):
    md = {
        "source_file": "d.pdf",
        "file_type": "pdf",
        "page_number": page,
        "chunk_type": "code",
        "content_classification": "code",
        "created_at": "2026-06-12T00:00:00Z",
    }
    if touched:
        md["code_repair_applied"] = True
    return {"chunk_id": cid, "doc_id": "d", "modality": "code", "content": content, "metadata": md}


def _write(path, chunks):
    path.write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")


def test_oracle_pass_when_repair_lifts_above_floor_and_pre(tmp_path):
    pre = tmp_path / "pre.jsonl"
    post = tmp_path / "post.jsonl"
    # pre: half mangled -> ~0.5 parse rate
    _write(pre, [_chunk("a", _GOOD, page=1, touched=False),
                 _chunk("b", _MANGLED, page=2, touched=False)])
    # post: repair-touched, all good -> 1.0
    _write(post, [_chunk("a2", _GOOD, page=1, touched=True),
                  _chunk("b2", _GOOD, page=2, touched=True)])
    rc = f1_oracle.main(["--book", "T", "--pre", str(pre), "--post", str(post),
                         "--artifacts-dir", str(tmp_path / "art")])
    assert rc == 0


def test_oracle_fails_below_floor(tmp_path):
    pre = tmp_path / "pre.jsonl"
    post = tmp_path / "post.jsonl"
    _write(pre, [_chunk("a", _MANGLED, page=1, touched=False)])
    # post touched but still mostly mangled -> below 0.85
    _write(post, [_chunk("a2", _GOOD, page=1, touched=True),
                  _chunk("b2", _MANGLED, page=2, touched=True),
                  _chunk("c2", _MANGLED, page=3, touched=True)])
    rc = f1_oracle.main(["--book", "T", "--pre", str(pre), "--post", str(post),
                         "--artifacts-dir", str(tmp_path / "art")])
    assert rc == 1


def test_oracle_fails_when_not_improved_over_pre(tmp_path):
    pre = tmp_path / "pre.jsonl"
    post = tmp_path / "post.jsonl"
    # pre already perfect -> post cannot strictly improve
    _write(pre, [_chunk("a", _GOOD, page=1, touched=False)])
    _write(post, [_chunk("a2", _GOOD, page=1, touched=True)])
    rc = f1_oracle.main(["--book", "T", "--pre", str(pre), "--post", str(post),
                         "--artifacts-dir", str(tmp_path / "art")])
    assert rc == 1  # post_rate 1.0 not strictly > pre_rate 1.0


def test_artifacts_written(tmp_path):
    pre = tmp_path / "pre.jsonl"
    post = tmp_path / "post.jsonl"
    _write(pre, [_chunk("a", _MANGLED, page=1, touched=False)])
    _write(post, [_chunk("a2", _GOOD, page=5, touched=True)])
    f1_oracle.main(["--book", "BK", "--pre", str(pre), "--post", str(post),
                    "--artifacts-dir", str(tmp_path / "art")])
    art = tmp_path / "art" / "BK_oracle_artifacts.txt"
    assert art.exists()
    assert "page 5" in art.read_text()
