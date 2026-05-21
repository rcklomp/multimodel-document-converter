"""v2.12 retrieval-regression pytest harness (production hybrid lane).

Wraps `scripts/retrieval_regression_v2_12.py` so the v2.12.0 production
retrieval shape (hybrid: dense + BM25 sparse + RRF + ModernBERT
rerank) is a pinned regression contract. The script writes/verifies
the fingerprint at `tests/fixtures/retrieval_regression_v2_12_hybrid.json`.

Three services are required for execution:
  - Qdrant on http://localhost:6333 with both
      `mmrag_v2_8__qwen3_dashscope` (dense) and
      `mmrag_v2_8__bm25_sparse`     (sparse-only side collection)
  - Dashscope reachable (DASHSCOPE_API_KEY env var set)
  - omlx-server reachable on 10.0.10.246:8000 with the
    `gte-reranker-modernbert-base-mlx` model loaded
    (MLX_API_KEY env var set)

All three are skipped (not failed) when unreachable so a clean
checkout without the full runtime still completes pytest. The v2.12
release contract requires this test to PASS on a live stack before
any v2.12.x tag push.

Drift semantics:
  - STRICT pass: top-3 chunk_ids match the baseline exactly.
  - PASS_WITH_DRIFT: top-1 doc_id matches but top-3 reshuffled.
  - FAIL: top-1 doc_id changed for any query.

To refresh after an intentional retrieval change (rebuild, reranker
swap, etc.):

  python scripts/retrieval_regression_v2_12.py --capture

Then commit the updated `retrieval_regression_v2_12_hybrid.json`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "retrieval_regression_v2_12.py"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "retrieval_regression_v2_12_hybrid.json"
QDRANT_URL = "http://localhost:6333"
DENSE_COLLECTION = "mmrag_v2_8__qwen3_dashscope"
SPARSE_COLLECTION = "mmrag_v2_8__bm25_sparse"
OMLX_URL = "http://10.0.10.246:8000"


def _qdrant_collection_reachable(name: str) -> bool:
    try:
        urllib.request.urlopen(f"{QDRANT_URL}/collections/{name}", timeout=3)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError):
        return False


def _dashscope_key_present() -> bool:
    return bool(os.environ.get("DASHSCOPE_API_KEY", "").strip())


def _mlx_key_present_and_omlx_reachable() -> bool:
    if not os.environ.get("MLX_API_KEY", "").strip():
        return False
    try:
        urllib.request.urlopen(f"{OMLX_URL}/v1/models", timeout=3)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError):
        return False


def test_fingerprint_fixture_exists() -> None:
    """The tracked v2.12 production baseline must live in the repo so
    the regression evidence is reproducible from tracked files
    (AGENT-EVIDENCE-01)."""
    assert FIXTURE.exists(), (
        f"v2.12 retrieval fingerprint missing at {FIXTURE.relative_to(REPO_ROOT)}; "
        "run `python scripts/retrieval_regression_v2_12.py --capture` to seed it."
    )
    payload = json.loads(FIXTURE.read_text())
    assert payload.get("engine_version") == "2.12.0"
    assert payload.get("pipeline") == "hybrid+rerank"
    assert payload.get("dense_collection") == DENSE_COLLECTION
    assert payload.get("sparse_collection") == SPARSE_COLLECTION
    assert payload.get("reranker_model") == "gte-reranker-modernbert-base-mlx"
    queries = payload.get("queries") or []
    assert len(queries) == 20, f"expected 20 queries in baseline, got {len(queries)}"
    for q in queries:
        top_k = q.get("top_k") or []
        assert top_k, f"query {q.get('id')} has empty top_k in baseline"
        assert all(r.get("chunk_id") for r in top_k), (
            f"query {q.get('id')} has missing chunk_id in baseline top-K"
        )
        # v2.12 fingerprint must include rerank_score on every row.
        assert all("rerank_score" in r for r in top_k), (
            f"query {q.get('id')} missing rerank_score on a top-K row"
        )


@pytest.mark.skipif(
    not _qdrant_collection_reachable(DENSE_COLLECTION),
    reason=f"Qdrant dense collection {DENSE_COLLECTION} unreachable",
)
@pytest.mark.skipif(
    not _qdrant_collection_reachable(SPARSE_COLLECTION),
    reason=f"Qdrant sparse collection {SPARSE_COLLECTION} unreachable",
)
@pytest.mark.skipif(
    not _dashscope_key_present(),
    reason="DASHSCOPE_API_KEY env var not set",
)
@pytest.mark.skipif(
    not _mlx_key_present_and_omlx_reachable(),
    reason="omlx-server unreachable or MLX_API_KEY not set",
)
def test_retrieval_regression_v2_12_against_fingerprint() -> None:
    """Run the v2.12 hybrid+rerank pipeline against the live stack and
    compare to the captured fingerprint.

    A non-zero exit means the top-1 doc_id changed for at least one
    query — that's a v2.12 retrieval-shape regression and must block
    promotion to a v2.12.x tag.
    """
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
        timeout=600,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    assert proc.returncode == 0, (
        f"retrieval_regression_v2_12.py exited {proc.returncode}; see captured "
        "output. If the drift is intentional, re-capture with "
        "`python scripts/retrieval_regression_v2_12.py --capture` and commit "
        "the updated fingerprint."
    )
