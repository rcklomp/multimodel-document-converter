"""v2.11 retrieval-regression pytest harness (production lane).

Wraps `scripts/retrieval_regression.py` so the v2.11.0 retrieval shape
(top-K results per query against `mmrag_v2_8__qwen3_dashscope`, built
with Dashscope `text-embedding-v4` 1024-dim) is a pinned regression
contract. The script writes/verifies the tracked fingerprint at
`tests/fixtures/retrieval_regression_v2_11_qwen3.json`.

**Role:** production retrieval-shape pin since the v2.11.0 embedder
swap (see `docs/DECISIONS.md` §"v2.11 Phase 1 Embedder Shootout
Outcome"). The legacy llava lane is covered by
`test_retrieval_regression_v2_10.py` during the 30-day rollback window.

Three services are required for execution:
  - Qdrant on http://localhost:6333 with the
    `mmrag_v2_8__qwen3_dashscope` collection
  - Dashscope reachable (DASHSCOPE_API_KEY env var set)
  - (no Ollama dependency in this lane)

All three are skipped (not failed) when unreachable so a clean checkout
without the runtime still completes pytest. The release contract
requires this test to PASS on a live stack before any v2.11.x tag push.

Drift semantics: same STRICT / PASS_WITH_DRIFT / FAIL as the legacy lane.

To refresh the production fingerprint after an intentional retrieval
change (corpus rebuild, embedder version bump, etc.):

  conda run -n mmrag-v2 python scripts/retrieval_regression.py --capture

(no flags needed — dashscope is the default provider since v2.11.0).
Then commit the updated `retrieval_regression_v2_11_qwen3.json`.
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
SCRIPT = REPO_ROOT / "scripts" / "retrieval_regression.py"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "retrieval_regression_v2_11_qwen3.json"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "mmrag_v2_8__qwen3_dashscope"


def _qdrant_collection_reachable() -> bool:
    try:
        urllib.request.urlopen(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=3)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError):
        return False


def _dashscope_key_present() -> bool:
    return bool(os.environ.get("DASHSCOPE_API_KEY", "").strip())


def test_fingerprint_fixture_exists() -> None:
    """The tracked v2.11 production baseline must live in the repo so the
    regression evidence is reproducible from tracked files
    (`AGENT-EVIDENCE-01`)."""
    assert FIXTURE.exists(), (
        f"v2.11 retrieval fingerprint missing at {FIXTURE.relative_to(REPO_ROOT)}; "
        "run `python scripts/retrieval_regression.py --capture` to seed it."
    )
    payload = json.loads(FIXTURE.read_text())
    assert payload.get("engine_version") == "2.11.0"
    assert payload.get("collection") == COLLECTION
    assert payload.get("provider") == "dashscope"
    assert payload.get("embed_model") == "text-embedding-v4"
    queries = payload.get("queries") or []
    assert len(queries) == 20, f"expected 20 queries in production baseline, got {len(queries)}"
    for q in queries:
        top_k = q.get("top_k") or []
        assert top_k, f"query {q.get('id')} has empty top_k in baseline"
        assert all(r.get("chunk_id") for r in top_k), (
            f"query {q.get('id')} has missing chunk_id in baseline top-K"
        )


@pytest.mark.skipif(
    not _qdrant_collection_reachable(),
    reason=f"Qdrant collection {COLLECTION} unreachable at {QDRANT_URL}",
)
@pytest.mark.skipif(
    not _dashscope_key_present(),
    reason="DASHSCOPE_API_KEY env var not set; cannot embed queries via dashscope",
)
def test_retrieval_regression_against_fingerprint(capsys) -> None:
    """Run the regression script's verify mode against the live stack.

    A non-zero exit means the top-1 doc_id changed for at least one
    query — that's a v2.11 retrieval-shape regression and must block
    promotion to a v2.11.x tag.
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
        f"retrieval_regression.py exited {proc.returncode}; see captured output. "
        "If the drift is intentional, re-capture with "
        "`python scripts/retrieval_regression.py --capture` and commit "
        "the updated fingerprint."
    )
