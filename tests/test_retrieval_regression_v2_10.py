"""v2.10 retrieval-regression pytest harness.

Wraps `scripts/retrieval_regression.py` so the v2.10.0-rc1 retrieval
shape (top-K results per query against `mmrag_v2_8`) is a pinned
regression contract. The script writes/verifies the tracked fingerprint
at `tests/fixtures/retrieval_regression_v2_10.json`.

Two services are required for execution:
  - Qdrant on http://localhost:6333 with the `mmrag_v2_8` collection
  - Ollama on http://localhost:11434 with the `llava` model present

Both are skipped (not failed) when unreachable so a clean checkout
without the runtime still completes pytest. The Phase 8 release
contract requires this test to PASS on a live local stack before any
v2.10.x tag push.

Drift semantics (per `scripts/retrieval_regression.py`):
  - STRICT pass: top-3 chunk_ids match the baseline exactly.
  - PASS_WITH_DRIFT: top-1 doc_id matches but top-3 reshuffled (this
    test treats it as a soft pass — surfaces a stderr note rather than
    failing).
  - FAIL: top-1 doc_id changed for any query.

To refresh the fingerprint after an intentional retrieval change
(rebuild, embedder swap, etc.):

  conda run -n mmrag-v2 python scripts/retrieval_regression.py --capture

Then commit the updated `retrieval_regression_v2_10.json`.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "retrieval_regression.py"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "retrieval_regression_v2_10.json"
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
COLLECTION = "mmrag_v2_8"


def _qdrant_collection_reachable() -> bool:
    try:
        urllib.request.urlopen(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=3)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError):
        return False


def _ollama_has_llava() -> bool:
    try:
        resp = urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3)
        data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError):
        return False
    return any(m.get("name", "").startswith("llava") for m in data.get("models", []))


def test_fingerprint_fixture_exists() -> None:
    """The tracked v2.10 baseline must live in the repo so the
    regression evidence is reproducible from tracked files
    (`AGENT-EVIDENCE-01`)."""
    assert FIXTURE.exists(), (
        f"v2.10 retrieval fingerprint missing at {FIXTURE.relative_to(REPO_ROOT)}; "
        "run `python scripts/retrieval_regression.py --capture` to seed it."
    )
    payload = json.loads(FIXTURE.read_text())
    assert payload.get("engine_version") == "2.10.0-rc1"
    assert payload.get("collection") == COLLECTION
    assert payload.get("embed_model") == "llava"
    queries = payload.get("queries") or []
    assert len(queries) == 20, f"expected 20 queries in baseline, got {len(queries)}"
    # Each query must have a non-empty top-K with chunk_ids.
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
    not _ollama_has_llava(),
    reason=f"Ollama at {OLLAMA_URL} does not have the llava model",
)
def test_retrieval_regression_against_fingerprint(capsys) -> None:
    """Run the regression script's verify mode against the live stack.

    A non-zero exit means the top-1 doc_id changed for at least one
    query — that's a v2.10 retrieval-shape regression and must block
    promotion to a final v2.10.0 tag.
    """
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
        timeout=600,
    )
    # Surface the script output in the pytest report so drift is visible.
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    assert proc.returncode == 0, (
        f"retrieval_regression.py exited {proc.returncode}; see captured output. "
        "If the drift is intentional, re-capture with "
        "`python scripts/retrieval_regression.py --capture` and commit "
        "the updated fingerprint."
    )
