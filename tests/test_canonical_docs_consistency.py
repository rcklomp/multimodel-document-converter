"""v2.16 Phase 0 step 6.5 — anti-drift bridge for CANONICAL_DOCS.

The canonical-docs list is duplicated across 5 consumer sites:

  scripts/rebuild_mmrag_v2_8_for_rc1.py (source of truth)
  scripts/synthetic_soak.py              (independent copy — soak harness)
  scripts/build_bm25_index.py            (imports from rebuild_mod)
  scripts/ingest_bm25_sparse.py          (imports from rebuild_mod)
  tests/test_rebuild_resume.py           (asserts length + sample)

If the source and the independent copy drift, BM25 parallel-mapping with
the dense collection breaks, and the soak harness measures against a
different doc set than the production rebuilds. This test detects drift
the moment it lands.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def test_synthetic_soak_canonical_docs_matches_rebuild_source() -> None:
    """synthetic_soak.CANONICAL_DOCS is a manually-maintained copy of
    rebuild_mmrag_v2_8_for_rc1.CANONICAL_DOCS. Both must hold the same
    set of basenames (order may differ — the rebuild walks in order,
    the soak samples stratified, so set-equality is the right invariant).
    """
    rebuild = _load("_rebuild_mod", SCRIPTS / "rebuild_mmrag_v2_8_for_rc1.py")
    soak = _load("_soak_mod", SCRIPTS / "synthetic_soak.py")
    rebuild_set = set(rebuild.CANONICAL_DOCS)
    soak_set = set(soak.CANONICAL_DOCS)
    only_in_rebuild = rebuild_set - soak_set
    only_in_soak = soak_set - rebuild_set
    assert not only_in_rebuild, (
        f"Drift: docs in rebuild_mmrag_v2_8_for_rc1.CANONICAL_DOCS but "
        f"missing from synthetic_soak.CANONICAL_DOCS: {sorted(only_in_rebuild)}"
    )
    assert not only_in_soak, (
        f"Drift: docs in synthetic_soak.CANONICAL_DOCS but missing "
        f"from rebuild_mmrag_v2_8_for_rc1.CANONICAL_DOCS: {sorted(only_in_soak)}"
    )


def test_canonical_docs_length_is_41_after_v2_16_phase_0() -> None:
    """Pin the v2.16 Phase 0 corpus expansion: 34 original + 7 new =
    41 canonical docs. Future cycle Phase 0 expansions should update
    this assertion + the test_rebuild_resume length pin together."""
    rebuild = _load("_rebuild_mod", SCRIPTS / "rebuild_mmrag_v2_8_for_rc1.py")
    assert len(rebuild.CANONICAL_DOCS) == 41


def test_no_duplicate_entries_in_canonical_docs() -> None:
    """Duplicates would silently double-weight the doc during BM25 build
    and skew dense-sparse parallel mapping."""
    rebuild = _load("_rebuild_mod", SCRIPTS / "rebuild_mmrag_v2_8_for_rc1.py")
    docs = rebuild.CANONICAL_DOCS
    assert len(docs) == len(set(docs)), (
        f"Duplicate entries in CANONICAL_DOCS: "
        f"{[d for d in docs if docs.count(d) > 1]}"
    )
