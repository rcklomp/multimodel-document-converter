"""v2.15 Phase N DoD gate — tests for verify_phase2_teardown.py.

The script asserts the 4 Abort Teardown Mandate cleanup items per
PLAN_V2.15.md §Phase 2. Tests cover:

  - PASS path on the actual production tree (Option F was chosen
    for v2.15, so the vacuous-pass early-exit fires)
  - PASS path under a synthetic "Option A chosen + clean teardown"
    repo layout (sandbox)
  - FAIL paths for each of the 4 missing items independently
  - Exit code semantics (0 / 1)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/verify_phase2_teardown.py"


def _run(repo_root: Path) -> tuple[int, str]:
    """Run the verify script with REPO_ROOT pointed at `repo_root`.

    The script discovers REPO_ROOT via `Path(__file__).resolve().parents[1]`,
    so we copy the script into a temp scripts/ subdir to redirect it.
    Exception: when `repo_root` IS the actual repo, just run the
    script in place (copy-to-self would SameFileError).
    """
    target_scripts = repo_root / "scripts"
    target_script = target_scripts / "verify_phase2_teardown.py"
    if target_script.resolve() != SCRIPT.resolve():
        target_scripts.mkdir(parents=True, exist_ok=True)
        shutil.copy(SCRIPT, target_script)
    result = subprocess.run(
        [sys.executable, str(target_script)],
        capture_output=True, text=True,
    )
    return result.returncode, result.stdout + result.stderr


def _scaffold_option_f(tmp_path: Path) -> Path:
    """Build a synthetic repo where Option F was chosen for v2.15.
    Vacuously passes per the script's early-exit logic."""
    (tmp_path / "src/mmrag_v2/engines").mkdir(parents=True, exist_ok=True)
    decisions = tmp_path / "docs/DECISIONS.md"
    decisions.parent.mkdir(parents=True, exist_ok=True)
    decisions.write_text(
        "# decisions\n"
        "## v2.15 Strategic Path — Option F Selected (2026-05-24)\n"
        "Chose F.\n"
    )
    return tmp_path


def _scaffold_option_a_complete(tmp_path: Path) -> Path:
    """Build a synthetic repo where Option A was chosen AND aborted
    AND teardown is complete. Should PASS."""
    engines = tmp_path / "src/mmrag_v2/engines"
    engines.mkdir(parents=True, exist_ok=True)
    # (a) production engines tree has NO pdfplumber import
    (engines / "docling_adapter.py").write_text("# clean docling adapter\n")
    # (b) experimental/README.md present with Mandate markers
    exp = engines / "experimental"
    exp.mkdir()
    (exp / "README.md").write_text(
        "# v2.15 Phase 2 Abort Teardown\n"
        "pdfplumber adapter parked here per the Mandate.\n"
    )
    # Optional: partial adapter actually parked here (script doesn't
    # require this, but real teardown would do it)
    (exp / "pdfplumber_adapter.py").write_text(
        "# partial pdfplumber work, parked\nimport pdfplumber\n"
    )
    # (c) test file marked skip
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_pdfplumber_adapter.py").write_text(
        "import pytest\n"
        "@pytest.mark.skip(reason='Phase 2 abort; see DECISIONS.md')\n"
        "def test_something():\n"
        "    pass\n"
    )
    # (d) DECISIONS.md has the strategic-path entry AND the teardown entry
    decisions = tmp_path / "docs/DECISIONS.md"
    decisions.parent.mkdir(parents=True, exist_ok=True)
    decisions.write_text(
        "# decisions\n"
        "## v2.15 Strategic Path — Option A Selected (2026-05-24)\n"
        "Chose A.\n"
        "## v2.15 Phase 2 abort + teardown (2026-06-01)\n"
        "Day-8 cap fired; teardown complete.\n"
    )
    return tmp_path


def test_option_f_vacuous_pass(tmp_path):
    """Option F → early exit, EXIT 0."""
    _scaffold_option_f(tmp_path)
    rc, out = _run(tmp_path)
    assert rc == 0
    assert "vacuously satisfied" in out


def test_option_a_complete_teardown_passes(tmp_path):
    """Full Option A teardown → all 4 checks pass, EXIT 0."""
    _scaffold_option_a_complete(tmp_path)
    rc, out = _run(tmp_path)
    assert rc == 0, f"expected PASS but got rc={rc}: {out}"
    assert "RESULT: PASS" in out


def test_fail_pdfplumber_import_in_production(tmp_path):
    """Production engines tree contains pdfplumber import → FAIL."""
    _scaffold_option_a_complete(tmp_path)
    # Inject a pdfplumber import in the production tree
    (tmp_path / "src/mmrag_v2/engines/leak.py").write_text(
        "import pdfplumber  # forbidden leak\n"
    )
    rc, out = _run(tmp_path)
    assert rc == 1, f"expected FAIL but got rc={rc}: {out}"
    assert "RESULT: FAIL" in out
    assert "pdfplumber" in out


def test_fail_missing_experimental_readme(tmp_path):
    """`engines/experimental/README.md` missing → FAIL."""
    _scaffold_option_a_complete(tmp_path)
    (tmp_path / "src/mmrag_v2/engines/experimental/README.md").unlink()
    rc, out = _run(tmp_path)
    assert rc == 1
    assert "RESULT: FAIL" in out


def test_fail_test_file_not_skipped(tmp_path):
    """test_pdfplumber_adapter.py exists but isn't @pytest.mark.skip → FAIL."""
    _scaffold_option_a_complete(tmp_path)
    (tmp_path / "tests/test_pdfplumber_adapter.py").write_text(
        "def test_something():\n    pass\n"  # NOT skipped
    )
    rc, out = _run(tmp_path)
    assert rc == 1
    assert "RESULT: FAIL" in out


def test_fail_missing_decisions_entry(tmp_path):
    """DECISIONS.md missing teardown entry → FAIL."""
    _scaffold_option_a_complete(tmp_path)
    # Rewrite DECISIONS.md without the teardown entry
    (tmp_path / "docs/DECISIONS.md").write_text(
        "## v2.15 Strategic Path — Option A Selected (2026-05-24)\n"
    )
    rc, out = _run(tmp_path)
    assert rc == 1
    assert "RESULT: FAIL" in out
    assert "v2.15 Phase 2 abort + teardown" in out


def test_missing_test_file_is_acceptable(tmp_path):
    """If test_pdfplumber_adapter.py was never created, nothing to skip
    — trivially passes that check."""
    _scaffold_option_a_complete(tmp_path)
    (tmp_path / "tests/test_pdfplumber_adapter.py").unlink()
    rc, out = _run(tmp_path)
    # Other checks still need to pass (they do per scaffold), and the
    # missing-test-file check trivially passes per script docstring
    assert rc == 0, f"expected PASS but got rc={rc}: {out}"
    assert "RESULT: PASS" in out


def test_production_repo_passes_with_option_f():
    """Real production repo (current state: Option F just selected) →
    PASS via the vacuous-exit. Ensures the script works in the live
    tree without needing a sandbox."""
    rc, out = _run(REPO_ROOT)
    assert rc == 0, f"expected PASS but got rc={rc}: {out}"
    # Either vacuous (current state) or full PASS would be acceptable
    assert "EXIT 0" in out or "RESULT: PASS" in out
