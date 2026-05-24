#!/usr/bin/env python3
"""v2.15 Phase N DoD gate — verify Phase 2 [A] abort teardown.

Per Round-6 Finding 5 + Round-8 Finding 6, the Phase 2 Abort
Teardown Mandate has 4 cleanup items the v2.15.0 tag DoD requires
to have all landed. This script is the programmatic gate; same
enforcement model as the calibration-freshness boolean.

Asserts (all must PASS):
  (a) no `pdfplumber` import in `src/mmrag_v2/engines/` non-
      experimental tree
  (b) `src/mmrag_v2/engines/experimental/README.md` exists AND
      contains the Mandate text
  (c) `tests/test_pdfplumber_adapter.py` marked `@pytest.mark.skip`
      (if it exists at all)
  (d) `docs/DECISIONS.md` contains a "v2.15 Phase 2 abort + teardown"
      entry

The 4 assertions correspond 1:1 to the 4 cleanup items in the
Mandate (PLAN_V2.15.md §Phase 2 [A] Abort Teardown Mandate).

EXIT CODES:
  0 — all assertions PASS (or Phase 2 wasn't entered, see below)
  1 — at least one assertion FAILED

EARLY-EXIT (returns 0 with "N/A" note): if Option F or Option E was
chosen for v2.15, Phase 2 was never entered, and the teardown gate
is vacuously satisfied. Detected via the strategic-decision entry
in DECISIONS.md.

DoD line: "If Phase 2 was aborted, scripts/verify_phase2_teardown.py
must report PASS"
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENGINES_DIR = _REPO_ROOT / "src/mmrag_v2/engines"
_EXPERIMENTAL_README = _ENGINES_DIR / "experimental/README.md"
_PDFPLUMBER_TEST = _REPO_ROOT / "tests/test_pdfplumber_adapter.py"
_DECISIONS = _REPO_ROOT / "docs/DECISIONS.md"


def _check_a_no_pdfplumber_import() -> tuple[bool, str]:
    """(a) No `pdfplumber` import in production engines tree."""
    if not _ENGINES_DIR.exists():
        return False, "src/mmrag_v2/engines/ not found (repo layout drift?)"
    bad: list[str] = []
    for py in _ENGINES_DIR.rglob("*.py"):
        # Skip the experimental subdirectory
        try:
            if "experimental" in py.relative_to(_ENGINES_DIR).parts:
                continue
        except ValueError:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if re.search(r"^\s*(?:import\s+pdfplumber|from\s+pdfplumber)",
                     text, re.MULTILINE):
            bad.append(str(py.relative_to(_REPO_ROOT)))
    if bad:
        return False, (
            f"Found pdfplumber import in production engines tree:\n  - "
            + "\n  - ".join(bad)
        )
    return True, "No pdfplumber import in production engines tree"


def _check_b_experimental_readme() -> tuple[bool, str]:
    """(b) `engines/experimental/README.md` exists AND contains
    the Mandate text marker."""
    if not _EXPERIMENTAL_README.exists():
        return False, (
            f"Missing {_EXPERIMENTAL_README.relative_to(_REPO_ROOT)}"
        )
    text = _EXPERIMENTAL_README.read_text(encoding="utf-8")
    # The Mandate header text is the canonical marker.
    markers = ["Abort Teardown", "Phase 2", "v2.15"]
    missing = [m for m in markers if m not in text]
    if missing:
        return False, (
            f"{_EXPERIMENTAL_README.relative_to(_REPO_ROOT)} missing "
            f"required marker(s): {missing}"
        )
    return True, f"{_EXPERIMENTAL_README.relative_to(_REPO_ROOT)} present + has Mandate markers"


def _check_c_test_marked_skip() -> tuple[bool, str]:
    """(c) `tests/test_pdfplumber_adapter.py` marked
    `@pytest.mark.skip(reason=...)` if the file exists.

    If the test file does NOT exist (Phase 2 never created it), this
    check trivially passes — there's nothing un-skipped to flag.
    """
    if not _PDFPLUMBER_TEST.exists():
        return True, (
            f"{_PDFPLUMBER_TEST.relative_to(_REPO_ROOT)} not present "
            f"(Phase 2 never created it; nothing to skip)"
        )
    text = _PDFPLUMBER_TEST.read_text(encoding="utf-8")
    if re.search(r"@pytest\.mark\.skip\s*\(\s*reason\s*=", text):
        return True, (
            f"{_PDFPLUMBER_TEST.relative_to(_REPO_ROOT)} marked @pytest.mark.skip(reason=...)"
        )
    return False, (
        f"{_PDFPLUMBER_TEST.relative_to(_REPO_ROOT)} exists but is NOT "
        f"marked with @pytest.mark.skip(reason=...)"
    )


def _check_d_decisions_entry() -> tuple[bool, str]:
    """(d) `docs/DECISIONS.md` contains 'v2.15 Phase 2 abort + teardown'
    section header."""
    if not _DECISIONS.exists():
        return False, "docs/DECISIONS.md not found"
    text = _DECISIONS.read_text(encoding="utf-8")
    if "v2.15 Phase 2 abort + teardown" in text:
        return True, "DECISIONS.md contains 'v2.15 Phase 2 abort + teardown' entry"
    return False, (
        "DECISIONS.md missing 'v2.15 Phase 2 abort + teardown' entry"
    )


def _phase2_was_entered() -> bool | None:
    """Inspect DECISIONS.md for a v2.15 strategic-path decision.

    Returns:
      True if Option A was chosen (Phase 2 was entered)
      False if Option E or F was chosen (Phase 2 never entered)
      None if no strategic decision is recorded yet
    """
    if not _DECISIONS.exists():
        return None
    text = _DECISIONS.read_text(encoding="utf-8")
    # Look for the v2.15 strategic-path entry
    m = re.search(
        r"##\s+v2\.15\s+Strategic\s+Path\s*[—\-]?\s*Option\s+([AEF])\s+Selected",
        text,
    )
    if m:
        return m.group(1) == "A"
    return None


def main() -> int:
    print("Phase 2 Teardown Verification")
    print("=" * 50)

    phase2_entered = _phase2_was_entered()
    if phase2_entered is False:
        print("Option E or F was chosen for v2.15 — Phase 2 never entered.")
        print("Teardown gate vacuously satisfied. EXIT 0.")
        return 0
    if phase2_entered is None:
        print("WARNING: No v2.15 strategic-path decision found in DECISIONS.md.")
        print("Running teardown checks anyway (defensive default).")

    checks = [
        ("(a) No pdfplumber import in production engines", _check_a_no_pdfplumber_import),
        ("(b) engines/experimental/README.md present with Mandate", _check_b_experimental_readme),
        ("(c) tests/test_pdfplumber_adapter.py marked skip (if present)", _check_c_test_marked_skip),
        ("(d) DECISIONS.md contains teardown entry", _check_d_decisions_entry),
    ]

    all_pass = True
    for name, check in checks:
        ok, msg = check()
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        print(f"         {msg}")
        if not ok:
            all_pass = False

    print("=" * 50)
    if all_pass:
        print("RESULT: PASS")
        return 0
    print("RESULT: FAIL — abort teardown incomplete, v2.15.0 tag BLOCKED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
