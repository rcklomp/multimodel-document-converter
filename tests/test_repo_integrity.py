"""Repo-integrity guards — preventive convention for the 2026-05-30 failure class.

This is the mechanical half of the "Committed-Truth" convention (see
``docs/REPO_INTEGRITY.md``). Each test below pins one *mechanically checkable*
invariant whose violation was observed in production this week. The guards run
in the always-on hosted CI job (no corpus / no GPU / no network) so a clean
clone of HEAD is proven sound before any heavy job spends time or VLM credits.

Philosophy (mirrors ``tests/test_v3_security.py``): a mechanical guard expressed
as a pytest assertion is worth more than a written rule, because it runs itself.
We assert against ``git ls-files`` (the *committed* tree), NOT the working
filesystem — that distinction is the whole point: a developer's dirty working
tree hides untracked-file bugs that only surface in a fresh clone.

Invariants and the failure each prevents:

  G1  import closure       — every module a TRACKED source imports is itself
                             tracked  → prevents "clean clone ModuleNotFoundError"
                             (failure #1, the worst one).
  G2  governance tracked   — the canonical docs CLAUDE.md names as Read-First are
                             tracked → a fresh clone / CI has governance (#2).
  G3  precedence applied    — any doc carrying a SUPERSEDED marker must name a
                             real, tracked superseding doc → conflicts resolve at
                             the source instead of being re-derived (#3).
  G4  contract liveness    — when a doc asserts "guarded by test <file>", that
                             test file exists, is tracked, and is not skipped →
                             a doc can't claim a guard that was deleted/skipped
                             (#4 invalidated-contract, part of #7 hollow-green).
  G5  no dangling paths    — repo-path references in committed governance docs
                             point at things that exist on disk → no references
                             to moved/deleted paths (#6).
  G6  skips are registered — every unconditional ``V3_DEFERRED`` test skip is
                             listed in ``docs/V3_DEFERRED_TESTS.md`` → behavioral
                             coverage can't silently rot off the books (#5 stale
                             status + #7 hollow-green via skipped tests).
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Set

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

# Canonical Read-First docs (CLAUDE.md §"Read First") + the agent contracts.
# These ARE the governance surface; a fresh clone with these missing is ungoverned.
CANONICAL_GOVERNANCE_DOCS = (
    "CLAUDE.md",
    "AGENTS.md",
    "docs/PROJECT_STATUS.md",
    "docs/ARCHITECTURE_V3_DRAFT_0.5.md",
    "docs/V3_EXECUTION_MANDATE.md",
    "docs/DECISIONS.md",
    "docs/QUALITY_GATES.md",
    "docs/TESTING.md",
    "docs/V3_DEFERRED_TESTS.md",
    "docs/REPO_INTEGRITY.md",
)


@lru_cache(maxsize=1)
def _tracked_files() -> Set[str]:
    """POSIX-relative paths of every file committed/staged in the repo.

    Uses ``git ls-files`` so the guard reasons about the *committed* tree, not
    the dirty working copy. This is what a fresh ``git clone`` would contain.
    """
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {line.strip() for line in out.splitlines() if line.strip()}


@lru_cache(maxsize=1)
def _tracked_python_modules() -> Set[str]:
    """Set of dotted module names that are *tracked* under ``src/``.

    e.g. ``src/mmrag_v3/engines/router.py`` -> ``mmrag_v3.engines.router`` and
    its package prefixes. Package ``__init__.py`` contributes the bare package.
    """
    modules: Set[str] = set()
    for rel in _tracked_files():
        if not rel.startswith("src/") or not rel.endswith(".py"):
            continue
        parts = Path(rel).relative_to("src").with_suffix("").parts
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            modules.add(".".join(parts))
    return modules


@lru_cache(maxsize=1)
def _top_level_packages() -> Set[str]:
    """First path component of every tracked module — the local top-level packages."""
    return {m.split(".", 1)[0] for m in _tracked_python_modules()}


def _module_is_local(dotted: str) -> bool:
    """True if ``dotted`` names one of this repo's own top-level packages."""
    return dotted.split(".", 1)[0] in _top_level_packages()


def _resolve_relative(source_rel: str, level: int, module: str | None) -> str:
    """Resolve a ``from . import x`` style import to a dotted module under src."""
    pkg_parts = Path(source_rel).relative_to("src").with_suffix("").parts
    # For a module file, its package is its parent; level 1 == current package.
    base_parts = list(pkg_parts[: -level]) if level <= len(pkg_parts) else []
    if module:
        base_parts.append(module)
    return ".".join(base_parts)


def _imports_of(source_rel: str) -> Set[str]:
    """Top-level, UNGUARDED dotted module names imported by a tracked source file.

    Scope is deliberately narrow — it matches exactly what breaks a fresh
    ``import <module>`` (failure #1): imports that execute at module-load time
    and are NOT wrapped in ``try/except``. Lazy imports inside functions and
    optional imports inside ``try`` blocks (the project's documented
    fail-graceful pattern for sibling-repo engines) are intentionally excluded —
    they do not break a clean clone's import.

    Captures the *module portion* only: ``import a.b.c`` -> ``a.b.c``;
    ``from a.b import c`` -> ``a.b`` (the symbol ``c`` may be a submodule or an
    attribute and is intentionally not resolved). Relative imports are absolutized.
    """
    tree = ast.parse((REPO_ROOT / source_rel).read_text(encoding="utf-8"))
    names: Set[str] = set()
    # Only module-body statements run at import time. Statements nested inside a
    # function/class/try are not part of the unconditional import surface.
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names.add(_resolve_relative(source_rel, node.level, node.module))
            elif node.module:
                names.add(node.module)
    return {n for n in names if n}


def _local_module_is_tracked(dotted: str) -> bool:
    """Whether a local dotted import would resolve against the *tracked* tree.

    For both ``import a.b.c`` and ``from a.b.c import name``, the dotted name
    captured here (``a.b.c`` — the module portion, never the imported symbol)
    must itself be a tracked module/package. A merely-tracked *ancestor* is NOT
    enough — that leniency is exactly what let the original
    ``import mmrag_v3.<leaf>`` bug slip through (the leaf module was untracked
    but the ancestor package existed). The genuinely-ambiguous
    ``from pkg import submod`` case is handled safely: ``pkg`` must resolve, and
    ``submod`` (a name, possibly an attribute) is intentionally not checked.
    """
    return dotted in _tracked_python_modules()


# ── G1: import closure — the clean-clone-fails bug ──────────────────────────


def test_g1_tracked_source_imports_are_tracked() -> None:
    """Every LOCAL module imported by a tracked src file must itself be tracked.

    Prevents failure #1: ``batch_processor`` imported ``mmrag_v3`` + a chunker
    that were never ``git add``-ed, so HEAD imported fine locally (dirty tree)
    but ``ModuleNotFoundError``-ed in a clean clone / CI.
    """
    violations = []
    for rel in sorted(f for f in _tracked_files() if f.startswith("src/") and f.endswith(".py")):
        for imported in sorted(_imports_of(rel)):
            if _module_is_local(imported) and not _local_module_is_tracked(imported):
                violations.append(f"{rel}: imports untracked local module {imported!r}")
    assert not violations, (
        "Tracked code imports an UNTRACKED local module — a clean clone of HEAD "
        "would fail to import. `git add` the missing module(s):\n  - "
        + "\n  - ".join(violations)
    )


# ── G2: governance docs are tracked ─────────────────────────────────────────


def test_g2_canonical_governance_docs_are_tracked() -> None:
    """The Read-First governance set must be in the committed tree."""
    tracked = _tracked_files()
    missing = [d for d in CANONICAL_GOVERNANCE_DOCS if d not in tracked]
    assert not missing, (
        "Canonical governance docs are NOT git-tracked — a fresh clone / CI "
        "would have zero governance. `git add` them:\n  - " + "\n  - ".join(missing)
    )


# ── G3: precedence is applied at the source ─────────────────────────────────

# A doc that is overruled by another MUST carry this marker naming the winner,
# so an agent never has to re-derive a contradiction (failure #3). Format:
#   SUPERSEDED ... by `docs/V3_EXECUTION_MANDATE.md`
_SUPERSEDED_RE = re.compile(r"SUPERSEDED[^\n`]*by\s+`([^`]+)`", re.IGNORECASE)


def test_g3_superseded_markers_name_a_real_doc() -> None:
    """Every ``SUPERSEDED ... by `X``` marker must point at a tracked doc."""
    tracked = _tracked_files()
    violations = []
    for rel in sorted(d for d in tracked if d.endswith(".md")):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        for m in _SUPERSEDED_RE.finditer(text):
            target = m.group(1).strip()
            if target not in tracked:
                violations.append(f"{rel}: SUPERSEDED-by points at missing doc {target!r}")
    assert not violations, (
        "A precedence (SUPERSEDED-by) marker names a doc that is not tracked — "
        "the conflict cannot be resolved at the source:\n  - " + "\n  - ".join(violations)
    )


# ── G4: documented test-guard contracts are live ────────────────────────────

# Matches doc prose like: guarded by `tests/test_v3_security.py`
#                         AST-guarded by ``tests/test_x.py``
_GUARD_RE = re.compile(r"guarded by\s+`+\s*(tests/test_[A-Za-z0-9_]+\.py)\s*`+", re.IGNORECASE)


@lru_cache(maxsize=1)
def _unconditionally_skipped_test_files() -> Set[str]:
    """Tracked test files whose WHOLE module is unconditionally skipped.

    Detects ``pytestmark = ...mark.skip(...)`` (no ``if``). ``skipif`` is a
    legitimate runtime guard (corpus/GPU absence) and is deliberately excluded.
    """
    skipped: Set[str] = set()
    for rel in _tracked_files():
        if not (rel.startswith("tests/") and rel.endswith(".py")):
            continue
        try:
            tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
            ):
                call = node.value
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "skip"  # NOT skipif
                ):
                    skipped.add(rel)
    return skipped


def test_g4_documented_guard_tests_exist_and_run() -> None:
    """A doc that says "guarded by test T" must have T tracked and not skipped.

    Prevents failure #4: a refactor invalidated a contract, but the doc kept
    asserting it while test T had been skipped (a hollow guarantee).
    """
    tracked = _tracked_files()
    skipped = _unconditionally_skipped_test_files()
    violations = []
    for rel in sorted(d for d in tracked if d.endswith(".md")):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        for m in _GUARD_RE.finditer(text):
            t = m.group(1).strip()
            if t not in tracked:
                violations.append(f"{rel}: claims guard {t!r} but it is not tracked")
            elif t in skipped:
                violations.append(
                    f"{rel}: claims guard {t!r} but that test is unconditionally SKIPPED"
                )
    assert not violations, (
        "A doc asserts a test-guard contract that is dead (missing or skipped). "
        "Fix the doc or restore the guard:\n  - " + "\n  - ".join(violations)
    )


# ── G5: no dangling repo-path references in governance docs ──────────────────

# Conservative: only inside-backtick tokens that look like a repo path we own
# (start with src/ tests/ scripts/ docs/) — avoids flagging prose or URLs.
_PATH_RE = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|md|sh|yml|yaml|json|toml|cfg))`")
# Live, must-exist references: code/artifact trees + LIVE top-level docs.
# Deliberately EXCLUDED: docs/archive/** and docs/.archive/** are the
# intentionally quarantined history (`.aiignore`-blocked) — a doc may cite a
# now-archived report as a historical breadcrumb, and DECISIONS.md is an
# append-only log full of such citations. Flagging those would be noise, not a
# real "moved/deleted live path" bug. We pin the paths that must stay live.
_OWNED_PREFIXES = ("src/", "tests/", "scripts/")
_LIVE_DOC_RE = re.compile(r"^docs/[^/]+\.md$")  # top-level docs only, not docs/archive/**
# Convention escape hatch: a path the doc explicitly annotates (on the SAME
# line) as not-yet-built / planned is an intentional forward reference, not a
# dangling one. This keeps the guard honest without exempting whole docs.
_PLANNED_MARKERS = ("not yet built", "not yet", "(planned)", "planned —", "to be built")


def _line_marks_planned(line: str) -> bool:
    low = line.lower()
    return any(marker in low for marker in _PLANNED_MARKERS)


def _is_live_owned_path(ref: str) -> bool:
    if ref.startswith(_OWNED_PREFIXES):
        return True
    return bool(_LIVE_DOC_RE.match(ref))


# G5 checks docs that are CONTRACTS ABOUT CURRENT STATE — their path
# references must resolve now. Two governance docs are deliberately exempt
# because their genre legitimately names paths that do not currently exist:
#   - ARCHITECTURE_V3_DRAFT_0.5.md is a forward-looking *charter* (target
#     state) that names planned-but-unbuilt artifacts.
#   - DECISIONS.md is an append-only historical *log* that cites reports later
#     archived/renamed; those are breadcrumbs, not live contracts.
# Excluding them keeps G5 a high-signal "current contract is intact" guard.
_G5_CURRENT_STATE_DOCS = tuple(
    d
    for d in CANONICAL_GOVERNANCE_DOCS
    if d not in ("docs/ARCHITECTURE_V3_DRAFT_0.5.md", "docs/DECISIONS.md")
)


def test_g5_governance_doc_paths_resolve() -> None:
    """Live repo-path references in current-state docs must point at things that exist.

    Prevents failure #6: committed docs referencing repo paths later moved or
    deleted (dangling references that mislead the next session). Scoped to
    LIVE owned paths (src/ tests/ scripts/ + top-level docs/*.md) in
    current-state docs; the forward-looking charter, the append-only
    DECISIONS log, and quarantined docs/archive history are out of scope.
    """
    violations = []
    for rel in _G5_CURRENT_STATE_DOCS:
        doc = REPO_ROOT / rel
        if not doc.is_file():
            continue  # G2 already covers missing docs
        for line in doc.read_text(encoding="utf-8").splitlines():
            if _line_marks_planned(line):
                continue  # explicit forward reference — not dangling
            for m in _PATH_RE.finditer(line):
                ref = m.group(1)
                base = ref.split("::", 1)[0].split("#", 1)[0]  # strip ::node / #anchor
                if not _is_live_owned_path(base):
                    continue
                if not (REPO_ROOT / base).exists():
                    violations.append(f"{rel}: dangling path reference {ref!r}")
    assert not violations, (
        "Canonical governance docs reference LIVE repo paths that no longer exist "
        "(moved/deleted). Fix the reference or restore the path:\n  - "
        + "\n  - ".join(sorted(set(violations)))
    )


# ── G6: every V3_DEFERRED skip is registered in the manifest ─────────────────


@lru_cache(maxsize=1)
def _registered_deferred_tests() -> Set[str]:
    """Test file basenames listed in docs/V3_DEFERRED_TESTS.md."""
    text = (REPO_ROOT / "docs" / "V3_DEFERRED_TESTS.md").read_text(encoding="utf-8")
    return set(re.findall(r"tests/(test_[A-Za-z0-9_]+\.py)", text))


def test_g6_v3_deferred_skips_are_registered() -> None:
    """A test skipped with reason "V3_DEFERRED" must appear in the manifest.

    Prevents the hollow-green / stale-status path (failures #5, #7): a quietly
    skipped behavioral test drops off the books. Registration keeps the
    deferred surface auditable — you can't lose 64 skipped tests in silence.
    """
    manifest = _registered_deferred_tests()
    unregistered = []
    for rel in sorted(_unconditionally_skipped_test_files()):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        if "V3_DEFERRED" in text and Path(rel).name not in manifest:
            unregistered.append(rel)
    assert not unregistered, (
        "Unconditionally-skipped V3_DEFERRED tests are missing from "
        "docs/V3_DEFERRED_TESTS.md — add a line per test so the deferred "
        "coverage stays auditable:\n  - " + "\n  - ".join(unregistered)
    )
