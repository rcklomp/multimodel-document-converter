"""V3 architectural security tests.

These tests are the architectural firewall around the V3 vision-native
engine. They guarantee that the new ``mmrag_v3`` namespace cannot
silently re-couple to Docling or to v2.x extraction / heuristic
modules, and that every engine in the namespace honors the V3
extraction contract::

    extract(file_path: str) -> UniversalDocument

The only v2.x module the V3 engines are allowed to touch is
``mmrag_v2.universal.*`` — that subtree is the canonical UIR contract,
not a v2.x extraction module. Any other ``mmrag_v2.*`` import is a
legacy coupling and is rejected.

Per the V3 EXECUTION MANDATE: this file MUST stay green at every
phase boundary.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import typing
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import pytest

from mmrag_v2.universal.intermediate import UniversalDocument

V3_ROOT = Path(__file__).resolve().parent.parent / "src" / "mmrag_v3"

# Vision-only / glue files — no docling, no v2.x legacy.
V3_ENGINE_FILES: List[Path] = [
    V3_ROOT / "engines" / "vlm_native.py",
    V3_ROOT / "engines" / "vlm_provider.py",
    V3_ROOT / "engines" / "router.py",
    V3_ROOT / "engines" / "mineru_native.py",
]

# Files explicitly authorized to import docling (single boundary).
# Still must NOT import any v2.x legacy extraction module.
V3_DOCLING_BOUNDARY_FILES: List[Path] = [
    V3_ROOT / "engines" / "docling_fast.py",
]

BANNED_TOP_LEVEL_PREFIXES: Tuple[str, ...] = ("docling",)

# A v2.x import is allowed ONLY if it targets the format-agnostic UIR
# contract. Any other mmrag_v2.* subpackage is a legacy coupling that
# the V3 vision-native engine must not depend on.
ALLOWED_V2_PREFIXES: Tuple[str, ...] = ("mmrag_v2.universal",)


def _imported_modules(source_path: Path) -> Set[str]:
    """Return the set of fully-qualified module names imported by a file."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # Relative import — resolve against the source file's package.
                package_parts = source_path.relative_to(
                    V3_ROOT.parent
                ).with_suffix("").parts[: -node.level]
                base = ".".join(package_parts)
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            if module:
                names.add(module)
    return names


def _check_no_banned_imports(source_path: Path) -> Iterable[str]:
    """Yield human-readable violations for banned imports in source_path."""
    for module in sorted(_imported_modules(source_path)):
        top = module.split(".", 1)[0]
        if top in BANNED_TOP_LEVEL_PREFIXES:
            yield f"{source_path.name}: banned top-level import {module!r}"
        if module == "mmrag_v2" or module.startswith("mmrag_v2."):
            if not any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in ALLOWED_V2_PREFIXES
            ):
                yield (
                    f"{source_path.name}: legacy v2.x import {module!r} "
                    f"(allowed v2.x prefixes: {ALLOWED_V2_PREFIXES})"
                )


@pytest.mark.parametrize(
    "source_path",
    V3_ENGINE_FILES + V3_DOCLING_BOUNDARY_FILES,
    ids=lambda p: p.name,
)
def test_v3_engine_files_exist(source_path: Path) -> None:
    assert source_path.is_file(), f"Missing V3 engine source: {source_path}"


@pytest.mark.parametrize("source_path", V3_ENGINE_FILES, ids=lambda p: p.name)
def test_v3_engine_no_docling_or_legacy_imports(source_path: Path) -> None:
    violations = list(_check_no_banned_imports(source_path))
    assert not violations, (
        "V3 engine imports violate the architectural firewall:\n  - "
        + "\n  - ".join(violations)
    )


@pytest.mark.parametrize(
    "source_path", V3_DOCLING_BOUNDARY_FILES, ids=lambda p: p.name
)
def test_v3_docling_boundary_blocks_v2_legacy_imports(source_path: Path) -> None:
    """Docling-boundary files may import docling but must not import v2.x legacy."""
    violations: List[str] = []
    for module in sorted(_imported_modules(source_path)):
        if module == "mmrag_v2" or module.startswith("mmrag_v2."):
            if not any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in ALLOWED_V2_PREFIXES
            ):
                violations.append(
                    f"{source_path.name}: legacy v2.x import {module!r}"
                )
    assert not violations, (
        "Docling-boundary file imports v2.x legacy:\n  - "
        + "\n  - ".join(violations)
    )


def test_v3_namespace_is_importable() -> None:
    """The new namespace must import cleanly so downstream code can rely on it."""
    importlib.import_module("mmrag_v3")
    importlib.import_module("mmrag_v3.engines")
    importlib.import_module("mmrag_v3.engines.vlm_provider")
    importlib.import_module("mmrag_v3.engines.vlm_native")
    importlib.import_module("mmrag_v3.engines.docling_fast")
    importlib.import_module("mmrag_v3.engines.router")


def test_vlm_native_engine_honors_extraction_contract() -> None:
    """``VlmNativeEngine.extract`` must take str and return UniversalDocument."""
    module = importlib.import_module("mmrag_v3.engines.vlm_native")
    engine_cls = getattr(module, "VlmNativeEngine", None)
    assert engine_cls is not None, "VlmNativeEngine class missing from vlm_native"

    extract = getattr(engine_cls, "extract", None)
    assert extract is not None, "VlmNativeEngine.extract method missing"
    assert callable(extract), "VlmNativeEngine.extract must be callable"

    signature = inspect.signature(extract)
    params = list(signature.parameters.values())
    # self + file_path
    assert len(params) >= 2, (
        f"VlmNativeEngine.extract must accept (self, file_path), got {params}"
    )
    file_path_param = params[1]
    assert file_path_param.name == "file_path", (
        f"second parameter must be 'file_path', got {file_path_param.name!r}"
    )

    # `from __future__ import annotations` keeps annotations as strings;
    # resolve them through typing.get_type_hints so the assertion sees
    # the real classes the implementation promised to honor.
    hints = typing.get_type_hints(extract)
    assert hints.get("file_path") is str, (
        f"file_path annotation must resolve to str, got {hints.get('file_path')!r}"
    )
    assert hints.get("return") is UniversalDocument, (
        "extract must return UniversalDocument, got "
        f"{hints.get('return')!r}"
    )


def test_hybrid_engine_honors_extraction_contract() -> None:
    """``HybridEngine.extract`` must take str and return UniversalDocument."""
    module = importlib.import_module("mmrag_v3.engines.router")
    engine_cls = getattr(module, "HybridEngine", None)
    assert engine_cls is not None, "HybridEngine class missing from router"
    extract = getattr(engine_cls, "extract", None)
    assert callable(extract), "HybridEngine.extract must be callable"

    hints = typing.get_type_hints(extract)
    assert hints.get("file_path") is str, (
        f"file_path annotation must resolve to str, got {hints.get('file_path')!r}"
    )
    assert hints.get("return") is UniversalDocument, (
        "extract must return UniversalDocument, got "
        f"{hints.get('return')!r}"
    )


def test_docling_fast_engine_honors_extraction_contract() -> None:
    """``DoclingFastEngine.extract`` must take str and return UniversalDocument."""
    module = importlib.import_module("mmrag_v3.engines.docling_fast")
    engine_cls = getattr(module, "DoclingFastEngine", None)
    assert engine_cls is not None, "DoclingFastEngine class missing"
    extract = getattr(engine_cls, "extract", None)
    assert callable(extract), "DoclingFastEngine.extract must be callable"

    hints = typing.get_type_hints(extract)
    assert hints.get("file_path") is str, (
        f"file_path annotation must resolve to str, got {hints.get('file_path')!r}"
    )
    assert hints.get("return") is UniversalDocument, (
        "extract must return UniversalDocument, got "
        f"{hints.get('return')!r}"
    )


def test_vlm_provider_exports_describe_method() -> None:
    """The provider must expose the modular ``describe`` entry point."""
    module = importlib.import_module("mmrag_v3.engines.vlm_provider")
    provider_cls = getattr(module, "VlmProvider", None)
    assert provider_cls is not None, "VlmProvider class missing from vlm_provider"
    describe = getattr(provider_cls, "describe", None)
    assert callable(describe), "VlmProvider.describe must be callable"
