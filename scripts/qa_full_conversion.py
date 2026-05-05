#!/usr/bin/env python3
"""
Deterministic full-conversion QA wrapper.

This script is intended to be the reusable "first line" quality gate for a
single converted document. It runs the existing project QA scripts, then adds
document-level anomaly checks that catch localized failures the broad gates can
miss: missing pages, extreme per-page outliers, duplicate long text, corrupted
tables/text, and unusable image descriptions.

Usage:
    python scripts/qa_full_conversion.py output/Doc/ingestion.jsonl
    python scripts/qa_full_conversion.py output/Doc/ingestion.jsonl --source-pdf data/file.pdf

Exit codes:
    0 - QA_PASS
    1 - QA_WARN or QA_FAIL (default: warnings are non-zero)
    2 - usage / input error
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"

PLACEHOLDER_IMAGE_RE = re.compile(
    r"^\s*(?:\[figure on page|\[image|\[vlm_failed|figure on page)",
    re.IGNORECASE,
)
PLACEHOLDER_VISUAL_RE = re.compile(
    r"^\s*(?:\[figure on page|\[image|\[vlm_failed|figure on page)",
    re.IGNORECASE,
)
CORRUPTION_PATTERNS = (
    re.compile(r"\ufffd"),
    re.compile(r"[\u2014]{6,}"),
    re.compile(r"[\u2122]{2,}"),
    re.compile(r"[CS]{10,}"),
    re.compile(r"\b[BSQ][0-9]th"),
    re.compile(r"\bFe35\b"),
    re.compile(r"\bF1SC\b"),
    re.compile(r"\bNCOCOC\b"),
)


@dataclass
class Issue:
    severity: str
    code: str
    message: str


@dataclass
class ScriptResult:
    name: str
    exit_code: int
    output: str

    @property
    def has_failure_marker(self) -> bool:
        markers = (
            "AUDIT_FAIL",
            "UNIVERSAL_FAIL",
            "SEMANTIC_FAIL",
            "GATE_FAIL",
            "Traceback",
        )
        return any(marker in self.output for marker in markers)

    @property
    def is_clean(self) -> bool:
        return self.exit_code == 0 and not self.has_failure_marker


def _load_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {}
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {lineno}: invalid JSON: {exc}") from exc
            if obj.get("object_type") == "ingestion_metadata":
                metadata = obj
            else:
                chunks.append(obj)
    return metadata, chunks


def _chunk_page(chunk: dict[str, Any]) -> Optional[int]:
    meta = chunk.get("metadata") or {}
    page = meta.get("page_number") or chunk.get("page_number")
    try:
        return int(page) if page is not None else None
    except (TypeError, ValueError):
        return None


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _is_blankish_visual_description(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if PLACEHOLDER_VISUAL_RE.search(t):
        return True
    if len(t) < 20 and "layout" not in t.lower():
        return True
    return False


def _read_pdf_page_count(path: Path) -> Optional[int]:
    try:
        import pymupdf  # type: ignore

        doc = pymupdf.open(path)
        try:
            return int(doc.page_count)
        finally:
            doc.close()
    except Exception:
        return None


def _read_blank_pages_in_source(path: Path) -> set[int]:
    """Return the set of source-PDF page numbers (1-based) that are
    genuinely empty — no extractable text, no images, no useful blocks.

    Books often have blank versos / chapter dividers. Counting those
    as "missing chunks" would false-alarm the gate. We only fail the
    gate on pages that have real content in the source but no output
    chunk.
    """
    blank: set[int] = set()
    try:
        import pymupdf  # type: ignore

        doc = pymupdf.open(path)
    except Exception:
        return blank
    try:
        for idx in range(doc.page_count):
            page = doc.load_page(idx)
            text = (page.get_text("text") or "").strip()
            images = page.get_images(full=True)
            blocks = page.get_text("blocks") or []
            # "Blank" = no extractable text AND no images AND no
            # meaningful drawing blocks. A single empty block is
            # typical PyMuPDF output for blank pages.
            non_trivial_blocks = [b for b in blocks if isinstance(b, tuple) and len(b) >= 5 and (b[4] or "").strip()]
            if not text and not images and not non_trivial_blocks:
                blank.add(idx + 1)
    finally:
        doc.close()
    return blank


def _run_script(script_name: str, jsonl_path: Path) -> ScriptResult:
    script_path = SCRIPT_DIR / script_name
    proc = subprocess.run(
        [sys.executable, str(script_path), str(jsonl_path)],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return ScriptResult(script_name, proc.returncode, proc.stdout.strip())


def _iter_modality(chunks: Iterable[dict[str, Any]], modality: str) -> Iterable[dict[str, Any]]:
    for chunk in chunks:
        if chunk.get("modality") == modality:
            yield chunk


def _page_coverage_issues(
    metadata: dict[str, Any],
    chunks: list[dict[str, Any]],
    source_pdf: Optional[Path],
    allow_missing_pages: bool,
) -> list[Issue]:
    issues: list[Issue] = []
    total_pages = _read_pdf_page_count(source_pdf) if source_pdf else None
    if total_pages is None:
        try:
            total_pages = int(metadata.get("total_pages") or 0)
        except (TypeError, ValueError):
            total_pages = None

    if not total_pages:
        issues.append(
            Issue("WARN", "PAGE_COUNT_UNKNOWN", "Could not determine source page count.")
        )
        return issues

    pages_with_chunks = {p for p in (_chunk_page(c) for c in chunks) if p}
    missing_all = [p for p in range(1, total_pages + 1) if p not in pages_with_chunks]

    # If we have the source PDF, separate genuinely-blank pages from
    # real content-loss. Blank pages are advisory; content-loss is a
    # hard fail. Without the source, we cannot distinguish — the
    # whole list stays at the original severity.
    blank_pages = _read_blank_pages_in_source(source_pdf) if source_pdf else set()
    missing_with_content = [p for p in missing_all if p not in blank_pages]
    missing_blank = [p for p in missing_all if p in blank_pages]

    if missing_blank:
        preview = ", ".join(str(p) for p in missing_blank[:10])
        suffix = "" if len(missing_blank) <= 10 else f" ... (+{len(missing_blank) - 10})"
        issues.append(
            Issue(
                "INFO",
                "MISSING_PAGES_BLANK",
                f"{len(missing_blank)} blank source page(s) have no chunks (acceptable): "
                f"{preview}{suffix}",
            )
        )
    if missing_with_content:
        severity = "WARN" if allow_missing_pages else "FAIL"
        preview = ", ".join(str(p) for p in missing_with_content[:20])
        suffix = (
            "" if len(missing_with_content) <= 20
            else f" ... (+{len(missing_with_content) - 20})"
        )
        issues.append(
            Issue(
                severity,
                "MISSING_PAGES",
                f"{len(missing_with_content)} non-blank source page(s) have no chunks: "
                f"{preview}{suffix}",
            )
        )
    return issues


def _duplicate_text_issues(
    chunks: list[dict[str, Any]],
    min_duplicate_chars: int,
) -> list[Issue]:
    issues: list[Issue] = []
    per_page: dict[int, Counter[str]] = defaultdict(Counter)
    examples: dict[str, tuple[int, str]] = {}

    for chunk in _iter_modality(chunks, "text"):
        page = _chunk_page(chunk)
        if not page:
            continue
        text = _normalized_text(chunk.get("content") or "")
        if len(text) < min_duplicate_chars:
            continue
        per_page[page][text] += 1
        examples.setdefault(text, (page, str(chunk.get("chunk_id") or "")))

    duplicate_excess = 0
    duplicate_groups = 0
    example_bits: list[str] = []
    for page, counter in per_page.items():
        for text, count in counter.items():
            if count <= 1:
                continue
            duplicate_groups += 1
            duplicate_excess += count - 1
            if len(example_bits) < 5:
                _, chunk_id = examples[text]
                sample = text[:90]
                example_bits.append(
                    f"p{page} {count}x first={chunk_id} sample={sample!r}"
                )

    if duplicate_excess:
        issues.append(
            Issue(
                "FAIL",
                "DUPLICATE_LONG_TEXT",
                f"{duplicate_excess} duplicate long text chunk(s) across "
                f"{duplicate_groups} group(s): " + "; ".join(example_bits),
            )
        )
    return issues


def _page_outlier_issues(chunks: list[dict[str, Any]]) -> list[Issue]:
    issues: list[Issue] = []
    page_chunk_counts: Counter[int] = Counter()
    page_text_chars: Counter[int] = Counter()

    for chunk in chunks:
        page = _chunk_page(chunk)
        if not page:
            continue
        page_chunk_counts[page] += 1
        if chunk.get("modality") == "text":
            page_text_chars[page] += len(chunk.get("content") or "")

    if not page_chunk_counts:
        return [Issue("FAIL", "NO_PAGE_CHUNKS", "No chunks have page numbers.")]

    chunk_values = list(page_chunk_counts.values())
    chunk_median = statistics.median(chunk_values)
    chunk_limit = max(30, int(chunk_median * 8))
    chunk_outliers = [
        (p, c) for p, c in page_chunk_counts.items() if c > chunk_limit
    ]
    if chunk_outliers:
        top = sorted(chunk_outliers, key=lambda item: item[1], reverse=True)[:8]
        issues.append(
            Issue(
                "FAIL",
                "PAGE_CHUNK_OUTLIER",
                f"page chunk count median={chunk_median:.1f}, limit={chunk_limit}; "
                + ", ".join(f"p{p}={c}" for p, c in top),
            )
        )

    text_values = [v for v in page_text_chars.values() if v > 0]
    if text_values:
        text_median = statistics.median(text_values)
        text_limit = max(50000, int(text_median * 12))
        text_outliers = [
            (p, c) for p, c in page_text_chars.items() if c > text_limit
        ]
        if text_outliers:
            top = sorted(text_outliers, key=lambda item: item[1], reverse=True)[:8]
            issues.append(
                Issue(
                    "FAIL",
                    "PAGE_TEXT_OUTLIER",
                    f"page text chars median={text_median:.1f}, limit={text_limit}; "
                    + ", ".join(f"p{p}={c}" for p, c in top),
                )
            )
    return issues


def _corruption_issues(chunks: list[dict[str, Any]]) -> list[Issue]:
    issues: list[Issue] = []
    bad: list[tuple[str, int, str, int, str]] = []
    for chunk in chunks:
        if chunk.get("modality") not in ("text", "table"):
            continue
        content = chunk.get("content") or ""
        if not content:
            continue
        replacement_ratio = content.count("\ufffd") / max(1, len(content))
        pattern_hit = any(pattern.search(content) for pattern in CORRUPTION_PATTERNS)
        if replacement_ratio > 0.005 or pattern_hit:
            bad.append(
                (
                    str(chunk.get("modality") or ""),
                    _chunk_page(chunk) or 0,
                    str(chunk.get("chunk_id") or ""),
                    len(content),
                    _normalized_text(content)[:110],
                )
            )

    if bad:
        page_counts = Counter(page for _, page, _, _, _ in bad)
        examples = "; ".join(
            f"{mod} p{page} {chunk_id} len={length} sample={sample!r}"
            for mod, page, chunk_id, length, sample in bad[:5]
        )
        top_pages = ", ".join(f"p{p}={c}" for p, c in page_counts.most_common(8))
        issues.append(
            Issue(
                "FAIL",
                "LOCALIZED_CORRUPTION",
                f"{len(bad)} text/table chunk(s) match corruption patterns "
                f"({top_pages}). Examples: {examples}",
            )
        )
    return issues


def _image_issues(
    chunks: list[dict[str, Any]],
    max_hard_fallback_ratio: float,
    require_image_descriptions: bool,
) -> list[Issue]:
    issues: list[Issue] = []
    images = list(_iter_modality(chunks, "image"))
    if not images:
        return issues

    placeholder_content = 0
    missing_visual = 0
    hard_fallback = 0
    pending = 0
    examples: list[str] = []

    for chunk in images:
        meta = chunk.get("metadata") or {}
        content = chunk.get("content") or ""
        visual_description = (
            meta.get("visual_description") or chunk.get("visual_description") or ""
        )
        vision_status = meta.get("vision_status") or chunk.get("vision_status")
        if PLACEHOLDER_IMAGE_RE.search(content):
            placeholder_content += 1
        if vision_status == "hard_fallback":
            hard_fallback += 1
        if vision_status == "pending":
            pending += 1
        if _is_blankish_visual_description(visual_description):
            missing_visual += 1
            if len(examples) < 5:
                examples.append(
                    f"p{_chunk_page(chunk)} {chunk.get('chunk_id')} "
                    f"status={vision_status} desc={visual_description[:60]!r}"
                )

    if pending:
        issues.append(
            Issue("FAIL", "VISION_PENDING", f"{pending} image chunk(s) still pending VLM.")
        )

    hard_ratio = hard_fallback / len(images)
    if hard_ratio > max_hard_fallback_ratio:
        issues.append(
            Issue(
                "WARN",
                "VISION_HARD_FALLBACK_RATE",
                f"{hard_fallback}/{len(images)} image chunks are hard_fallback "
                f"({hard_ratio:.1%}; limit {max_hard_fallback_ratio:.1%}).",
            )
        )

    if require_image_descriptions and missing_visual:
        issues.append(
            Issue(
                "FAIL",
                "IMAGE_DESCRIPTION_UNUSABLE",
                f"{missing_visual}/{len(images)} image chunks lack useful "
                f"visual_description. Examples: {'; '.join(examples)}",
            )
        )
    elif missing_visual:
        issues.append(
            Issue(
                "WARN",
                "IMAGE_DESCRIPTION_UNUSABLE",
                f"{missing_visual}/{len(images)} image chunks lack useful "
                f"visual_description. Examples: {'; '.join(examples)}",
            )
        )

    if placeholder_content:
        issues.append(
            Issue(
                "WARN",
                "IMAGE_CONTENT_PLACEHOLDER",
                f"{placeholder_content}/{len(images)} image chunks have placeholder "
                "canonical content. This is acceptable only if downstream embedding "
                "uses visual_description.",
            )
        )
    return issues


def _asset_issues(jsonl_path: Path, chunks: list[dict[str, Any]]) -> list[Issue]:
    issues: list[Issue] = []
    output_root = jsonl_path.parent
    missing: list[str] = []
    tiny: list[str] = []

    for chunk in chunks:
        asset_ref = chunk.get("asset_ref")
        if not asset_ref:
            continue
        rel = asset_ref.get("file_path")
        if not rel:
            missing.append(f"{chunk.get('chunk_id')}: empty asset_ref.file_path")
            continue
        path = output_root / rel
        if not path.exists():
            missing.append(f"{chunk.get('chunk_id')}: {rel}")
            continue
        size = path.stat().st_size
        if size < 1000:
            tiny.append(f"{chunk.get('chunk_id')}: {rel} ({size} bytes)")

    if missing:
        issues.append(
            Issue(
                "FAIL",
                "ASSET_MISSING",
                f"{len(missing)} asset reference(s) missing. "
                + "; ".join(missing[:5]),
            )
        )
    if tiny:
        issues.append(
            Issue(
                "WARN",
                "ASSET_TINY",
                f"{len(tiny)} asset file(s) are under 1000 bytes. "
                + "; ".join(tiny[:5]),
            )
        )
    return issues


def _table_issues(chunks: list[dict[str, Any]]) -> list[Issue]:
    issues: list[Issue] = []
    bad: list[str] = []
    for chunk in _iter_modality(chunks, "table"):
        content = chunk.get("content") or ""
        if not content:
            bad.append(f"p{_chunk_page(chunk)} {chunk.get('chunk_id')}: empty")
            continue
        repl_ratio = content.count("\ufffd") / max(1, len(content))
        row_lengths = [len(line) for line in content.splitlines() if line.strip()]
        max_row = max(row_lengths) if row_lengths else 0
        avg_row = sum(row_lengths) / len(row_lengths) if row_lengths else 0
        if repl_ratio > 0.001 or len(content) > 30000 or max_row > max(3000, avg_row * 6):
            bad.append(
                f"p{_chunk_page(chunk)} {chunk.get('chunk_id')}: "
                f"len={len(content)} repl={repl_ratio:.2%} max_row={max_row}"
            )
    if bad:
        issues.append(
            Issue(
                "FAIL",
                "TABLE_CORRUPTION",
                f"{len(bad)} table chunk(s) look corrupted. " + "; ".join(bad[:5]),
            )
        )
    return issues


def _print_script_results(results: list[ScriptResult]) -> None:
    print("Existing QA Scripts")
    print("-------------------")
    for result in results:
        status = "PASS" if result.is_clean else "FAIL"
        print(f"[{status}] {result.name} exit={result.exit_code}")
        if result.output:
            for line in result.output.splitlines()[-12:]:
                print(f"  {line}")
        print()


def _print_issues(title: str, issues: list[Issue]) -> None:
    print(title)
    print("-" * len(title))
    if not issues:
        print("No additional deterministic issues found.")
        return
    for issue in issues:
        print(f"[{issue.severity}] {issue.code}: {issue.message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ingestion_jsonl", type=Path)
    parser.add_argument("--source-pdf", type=Path)
    parser.add_argument(
        "--allow-missing-pages",
        action="store_true",
        help="downgrade pages with zero chunks from FAIL to WARN",
    )
    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="return exit 0 for QA_WARN; QA_FAIL still returns 1",
    )
    parser.add_argument(
        "--min-duplicate-chars",
        type=int,
        default=120,
        help="minimum normalized text length for duplicate-content checks",
    )
    parser.add_argument(
        "--max-hard-fallback-ratio",
        type=float,
        default=0.05,
        help="warning threshold for image hard_fallback ratio",
    )
    parser.add_argument(
        "--no-require-image-descriptions",
        action="store_true",
        help="downgrade unusable image descriptions from FAIL to WARN",
    )
    args = parser.parse_args()

    jsonl_path = args.ingestion_jsonl.resolve()
    if not jsonl_path.exists():
        print(f"ERROR: JSONL not found: {jsonl_path}", file=sys.stderr)
        return 2
    if args.source_pdf and not args.source_pdf.exists():
        print(f"ERROR: source PDF not found: {args.source_pdf}", file=sys.stderr)
        return 2

    try:
        metadata, chunks = _load_jsonl(jsonl_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    script_results = [
        _run_script("qa_conversion_audit.py", jsonl_path),
        _run_script("qa_universal_invariants.py", jsonl_path),
        _run_script("qa_ingestion_hygiene.py", jsonl_path),
        _run_script("qa_semantic_fidelity.py", jsonl_path),
    ]

    issues: list[Issue] = []
    for result in script_results:
        if result.name in {"qa_conversion_audit.py", "qa_universal_invariants.py"}:
            if not result.is_clean:
                issues.append(
                    Issue(
                        "FAIL",
                        "SCRIPT_GATE_FAIL",
                        f"{result.name} reported failure (exit={result.exit_code})",
                    )
                )
        elif not result.is_clean:
            issues.append(
                Issue(
                    "WARN",
                    "SCRIPT_ADVISORY_FAIL",
                    f"{result.name} reported advisory failure (exit={result.exit_code})",
                )
            )

    issues.extend(
        _page_coverage_issues(
            metadata,
            chunks,
            args.source_pdf.resolve() if args.source_pdf else None,
            args.allow_missing_pages,
        )
    )
    issues.extend(_duplicate_text_issues(chunks, args.min_duplicate_chars))
    issues.extend(_page_outlier_issues(chunks))
    issues.extend(_corruption_issues(chunks))
    issues.extend(
        _image_issues(
            chunks,
            args.max_hard_fallback_ratio,
            not args.no_require_image_descriptions,
        )
    )
    issues.extend(_asset_issues(jsonl_path, chunks))
    issues.extend(_table_issues(chunks))

    fail_count = sum(1 for issue in issues if issue.severity == "FAIL")
    warn_count = sum(1 for issue in issues if issue.severity == "WARN")
    final_status = "QA_FAIL" if fail_count else ("QA_WARN" if warn_count else "QA_PASS")

    print("=" * 72)
    print("FULL CONVERSION QA")
    print("=" * 72)
    print(f"jsonl: {jsonl_path}")
    print(f"source_pdf: {args.source_pdf.resolve() if args.source_pdf else 'not provided'}")
    print(
        "metadata: "
        f"source={metadata.get('source_file', 'unknown')} "
        f"profile={metadata.get('profile_type', 'unknown')} "
        f"pages={metadata.get('total_pages', 'unknown')} "
        f"declared_chunks={metadata.get('chunk_count', 'unknown')}"
    )
    print(
        "chunks: "
        f"total={len(chunks)} "
        f"text={sum(1 for c in chunks if c.get('modality') == 'text')} "
        f"image={sum(1 for c in chunks if c.get('modality') == 'image')} "
        f"table={sum(1 for c in chunks if c.get('modality') == 'table')}"
    )
    print()
    _print_script_results(script_results)
    _print_issues("Additional Deterministic Checks", issues)
    print()
    print(f"{final_status}: failures={fail_count} warnings={warn_count}")

    if final_status == "QA_PASS":
        return 0
    if final_status == "QA_WARN" and args.allow_warnings:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
