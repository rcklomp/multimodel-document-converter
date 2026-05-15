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


@dataclass(frozen=True)
class EpubChapterInfo:
    index: int
    name: str
    text_len: int
    text_sample: str


# Plan v2.9 Phase G (2026-05-11) — advisory-warning allowance.
# WARN codes in this set are documented allowed-advisories per
# `docs/QUALITY_GATES.md` "Advisory Warning Classes". When EVERY
# remaining WARN issue's code is in this set AND there are zero
# FAILs, the gate emits `QA_PASS_WITH_ADVISORIES` rather than
# `QA_WARN`. The advisory warnings remain visible in the output for
# operator awareness but do not block ship per Goal 1's "promote into
# an explicit pass variant" provision.
#
# Codes are unconditional unless noted in the per-code check at
# `_warn_is_documented_advisory`.
_ALLOWED_ADVISORY_WARN_CODES = frozenset(
    {
        "ASSET_TINY",
        "PAGE_COUNT_UNKNOWN",
        "SCRIPT_ADVISORY_FAIL",
        "VISION_HARD_FALLBACK_RATE",  # conditional: all hard_fallbacks must carry the F4 sentinel
        # v2.10 Phase 7: EPUB lane. Allowed only for WARN-severity
        # MISSING_CHAPTERS raised by _epub_chapter_coverage_issues when
        # every missing chapter is an edge-only low-content structural
        # spine item. Internal/content-bearing missing chapters are FAIL.
        "MISSING_CHAPTERS",
    }
)

# F4 sentinel from `scripts/enrich_image_chunks_v29.py` —
# `SHORT_RESPONSE_SENTINEL`. A hard_fallback with this sentinel is a
# documented "complex asset, VLM produced terse description after
# retry" case — legitimate signal, not a defect.
_F4_HARD_FALLBACK_SENTINEL = "complex_asset_short_response_after_retry"


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


def _is_blankish_visual_description(
    text: str,
    chunk: Optional[dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> bool:
    """Phase 3 Step 3: complexity-aware blankish check.

    Hard-fallback exemption (F4 contract): when a chunk has
    ``vision_status="hard_fallback"`` AND both ``vision_error`` and
    ``vision_provider_used`` set, the description is a documented
    no-VLM-signal state, not a missing description; exempt from
    blankish checks.

    Complexity-aware short-description rule (replaces the prior flat
    ``len < 20 and "layout" not in t.lower()`` rule):
    - simple asset → short description acceptable, NOT blankish.
    - complex / text_heavy asset → short description IS blankish.

    The plain-text backward-compatible call (no ``chunk``) preserves
    the original behavior so existing callers keep working.
    """
    # F4 hard-fallback exemption FIRST — placeholder-shaped content is
    # the contract on hard_fallback chunks, not a fault.
    if chunk is not None:
        meta = chunk.get("metadata") or {}
        if (
            meta.get("vision_status") == "hard_fallback"
            and (meta.get("vision_error") or "").strip()
            and (meta.get("vision_provider_used") or "").strip()
        ):
            return False

    t = (text or "").strip()
    if not t:
        return True
    if PLACEHOLDER_VISUAL_RE.search(t):
        return True

    # Short-description complexity gate.
    if len(t) < 20 and "layout" not in t.lower():
        if chunk is None:
            # Backward-compat: no chunk → fall back to flat short rule.
            return True
        try:
            from mmrag_v2.vision.asset_complexity import classify_asset_complexity
        except ImportError:  # pragma: no cover — editable install always present.
            return True
        complexity = classify_asset_complexity(chunk, output_dir=output_dir).complexity
        # simple asset → short description allowed; otherwise short = blankish.
        return complexity != "simple"
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


def _is_epub_source(path: Optional[Path]) -> bool:
    """v2.10 Phase 7: detect EPUB source paths. Producer
    (`processor._epub_to_html` + `_apply_epub_synthetic_pagination`)
    emits synthetic page numbers (`chapter_1based * 1000 + position // 5`)
    for EPUBs, so the PDF page-coverage check does not apply.
    """
    return path is not None and path.suffix.lower() == ".epub"


_EPUB_STRUCTURAL_NAME_RE = re.compile(
    r"(?:^|[/_.-])(?:cover|titlepage|title|colophon|copyright|dedication|"
    r"imprint|toc|contents|nav)(?:[/_.-]|$)",
    re.IGNORECASE,
)
_EPUB_STRUCTURAL_TEXT_RE = re.compile(
    r"(?:©|\b(?:colofon|colophon|copyright|all rights reserved|alle rechten "
    r"voorbehouden|isbn|uitgever|publisher|published by|cover|omslag|"
    r"title page|inhoudsopgave)\b)",
    re.IGNORECASE,
)


def _epub_spine_chapters(path: Path) -> Optional[list[EpubChapterInfo]]:
    """v2.10 Phase 7: enumerate non-empty spine chapters via ebooklib —
    mirrors the producer's chapter enumeration in
    `processor._epub_to_html` so the gate's expected chapter set lines
    up with what the producer emits.
    """
    try:
        import ebooklib  # type: ignore
        from ebooklib import epub  # type: ignore
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        return None
    try:
        book = epub.read_epub(str(path), options={"ignore_ncx": True})
    except Exception:
        return None
    chapters: list[EpubChapterInfo] = []
    for sp_entry in book.spine:
        idref = sp_entry[0] if isinstance(sp_entry, tuple) else sp_entry
        item = book.get_item_with_id(idref)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        try:
            soup = BeautifulSoup(item.get_content(), "html.parser")
        except Exception:
            continue
        body = soup.find("body")
        if body is None:
            continue
        for tag in body.find_all(["svg", "script", "style"]):
            tag.decompose()
        inner = body.decode_contents()
        if inner.strip():
            text = " ".join(body.get_text(" ", strip=True).split())
            chapters.append(
                EpubChapterInfo(
                    index=len(chapters) + 1,
                    name=item.get_name() or f"chapter_{len(chapters) + 1}",
                    text_len=len(text),
                    text_sample=text[:160],
                )
            )
    return chapters


def _epub_chapter_count(path: Path) -> Optional[int]:
    chapters = _epub_spine_chapters(path)
    if chapters is None:
        return None
    return len(chapters)


def _is_low_value_epub_structural_chapter(chapter: EpubChapterInfo) -> bool:
    """Return True only for EPUB spine items that are reasonable advisory
    candidates when stripped by the HTML parser.

    The carve-out is intentionally narrow: short leading/trailing title,
    copyright, colophon, cover, TOC/nav, or blank wrapper items. Real
    internal/content-bearing chapters still fail chapter coverage.
    """
    name = chapter.name or ""
    sample = chapter.text_sample or ""
    if chapter.text_len == 0:
        return True
    if chapter.text_len <= 1200 and (
        _EPUB_STRUCTURAL_NAME_RE.search(name)
        or _EPUB_STRUCTURAL_TEXT_RE.search(sample)
    ):
        return True
    return False


def _missing_epub_chapters_are_advisory(
    missing: list[int],
    chapters: list[EpubChapterInfo],
) -> bool:
    """MISSING_CHAPTERS is advisory only for low-value edge chapters.

    Missing internal chapters indicate real content loss and remain FAIL.
    Edge means before the first covered chapter or after the last covered
    chapter; scattered gaps do not qualify.
    """
    if not missing or not chapters:
        return False
    missing_set = set(missing)
    all_indices = {chapter.index for chapter in chapters}
    covered = all_indices - missing_set
    if not covered:
        return False
    first_covered = min(covered)
    last_covered = max(covered)
    edge_missing = {
        idx
        for idx in all_indices
        if idx < first_covered or idx > last_covered
    }
    if missing_set != edge_missing:
        return False
    by_index = {chapter.index: chapter for chapter in chapters}
    return all(
        _is_low_value_epub_structural_chapter(by_index[idx])
        for idx in missing
    )


def _epub_chapter_coverage_issues(
    epub_path: Path,
    chunks: list[dict[str, Any]],
    allow_missing: bool,
) -> list[Issue]:
    """v2.10 Phase 7: EPUB lane chapter coverage.

    Producer synthetic mapping is ``page_number = chapter_1based * 1000
    + position // 5``, so a chunk's chapter index is ``page_number //
    1000``. Missing chapters are advisory only when every missing item
    is an edge-only low-content structural spine item (title page,
    colophon, copyright stub, blank wrapper, etc.). Internal gaps or
    content-bearing missing chapters are FAIL regardless of
    ``--allow-missing-pages``.
    """
    issues: list[Issue] = []
    chapters = _epub_spine_chapters(epub_path)
    if not chapters:
        issues.append(
            Issue(
                "WARN",
                "PAGE_COUNT_UNKNOWN",
                f"Could not enumerate EPUB chapters in {epub_path.name}; "
                "skipping chapter coverage check.",
            )
        )
        return issues
    chapter_count = len(chapters)
    chapters_with_chunks: set[int] = set()
    for c in chunks:
        page = _chunk_page(c)
        if page:
            chapters_with_chunks.add(page // 1000)
    missing = [
        ch for ch in range(1, chapter_count + 1) if ch not in chapters_with_chunks
    ]
    if missing:
        severity = (
            "WARN"
            if _missing_epub_chapters_are_advisory(missing, chapters)
            else "FAIL"
        )
        preview = ", ".join(str(c) for c in missing[:20])
        suffix = "" if len(missing) <= 20 else f" ... (+{len(missing) - 20})"
        missing_names = ", ".join(
            f"{ch.index}={ch.name}" for ch in chapters if ch.index in set(missing[:5])
        )
        reason = (
            "edge low-content structural spine item(s)"
            if severity == "WARN"
            else "internal or content-bearing spine item(s)"
        )
        issues.append(
            Issue(
                severity,
                "MISSING_CHAPTERS",
                f"{len(missing)} EPUB chapter(s) have no chunks "
                f"(expected 1..{chapter_count}): {preview}{suffix}; "
                f"{reason}; {missing_names}",
            )
        )
    return issues


_INTENTIONALLY_BLANK_PATTERN = re.compile(
    r"this\s+page\s+(?:is\s+)?intentionally\s+left\s+blank",
    re.IGNORECASE,
)


# Phase B4.a (2026-05-11) thresholds. The render-based blank check
# fires when a page's rasterized pixel statistics indicate a
# near-white surface (no substantive ink coverage) AND its text-layer
# content is below a small cap. Empirically verified safe on the v2.9
# 34-doc corpus: 0/15 real body-text content pages trigger; catches
# Python_Distilled's ~697 publisher-template placeholder pages,
# Devlin p2/p264, Chaubal p4, and similar near-blank pages with no
# retrieval value.
_RENDER_BLANK_MEAN_MIN = 245.0
_RENDER_BLANK_STD_MAX = 20.0
_RENDER_BLANK_TEXT_CAP = 200
_RENDER_BLANK_DPI_SCALE = 0.5  # 36 dpi at default page size; ~10 ms/page
_RENDER_BLANK_NO_TEXT_MEAN_MIN = 250.0  # stricter threshold for the
# zero-text variant (`_page_is_no_text_image_only_placeholder`)
# because removing the text-existence-as-signal requires a more
# certain "this is white" mean.


def _is_intentionally_blank_text(text: str) -> bool:
    """Phase B2 (2026-05-11): some publisher templates emit chapter-
    divider versos containing only the boilerplate "This page
    intentionally left blank" (sometimes duplicated by a backing
    layer). The gate's strict blank check (no text + no images + no
    blocks) does not classify these pages as blank because the
    boilerplate text is present. The pipeline correctly emits no
    chunk for these pages (the content has no semantic value), so the
    gate incorrectly flags them as MISSING_PAGES.

    Recognize the boilerplate as blank-equivalent so the gate marks
    these pages as advisory (MISSING_PAGES_BLANK) rather than a hard
    MISSING_PAGES FAIL. The regex is restrictive enough to avoid
    false positives: it requires the literal four-word phrase
    "intentionally left blank" with optional "is" between "page" and
    "intentionally", and the only matched text must be effectively
    the boilerplate (≤ 2x the boilerplate length, indicating no
    real content surrounds it).
    """
    stripped = text.strip()
    if not stripped:
        return False
    if not _INTENTIONALLY_BLANK_PATTERN.search(stripped):
        return False
    # Allow duplication (publisher backing-layer pattern) but reject
    # pages where the boilerplate is just one detail among real text.
    return len(stripped) <= 120


def _page_render_stats(page: Any) -> Optional[tuple[float, float]]:
    """Return (mean, std) of the rasterized page in grayscale, or
    None when rendering fails."""
    try:
        import pymupdf  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
        import io
    except Exception:
        return None
    try:
        matrix = pymupdf.Matrix(_RENDER_BLANK_DPI_SCALE, _RENDER_BLANK_DPI_SCALE)
        pix = page.get_pixmap(matrix=matrix)
        pil = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
        arr = np.array(pil)
    except Exception:
        return None
    return float(arr.mean()), float(arr.std())


def _page_render_is_near_blank(page: Any) -> bool:
    """Phase B4.a (2026-05-11): rasterize the page at low DPI and
    check whether the rendered pixels are near-white (high mean, low
    standard deviation). Catches publisher-template placeholder pages
    that have a single full-page image (so the text-only blank check
    fails) but visually carry no content.

    Empirically validated on the v2.9 corpus: zero false positives on
    15 sampled real body-text content pages (Harry/Bourne/Cronin/
    Adedeji/Combat); catches Python_Distilled's ~615 publisher
    placeholder pages, Devlin p2/p264, and similar.

    See `docs/DECISIONS.md` "Retrieval-Value Test": a page rendered
    as near-white has no retrieval-relevant content; the few text
    characters that may be present (page number, watermark) are
    metadata, not content.
    """
    stats = _page_render_stats(page)
    if stats is None:
        return False
    mean, std = stats
    return mean > _RENDER_BLANK_MEAN_MIN and std < _RENDER_BLANK_STD_MAX


def _page_is_no_text_image_only_placeholder(page: Any) -> bool:
    """Phase B4.a refinement (2026-05-11): a stricter sibling of
    `_page_render_is_near_blank`. Fires when the page has NO
    text-layer content at all AND at least one image AND the render
    mean is above `_RENDER_BLANK_NO_TEXT_MEAN_MIN` (250).

    This catches Python_Distilled placeholder pages that sit at the
    `_RENDER_BLANK_STD_MAX` boundary (std ~20-23). Because the
    page has zero text-layer content, the std band can be widened
    without risking real-content false positives: a true content
    page would have text. Combined with the high mean, the rule
    is principle-based (no retrieval signal: no text + nearly all
    white pixels = placeholder).

    Verified against real-content reference pages: all real body
    pages sampled (Harry/Bourne/Cronin/Adedeji/Combat) have text
    content, so the `text_len == 0` precondition excludes them.
    Earthship p109 (real diagram, text_len=0, mean=128) is excluded
    by the mean>250 precondition.
    """
    try:
        text = (page.get_text("text") or "").strip()
    except Exception:
        return False
    if text:
        return False
    try:
        images = page.get_images(full=True)
    except Exception:
        images = []
    if not images:
        return False
    stats = _page_render_stats(page)
    if stats is None:
        return False
    mean, _std = stats
    return mean > _RENDER_BLANK_NO_TEXT_MEAN_MIN


def _read_blank_pages_in_source(path: Path) -> set[int]:
    """Return the set of source-PDF page numbers (1-based) that are
    genuinely empty — no extractable text, no images, no useful blocks.

    Books often have blank versos / chapter dividers. Counting those
    as "missing chunks" would false-alarm the gate. We only fail the
    gate on pages that have real content in the source but no output
    chunk.

    Phase B2 extension: also classify "intentionally left blank"
    boilerplate-only pages as blank-equivalent (see
    `_is_intentionally_blank_text`).

    Phase B4.a extension (2026-05-11): also classify pages whose
    rasterized render is near-white (no substantive ink coverage)
    AND whose text-layer content is below `_RENDER_BLANK_TEXT_CAP`
    chars as blank-equivalent. Catches publisher-template
    placeholder pages with a single near-white full-page image (the
    "697 Python_Distilled missing pages" class).
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
                continue
            # Phase B2: "intentionally left blank" boilerplate-only
            # pages count as blank-equivalent. Images on such pages
            # would be a publisher logo or page-number marker; ignore.
            if not non_trivial_blocks and _is_intentionally_blank_text(text):
                blank.add(idx + 1)
                continue
            # Even with non-trivial blocks, accept if the only block
            # text is the boilerplate.
            if non_trivial_blocks and all(
                _is_intentionally_blank_text(b[4] or "")
                for b in non_trivial_blocks
            ) and _is_intentionally_blank_text(text):
                blank.add(idx + 1)
                continue
            # Phase B4.a: render-based near-blank check. The page
            # may have a full-page image (so `images` is non-empty)
            # but the rendered pixels are nearly all white. Only
            # apply when text-layer is below the cap so that a real
            # text page with a coincidentally light background is
            # not misclassified.
            if len(text) < _RENDER_BLANK_TEXT_CAP and _page_render_is_near_blank(page):
                blank.add(idx + 1)
                continue
            # Phase B4.a refinement: stricter zero-text + image-only
            # placeholder check. Picks up the boundary cases the
            # render-near-blank check misses (page placeholders that
            # have a slightly higher std but still no real content).
            if _page_is_no_text_image_only_placeholder(page):
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
    output_dir: Optional[Path] = None,
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
        # F4 hard-fallback exemption: a chunk with vision_status="hard_fallback"
        # AND both vision_error and vision_provider_used recorded is a
        # documented no-VLM-signal state, NOT a placeholder row. Skip the
        # placeholder-content tally so the gate doesn't double-count it.
        is_documented_hard_fallback = (
            vision_status == "hard_fallback"
            and bool((meta.get("vision_error") or "").strip())
            and bool((meta.get("vision_provider_used") or "").strip())
        )
        if PLACEHOLDER_IMAGE_RE.search(content) and not is_documented_hard_fallback:
            placeholder_content += 1
        if vision_status == "hard_fallback":
            hard_fallback += 1
        if vision_status == "pending":
            pending += 1
        if _is_blankish_visual_description(
            visual_description, chunk=chunk, output_dir=output_dir
        ):
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


def _warn_is_documented_advisory(
    issue: Issue,
    chunks: list[dict[str, Any]],
) -> bool:
    """Phase G (2026-05-11): return True when a WARN-level issue is
    documented in `docs/QUALITY_GATES.md` "Advisory Warning Classes"
    as an allowed advisory that does not block QA_PASS_WITH_ADVISORIES.

    Most codes are unconditional. Conditional codes:
    - `MISSING_CHAPTERS`: allowed only for the explicit edge low-content
      structural-chapter message emitted by `_epub_chapter_coverage_issues`.
    - `VISION_HARD_FALLBACK_RATE`: allowed only when every hard_fallback
      image chunk in the corpus carries the F4 sentinel
      (`complex_asset_short_response_after_retry`). A
      non-F4-sentinelled hard_fallback indicates a real defect (asset
      file missing, VLM API failure, validator rejection) and continues
      to block PASS.
    """
    if issue.severity != "WARN":
        return False
    if issue.code not in _ALLOWED_ADVISORY_WARN_CODES:
        return False
    if issue.code == "MISSING_CHAPTERS":
        return "edge low-content structural spine item" in issue.message
    if issue.code != "VISION_HARD_FALLBACK_RATE":
        return True

    # Conditional check for VISION_HARD_FALLBACK_RATE: every hard_fallback
    # chunk must carry the F4 sentinel.
    hard_fallback_chunks = [
        c
        for c in chunks
        if c.get("modality") == "image"
        and ((c.get("metadata") or {}).get("vision_status") == "hard_fallback")
    ]
    if not hard_fallback_chunks:
        # No hard_fallbacks present yet a VISION_HARD_FALLBACK_RATE WARN
        # was raised — defensively treat as non-advisory.
        return False
    for chunk in hard_fallback_chunks:
        meta = chunk.get("metadata") or {}
        if meta.get("vision_error") != _F4_HARD_FALLBACK_SENTINEL:
            return False
    return True


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

    source = args.source_pdf.resolve() if args.source_pdf else None
    if _is_epub_source(source):
        # v2.10 Phase 7: EPUB sources use synthetic per-chapter page
        # numbers; the PDF page-coverage check does not apply. Validate
        # chapter coverage instead via ebooklib spine enumeration.
        issues.extend(
            _epub_chapter_coverage_issues(source, chunks, args.allow_missing_pages)
        )
    else:
        issues.extend(
            _page_coverage_issues(
                metadata,
                chunks,
                source,
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
            output_dir=jsonl_path.parent,
        )
    )
    issues.extend(_asset_issues(jsonl_path, chunks))
    issues.extend(_table_issues(chunks))

    fail_count = sum(1 for issue in issues if issue.severity == "FAIL")
    warn_issues = [issue for issue in issues if issue.severity == "WARN"]
    warn_count = len(warn_issues)
    # Phase G: split WARNs into documented-allowed advisories vs. real.
    # When fail_count == 0 AND every WARN is documented-advisory, the
    # gate emits QA_PASS_WITH_ADVISORIES rather than QA_WARN per the
    # `docs/QUALITY_GATES.md` "Advisory Warning Classes" allowance.
    disallowed_warn_count = sum(
        1
        for issue in warn_issues
        if not _warn_is_documented_advisory(issue, chunks)
    )
    advisory_warn_count = warn_count - disallowed_warn_count
    if fail_count:
        final_status = "QA_FAIL"
    elif disallowed_warn_count:
        final_status = "QA_WARN"
    elif advisory_warn_count:
        final_status = "QA_PASS_WITH_ADVISORIES"
    else:
        final_status = "QA_PASS"

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

    if final_status in ("QA_PASS", "QA_PASS_WITH_ADVISORIES"):
        return 0
    if final_status == "QA_WARN" and args.allow_warnings:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
