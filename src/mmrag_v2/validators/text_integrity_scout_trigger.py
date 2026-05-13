"""Per-batch trigger for TextIntegrityScout (Phase 2 / PLAN_V2.10.md).

Doc-level token-balance variance is the historical trigger for the scout
(see `BatchProcessor._run_text_integrity_scout`). On large documents
where only a handful of pages have missing content, the per-doc
variance can land well inside the tolerance band while individual
batches still carry localized shortfalls (e.g. Fluent_Python: 770 pages,
6 missing pages, doc-variance ~0.2%, batch-variance up to 74%).

This module computes a per-batch shortfall classification using
universal geometric rules over emitted-vs-source page text. It does not
weaken any quality gate, and it returns a Boolean to be ORed with the
existing doc-level gate so the scout is allowed to run when localized
shortfall is detected.

Trigger (either rule, gated by a non-trivial source-text floor):

  (A) Per-batch variance:
        batch_source_chars >= MIN_SOURCE_CHARS
        and (batch_source_chars - batch_chunk_chars) / batch_source_chars
            >= VARIANCE_PCT

  (B) Per-batch missing-page count:
        batch_source_chars >= MIN_SOURCE_CHARS
        and number_of_pages_in_batch where
            per_page_source_chars >= MIN_PAGE_SOURCE_CHARS
            and per_page_chunk_chars <  MIN_PAGE_CHUNK_CHARS
          >= MIN_MISSING_PAGES

Thresholds are corpus-validated by
`scripts/probe_phase2_scout_threshold.py`; do not edit them without
running the probe and recording the diff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


# Defaults validated against the 32-doc canonical PDF corpus
# (probe_phase2_scout_threshold.py, 2026-05-12). See PLAN_V2.10.md
# §"Phase 2 — TEXT_INTEGRITY_SCOUT_FULL_DOC_SENSITIVITY".
VARIANCE_PCT: float = 0.30
MIN_SOURCE_CHARS: int = 500
MIN_MISSING_PAGES: int = 2
MIN_PAGE_SOURCE_CHARS: int = 100
MIN_PAGE_CHUNK_CHARS: int = 50


@dataclass(frozen=True)
class BatchShortfall:
    """Coverage shape for one batch's page range."""

    batch_index: int
    start_page: int
    end_page: int
    source_chars: int
    chunk_chars: int
    variance_ratio: float
    missing_pages: Tuple[int, ...]

    def fires(
        self,
        *,
        variance_pct: float = VARIANCE_PCT,
        min_source_chars: int = MIN_SOURCE_CHARS,
        min_missing_pages: int = MIN_MISSING_PAGES,
    ) -> bool:
        if self.source_chars < min_source_chars:
            return False
        if self.variance_ratio >= variance_pct:
            return True
        if len(self.missing_pages) >= min_missing_pages:
            return True
        return False


def classify_batches(
    batches: Iterable[Tuple[int, int, int]],
    source_chars_per_page: Dict[int, int],
    chunk_chars_per_page: Dict[int, int],
    *,
    min_page_source_chars: int = MIN_PAGE_SOURCE_CHARS,
    min_page_chunk_chars: int = MIN_PAGE_CHUNK_CHARS,
) -> List[BatchShortfall]:
    """Compute per-batch shortfall shape.

    Args:
        batches: iterable of (batch_index, start_page, end_page) triples.
            Page numbers are 1-indexed and inclusive.
        source_chars_per_page: 1-indexed page -> raw source text char count.
        chunk_chars_per_page: 1-indexed page -> total emitted TEXT chunk char
            count.

    Returns:
        One BatchShortfall per input batch (in input order).
    """
    out: List[BatchShortfall] = []
    for batch_index, start_page, end_page in batches:
        src = 0
        chk = 0
        missing: List[int] = []
        for p in range(start_page, end_page + 1):
            psrc = source_chars_per_page.get(p, 0)
            pchk = chunk_chars_per_page.get(p, 0)
            src += psrc
            chk += pchk
            if psrc >= min_page_source_chars and pchk < min_page_chunk_chars:
                missing.append(p)
        variance = ((src - chk) / src) if src > 0 else 0.0
        out.append(
            BatchShortfall(
                batch_index=batch_index,
                start_page=start_page,
                end_page=end_page,
                source_chars=src,
                chunk_chars=chk,
                variance_ratio=variance,
                missing_pages=tuple(missing),
            )
        )
    return out


def any_batch_fires(
    batches: Iterable[Tuple[int, int, int]],
    source_chars_per_page: Dict[int, int],
    chunk_chars_per_page: Dict[int, int],
    *,
    variance_pct: float = VARIANCE_PCT,
    min_source_chars: int = MIN_SOURCE_CHARS,
    min_missing_pages: int = MIN_MISSING_PAGES,
    min_page_source_chars: int = MIN_PAGE_SOURCE_CHARS,
    min_page_chunk_chars: int = MIN_PAGE_CHUNK_CHARS,
) -> Tuple[bool, List[BatchShortfall]]:
    """Return (fires, all_batch_shapes).

    `fires` is True iff at least one batch satisfies the trigger.
    `all_batch_shapes` contains every batch shape (firing or not) so
    callers can log telemetry.
    """
    shapes = classify_batches(
        batches=batches,
        source_chars_per_page=source_chars_per_page,
        chunk_chars_per_page=chunk_chars_per_page,
        min_page_source_chars=min_page_source_chars,
        min_page_chunk_chars=min_page_chunk_chars,
    )
    fired = any(
        s.fires(
            variance_pct=variance_pct,
            min_source_chars=min_source_chars,
            min_missing_pages=min_missing_pages,
        )
        for s in shapes
    )
    return fired, shapes
