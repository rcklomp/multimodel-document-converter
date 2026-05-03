#!/usr/bin/env python3
"""
Universal invariant checker for ingestion.jsonl.

These checks apply to every document type and profile — they are not
semantic judgements (those are profile-specific) but structural guarantees
that the output schema must always satisfy.

Exit codes:
  0 — UNIVERSAL_PASS
  1 — UNIVERSAL_FAIL
  2 — usage / file error
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


MODALITIES = ("text", "image", "table", "other")


@dataclass
class BBoxStats:
    total: int = 0
    with_bbox: int = 0
    missing_bbox: int = 0
    invalid_bbox: int = 0
    bbox_missing_dims: int = 0
    min_x: Optional[int] = None
    max_x: Optional[int] = None
    min_y: Optional[int] = None
    max_y: Optional[int] = None
    min_area: Optional[int] = None
    max_area: Optional[int] = None
    total_area: int = 0

    def record_missing(self) -> None:
        self.total += 1
        self.missing_bbox += 1

    def record_invalid(self) -> None:
        self.total += 1
        self.invalid_bbox += 1

    def record_valid(self, bbox: List[int], has_page_dims: bool) -> None:
        self.total += 1
        self.with_bbox += 1
        if not has_page_dims:
            self.bbox_missing_dims += 1

        x0, y0, x1, y1 = bbox
        area = (x1 - x0) * (y1 - y0)
        self.total_area += area
        self.min_x = x0 if self.min_x is None else min(self.min_x, x0)
        self.max_x = x1 if self.max_x is None else max(self.max_x, x1)
        self.min_y = y0 if self.min_y is None else min(self.min_y, y0)
        self.max_y = y1 if self.max_y is None else max(self.max_y, y1)
        self.min_area = area if self.min_area is None else min(self.min_area, area)
        self.max_area = area if self.max_area is None else max(self.max_area, area)

    @property
    def avg_area(self) -> Optional[float]:
        if self.with_bbox == 0:
            return None
        return self.total_area / self.with_bbox


@dataclass
class UniversalResult:
    # Document-level metadata (from ingestion_metadata record)
    doc_id: Optional[str] = None
    profile_type: Optional[str] = None
    domain: Optional[str] = None
    is_scan: Optional[bool] = None
    total_pages: Optional[int] = None
    chunk_count_declared: Optional[int] = None

    # Chunk counts
    total_chunks: int = 0
    text_chunks: int = 0
    image_chunks: int = 0
    table_chunks: int = 0

    # Invariant violations
    null_chunk_type_text: int = 0       # text chunk with chunk_type=null/missing
    invalid_bbox: int = 0               # bbox values outside [0, 1000]
    bbox_missing_dims: int = 0          # spatial_metadata present but no page_width/page_height
    empty_content_text: int = 0         # text chunk with empty/whitespace-only content
    missing_modality: int = 0           # chunk with no modality field

    # Examples for debugging
    examples: List[str] = field(default_factory=list)
    bbox_stats: Dict[str, BBoxStats] = field(
        default_factory=lambda: {modality: BBoxStats() for modality in MODALITIES}
    )

    def add_example(self, msg: str) -> None:
        if len(self.examples) < 5:
            self.examples.append(msg)


def check(path: Path) -> UniversalResult:
    r = UniversalResult()

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                r.add_example(f"line {lineno}: JSON parse error: {e}")
                continue

            # ── ingestion_metadata record ────────────────────────────────
            if obj.get("object_type") == "ingestion_metadata":
                r.doc_id = obj.get("doc_id")
                r.profile_type = obj.get("profile_type")
                r.domain = obj.get("domain")
                r.is_scan = obj.get("is_scan")
                r.total_pages = obj.get("total_pages")
                r.chunk_count_declared = obj.get("chunk_count")
                continue

            # ── chunk records ────────────────────────────────────────────
            r.total_chunks += 1
            modality = obj.get("modality")
            if not modality:
                r.missing_modality += 1
                r.add_example(f"line {lineno}: missing modality, chunk_id={obj.get('chunk_id')}")
                continue

            if modality == "text":
                r.text_chunks += 1
            elif modality == "image":
                r.image_chunks += 1
            elif modality == "table":
                r.table_chunks += 1

            meta = obj.get("metadata") or {}
            chunk_id = obj.get("chunk_id", f"line_{lineno}")

            # ── Invariant 1: text chunks must have chunk_type ────────────
            if modality == "text":
                ct = meta.get("chunk_type")
                if not ct:
                    r.null_chunk_type_text += 1
                    r.add_example(
                        f"chunk_id={chunk_id} page={meta.get('page_number')}: "
                        f"text chunk missing chunk_type"
                    )

            # ── Invariant 2: text chunks must not be empty ───────────────
            if modality == "text":
                content = obj.get("content") or ""
                if not content.strip():
                    r.empty_content_text += 1
                    r.add_example(f"chunk_id={chunk_id}: empty text chunk content")

            # ── Invariant 3: bbox values must be integers in [0, 1000] ───
            spatial = _extract_spatial_metadata(meta)
            bbox = spatial.get("bbox")
            bbox_modality = modality if modality in r.bbox_stats else "other"
            bbox_valid = _is_valid_bbox(bbox)
            bbox_present = bbox is not None

            if bbox_present and not bbox_valid:
                r.invalid_bbox += 1
                r.bbox_stats[bbox_modality].record_invalid()
                r.add_example(
                    f"chunk_id={chunk_id}: invalid bbox {bbox!r} "
                    f"(expected 4 ints in [0,1000] with x1>x0 and y1>y0)"
                )
            elif bbox_valid:
                pw = spatial.get("page_width")
                ph = spatial.get("page_height")
                has_page_dims = pw is not None and ph is not None
                r.bbox_stats[bbox_modality].record_valid(bbox, has_page_dims)
            else:
                r.bbox_stats[bbox_modality].record_missing()

            # ── Invariant 4: spatial_metadata with bbox needs page dims ──
            if bbox_valid:
                pw = spatial.get("page_width")
                ph = spatial.get("page_height")
                if pw is None or ph is None:
                    r.bbox_missing_dims += 1
                    r.add_example(
                        f"chunk_id={chunk_id}: spatial_metadata has bbox but "
                        f"page_width={pw} page_height={ph}"
                    )

    return r


def _extract_spatial_metadata(meta: dict) -> dict:
    """Read current and legacy spatial metadata keys from emitted JSONL."""
    spatial = meta.get("spatial")
    if isinstance(spatial, dict):
        return spatial
    spatial_metadata = meta.get("spatial_metadata")
    if isinstance(spatial_metadata, dict):
        return spatial_metadata
    return {}


def _is_valid_bbox(bbox: object) -> bool:
    """Validate REQ-COORD-01/02 shape, type, range, and geometry."""
    if not isinstance(bbox, list):
        return False
    if len(bbox) != 4:
        return False
    if not all(type(val) is int and 0 <= val <= 1000 for val in bbox):
        return False
    x0, y0, x1, y1 = bbox
    return x1 > x0 and y1 > y0


def _format_optional_int(value: Optional[int]) -> str:
    return "n/a" if value is None else str(value)


def _format_optional_float(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _format_bbox_stats(modality: str, stats: BBoxStats) -> str:
    return (
        f"bbox_stats[{modality}]: total={stats.total} "
        f"with_bbox={stats.with_bbox} missing_bbox={stats.missing_bbox} "
        f"invalid_bbox={stats.invalid_bbox} missing_page_dims={stats.bbox_missing_dims} "
        f"x_range={_format_optional_int(stats.min_x)}..{_format_optional_int(stats.max_x)} "
        f"y_range={_format_optional_int(stats.min_y)}..{_format_optional_int(stats.max_y)} "
        f"area_min={_format_optional_int(stats.min_area)} "
        f"area_avg={_format_optional_float(stats.avg_area)} "
        f"area_max={_format_optional_int(stats.max_area)}"
    )


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: qa_universal_invariants.py path/to/ingestion.jsonl", file=sys.stderr)
        return 2

    path = Path(argv[1])
    if not path.exists():
        print(f"error: not found: {path}", file=sys.stderr)
        return 2

    r = check(path)

    print(f"file: {path}")
    if r.profile_type:
        print(
            f"profile={r.profile_type}  domain={r.domain}  "
            f"is_scan={r.is_scan}  total_pages={r.total_pages}  "
            f"declared_chunks={r.chunk_count_declared}"
        )
    print(
        f"chunks_total={r.total_chunks}  "
        f"text={r.text_chunks}  image={r.image_chunks}  table={r.table_chunks}"
    )
    print(f"null_chunk_type_on_text={r.null_chunk_type_text}")
    print(f"invalid_bbox={r.invalid_bbox}")
    print(f"bbox_missing_page_dims={r.bbox_missing_dims}")
    print(f"empty_text_chunks={r.empty_content_text}")
    print(f"missing_modality={r.missing_modality}")
    for modality in MODALITIES:
        print(_format_bbox_stats(modality, r.bbox_stats[modality]))

    fails = []
    if r.null_chunk_type_text > 0:
        fails.append(f"null_chunk_type_on_text={r.null_chunk_type_text}")
    if r.invalid_bbox > 0:
        fails.append(f"invalid_bbox={r.invalid_bbox}")
    if r.empty_content_text > 0:
        fails.append(f"empty_text_chunks={r.empty_content_text}")
    if r.missing_modality > 0:
        fails.append(f"missing_modality={r.missing_modality}")

    # bbox_missing_dims is a warning, not a hard fail — some OCR paths
    # don't produce spatial metadata at all, which is acceptable.
    if r.bbox_missing_dims > 0:
        print(f"WARN: bbox_missing_page_dims={r.bbox_missing_dims} (advisory, not a hard fail)")

    if r.examples:
        for ex in r.examples:
            print(f"  example: {ex}")

    if fails:
        print("UNIVERSAL_FAIL: " + "; ".join(fails))
        return 1

    print("UNIVERSAL_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
