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
from typing import List, Optional


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
            spatial = meta.get("spatial_metadata") or {}
            bbox = spatial.get("bbox")
            if bbox and isinstance(bbox, list):
                for val in bbox:
                    if not isinstance(val, int) or val < 0 or val > 1000:
                        r.invalid_bbox += 1
                        r.add_example(
                            f"chunk_id={chunk_id}: invalid bbox value {val!r} "
                            f"(expected int in [0,1000])"
                        )
                        break

            # ── Invariant 4: spatial_metadata with bbox needs page dims ──
            if bbox:
                pw = spatial.get("page_width")
                ph = spatial.get("page_height")
                if pw is None or ph is None:
                    r.bbox_missing_dims += 1
                    r.add_example(
                        f"chunk_id={chunk_id}: spatial_metadata has bbox but "
                        f"page_width={pw} page_height={ph}"
                    )

    return r


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
