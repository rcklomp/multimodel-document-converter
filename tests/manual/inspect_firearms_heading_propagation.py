#!/usr/bin/env python3
"""Inspect Firearms heading propagation in the emitted ingestion JSONL.

Manual diagnostic for PLAN_V2.10 Phase 6 (OCR_PATH_HEADING_PROPAGATION).
Modelled on the Phase 5 Devlin probe: walks text chunks in (page, position)
order, prints per-row attribution, summarises null-parent ratios per batch,
and reports the top-N parent_heading distribution. Critically it also runs
`is_valid_heading` against the top attributions so an OCR-lane fix that
re-introduces the v1 garbage failure mode is visible immediately.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_JSONL = Path("output/Firearms/ingestion.jsonl")
DEFAULT_SOURCE_PDF = Path("data/technical_manual/Firearms.pdf")


class LinePrinter:
    def __init__(self) -> None:
        self.line_no = 0

    def emit(self, text: str = "") -> None:
        self.line_no += 1
        print(f"{self.line_no:04d}: {text}")


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as fh:
        for row_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            yield row_no, json.loads(line)


def _is_text_chunk(obj: dict[str, Any]) -> bool:
    return obj.get("modality") == "text"


def _metadata_batch_size(metadata: dict[str, Any], default: int) -> int:
    for key in ("batch_size", "pages_per_batch"):
        value = metadata.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return default


def _first_line(text: str, max_chars: int = 80) -> str:
    one_line = " ".join((text or "").split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max_chars - 1] + "..."


def _parent_heading(chunk: dict[str, Any]) -> Any:
    return (
        chunk.get("metadata", {})
        .get("hierarchy", {})
        .get("parent_heading")
    )


def _chunk_type(chunk: dict[str, Any]) -> str:
    return (
        chunk.get("metadata", {}).get("chunk_type")
        or chunk.get("chunk_type")
        or ""
    )


def _extraction_method(chunk: dict[str, Any]) -> str:
    return chunk.get("metadata", {}).get("extraction_method") or ""


def _page_number(chunk: dict[str, Any]) -> int:
    page = chunk.get("metadata", {}).get("page_number")
    return int(page or 0)


def _position(chunk: dict[str, Any], row_no: int, ordinal: int) -> int:
    value = chunk.get("position") or chunk.get("metadata", {}).get("position")
    if isinstance(value, int):
        return value
    return ordinal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--source-pdf", type=Path, default=DEFAULT_SOURCE_PDF)
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    from mmrag_v2.state.context_state import is_valid_heading

    printer = LinePrinter()
    if not args.jsonl.exists():
        raise SystemExit(f"JSONL not found: {args.jsonl}")

    metadata: dict[str, Any] = {}
    text_rows: list[tuple[int, int, dict[str, Any]]] = []
    for row_no, obj in _iter_jsonl(args.jsonl):
        if obj.get("object_type") == "ingestion_metadata":
            metadata = obj
            continue
        if _is_text_chunk(obj):
            text_rows.append((row_no, len(text_rows), obj))

    batch_size = _metadata_batch_size(metadata, args.batch_size)
    total_pages = int(metadata.get("total_pages") or 0)
    chunk_count = int(metadata.get("chunk_count") or 0)

    printer.emit(f"jsonl={args.jsonl}")
    printer.emit(
        f"source_file={metadata.get('source_file', '<unknown>')} "
        f"profile={metadata.get('profile_type', '<unknown>')} "
        f"total_pages={total_pages} batch_size={batch_size} "
        f"metadata_chunk_count={chunk_count} text_chunks={len(text_rows)}"
    )
    printer.emit(
        "columns: page position jsonl_row parent_heading chunk_type "
        "extraction_method first_80_chars"
    )

    none_by_batch: Counter[int] = Counter()
    total_by_batch: Counter[int] = Counter()
    none_rows: list[tuple[int, int, int, str, str, str]] = []
    heading_counts: Counter[str] = Counter()
    last_batch: int | None = None

    for row_no, ordinal, chunk in sorted(
        text_rows,
        key=lambda item: (_page_number(item[2]), _position(item[2], item[0], item[1])),
    ):
        page = _page_number(chunk)
        pos = _position(chunk, row_no, ordinal)
        batch_index = ((page - 1) // batch_size) + 1 if page > 0 else 0
        batch_start = ((batch_index - 1) * batch_size) + 1 if batch_index > 0 else 0
        batch_end = batch_start + batch_size - 1 if batch_start else 0

        if last_batch is None or batch_index != last_batch:
            printer.emit("")
            printer.emit(
                f"========== BATCH {batch_index} pages {batch_start}-{batch_end} =========="
            )
            last_batch = batch_index

        parent = _parent_heading(chunk)
        ctype = _chunk_type(chunk)
        method = _extraction_method(chunk)
        content = _first_line(chunk.get("content") or "")
        total_by_batch[batch_index] += 1

        marker = "!!NONE!!" if parent is None else "        "
        if parent is None:
            none_by_batch[batch_index] += 1
            none_rows.append((page, pos, row_no, ctype, method, content))
        else:
            heading_counts[str(parent)] += 1

        printer.emit(
            f"{marker} page={page:03d} position={pos:04d} row={row_no:05d} "
            f"parent={parent!r} chunk_type={ctype!r} "
            f"method={method!r} text={content!r}"
        )

    printer.emit("")
    printer.emit("========== NONE PARENT SUMMARY BY BATCH ==========")
    for batch_index in sorted(total_by_batch):
        none = none_by_batch[batch_index]
        total = total_by_batch[batch_index]
        printer.emit(
            f"batch={batch_index:02d} none_parent={none:03d}/{total:03d} "
            f"ratio={none / total if total else 0:.3f}"
        )

    printer.emit("")
    printer.emit("========== FIRST 40 NONE-PARENT ROWS ==========")
    for page, pos, row_no, ctype, method, content in none_rows[:40]:
        printer.emit(
            f"none_parent page={page:03d} position={pos:04d} row={row_no:05d} "
            f"chunk_type={ctype!r} method={method!r} text={content!r}"
        )
    printer.emit(f"total_none_parent_rows={len(none_rows)}")

    printer.emit("")
    printer.emit(f"========== TOP {args.top_n} PARENT HEADING DISTRIBUTION ==========")
    total_attributed = sum(heading_counts.values())
    null_count = sum(none_by_batch.values())
    printer.emit(
        f"text_chunks={len(text_rows)} attributed={total_attributed} "
        f"null={null_count} null_ratio={null_count / max(len(text_rows), 1):.3f}"
    )
    for heading, count in heading_counts.most_common(args.top_n):
        valid = is_valid_heading(heading)
        marker = "OK   " if valid else "FAIL "
        printer.emit(
            f"{marker} count={count:03d} valid={valid!s:>5} parent={heading!r}"
        )

    printer.emit("")
    printer.emit("========== TOP-N HUMAN-READABILITY CHECK (Phase 6 quality gate) ==========")
    printer.emit("Top 5 attributed headings must all be human-readable chapter/section titles.")
    top5 = heading_counts.most_common(5)
    failures: list[str] = []
    for heading, count in top5:
        if not is_valid_heading(heading):
            failures.append(heading)
            printer.emit(f"FAIL: '{heading}' (count={count}) rejected by is_valid_heading")
    if not failures:
        printer.emit(f"PASS: all top-{len(top5)} headings pass is_valid_heading")
    else:
        printer.emit(f"FAIL_COUNT={len(failures)} of {len(top5)}")

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
