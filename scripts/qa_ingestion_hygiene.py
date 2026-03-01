#!/usr/bin/env python3
"""
Quick hygiene scanner for ingestion.jsonl.

Goal: provide fast, repeatable feedback without rerunning multi-hour conversions.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _has_ctrl(s: str) -> bool:
    return any((ord(c) < 32 and c not in ("\n", "\t")) or ord(c) == 127 for c in s)


def _ctrl_count(s: str) -> int:
    return sum(1 for c in s if ((ord(c) < 32 and c not in ("\n", "\t")) or ord(c) == 127))


@dataclass
class Stats:
    total: int = 0
    text: int = 0
    image: int = 0
    table: int = 0

    short_lt_30: int = 0
    long_gt_1500: int = 0
    max_len: int = 0
    sum_len: int = 0

    ctrl_chunks: int = 0
    ctrl_total: int = 0
    page_num_art_chunks: int = 0

    code_chunks: int = 0
    code_newlines_zero: int = 0
    code_like_not_code: int = 0
    code_like_total: int = 0
    code_fragment_micro: int = 0

    micro_non_label_chunks: int = 0
    oversize_chunks: int = 0
    label_chunks: int = 0
    orphan_label_chunks: int = 0

    chunk_type_counts: Counter = field(default_factory=Counter)


_PAGE_NUM_LINE = re.compile(r"(?m)^\s*\d{1,4}\s*$")
_CODE_SIG = re.compile(
    r"(?m)^\s*(def|class|import|from|return|yield|if\s+__name__|async\s+def)\b"
)
_LABEL_LIKE = re.compile(r"^[A-Z][A-Za-z0-9/&()' .,-]{1,50}:?$")
_CODE_INLINE = re.compile(
    r"(```|::|\bdef\s+\w+\(|\bclass\s+\w+|"
    r"\bimport\s+\w+|\bfrom\s+\w+\s+import\b|"
    r"[{}<>]=?|==|!=|:=|->|=>|\breturn\b)"
)


def _is_code_chunk(meta: dict) -> bool:
    ct = str((meta or {}).get("chunk_type") or "").lower()
    cc = str((meta or {}).get("content_classification") or "").lower()
    return ct == "code" or cc == "code"


def _is_label_like(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    if len(s) > 60:
        return False
    if "\n" in s:
        return False
    if _PAGE_NUM_LINE.fullmatch(s):
        return False
    if not _LABEL_LIKE.match(s):
        return False

    # Field-value lines like "Origin: United States" are complete records,
    # not orphan labels that require a following body chunk.
    if ":" in s and not s.endswith(":"):
        return False

    # Non-colon headers should be short title-like labels, not prose.
    if ":" not in s:
        words = [w for w in re.split(r"\s+", s) if w]
        if len(words) > 6:
            return False
        if any(w.endswith((".", "?", "!")) for w in words):
            return False
    return True


def _looks_code_like(text: str) -> bool:
    s = text or ""
    if not s.strip():
        return False
    if _CODE_SIG.search(s):
        return True
    return bool(_CODE_INLINE.search(s))


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def scan(path: Path) -> Tuple[Stats, Dict[str, Optional[str]]]:
    st = Stats()
    first_ctrl: Optional[str] = None
    first_page_num: Optional[str] = None
    first_flat_code: Optional[str] = None

    text_rows: List[Tuple[str, int, str, dict]] = []

    for o in iter_jsonl(path):
        # Skip the document-level metadata record (first line in v2.6+ JSONLs).
        if o.get("object_type") == "ingestion_metadata":
            continue
        st.total += 1
        modality = o.get("modality")
        if modality == "text":
            st.text += 1
        elif modality == "image":
            st.image += 1
        elif modality == "table":
            st.table += 1

        meta = o.get("metadata") or {}
        ct = meta.get("chunk_type")
        if ct:
            st.chunk_type_counts[ct] += 1

        if modality != "text":
            continue

        c = o.get("content") or ""
        chunk_id = str(o.get("chunk_id") or "")
        page_number = int(meta.get("page_number") or 0)
        text_rows.append((chunk_id, page_number, c, meta))
        L = len(c)
        st.sum_len += L
        st.max_len = max(st.max_len, L)
        if L < 30:
            st.short_lt_30 += 1
        if L > 1500:
            st.long_gt_1500 += 1
            st.oversize_chunks += 1

        label_like = _is_label_like(c)
        if label_like:
            st.label_chunks += 1
        if L < 30 and (not label_like) and (not _is_code_chunk(meta)):
            st.micro_non_label_chunks += 1

        if _has_ctrl(c):
            st.ctrl_chunks += 1
            st.ctrl_total += _ctrl_count(c)
            if first_ctrl is None:
                first_ctrl = f"{o.get('chunk_id')} page={meta.get('page_number')} sample={repr(c[:120])}"

        if _PAGE_NUM_LINE.search(c):
            st.page_num_art_chunks += 1
            if first_page_num is None:
                first_page_num = f"{o.get('chunk_id')} page={meta.get('page_number')} sample={repr(c[:160])}"

        code_like = _looks_code_like(c)
        if code_like:
            st.code_like_total += 1

        if ct == "code":
            st.code_chunks += 1
            if "\n" not in c:
                st.code_newlines_zero += 1
                if first_flat_code is None:
                    first_flat_code = (
                        f"{o.get('chunk_id')} page={meta.get('page_number')} sample={repr(c[:160])}"
                    )
        else:
            if code_like:
                st.code_like_not_code += 1
                if len(c.strip()) < 80:
                    st.code_fragment_micro += 1

    # Label attachment quality: label chunks should be followed by body-like text
    # within a local window (allowing a single page break).
    for i, (_cid, page_no, txt, _meta) in enumerate(text_rows):
        if not _is_label_like(txt):
            continue
        attached = False
        for j in range(i + 1, min(len(text_rows), i + 5)):
            if j >= len(text_rows):
                break
            _nid, npg, ntxt, _nmeta = text_rows[j]
            if npg not in (page_no, page_no + 1):
                continue
            ns = (ntxt or "").strip()
            if len(ns) >= 20 and not _is_label_like(ns):
                attached = True
                break
        if not attached:
            st.orphan_label_chunks += 1

    examples = {
        "first_ctrl": first_ctrl,
        "first_page_num_artifact": first_page_num,
        "first_flat_code": first_flat_code,
    }
    return st, examples


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: scripts/qa_ingestion_hygiene.py path/to/ingestion.jsonl", file=sys.stderr)
        return 2

    path = Path(argv[1])
    if not path.exists():
        print(f"error: not found: {path}", file=sys.stderr)
        return 2

    st, ex = scan(path)
    mean_len = (st.sum_len / st.text) if st.text else 0.0
    print(f"file: {path}")
    print(f"chunks_total: {st.total}  text: {st.text}  image: {st.image}  table: {st.table}")
    print(f"text_mean_len: {mean_len:.1f}  text_max_len: {st.max_len}")
    print(f"text_short_<30: {st.short_lt_30}  text_long_>1500: {st.long_gt_1500}")
    micro_ratio = (st.micro_non_label_chunks / st.text) if st.text else 0.0
    oversize_ratio = (st.oversize_chunks / st.text) if st.text else 0.0
    orphan_ratio = (st.orphan_label_chunks / st.label_chunks) if st.label_chunks else 0.0
    code_frag_ratio = (st.code_fragment_micro / st.code_like_total) if st.code_like_total else 0.0
    print(
        f"micro_non_label_chunks: {st.micro_non_label_chunks}  micro_non_label_ratio: {micro_ratio:.3f}"
    )
    print(f"oversize_chunks: {st.oversize_chunks}  oversize_ratio: {oversize_ratio:.3f}")
    print(f"label_chunks: {st.label_chunks}  orphan_label_chunks: {st.orphan_label_chunks}  orphan_label_ratio: {orphan_ratio:.3f}")
    print(f"ctrl_chunks: {st.ctrl_chunks}  ctrl_total: {st.ctrl_total}")
    print(f"page_num_artifact_chunks: {st.page_num_art_chunks}")
    print(
        f"code_chunks: {st.code_chunks}  code_flat(no_newlines): {st.code_newlines_zero}  "
        f"code_like_not_code: {st.code_like_not_code}"
    )
    print(
        f"code_like_total: {st.code_like_total}  code_fragment_micro: {st.code_fragment_micro}  "
        f"code_fragmentation_ratio: {code_frag_ratio:.3f}"
    )

    if st.chunk_type_counts:
        top = ", ".join(f"{k}={v}" for k, v in st.chunk_type_counts.most_common(8))
        print(f"chunk_type_top: {top}")

    for k, v in ex.items():
        if v:
            print(f"{k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
