#!/usr/bin/env python3
"""
Evaluate technical-manual hygiene gates on an ingestion.jsonl file.

Prints metrics and one explicit `GATE_PASS`/`GATE_FAIL` line.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9/&()' .,-]{1,50}:?$")
CODE_LIKE_RE = re.compile(
    r"(::|\bdef\s+\w+\(|\bclass\s+\w+|\bimport\s+\w+|"
    r"\bfrom\s+\w+\s+import\b|[{}<>]=?|==|!=|:=|->|=>|\breturn\b)"
)
INFIX_RE = re.compile(
    r"(?<![\n\r])(?<!^)"
    r"\b(?P<prev>[a-z][a-z'\-]{0,24})\s+"
    r"(?P<num>(?:[1-9]|[12]\d|3\d|40))\.\s+"
    r"(?P<next>[A-Za-z][A-Za-z'\-]*)"
)


def is_label_like(s: str) -> bool:
    s = (s or "").strip()
    if not s or len(s) > 60 or "\n" in s:
        return False
    if not LABEL_RE.match(s):
        return False
    if ":" in s and not s.endswith(":"):
        return False
    if ":" not in s:
        words = [w for w in re.split(r"\s+", s) if w]
        if len(words) > 6:
            return False
        if any(w.endswith((".", "?", "!")) for w in words):
            return False
    return True


def count_infix_artifacts(text: str) -> int:
    n = 0
    for m in INFIX_RE.finditer(text or ""):
        # Regex-only guardrails: keep it structural (no hardcoded lexical allowlists).
        prev = m.group("prev")
        nxt = m.group("next")
        start = m.start()
        left = (text or "")[max(0, start - 2):start]
        # Skip list starts / sentence starts where numbering is often legitimate.
        if left.endswith(("\n", "\r", ". ", ": ", "; ", "! ", "? ")):
            continue
        # Mid-sentence artifacts usually bridge normal words.
        if len(prev) <= 1 or len(nxt) <= 1:
            continue
        n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ingestion_jsonl", type=Path)
    parser.add_argument(
        "--doc-class",
        choices=("digital", "scanned"),
        required=True,
        help="Profile class for threshold selection",
    )
    args = parser.parse_args()

    path = args.ingestion_jsonl
    doc_class = args.doc_class

    strict = 0
    ref_bad = 0
    text = 0
    micro = 0
    oversize = 0
    labels = 0
    orphan = 0
    code_like_total = 0
    code_frag = 0
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("modality") != "text":
                continue
            text += 1
            md = o.get("metadata") or {}
            s = (o.get("content") or "").strip()
            rows.append((int(md.get("page_number") or 0), s, md))
            strict += count_infix_artifacts(s)

            rc = md.get("refined_content")
            if isinstance(rc, str) and rc:
                ref_bad += count_infix_artifacts(rc)

            label = is_label_like(s)
            if label:
                labels += 1

            is_code_chunk = (
                str(md.get("chunk_type") or "").lower() == "code"
                or str(md.get("content_classification") or "").lower() == "code"
            )
            is_code_like = bool(CODE_LIKE_RE.search(s))
            if is_code_like:
                code_like_total += 1

            if len(s) < 30 and (not label) and (not is_code_chunk):
                micro += 1
            if len(s) > 1500:
                oversize += 1
            if is_code_like and (not is_code_chunk) and len(s) < 80:
                code_frag += 1

    for i, (pg, s, _md) in enumerate(rows):
        if not is_label_like(s):
            continue
        attached = False
        for j in range(i + 1, min(len(rows), i + 5)):
            if j >= len(rows):
                break
            npg, ns, _nmd = rows[j]
            if npg not in (pg, pg + 1):
                continue
            if len(ns) >= 20 and not is_label_like(ns):
                attached = True
                break
        if not attached:
            orphan += 1

    micro_ratio = (micro / text) if text else 0.0
    oversize_ratio = (oversize / text) if text else 0.0
    orphan_ratio = (orphan / labels) if labels else 0.0
    code_frag_ratio = (code_frag / code_like_total) if code_like_total else 0.0

    print(f"infix_strict={strict}")
    print(f"refined_content_infix={ref_bad}")
    print(f"text_chunks={text}")
    print(f"micro_non_label_ratio={micro_ratio:.4f}")
    print(f"oversize_ratio={oversize_ratio:.4f}")
    print(f"orphan_label_ratio={orphan_ratio:.4f}")
    print(f"code_fragmentation_ratio={code_frag_ratio:.4f}")

    fails = []
    if strict > 0:
        fails.append(f"infix_strict={strict} (expected 0)")

    if doc_class == "digital":
        if micro_ratio > 0.12:
            fails.append(f"micro_non_label_ratio={micro_ratio:.3f} (>0.12)")
        if oversize_ratio > 0.01:
            fails.append(f"oversize_ratio={oversize_ratio:.3f} (>0.01)")
        if orphan_ratio > 0.20:
            fails.append(f"orphan_label_ratio={orphan_ratio:.3f} (>0.20)")
        if code_frag_ratio > 0.05:
            fails.append(f"code_fragmentation_ratio={code_frag_ratio:.3f} (>0.05)")
    else:
        if micro_ratio > 0.22:
            fails.append(f"micro_non_label_ratio={micro_ratio:.3f} (>0.22)")
        if oversize_ratio > 0.02:
            fails.append(f"oversize_ratio={oversize_ratio:.3f} (>0.02)")
        if orphan_ratio > 0.30:
            fails.append(f"orphan_label_ratio={orphan_ratio:.3f} (>0.30)")

    if fails:
        print("GATE_FAIL: " + "; ".join(fails))
    else:
        print("GATE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
