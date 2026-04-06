#!/usr/bin/env python3
"""
Evaluate technical-manual hygiene gates on an ingestion.jsonl file.

Prints metrics and one explicit `GATE_PASS`/`GATE_FAIL` line.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path


# Optional numeric prefix (e.g. "2.1 ", "10 ") allows numbered section headings
# like "2.1 Der horizontale Bruch" or "1 Introduction" to be recognised as labels
# rather than counted as micro-noise.
LABEL_RE = re.compile(r"^(?:\d[\d.]*\s+)?[A-Z][A-Za-z0-9/&()' .,-]{1,55}:?$")
CODE_LIKE_RE = re.compile(
    r"(::|\bdef\s+\w+\(|\bclass\s+\w+|\bimport\s+\w+|"
    r"\bfrom\s+\w+\s+import\b|[{}<>]=?|==|!=|:=|->|=>|\breturn\b)"
)
INFIX_RE = re.compile(
    r"(?<![\n\r])(?<!^)"
    r"\b(?P<prev>[a-z][a-z'\-]{0,24})\s+"
    r"(?P<num>(?:[1-9]|[12]\d|3\d|40))\.\s+"
    # Lowercase continuation only to avoid false positives on valid prose like
    # "chapter 3. Note" or section labels.
    r"(?P<next>[a-z][A-Za-z'\-]*)"
)


def _compact_counter(counter: Counter[str], limit: int = 4) -> str:
    if not counter:
        return "<none>"
    return ", ".join(f"{k}:{v}" for k, v in counter.most_common(limit))


def infer_doc_class(path: Path) -> tuple[str, str, Counter[str], Counter[str]]:
    digital_modalities = {"native_digital", "image_heavy"}
    modality_counts: Counter[str] = Counter()
    extraction_counts: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            # Skip the document-level metadata record (first line in v2.6+ JSONLs).
            if o.get("object_type") == "ingestion_metadata":
                continue
            md = o.get("metadata") or {}
            modality = str(md.get("document_modality") or "").strip().lower()
            extraction = str(md.get("extraction_method") or "").strip().lower()
            if modality:
                modality_counts[modality] += 1
            if extraction:
                extraction_counts[extraction] += 1

    if modality_counts:
        top_modality = modality_counts.most_common(1)[0][0]
        scanned_votes = sum(
            count
            for modality, count in modality_counts.items()
            if modality.startswith("scanned")
        )
        digital_votes = sum(
            count for modality, count in modality_counts.items() if modality in digital_modalities
        )

        if top_modality.startswith("scanned"):
            return "scanned", f"top_document_modality={top_modality}", modality_counts, extraction_counts
        if top_modality in digital_modalities:
            return "digital", f"top_document_modality={top_modality}", modality_counts, extraction_counts
        if scanned_votes > digital_votes:
            return "scanned", "scanned_modality_votes>digital_modality_votes", modality_counts, extraction_counts
        if digital_votes > 0:
            return "digital", "digital_modality_votes>=scanned_modality_votes", modality_counts, extraction_counts

    ocr_votes = sum(count for method, count in extraction_counts.items() if "ocr" in method)
    docling_votes = extraction_counts.get("docling", 0)
    if ocr_votes > docling_votes:
        return "scanned", "ocr_extraction_votes>docling_votes", modality_counts, extraction_counts
    return "digital", "fallback_default_digital", modality_counts, extraction_counts


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
        # Skip ordinal range patterns: "vom 6. bis 8. Oktober" (German date range),
        # "from 5. to 7." etc. are legitimate ordinal usage, not list artifacts.
        # Guard on prev (range-start prepositions) and next (range connectors).
        if prev in ("bis", "to", "from", "through", "vom", "von", "and", "or"):
            continue
        if nxt in ("bis", "to", "through"):
            continue
        n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ingestion_jsonl", type=Path)
    parser.add_argument(
        "--doc-class",
        choices=("digital", "scanned", "auto"),
        default="auto",
        help="Profile class for threshold selection; use 'auto' to infer from ingestion metadata",
    )
    args = parser.parse_args()

    path = args.ingestion_jsonl
    doc_class = args.doc_class

    # Read profile_type from ingestion_metadata (first JSONL line in v2.6+).
    # Used to apply profile-appropriate gate thresholds below.
    profile_type = "unknown"
    with path.open("r", encoding="utf-8") as _f:
        try:
            _first = json.loads(_f.readline())
            if _first.get("object_type") == "ingestion_metadata":
                profile_type = _first.get("profile_type") or "unknown"
        except Exception:
            pass
    print(f"profile_type={profile_type}")

    if doc_class == "auto":
        doc_class, reason, modality_counts, extraction_counts = infer_doc_class(path)
        print(f"doc_class={doc_class} (inferred)")
        print(f"doc_class_inference={reason}")
        print(f"document_modality_top={_compact_counter(modality_counts)}")
        print(f"extraction_method_top={_compact_counter(extraction_counts)}")
    else:
        print(f"doc_class={doc_class} (explicit)")

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
            # Skip the document-level metadata record (first line in v2.6+ JSONLs).
            if o.get("object_type") == "ingestion_metadata":
                continue
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

    # Profile-aware threshold overrides.
    # Academic whitepapers legitimately contain abbreviation/nomenclature tables
    # with many isolated short label-like entries (e.g. "EV  Electric vehicle"),
    # which inflate orphan_label_ratio well above the technical-manual baseline.
    # Digital magazines and technical reports often have more caption fragments
    # than a coding book, so micro_non_label is relaxed slightly.
    if doc_class == "digital":
        orphan_label_limit = 0.65 if profile_type == "academic_whitepaper" else 0.25
        micro_limit = 0.22 if profile_type in ("digital_magazine", "academic_whitepaper") else 0.12
    else:
        orphan_label_limit = 0.30
        micro_limit = 0.22

    fails = []
    if strict > 0:
        fails.append(f"infix_strict={strict} (expected 0)")

    if doc_class == "digital":
        if micro_ratio > micro_limit:
            fails.append(f"micro_non_label_ratio={micro_ratio:.3f} (>{micro_limit})")
        if oversize_ratio > 0.01:
            fails.append(f"oversize_ratio={oversize_ratio:.3f} (>0.01)")
        if orphan_ratio > orphan_label_limit:
            fails.append(f"orphan_label_ratio={orphan_ratio:.3f} (>{orphan_label_limit})")
        if code_frag_ratio > 0.05:
            fails.append(f"code_fragmentation_ratio={code_frag_ratio:.3f} (>0.05)")
    else:
        if micro_ratio > micro_limit:
            fails.append(f"micro_non_label_ratio={micro_ratio:.3f} (>{micro_limit})")
        if oversize_ratio > 0.02:
            fails.append(f"oversize_ratio={oversize_ratio:.3f} (>0.02)")
        if orphan_ratio > orphan_label_limit:
            fails.append(f"orphan_label_ratio={orphan_ratio:.3f} (>{orphan_label_limit})")

    if fails:
        print("GATE_FAIL: " + "; ".join(fails))
    else:
        print("GATE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
