#!/usr/bin/env python3
"""
Semantic-fidelity QA for ingestion JSONL outputs.

Structural gates (size/orphan/crash) are necessary but not sufficient.
This script checks whether extracted content is semantically useful for RAG:
- Image chunks are descriptive (not placeholders)
- Table chunks are structured (Markdown) and not placeholders
- Code chunks preserve multiline structure
- Detect simple cross-page heading->body anchor risks
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9/&()' .,-]{1,50}:?$")


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


def is_placeholder_image_or_table(content: str) -> bool:
    t = (content or "").strip().lower()
    if not t:
        return True
    if "extraction unavailable" in t:
        return True
    if re.match(r"^\[(figure|image|table)\b", t):
        return True
    # VLM failure sentinels — distinguishable from intentional no-VLM placeholders.
    if t.startswith("[vlm_failed"):
        return True
    # Extremely short "context only" strings should count as low-fidelity placeholders.
    if len(t) < 80 and ("figure on page" in t or "table on page" in t):
        return True
    return False


def is_markdown_table(content: str) -> bool:
    t = (content or "").strip()
    if not t:
        return False
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    if "|" not in lines[0]:
        return False
    # Separator line (---) is required for robust table parsing.
    return any(re.search(r"\|\s*-{2,}", ln) for ln in lines[1:3])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ingestion_jsonl", type=Path)
    parser.add_argument("--max-image-placeholder-ratio", type=float, default=0.20)
    parser.add_argument("--max-table-placeholder-ratio", type=float, default=0.20)
    parser.add_argument("--min-table-markdown-ratio", type=float, default=0.80)
    parser.add_argument("--max-code-flat-ratio", type=float, default=0.35)
    parser.add_argument("--min-code-indentation-fidelity", type=float, default=0.90)
    parser.add_argument("--max-cross-page-label-anchor-risk", type=float, default=0.10)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    with args.ingestion_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # Skip the document-level metadata record (first line in v2.6+ JSONLs).
    rows = [r for r in rows if r.get("object_type") != "ingestion_metadata"]

    images = [r for r in rows if r.get("modality") == "image"]
    tables = [r for r in rows if r.get("modality") == "table"]
    texts = [r for r in rows if r.get("modality") == "text"]

    image_placeholders = sum(
        1 for r in images if is_placeholder_image_or_table(r.get("content") or "")
    )
    table_placeholders = sum(
        1 for r in tables if is_placeholder_image_or_table(r.get("content") or "")
    )
    table_markdown = sum(1 for r in tables if is_markdown_table(r.get("content") or ""))

    code_chunks = []
    for r in texts:
        md = r.get("metadata") or {}
        if (
            str(md.get("chunk_type") or "").lower() == "code"
            or str(md.get("content_classification") or "").lower() == "code"
        ):
            code_chunks.append(r)
    code_flat = sum(1 for r in code_chunks if "\n" not in (r.get("content") or ""))
    code_with_indent_or_repl = 0
    for r in code_chunks:
        s = r.get("content") or ""
        lines = [ln for ln in s.splitlines() if ln.strip()]
        has_indented_line = any(ln.startswith(("    ", "\t")) for ln in lines)
        has_repl = ">>>" in s or re.search(r"(?m)^\s*\.\.\.\s", s) is not None
        if has_indented_line or has_repl:
            code_with_indent_or_repl += 1

    text_rows = []
    for r in texts:
        md = r.get("metadata") or {}
        text_rows.append((int(md.get("page_number") or 0), (r.get("content") or "").strip()))

    cross_page_anchor_risk = 0
    for i in range(len(text_rows) - 1):
        pg, s = text_rows[i]
        npg, ns = text_rows[i + 1]
        if npg == pg + 1 and is_label_like(s) and len(ns) >= 40 and not is_label_like(ns):
            cross_page_anchor_risk += 1

    image_placeholder_ratio = (image_placeholders / len(images)) if images else 0.0
    image_with_description = sum(
        1 for r in images
        if r.get("visual_description") or (r.get("metadata") or {}).get("visual_description")
    )
    image_description_coverage = (image_with_description / len(images)) if images else 1.0
    table_placeholder_ratio = (table_placeholders / len(tables)) if tables else 0.0
    table_markdown_ratio = (table_markdown / len(tables)) if tables else 1.0
    code_flat_ratio = (code_flat / len(code_chunks)) if code_chunks else 0.0
    code_indentation_fidelity = (
        code_with_indent_or_repl / len(code_chunks) if code_chunks else 1.0
    )
    # Normalize cross-page risk by number of text chunks to keep threshold stable.
    cross_page_anchor_risk_ratio = (
        cross_page_anchor_risk / len(text_rows) if text_rows else 0.0
    )

    print(
        f"images={len(images)} image_placeholder_ratio={image_placeholder_ratio:.4f} "
        f"image_description_coverage={image_description_coverage:.4f}"
    )
    print(
        f"tables={len(tables)} table_placeholder_ratio={table_placeholder_ratio:.4f} "
        f"table_markdown_ratio={table_markdown_ratio:.4f}"
    )
    print(f"code_chunks={len(code_chunks)} code_flat_ratio={code_flat_ratio:.4f}")
    print(f"code_indentation_fidelity={code_indentation_fidelity:.4f}")
    print(
        "cross_page_label_anchor_risk="
        f"{cross_page_anchor_risk} ratio={cross_page_anchor_risk_ratio:.4f}"
    )

    fails: List[str] = []
    if image_placeholder_ratio > args.max_image_placeholder_ratio:
        fails.append(
            f"image_placeholder_ratio={image_placeholder_ratio:.3f} "
            f"(>{args.max_image_placeholder_ratio:.2f})"
        )
    if images and image_description_coverage < 0.80:
        fails.append(
            f"image_description_coverage={image_description_coverage:.3f} (<0.80)"
        )
    if len(tables) > 0:
        if table_placeholder_ratio > args.max_table_placeholder_ratio:
            fails.append(
                f"table_placeholder_ratio={table_placeholder_ratio:.3f} "
                f"(>{args.max_table_placeholder_ratio:.2f})"
            )
        if table_markdown_ratio < args.min_table_markdown_ratio:
            fails.append(
                f"table_markdown_ratio={table_markdown_ratio:.3f} "
                f"(<{args.min_table_markdown_ratio:.2f})"
            )
    # Require at least 3 code chunks before penalising flat ratio — a single
    # flat snippet in a 20-page sample is a statistical artefact, not a bug.
    if len(code_chunks) >= 3 and code_flat_ratio > args.max_code_flat_ratio:
        fails.append(
            f"code_flat_ratio={code_flat_ratio:.3f} "
            f"(>{args.max_code_flat_ratio:.2f})"
        )
    if len(code_chunks) >= 3 and code_indentation_fidelity < args.min_code_indentation_fidelity:
        fails.append(
            f"code_indentation_fidelity={code_indentation_fidelity:.3f} "
            f"(<{args.min_code_indentation_fidelity:.2f})"
        )
    if cross_page_anchor_risk_ratio > args.max_cross_page_label_anchor_risk:
        fails.append(
            f"cross_page_label_anchor_risk_ratio={cross_page_anchor_risk_ratio:.3f} "
            f"(>{args.max_cross_page_label_anchor_risk:.2f})"
        )

    if fails:
        print("SEMANTIC_FAIL: " + "; ".join(fails))
    else:
        print("SEMANTIC_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
