#!/usr/bin/env python3
"""
Fix OCR-induced infix list numbering artifacts in JSONL text chunks.

Example artifact:
  "... retained by a 16. vertical screw ..."
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Tuple


INFIX_PATTERN = re.compile(
    r"(?<![\n\r])(?<!^)"
    r"(?P<prev>\b[a-z][a-z'\-]{0,24})\s+"
    r"(?P<num>(?:[1-9]|[12]\d|3\d|40))\.\s+"
    r"(?P<next>[A-Za-z][A-Za-z'\-]*)"
)


def _fix_text(text: str) -> Tuple[str, int]:
    """Return (fixed_text, replacements_count)."""
    replacements = 0

    def _repl(match: "re.Match[str]") -> str:
        nonlocal replacements
        prev = match.group("prev")
        nxt = match.group("next")
        start = match.start()
        left = text[max(0, start - 2):start]
        # Keep obvious sentence/list starts; fix only mid-sentence artifacts.
        if left.endswith(("\n", "\r", ". ", ": ", "; ", "! ", "? ")):
            return match.group(0)
        if len(prev) <= 1 or len(nxt) <= 1:
            return match.group(0)
        replacements += 1
        return f"{prev} {nxt}"

    return INFIX_PATTERN.sub(_repl, text), replacements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path, help="Path to input ingestion.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output JSONL (default: <input>.fixed.jsonl)",
    )
    args = parser.parse_args()

    in_path = args.input_jsonl
    out_path = args.output or in_path.with_name(in_path.stem + ".fixed.jsonl")

    total_rows = 0
    text_rows = 0
    changed_rows = 0
    replacement_total = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            total_rows += 1
            row = json.loads(line)
            if row.get("modality") == "text":
                text_rows += 1
                content = row.get("content") or ""
                fixed, replaced = _fix_text(content)
                if replaced > 0:
                    row["content"] = fixed
                    changed_rows += 1
                    replacement_total += replaced
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"input={in_path}")
    print(f"output={out_path}")
    print(f"total_rows={total_rows}")
    print(f"text_rows={text_rows}")
    print(f"changed_rows={changed_rows}")
    print(f"replacements={replacement_total}")


if __name__ == "__main__":
    main()
