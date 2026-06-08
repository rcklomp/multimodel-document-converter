"""Engine-agnostic Markdown-table normalization.

Shared by the MinerU adapter (HTML-table transcode) and the UIR chunker
(separator repair for either engine's pipe tables). A TABLE chunk must be a
well-formed Markdown grid with a ``|---|`` separator row to pass the
table-format gate; MinerU and Qwen both occasionally emit pipe-delimited rows
WITHOUT that separator (FluentPython p17). Normalizing at the chunker chokepoint
covers every engine, present and future.
"""

from __future__ import annotations

import re
from typing import List, Optional


def rows_to_markdown_grid(rows: List[List[str]]) -> Optional[str]:
    """Emit a rectangular Markdown grid from parsed rows (None if empty).

    First row becomes the header, followed by a ``| --- |`` separator and the
    body rows. Ragged rows are right-padded to the widest row so the grid is
    rectangular.
    """
    rows = [r for r in rows if r]
    if not rows:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(r) + " |" for r in rows[1:])
    return "\n".join(lines)


def _split_pipe_row(line: str) -> Optional[List[str]]:
    """Split a ``| a | b | c |`` line into trimmed cells (None if not piped).

    Splits on UNESCAPED pipes only so a literal cell pipe written ``\\|`` (regex,
    units, code) is not treated as a column boundary.
    """
    s = line.strip()
    if "|" not in s:
        return None
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", s)]


_SEP_CELL_RE = re.compile(r":?-{2,}:?$")


def _is_separator_cells(cells: List[str]) -> bool:
    """True if every cell is a Markdown separator token (``---`` / ``:--:``).

    Requires 2+ dashes per cell, matching the table-format gate's ``-{2,}``
    rule, so a single-dash DATA row (``| - | - |`` N/A markers) is not mistaken
    for a separator and a malformed single-dash separator still gets repaired.
    """
    return bool(cells) and all(_SEP_CELL_RE.fullmatch(c.strip()) for c in cells)


def normalize_pipe_table(content: str) -> Optional[str]:
    """Insert a missing header separator into a pipe-delimited table.

    Returns the normalized content, or None when the content is not a
    separator-less pipe table (so the caller keeps the original verbatim -
    content is never lost on a parse miss). Guards:
    - tolerates a leading title / trailing caption line that is not itself
      pipe-delimited (the pipe rows are a contiguous block);
    - treats the table as already-separated only when the SECOND grid row is a
      separator (a later all-dashes DATA row, e.g. ``| - | - |`` N/A markers, is
      not mistaken for one);
    - only reshapes a RECTANGULAR grid (all rows the same column count); a ragged
      split signals an unescaped pipe inside a cell, so it bails rather than
      shifting data into the wrong columns and shipping a corrupt-but-gated grid.
    """
    lines = [ln for ln in content.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    parsed = [_split_pipe_row(ln) for ln in lines]
    pipe_idx = [i for i, p in enumerate(parsed) if p is not None]
    if len(pipe_idx) < 2:
        return None
    lo, hi = pipe_idx[0], pipe_idx[-1]
    # A non-pipe line interspersed INSIDE the table block is ambiguous - bail.
    if any(parsed[i] is None for i in range(lo, hi + 1)):
        return None
    rows = [parsed[i] for i in range(lo, hi + 1)]
    if _is_separator_cells(rows[1]):  # canonical separator position only
        return None
    if len({len(r) for r in rows}) != 1:  # ragged => ambiguous pipe-in-cell
        return None
    grid = rows_to_markdown_grid(rows)
    if grid is None:
        return None
    lead = lines[:lo]
    tail = lines[hi + 1 :]
    return "\n".join(lead + grid.splitlines() + tail)


def ensure_table_separator(content: str) -> str:
    """Return table content with a Markdown separator row, else unchanged.

    The engine-agnostic chunker entry point: idempotent on already-valid grids
    and a no-op on non-pipe content (HTML, prose), so it is safe to call on
    every TABLE chunk regardless of which engine produced it.
    """
    if not content:
        return content
    fixed = normalize_pipe_table(content)
    return fixed if fixed is not None else content
