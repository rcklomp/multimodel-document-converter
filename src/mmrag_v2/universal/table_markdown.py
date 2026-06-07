"""Engine-agnostic Markdown-table normalization.

Shared by the MinerU adapter (HTML-table transcode) and the UIR chunker
(separator repair for either engine's pipe tables). A TABLE chunk must be a
well-formed Markdown grid with a ``|---|`` separator row to pass the
table-format gate; MinerU and Qwen both occasionally emit pipe-delimited rows
WITHOUT that separator (FluentPython p17). Normalizing at the chunker chokepoint
covers every engine, present and future.
"""

from __future__ import annotations

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
    """Split a ``| a | b | c |`` line into trimmed cells (None if not piped)."""
    s = line.strip()
    if "|" not in s:
        return None
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_separator_cells(cells: List[str]) -> bool:
    """True if every cell is a Markdown separator token (``---`` / ``:--:``)."""
    return bool(cells) and all(c and set(c) <= {"-", ":"} and "-" in c for c in cells)


def normalize_pipe_table(content: str) -> Optional[str]:
    """Insert a missing header separator into a pipe-delimited table.

    Returns the normalized grid, or None when the content is not a
    separator-less pipe table (already has a separator, fewer than two rows, or
    a line is not pipe-delimited) so the caller keeps the original verbatim -
    content is never lost on a parse miss.
    """
    lines = [ln for ln in content.splitlines() if ln.strip()]
    parsed = [_split_pipe_row(ln) for ln in lines]
    if len(parsed) < 2 or any(p is None for p in parsed):
        return None
    if any(_is_separator_cells(cells) for cells in parsed if cells is not None):
        return None
    return rows_to_markdown_grid([p for p in parsed if p is not None])


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
