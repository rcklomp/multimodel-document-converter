"""50-chunk human-labeled golden set for the LLM dominance criterion.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.3 dominance criterion #5
(NEW in 0.4). Mitigates R19 sanitizer-judge correlated-failure-mode.

Foundation-session status: SCHEMA + LOADER ONLY (no labels yet).

The golden set is built once during Phase B B2 and never modified
afterward — modifying it is gate-weakening per DECISIONS.md. This
module ships the immutable JSONL schema + loader so Phase B can drop
in labels and have a working scorer immediately.

Each entry encodes the operator's manual selection between three options:

    {
      "chunk_id":      str,
      "raw":           str,        # original UIR chunk content
      "heuristic":     str,        # post-heuristic-stack output
      "llm":           str,        # post-LLM-sanitization output
      "preferred":     "raw" | "heuristic" | "llm",
      "rationale":     str,        # operator's note on why preferred wins
      "modality":      str,        # Modality enum value
      "doc_id":        str,
      "labeled_at":    str,        # ISO-8601 timestamp
      "labeled_by":    str,        # operator handle
    }

Per Charter dominance criterion #5: LLM must show ≥ heuristic by ≥5pp
absolute on this golden set, where 1 point = 1 chunk preferring LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional


GOLDEN_SET_SIZE = 50  # Charter §3.3 dominance criterion #5
PREFERRED_OPTIONS = ("raw", "heuristic", "llm")

DEFAULT_GOLDEN_SET_PATH = Path("docs/PHASE_B_GOLDEN_SET.jsonl")


@dataclass(frozen=True)
class GoldenEntry:
    """One labeled golden-set entry."""

    chunk_id: str
    raw: str
    heuristic: str
    llm: str
    preferred: Literal["raw", "heuristic", "llm"]
    rationale: str
    modality: str
    doc_id: str
    labeled_at: str
    labeled_by: str

    def __post_init__(self) -> None:
        if self.preferred not in PREFERRED_OPTIONS:
            raise ValueError(
                f"GoldenEntry.preferred={self.preferred!r} must be one of "
                f"{PREFERRED_OPTIONS}"
            )


def load_golden_set(path: Path = DEFAULT_GOLDEN_SET_PATH) -> List[GoldenEntry]:
    """Load the JSONL-encoded golden set from disk.

    Returns an empty list when the file does not exist (foundation-session
    state). Phase B B2 writes the file once, then this loader is the
    only read path.
    """
    if not path.exists():
        return []
    entries: List[GoldenEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Golden set malformed at line {line_no}: {exc}"
                ) from exc
            entries.append(GoldenEntry(**payload))
    return entries


@dataclass(frozen=True)
class DominanceScore:
    """Result of scoring LLM vs heuristic against the golden set."""

    n: int  # total entries scored
    llm_preferred: int
    heuristic_preferred: int
    raw_preferred: int
    llm_minus_heuristic_pp: float  # (llm - heuristic) / n × 100

    def passes_dominance(self, threshold_pp: float = 5.0) -> bool:
        """Charter §3.3 #5: LLM ≥ heuristic by ≥5pp absolute."""
        return self.llm_minus_heuristic_pp >= threshold_pp


def score_against_golden_set(
    entries: Optional[List[GoldenEntry]] = None,
    path: Path = DEFAULT_GOLDEN_SET_PATH,
) -> DominanceScore:
    """Compute the LLM-vs-heuristic dominance score per Charter §3.3 #5.

    The "preferred" label on each entry is the ground truth; the score
    counts how many entries each candidate "wins".
    """
    if entries is None:
        entries = load_golden_set(path)
    n = len(entries)
    if n == 0:
        return DominanceScore(
            n=0,
            llm_preferred=0,
            heuristic_preferred=0,
            raw_preferred=0,
            llm_minus_heuristic_pp=0.0,
        )
    counts = {"raw": 0, "heuristic": 0, "llm": 0}
    for e in entries:
        counts[e.preferred] += 1
    pp = (counts["llm"] - counts["heuristic"]) / n * 100.0
    return DominanceScore(
        n=n,
        llm_preferred=counts["llm"],
        heuristic_preferred=counts["heuristic"],
        raw_preferred=counts["raw"],
        llm_minus_heuristic_pp=pp,
    )
