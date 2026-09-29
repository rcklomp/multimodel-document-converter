"""Structural regression-guard metrics over ingestion JSONL rows (advisory).

PLAN_QUALITY_REMEDIATION_V1 WP-0.2. These functions describe defect classes that the
hard gates cannot see (a defective IRJET conversion passed with failures=0, warnings=0):
stale neighbour snippets, headings embedded inside chunk bodies, running header/footer
lines glued into text, reference lists cut mid-entry, and figures referenced in text
without an image chunk.

Two rules govern how the numbers may be used (AGENT-GATE-PROGRESSION, AGENT-INTEGRITY-01):

* They are ADVISORY regression guards. Some of them restate the predicate of the fix they
  protect (``heading_inside_body`` mirrors the heading-section rule), so they read zero
  by construction after that fix. They are therefore NEVER the acceptance evidence; the
  acceptance evidence is anchored in the source PDF (``scripts/qa_gold_anchor_smoke.py``).
* Pure functions over plain dict rows: no file access, no network, no shell. The host
  script wraps every call so an unexpected row shape can never flip the strict gate.

A "row" is one parsed JSONL chunk object (the ``ingestion_metadata`` record excluded).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Row = Dict[str, Any]

_SKELETON_RE = re.compile(r"[\s\x00-\x1f]+")
_WS_RE = re.compile(r"\s+")


def _meta(row: Row) -> Dict[str, Any]:
    return row.get("metadata") or {}


def _page(row: Row) -> Optional[int]:
    p = _meta(row).get("page_number")
    return p if isinstance(p, int) else None


def _content(row: Row) -> str:
    return row.get("content") or ""


def _text_rows(rows: Sequence[Row]) -> List[Row]:
    return [r for r in rows if r.get("modality") == "text"]


def _skeleton(text: str, n: int) -> str:
    """First ``n`` characters with ALL whitespace and control characters removed.

    Export sanitizing strips control characters while the snippet was cut before it, so
    a spacing-sensitive comparison would report false orphans; the skeleton is immune.
    """
    return _SKELETON_RE.sub("", text)[:n]


# --------------------------------------------------------------------------- snippets
@dataclass(frozen=True)
class SnippetReport:
    orphans: List[str] = field(default_factory=list)  # snippet does not match the real successor
    coverage_gaps: List[str] = field(default_factory=list)  # successor exists but snippet is empty
    checked: int = 0
    present: int = 0  # rows carrying a non-empty snippet; 0 means the output predates the feature


def snippet_consistency(rows: Sequence[Row], compare_chars: int = 60) -> SnippetReport:
    """Compare every chunk's ``semantic_context.next_text_snippet`` with its real successor.

    An orphan is a non-empty snippet that is not a (whitespace/control-insensitive) prefix
    of the following row's content, or any non-empty snippet on the last row. The comparison
    is on whitespace/control-free skeletons of the first ``compare_chars`` characters. A coverage
    gap is a non-last row whose successor has content while the snippet is empty; it
    exists so that nulling every snippet cannot score as "no orphans". Outputs that predate
    the snippet feature carry no snippets at all (``present == 0``) and read a coverage gap
    of nearly 100%: interpret the gap only when ``present > 0`` or on a fresh output.
    """
    orphans: List[str] = []
    gaps: List[str] = []
    present = 0
    for i, row in enumerate(rows):
        sc = row.get("semantic_context") or {}
        snippet = _skeleton(sc.get("next_text_snippet") or "", compare_chars)
        successor = rows[i + 1] if i + 1 < len(rows) else None
        succ_text = _skeleton(_content(successor), compare_chars) if successor else ""
        cid = str(row.get("chunk_id") or i)
        if snippet:
            present += 1
            if successor is None or not succ_text:
                orphans.append(cid)
            else:
                n = min(len(snippet), len(succ_text))
                if snippet[:n] != succ_text[:n]:
                    orphans.append(cid)
        elif succ_text:
            gaps.append(cid)
    return SnippetReport(orphans=orphans, coverage_gaps=gaps, checked=len(rows), present=present)


# --------------------------------------------------------------------------- headings
def heading_inside_body(rows: Sequence[Row]) -> List[str]:
    """TEXT chunks where a non-first line equals a heading used as some chunk's parent.

    MIRROR of the heading-section rule (a heading starts its own chunk): zero by
    construction after that fix. A regression guard only.
    """
    texts = _text_rows(rows)
    headings = {
        ((_meta(r).get("hierarchy") or {}).get("parent_heading") or "").strip() for r in texts
    }
    headings.discard("")
    flagged: List[str] = []
    for r in texts:
        lines = [ln.strip() for ln in _content(r).split("\n")]
        if any(ln in headings for ln in lines[1:]):
            flagged.append(str(r.get("chunk_id")))
    return flagged


# --------------------------------------------------------------------------- furniture
_DIGITS_RE = re.compile(r"\d+")


def _norm_line(line: str) -> str:
    return _DIGITS_RE.sub("#", _WS_RE.sub(" ", line.strip())).lower()


def furniture_line_chunks(
    rows: Sequence[Row], min_pages: int = 3, min_len: int = 12
) -> List[Tuple[str, str]]:
    """TEXT chunks containing a line whose digit-normalized form repeats on >= ``min_pages`` pages.

    Output-derived UPPER BOUND on running header/footer contamination: legitimate
    repeats (a caption pattern repeated per page) count too. Independent of position and
    of chunk length, unlike the chunk-level F1 filter this guards. Returns
    ``(chunk_id, line)`` pairs.
    """
    pages_by_line: Dict[str, set] = {}
    for r in _text_rows(rows):
        pg = _page(r)
        for ln in _content(r).split("\n"):
            n = _norm_line(ln)
            if len(n) >= min_len:
                pages_by_line.setdefault(n, set()).add(pg)
    repeating = {n for n, pgs in pages_by_line.items() if len(pgs) >= min_pages}
    flagged: List[Tuple[str, str]] = []
    for r in _text_rows(rows):
        for ln in _content(r).split("\n"):
            if _norm_line(ln) in repeating:
                flagged.append((str(r.get("chunk_id")), ln.strip()))
                break
    return flagged


# --------------------------------------------------------------------------- references
_REF_HEADING_RE = re.compile(
    r"^\W*(?:\d+\.?\s*)?(?:references|bibliography|literature cited|works cited|"
    r"literaturverzeichnis|referenties|literatuur)\b",
    re.I,
)
# A label at a line start, or inline right after a sentence end, followed by a capital.
_REF_LABEL_RE = re.compile(r"(?:^|(?<=\n)|(?<=[.] ))\[(\d+)\]\s+(?=[A-Z])")
_LABEL_ANYWHERE_RE = re.compile(r"\[\d+\]")
# A citation tail is complete when it ends in a year, a page range, or "no. N".
_ENTRY_COMPLETE_RE = re.compile(
    r"(?:\b(?:19|20)\d{2}|\bpp?\.?\s*\d+\s*[-–]\s*\d+|\bno\.?\s*\d+)\W*$"
)


def _is_reference_row(row: Row) -> bool:
    heading = (_meta(row).get("hierarchy") or {}).get("parent_heading") or ""
    if _REF_HEADING_RE.match(heading):
        return True
    return len(_LABEL_ANYWHERE_RE.findall(_content(row))) >= 3


@dataclass(frozen=True)
class ReferenceReport:
    order_breaks: int = 0  # a label smaller than its predecessor, in chunk order
    split_entries: int = 0  # an entry that continues into a chunk that does not start with a label
    reference_rows: int = 0


def reference_entry_integrity(rows: Sequence[Row]) -> ReferenceReport:
    """Reference-list integrity inside references-class TEXT chunks.

    Counts label-order breaks (labels found at line starts AND inline after a sentence
    end, so a list emitted as one inline element is seen) and entries cut across chunks.
    Body citations such as "Reference [10] presented ..." are not labels (no preceding
    sentence end, not followed by a capital after a line start).
    """
    ref_rows = [r for r in rows if r.get("modality") == "text" and _is_reference_row(r)]
    order_breaks = 0
    split = 0
    prev_label: Optional[int] = None
    for idx, r in enumerate(ref_rows):
        content = _content(r)
        labels = [int(m.group(1)) for m in _REF_LABEL_RE.finditer(content)]
        for lab in labels:
            if prev_label is not None and lab < prev_label:
                order_breaks += 1
            prev_label = lab
        matches = list(_REF_LABEL_RE.finditer(content))
        if matches and idx + 1 < len(ref_rows):
            tail = content[matches[-1].start() :].strip()
            nxt = _content(ref_rows[idx + 1]).lstrip()
            starts_with_label = bool(re.match(r"\[\d+\]\s+[A-Z]", nxt))
            if not starts_with_label and not _ENTRY_COMPLETE_RE.search(tail):
                split += 1
    return ReferenceReport(
        order_breaks=order_breaks, split_entries=split, reference_rows=len(ref_rows)
    )


# --------------------------------------------------------------------------- figures
# A panel letter must be attached to the number ("10a", "10(a)"); a detached capital such as
# the "I" of "Fig. 6 I-V characteristic curve" is title text, not a panel.
_CAPTION_RE = re.compile(
    r"^\s*(?:Fig(?:ure)?\.?)\s*(\d+)(?:\((?-i:[a-z])\)|(?-i:[a-z])(?![A-Za-z]))?\s*[:.]?\s*(.*)$",
    re.I,
)


def _caption_numbers(text: str) -> set:
    nums = set()
    for ln in text.split("\n"):
        if len(ln) > 120:
            continue
        m = _CAPTION_RE.match(ln)
        if not m:
            continue
        rest = m.group(2).strip()
        # A caption title starts with a capital; "Fig. 4 shows ..." is a body sentence.
        if rest and rest[0].isupper():
            nums.add(int(m.group(1)))
    return nums


def figure_deficit_pages(rows: Sequence[Row]) -> List[Tuple[int, List[int], int]]:
    """Pages whose distinct figure captions outnumber the IMAGE chunks on the page.

    ``(page, sorted caption numbers, image chunks on the page)``. Captions folded into an
    IMAGE chunk's own content count as referenced numbers as well, so a figure whose
    caption travelled with its image is not reported.
    """
    by_page_text: Dict[int, set] = {}
    by_page_img_nums: Dict[int, set] = {}
    by_page_imgs: Dict[int, int] = {}
    for r in rows:
        pg = _page(r)
        if pg is None:
            continue
        if r.get("modality") == "text":
            by_page_text.setdefault(pg, set()).update(_caption_numbers(_content(r)))
        elif r.get("modality") == "image":
            by_page_imgs[pg] = by_page_imgs.get(pg, 0) + 1
            by_page_img_nums.setdefault(pg, set()).update(_caption_numbers(_content(r)))
    out: List[Tuple[int, List[int], int]] = []
    for pg in sorted(set(by_page_text) | set(by_page_img_nums)):
        nums = by_page_text.get(pg, set()) | by_page_img_nums.get(pg, set())
        n_img = by_page_imgs.get(pg, 0)
        if len(nums) > n_img:
            out.append((pg, sorted(nums), n_img))
    return out


# --------------------------------------------------------------------------- aggregate
@dataclass(frozen=True)
class StructuralOutcomes:
    n_text: int
    n_rows: int
    snippets: SnippetReport
    heading_inside_body: List[str]
    furniture_lines: List[Tuple[str, str]]
    references: ReferenceReport
    figure_deficits: List[Tuple[int, List[int], int]]

    def summary_lines(self) -> Iterable[str]:
        t = max(self.n_text, 1)
        yield (
            f"orphan_snippet_chunks={len(self.snippets.orphans)} "
            f"snippet_coverage_gap={len(self.snippets.coverage_gaps)} "
            f"snippets_present={self.snippets.present} rows={self.snippets.checked}"
        )
        yield (
            f"heading_inside_body={len(self.heading_inside_body)} "
            f"heading_inside_body_ratio={len(self.heading_inside_body) / t:.4f}"
        )
        yield (
            f"furniture_line_chunks={len(self.furniture_lines)} "
            f"furniture_line_ratio={len(self.furniture_lines) / t:.4f}"
        )
        yield (
            f"reference_rows={self.references.reference_rows} "
            f"reference_order_breaks={self.references.order_breaks} "
            f"reference_split_entries={self.references.split_entries}"
        )
        yield (
            f"figure_deficit_pages={len(self.figure_deficits)} "
            f"detail={[(p, n, i) for p, n, i in self.figure_deficits]}"
        )


def compute_all(rows: Sequence[Row]) -> StructuralOutcomes:
    """All guard metrics for one document's rows (metadata record already removed)."""
    return StructuralOutcomes(
        n_text=len(_text_rows(rows)),
        n_rows=len(rows),
        snippets=snippet_consistency(rows),
        heading_inside_body=heading_inside_body(rows),
        furniture_lines=furniture_line_chunks(rows),
        references=reference_entry_integrity(rows),
        figure_deficits=figure_deficit_pages(rows),
    )
