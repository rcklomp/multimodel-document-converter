"""Running page furniture (headers, footers, folios) detected at ELEMENT level.

PLAN_QUALITY_REMEDIATION_V1 WP-B2. The chunk-level filter (BatchProcessor._filter_running_furniture,
F1) runs last, only sees chunks of <= 70 characters, and cannot help once the chunker has glued a
running header into a body chunk: on IRJET 11 of 30 text chunks carried the journal header, footer
or page folio. This module finds the furniture ELEMENTS before they are chunked.

Signals (in this order):
  1. the engine's own label (MinerU ``header`` / ``footer`` / ``page_number``; a VLM ``footer`` or
     ``header`` type preserved as ``original_vlm_type``); short elements only.
  2. repetition: a digit-normalized, whitespace-collapsed signature (difflib ratio >= 0.9) that
     sits in the top-k or bottom-k TEXT elements of >= 3 distinct pages. The position is a RANK on
     the page, not an absolute band: VLM frames are compressed (measured 0.62 in y), so a band that
     works for Docling misses VLM output.
Never removed: heading-labelled elements, elements whose text equals a heading string of the
document (in-page heading or TOC entry: a book's running header is often its chapter title and is
then the only source of the section heading), code/form-promoted elements, non-TEXT elements, and
the last remaining element(s) of a page.

Pure functions; nothing here mutates the document.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Engine labels that name page furniture. MinerU2.5 emits type "header"/"footer"/"page_number"
# (kept verbatim as source_label); Docling's layout labels are page_header/page_footer.
FURNITURE_LABELS = frozenset(
    {
        "header",
        "footer",
        "page_header",
        "page_footer",
        "page_number",
        "running_header",
        "running_footer",
        "folio",
    }
)

# Furniture is short. The IRJET journal header is 151 characters once the wrapped line is joined,
# so the chunk-level filter's 70-character cap missed exactly the case that matters.
MAX_FURNITURE_CHARS = 200
# Rank window measured on 40 local outputs (k=2/3/4/6: 872/1017/1111/1245 elements flagged, no output
# gained a headless chunk at any k). A running header split into several short elements (MinerU
# reconstruction: 6 fragments) needs a window wider than 2; beyond 4 the gain is only the same block.
# A stricter 'same exact rank on every page' variant flagged 57 fewer elements but left 11 more IRJET
# furniture strings in the output, so the top/bottom ZONE is used.
TOP_BOTTOM_RANKS = 4
MIN_REPEAT_PAGES = 3
SIMILARITY_MIN = 0.9

# A caption is content even when the same sub-caption repeats on several figure pages
# ("(a) Normalized throughput. Higher is better." x5 on AIOS); furniture never starts like this.
_CAPTION_START_RE = re.compile(
    r"^(?:fig(?:ure)?s?\.?|table|tab\.)\s*\(?\d|^\([a-z]\)\s+[a-z]", re.IGNORECASE
)
_PROTECTED_LABEL_PARTS = ("caption", "footnote")

_HEADING_LABELS = frozenset(
    {"heading", "title", "section_header", "section_heading", "subtitle", "chapter_title"}
)


@dataclass(frozen=True)
class FurnitureDrop:
    """One removed element, for the drop report and QA-CHECK-01 accounting."""

    page: int
    position: int  # index into page.elements
    element_index: int
    text: str
    rule: str  # "label" | "repetition"
    signature: str


def normalize_signature(text: str) -> str:
    """Digit-normalized, whitespace-collapsed, lowercase form (the page number may vary)."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", (text or "").strip())).lower()


def _label(element: Any) -> str:
    return (getattr(element, "source_label", "") or "").lower().replace("-", "_").replace(" ", "_")


def _original_vlm_type(element: Any) -> str:
    md = getattr(element, "metadata", None) or {}
    return str(md.get("original_vlm_type") or "").lower().replace("-", "_").replace(" ", "_")


def _is_text_body(element: Any) -> bool:
    """A plain TEXT element with content (the only kind furniture detection looks at)."""
    etype = getattr(getattr(element, "type", None), "value", getattr(element, "type", None))
    if etype != "text" or not (getattr(element, "content", "") or "").strip():
        return False
    md = getattr(element, "metadata", None) or {}
    return md.get("promoted_modality") not in ("code", "form")


def _norm_heading(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace(" ", " ")).strip().lower()


def protected_heading_strings(
    pages: Sequence[Any], toc_headings: Optional[Dict[Any, Any]] = None
) -> Set[str]:
    """Every heading string of the document: heading-labelled elements plus TOC entries."""
    out: Set[str] = set()
    for page in pages:
        for e in page.elements:
            if _label(e) in _HEADING_LABELS and (e.content or "").strip():
                out.add(_norm_heading(e.content))
    if toc_headings:
        heading_map = toc_headings.get("__heading_map__") or {}
        out.update(_norm_heading(str(k)) for k in heading_map)
        for key, crumbs in toc_headings.items():
            if isinstance(crumbs, (list, tuple)):
                out.update(_norm_heading(str(c)) for c in crumbs if isinstance(c, str))
    out.discard("")
    return out


def _clusters(signatures: Iterable[str]) -> Dict[str, str]:
    """Map each signature to a representative: signatures within SIMILARITY_MIN share one."""
    reps: List[str] = []
    mapping: Dict[str, str] = {}
    for sig in sorted(set(signatures)):
        for rep in reps:
            if sig == rep or difflib.SequenceMatcher(None, sig, rep).ratio() >= SIMILARITY_MIN:
                mapping[sig] = rep
                break
        else:
            reps.append(sig)
            mapping[sig] = sig
    return mapping


def find_running_furniture(
    pages: Sequence[Any],
    *,
    toc_headings: Optional[Dict[Any, Any]] = None,
    use_labels: bool = True,
    use_repetition: bool = True,
) -> List[FurnitureDrop]:
    """Return the furniture elements of ``pages`` (never mutates them)."""
    protected = protected_heading_strings(pages, toc_headings)

    def eligible(e: Any) -> bool:
        return (
            _is_text_body(e)
            and _label(e) not in _HEADING_LABELS
            and not any(part in _label(e) for part in _PROTECTED_LABEL_PARTS)
            and not _CAPTION_START_RE.match(e.content.strip())
            and len(e.content.strip()) <= MAX_FURNITURE_CHARS
            and _norm_heading(e.content) not in protected
        )

    flagged: Dict[Tuple[int, int], FurnitureDrop] = {}

    if use_labels:
        for page in pages:
            for pos, e in enumerate(page.elements):
                lab = _label(e)
                if (lab in FURNITURE_LABELS or _original_vlm_type(e) in FURNITURE_LABELS) and (
                    eligible(e)
                ):
                    flagged[(page.page_number, pos)] = FurnitureDrop(
                        page.page_number,
                        pos,
                        e.element_index,
                        e.content.strip(),
                        "label",
                        normalize_signature(e.content),
                    )

    if use_repetition:
        slots: List[Tuple[int, int, str, str]] = []  # (page, pos, zone, signature)
        for page in pages:
            body = [(pos, e) for pos, e in enumerate(page.elements) if _is_text_body(e)]
            n = len(body)
            for rank, (pos, e) in enumerate(body):
                zones = []
                if rank < TOP_BOTTOM_RANKS:
                    zones.append("top")
                if rank >= n - TOP_BOTTOM_RANKS:
                    zones.append("bottom")
                if not zones or not eligible(e):
                    continue
                sig = normalize_signature(e.content)
                for zone in zones:
                    slots.append((page.page_number, pos, zone, sig))
        rep = _clusters(s[3] for s in slots)
        pages_by_key: Dict[Tuple[str, str], Set[int]] = {}
        for page_no, _pos, zone, sig in slots:
            pages_by_key.setdefault((zone, rep[sig]), set()).add(page_no)
        by_page = {p.page_number: p for p in pages}
        for page_no, pos, zone, sig in slots:
            if len(pages_by_key[(zone, rep[sig])]) >= MIN_REPEAT_PAGES:
                key = (page_no, pos)
                if key not in flagged:
                    e = by_page[page_no].elements[pos]
                    flagged[key] = FurnitureDrop(
                        page_no, pos, e.element_index, e.content.strip(), "repetition", sig
                    )

    # Page-coverage guard: a page never loses every content element to this pass.
    drops: List[FurnitureDrop] = []
    for page in pages:
        page_flagged = [k for k in flagged if k[0] == page.page_number]
        if not page_flagged:
            continue
        content_positions = {pos for pos, e in enumerate(page.elements) if _is_text_body(e)}
        content_positions |= {
            pos for pos, e in enumerate(page.elements) if not _is_text_body(e) and _has_payload(e)
        }
        if content_positions and content_positions <= {k[1] for k in page_flagged}:
            continue
        drops.extend(flagged[k] for k in sorted(page_flagged, key=lambda k: k[1]))
    return drops


def _has_payload(element: Any) -> bool:
    """A non-text element (image/table/code/form) that keeps its page populated."""
    return bool((getattr(element, "content", "") or "").strip()) or (
        getattr(getattr(element, "type", None), "value", None) in ("image", "table")
    )
