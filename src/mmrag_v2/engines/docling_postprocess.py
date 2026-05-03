"""Post-Docling sanity passes for native-digital documents.

Mirrors community precedent (`docling-hierarchical-pdf`'s ResultPostprocessor):
the Docling team has publicly stated that built-in reading-order, drop-cap
handling, and OCR gating are not on the near-term roadmap, so a surgical
post-pass is the supported fix.

Each stage is independently togglable via PdfConversionPlan fields:

  reading_order_strategy: "docling_native"          - no-op (default)
                          "y_sort"                  - Phase 1 only
                          "y_sort_with_dropcap"     - Phase 1 + Phase 2

Upstream tracking (Docling 2.86, May 2026):

  reading order:   https://github.com/docling-project/docling/discussions/2791
                   https://github.com/docling-project/docling/issues/1203
                   https://github.com/docling-project/docling/issues/2245

When Docling ships built-in reading-order, replace this pass with their flag
and verify the HARRY pages 1-30 acceptance fixture still passes.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_Y_SORT_STRATEGIES = frozenset({"y_sort", "y_sort_with_dropcap"})
_DROPCAP_STRATEGIES = frozenset({"y_sort_with_dropcap"})


def _first_prov(item: Any) -> Any:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    if isinstance(prov, list):
        return prov[0] if prov else None
    return prov


def _page_no(item: Any) -> Optional[int]:
    prov = _first_prov(item)
    if prov is None:
        return None
    page = getattr(prov, "page_no", None)
    return int(page) if page is not None else None


def _y_sort_key(item: Any) -> Optional[Tuple[float, float]]:
    """Reading-order sort key for a Docling item.

    BOTTOMLEFT-origin bboxes (Docling default for PDF pages): top-to-bottom
    means decreasing `t`, so we return `(-t, l)`. TOPLEFT-origin bboxes
    (used by some downstream conversions) flip the sign: top-to-bottom
    means increasing `t`, so we return `(t, l)`.

    Returns None when the item has no prov bbox; callers fall back to the
    item's original document order.
    """
    prov = _first_prov(item)
    if prov is None:
        return None
    bbox = getattr(prov, "bbox", None)
    if bbox is None:
        return None
    t = getattr(bbox, "t", None)
    l = getattr(bbox, "l", None)
    if t is None or l is None:
        return None
    origin = getattr(bbox, "coord_origin", None)
    origin_name = (getattr(origin, "name", None) or str(origin or "")).upper()
    if "TOPLEFT" in origin_name:
        return (float(t), float(l))
    return (-float(t), float(l))


def apply_reading_order_sort(doc: Any, plan: Any) -> Any:
    """Re-sort the document body's children per page by reading order.

    Items on a given page are sorted by `(-bbox.t, bbox.l)` (BOTTOMLEFT
    origin) or `(bbox.t, bbox.l)` (TOPLEFT). Items lacking a prov bbox keep
    their original document order via a stable secondary key. Pages stay in
    ascending order: an item on page 14 never reorders ahead of a page 13
    item.

    The pass is a no-op unless `plan.reading_order_strategy` is `"y_sort"`
    or `"y_sort_with_dropcap"`. Returns the same `doc` instance so callers
    can chain stages.
    """
    strategy = getattr(plan, "reading_order_strategy", "docling_native")
    if strategy not in _Y_SORT_STRATEGIES:
        return doc

    body = getattr(doc, "body", None)
    if body is None:
        return doc
    children = list(getattr(body, "children", None) or ())
    if not children:
        return doc

    resolved: List[Tuple[int, Any, Any]] = []
    for original_idx, ref in enumerate(children):
        resolve = getattr(ref, "resolve", None)
        item = None
        if callable(resolve):
            try:
                item = resolve(doc)
            except Exception:
                item = None
        resolved.append((original_idx, ref, item))

    page_buckets: Dict[int, List[Tuple[int, Any, Any]]] = {}
    no_page: List[Tuple[int, Any, Any]] = []
    for entry in resolved:
        _, _, item = entry
        page = _page_no(item) if item is not None else None
        if page is None:
            no_page.append(entry)
        else:
            page_buckets.setdefault(page, []).append(entry)

    def _sort_key(entry: Tuple[int, Any, Any]) -> Tuple[int, float, float, int]:
        original_idx, _, item = entry
        ysort = _y_sort_key(item) if item is not None else None
        if ysort is None:
            return (1, 0.0, 0.0, original_idx)
        return (0, ysort[0], ysort[1], original_idx)

    new_children: List[Any] = []
    for page in sorted(page_buckets):
        bucket = page_buckets[page]
        bucket.sort(key=_sort_key)
        new_children.extend(ref for _, ref, _ in bucket)
    new_children.extend(ref for _, ref, _ in no_page)

    body.children = new_children
    logger.info(
        "[POSTPROC] reading_order_sort applied: %d items across %d pages (strategy=%s)",
        len(new_children),
        len(page_buckets),
        strategy,
    )
    return doc


def _bbox_y_band(item: Any) -> Optional[Tuple[float, float, str]]:
    """Return (low_y, high_y, origin_name) for an item's first prov bbox."""
    prov = _first_prov(item)
    if prov is None:
        return None
    bbox = getattr(prov, "bbox", None)
    if bbox is None:
        return None
    t = getattr(bbox, "t", None)
    b = getattr(bbox, "b", None)
    if t is None or b is None:
        return None
    origin = getattr(bbox, "coord_origin", None)
    origin_name = (getattr(origin, "name", None) or str(origin or "")).upper()
    low = min(float(t), float(b))
    high = max(float(t), float(b))
    return (low, high, origin_name)


def _y_overlaps(a: Tuple[float, float, str], b: Tuple[float, float, str]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _is_dropcap_candidate(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    return len(stripped) == 1 and stripped.isalpha() and stripped.isupper()


def _starts_lowercase(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    for ch in text:
        if ch.isalpha():
            return ch.islower()
    return False


_TRAILING_DROPCAP_RE = re.compile(r"\s+([A-Z])\s*$")
_LEADING_LOWERCASE_RE = re.compile(r"^\s*([a-z])")


def _heal_inline_trailing_dropcap(item: Any) -> bool:
    """Move a trailing single-uppercase glyph to the front of a paragraph.

    Docling 2.86 routinely keeps drop-cap "M" glued to the END of the
    same TextItem whose body starts lowercase ("r. and Mrs. Dursley...
    nonsense. M"). Pattern: the text starts with a single lowercase letter
    after no leading whitespace AND ends with whitespace + a lone uppercase
    letter. When matched, prepend the trailing uppercase to the front and
    strip the trailing copy. Idempotent: a healed text starts uppercase
    and won't re-match.

    Returns True when the item's text was modified.
    """
    text = getattr(item, "text", None)
    if not isinstance(text, str) or len(text) < 4:
        return False
    head_match = _LEADING_LOWERCASE_RE.match(text)
    tail_match = _TRAILING_DROPCAP_RE.search(text)
    if not head_match or not tail_match:
        return False
    cap = tail_match.group(1)
    healed = cap + text[: tail_match.start()].rstrip() + text[tail_match.end():]
    # Defensive: only commit if the result looks like a real heal (still
    # starts with the same lowercase fragment after the cap).
    if not healed[:1].isupper() or healed[1:2] != head_match.group(1):
        return False
    try:
        item.text = healed
        if hasattr(item, "orig"):
            item.orig = healed
    except Exception:
        return False
    return True


def apply_dropcap_promotion(doc: Any, plan: Any) -> Any:
    """Merge stray drop-cap glyphs into the body paragraph they should lead.

    Detects a TextItem whose stripped text is a single uppercase letter and
    whose y-band overlaps an adjacent TextItem (predecessor or successor in
    body.children) that begins with a lowercase fragment - the signature of
    a literary drop cap that Docling's layout model split off as its own
    item. Prepends the letter to the neighbor's text and removes the
    standalone item from `body.children`.

    No-op unless `plan.reading_order_strategy == "y_sort_with_dropcap"`.
    Drop-cap detection has no community precedent; the rule deliberately
    requires both a single-letter uppercase glyph AND a lowercase neighbor
    so that legitimate single-letter content (sentence "I", section "A.")
    survives.
    """
    strategy = getattr(plan, "reading_order_strategy", "docling_native")
    if strategy not in _DROPCAP_STRATEGIES:
        return doc

    body = getattr(doc, "body", None)
    if body is None:
        return doc
    children = list(getattr(body, "children", None) or ())
    if len(children) < 2:
        # Even with no neighbors, run the inline trailing-glyph heal across
        # any items the body has (e.g. a single TextItem whose text already
        # bundles the drop cap at the tail).
        for ref in children:
            resolve = getattr(ref, "resolve", None)
            item = resolve(doc) if callable(resolve) else None
            if item is not None:
                _heal_inline_trailing_dropcap(item)
        return doc

    resolved: List[Tuple[Any, Any]] = []
    for ref in children:
        resolve = getattr(ref, "resolve", None)
        item = resolve(doc) if callable(resolve) else None
        resolved.append((ref, item))

    keep: List[bool] = [True] * len(resolved)
    promotions = 0

    inline_healed = 0
    for _, item in resolved:
        if item is None:
            continue
        if _heal_inline_trailing_dropcap(item):
            inline_healed += 1

    for idx, (_, item) in enumerate(resolved):
        if not keep[idx] or item is None:
            continue
        if not _is_dropcap_candidate(getattr(item, "text", None)):
            continue
        cap_page = _page_no(item)
        cap_band = _bbox_y_band(item)
        if cap_page is None or cap_band is None:
            continue

        for neighbor_idx in (idx + 1, idx - 1):
            if not 0 <= neighbor_idx < len(resolved):
                continue
            if not keep[neighbor_idx]:
                continue
            neighbor = resolved[neighbor_idx][1]
            if neighbor is None or neighbor is item:
                continue
            if _page_no(neighbor) != cap_page:
                continue
            neighbor_text = getattr(neighbor, "text", "")
            if not _starts_lowercase(neighbor_text):
                continue
            neighbor_band = _bbox_y_band(neighbor)
            if neighbor_band is None or not _y_overlaps(cap_band, neighbor_band):
                continue

            cap_letter = (getattr(item, "text", "") or "").strip()
            merged = f"{cap_letter}{neighbor_text}"
            try:
                neighbor.text = merged
                if hasattr(neighbor, "orig"):
                    neighbor.orig = merged
            except Exception:
                continue
            keep[idx] = False
            promotions += 1
            break

    if inline_healed:
        logger.info(
            "[POSTPROC] dropcap_promotion healed %d inline trailing glyph(s)",
            inline_healed,
        )
    if promotions == 0:
        return doc

    body.children = [children[i] for i in range(len(children)) if keep[i]]
    logger.info("[POSTPROC] dropcap_promotion merged %d standalone glyph(s)", promotions)
    return doc


def apply_postprocessors(doc: Any, plan: Any) -> Any:
    """Run all enabled post-Docling stages in the documented order.

    Order: label_filter (Phase 3, serializer-side) -> dropcap_promoter
           (Phase 2) -> reading_order_sort (Phase 1).

    Filtering label noise first lets the drop-cap heuristic ignore stray
    OTHER/UNKNOWN items; promoting drop caps before the y-sort means the
    final sort sees the merged paragraph as a single anchored item.
    """
    apply_dropcap_promotion(doc, plan)
    apply_reading_order_sort(doc, plan)
    return doc
