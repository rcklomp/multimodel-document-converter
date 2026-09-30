"""Itemized ledger of every IMAGE chunk the export chain removes.

PLAN_QUALITY_REMEDIATION_V1 WP-A2b. The export chain drops IMAGE chunks at a dozen sites (blank
asset, icon-class, thin strip, no-visual sentinel, full-page editorial, pHash duplicate, ...), each
with its own log line, so a described figure that vanished (IRJET: "Dropped 7 icon-class image
chunk(s)") left no per-document account. The ledger records the reason and the asset for each drop
and states the invariant

    IMAGE chunks in  ==  IMAGE chunks written  +  itemized drops

so a drop site nobody wrapped shows up as ``unaccounted`` instead of as silence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

_MAX_ASSETS_PER_REASON = 12


@dataclass
class ImageDrop:
    reason: str
    chunk_id: str
    page: Optional[int]
    asset: Optional[str]


def _is_image(chunk: Any) -> bool:
    modality = getattr(chunk, "modality", None)
    return getattr(modality, "value", modality) == "image"


def _chunk_key(chunk: Any) -> str:
    return getattr(chunk, "chunk_id", None) or f"@{id(chunk)}"


def _describe(reason: str, chunk: Any) -> ImageDrop:
    meta = getattr(chunk, "metadata", None)
    asset_ref = getattr(chunk, "asset_ref", None)
    path = getattr(asset_ref, "file_path", None) if asset_ref else None
    return ImageDrop(
        reason=reason,
        chunk_id=_chunk_key(chunk),
        page=getattr(meta, "page_number", None) if meta else None,
        asset=path.split("/")[-1] if path else None,
    )


@dataclass
class ImageDropLedger:
    image_in: Optional[int] = None
    drops: List[ImageDrop] = field(default_factory=list)

    def begin(self, chunks: Sequence[Any]) -> None:
        """Fix the "in" side of the invariant (first call wins)."""
        if self.image_in is None:
            self.image_in = sum(1 for c in chunks if _is_image(c))

    def track(self, reason: str, before: Sequence[Any], after: Sequence[Any]) -> Any:
        """Record the IMAGE chunks present in ``before`` and absent from ``after``; return ``after``.

        Matching is by chunk_id as a multiset, so a filter that replaces a chunk object keeps it
        (same id) and a chunk_id dedup that removes the second of two equal ids is itemized.
        """
        gone = Counter(_chunk_key(c) for c in before if _is_image(c))
        gone.subtract(Counter(_chunk_key(c) for c in after if _is_image(c)))
        remaining = {key: n for key, n in gone.items() if n > 0}
        if remaining:
            for chunk in before:
                if not _is_image(chunk):
                    continue
                key = _chunk_key(chunk)
                if remaining.get(key, 0) > 0:
                    remaining[key] -= 1
                    self.drops.append(_describe(reason, chunk))
        return after

    def record(self, reason: str, chunk: Any) -> None:
        """Record one drop made inside a loop (no before/after lists available)."""
        if _is_image(chunk):
            self.drops.append(_describe(reason, chunk))

    def unaccounted(self, image_out: int) -> Optional[int]:
        if self.image_in is None:
            return None
        return self.image_in - image_out - len(self.drops)

    def by_reason(self) -> Dict[str, List[ImageDrop]]:
        grouped: Dict[str, List[ImageDrop]] = {}
        for drop in self.drops:
            grouped.setdefault(drop.reason, []).append(drop)
        return grouped

    def summary_lines(self, image_out: int) -> List[str]:
        """One header line plus one line per reason; empty when nothing was dropped or unexplained."""
        missing = self.unaccounted(image_out)
        if not self.drops and not missing:
            return []
        lines = [
            f"[IMAGE-DROPS] in={self.image_in} written={image_out} "
            f"itemized_drops={len(self.drops)} unaccounted={missing}"
        ]
        for reason, drops in sorted(self.by_reason().items()):
            names = [d.asset or f"p{d.page}:{d.chunk_id}" for d in drops[:_MAX_ASSETS_PER_REASON]]
            more = len(drops) - len(names)
            tail = f" (+{more} more)" if more > 0 else ""
            lines.append(f"[IMAGE-DROPS]   {reason}: {len(drops)} -> {', '.join(names)}{tail}")
        return lines
