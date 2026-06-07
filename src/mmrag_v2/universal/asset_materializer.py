"""Materialize on-disk PNG assets for vision-native IMAGE/TABLE UIR chunks.

The VLM-native engine *describes* image/table regions but never emits a binary
asset, so the resulting ``UIRChunk`` objects reach
``IngestionChunk.from_uir`` with ``asset_ref=None`` and fail QA-CHECK-05
(IMAGE/TABLE require ``asset_ref`` *and* ``spatial.bbox``). This helper renders
each region crop from the source PDF page (bbox is ``[0, COORD_SCALE]``
normalized in the page-portrait frame), saves a PNG under ``assets_dir``, and
sets the relative ``asset_ref`` on the chunk in place.

Single source of truth shared by ``batch_processor`` (production CLI path) and
``scripts/v3_batch_ingest.py`` (the grand-soak harness) so the crop logic of the
two paths cannot silently diverge - the divergence that produced 0/18 valid
baselines in the 2026-06-01 crucible soak (the soak never ran this step).

Crop-audit (PLAN_V3.1 pre-Crucible hardening): clamping a hallucinated VLM bbox
keeps the process alive but turns it into a pipeline that emits *garbage* PNGs
that still pass QA-CHECK-05 (the gate checks asset_ref/bbox PRESENCE, not crop
CORRECTNESS). To make drift measurable without a human eyeballing crops, every
crop is scored for three cheap fingerprints and aggregated into a document-level
``CropAuditReport`` that raises ``QA_WARN_CROP_DRIFT`` past a threshold:

* ``is_full_page_fallback`` - the bbox was missing or degenerate, so the whole
  page was rendered (a degraded-but-not-garbage asset; reported, not gated).
* ``is_edge_clamped`` - the bbox sits on the page boundary, the fingerprint of a
  coordinate that overflowed the frame and was clamped (also matches legitimate
  full-bleed art; see the caveat on ``EDGE_TOUCH_EPS``).
* ``is_low_information`` - the rendered crop is near-uniform white/black, the
  fingerprint of a bbox that landed on whitespace (hallucinated coordinates).

Coverage boundary (documented honestly): these catch edge-overflow drift and
whitespace-landing drift. They do NOT catch *interior misplacement* - a
plausible in-frame bbox that crops the wrong content-rich region - which needs a
semantic crop-vs-description check, out of scope here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF - pure renderer

from mmrag_v2.schema.ingestion_schema import COORD_SCALE, Modality

logger = logging.getLogger(__name__)

# Region crops are rendered at 2x the PDF point grid - enough resolution for a
# downstream VLM/human to read a figure or table without bloating disk.
DEFAULT_CROP_ZOOM = 2.0

# Blank/low-information thresholds MIRROR BatchProcessor._is_blank_asset (Plan
# v2.9 Phase E, 2026-05-11) so "blank" means the same thing across the codebase:
# near-uniform variance AND near-white or near-black mean.
_BLANK_STD_MAX = 10.0
_BLANK_MEAN_HI = 250.0
_BLANK_MEAN_LO = 5.0

# A normalized bbox edge within this many [0, COORD_SCALE] units of the frame is
# treated as "touching the page boundary". 0.5% of the page. NOTE: because
# vlm_native._project_bbox_to_uir already clamps raw VLM pixel coords to
# [0, COORD_SCALE], the original overflow magnitude is gone by the time a bbox
# reaches here; an edge-touch is the recoverable fingerprint of that clamp. It
# also matches legitimate full-bleed art, so it is a WARN signal, not a hard
# fail. (Precise overflow capture would live upstream in the projection.)
EDGE_TOUCH_EPS = COORD_SCALE * 0.005

# Document-level: fraction of crops flagged as edge-clamped OR low-information
# above which the document earns QA_WARN_CROP_DRIFT.
DEFAULT_CROP_DRIFT_WARN_THRESHOLD = 0.15

CROP_AUDIT_PASS = "CROP_AUDIT_PASS"
QA_WARN_CROP_DRIFT = "QA_WARN_CROP_DRIFT"


@dataclass
class CropHealth:
    """Per-crop health fingerprints for one materialized IMAGE/TABLE asset."""

    asset_ref: str
    page: int
    modality: str
    mean_luminance: float
    std_luminance: float
    is_full_page_fallback: bool
    is_edge_clamped: bool
    is_low_information: bool
    # Provenance of the crop rectangle (B1): "geometric" when cropped from a
    # PyMuPDF-detected object bbox (trusted coordinates), "vlm" when cropped
    # from the VLM-supplied bbox, "full_page" when no usable bbox existed.
    crop_source: str = "vlm"
    # B2: True when the drift-flagged crop was re-extracted to a full-page
    # render before persisting (the garbage crop was NOT written to disk). The
    # detection fingerprints above describe the ORIGINAL drifted crop; this
    # flag records that the persisted asset is the full-page fallback.
    reextracted: bool = False

    @property
    def is_drift_flagged(self) -> bool:
        """Doc-level gate criterion (the user's 'clamped or blank').

        Full-page fallback is reported but NOT counted as drift: it is a
        degraded-but-not-garbage asset (no usable bbox -> whole page), a
        different failure from a wrong/garbage crop.
        """
        return self.is_edge_clamped or self.is_low_information


@dataclass
class CropAuditReport:
    """Document-level aggregate of per-crop health, with the drift gate."""

    crops: List[CropHealth] = field(default_factory=list)
    warn_threshold: float = DEFAULT_CROP_DRIFT_WARN_THRESHOLD

    @property
    def rendered(self) -> int:
        return len(self.crops)

    @property
    def full_page_fallbacks(self) -> int:
        return sum(1 for c in self.crops if c.is_full_page_fallback)

    @property
    def edge_clamped(self) -> int:
        return sum(1 for c in self.crops if c.is_edge_clamped)

    @property
    def low_information(self) -> int:
        return sum(1 for c in self.crops if c.is_low_information)

    @property
    def drift_flagged(self) -> int:
        return sum(1 for c in self.crops if c.is_drift_flagged)

    @property
    def drift_rate(self) -> float:
        return self.drift_flagged / self.rendered if self.rendered else 0.0

    @property
    def exceeds_threshold(self) -> bool:
        return self.rendered > 0 and self.drift_rate > self.warn_threshold

    @property
    def gate_status(self) -> str:
        return QA_WARN_CROP_DRIFT if self.exceeds_threshold else CROP_AUDIT_PASS

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for meta.json. Lists only the suspect crops to stay small."""
        return {
            "gate_status": self.gate_status,
            "rendered": self.rendered,
            "drift_flagged": self.drift_flagged,
            "drift_rate": round(self.drift_rate, 4),
            "warn_threshold": self.warn_threshold,
            "full_page_fallbacks": self.full_page_fallbacks,
            "edge_clamped": self.edge_clamped,
            "low_information": self.low_information,
            "suspect_assets": [
                {
                    "asset_ref": c.asset_ref,
                    "page": c.page,
                    "modality": c.modality,
                    "crop_source": c.crop_source,
                    "mean_luminance": round(c.mean_luminance, 1),
                    "std_luminance": round(c.std_luminance, 1),
                    "is_full_page_fallback": c.is_full_page_fallback,
                    "is_edge_clamped": c.is_edge_clamped,
                    "is_low_information": c.is_low_information,
                    "reextracted": c.reextracted,
                }
                for c in self.crops
                if c.is_drift_flagged or c.is_full_page_fallback
            ],
        }


def _luminance_from_png(png_bytes: bytes) -> Tuple[float, float]:
    """(mean, std) grayscale luminance of a PNG, via PIL convert("L").

    Uses the same grayscale conversion as BatchProcessor._is_blank_asset so the
    blank definition is identical. Returns (255.0, 0.0) - i.e. "blank" - on any
    failure, so an unanalyzable crop is conservatively flagged for review.
    """
    try:
        import io

        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(png_bytes)) as img:
            arr = np.asarray(img.convert("L"))
        return float(arr.mean()), float(arr.std())
    except Exception:  # pragma: no cover - defensive
        return 255.0, 0.0


def _is_low_information(mean_lum: float, std_lum: float) -> bool:
    """Blank/low-information crop fingerprint (shared blank definition)."""
    return std_lum < _BLANK_STD_MAX and (mean_lum > _BLANK_MEAN_HI or mean_lum < _BLANK_MEAN_LO)


def _bbox_touches_edge(bbox: Sequence[float]) -> bool:
    """True if a [0, COORD_SCALE] bbox sits on the page frame (drift fingerprint)."""
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
    return (
        x0 <= EDGE_TOUCH_EPS
        or y0 <= EDGE_TOUCH_EPS
        or x1 >= COORD_SCALE - EDGE_TOUCH_EPS
        or y1 >= COORD_SCALE - EDGE_TOUCH_EPS
    )


def _geometric_candidates(page: "fitz.Page", modality: Modality) -> List["fitz.Rect"]:
    """Detected-object bboxes (page coords) for B1 deterministic cropping.

    The VLM is good at semantics, bad at coordinates, so when the page yields
    its own detectable objects we crop from THOSE rather than the hallucinated
    VLM bbox. IMAGE -> embedded raster rects (``get_image_info``); TABLE ->
    ``find_tables`` bboxes. Returns only non-degenerate rects; [] when the page
    has no detectable object of that kind (then the VLM bbox is used).
    """
    rects: List["fitz.Rect"] = []
    try:
        if modality == Modality.IMAGE:
            for info in page.get_image_info():
                bb = info.get("bbox")
                if bb:
                    rects.append(fitz.Rect(bb))
        elif modality == Modality.TABLE:
            for table in page.find_tables().tables:
                if table.bbox:
                    rects.append(fitz.Rect(table.bbox))
    except Exception:  # pragma: no cover - defensive: PyMuPDF API drift
        return []
    return [r for r in rects if r.width > 2.0 and r.height > 2.0]


def _pick_geometric_clip(
    candidates: List["fitz.Rect"],
    consumed: set,
    vlm_clip: "Optional[fitz.Rect]",
) -> "Optional[Tuple[int, fitz.Rect]]":
    """Choose the best unconsumed detected object for this chunk.

    Prefer the candidate that overlaps the VLM's (rough, semantically-placed)
    bbox most; with no overlap or no VLM bbox, fall back to the largest
    remaining object (most likely the primary figure/table). Returns
    ``(index, rect)`` or ``None`` when every candidate is already consumed.
    """
    avail = [(i, r) for i, r in enumerate(candidates) if i not in consumed]
    if not avail:
        return None

    def _overlap(rect: "fitz.Rect") -> float:
        if vlm_clip is None:
            return 0.0
        inter = rect & vlm_clip
        return 0.0 if inter.is_empty else inter.width * inter.height

    if vlm_clip is not None:
        best_idx, best_rect = max(avail, key=lambda ir: _overlap(ir[1]))
        if _overlap(best_rect) > 0.0:
            return best_idx, best_rect
    # No overlap signal: take the largest remaining real object.
    best_idx, best_rect = max(avail, key=lambda ir: ir[1].width * ir[1].height)
    return best_idx, best_rect


def materialize_visual_assets(
    uir_chunks: Sequence[Any],
    source_pdf: Path | str,
    assets_dir: Path | str,
    *,
    doc_hash: str,
    page_offset: int = 0,
    asset_ref_prefix: str = "assets",
    zoom: float = DEFAULT_CROP_ZOOM,
    warn_threshold: float = DEFAULT_CROP_DRIFT_WARN_THRESHOLD,
) -> CropAuditReport:
    """Render IMAGE/TABLE region crops, set ``asset_ref``, and audit each crop.

    Args:
        uir_chunks: UIRChunk objects from the chunker (mutated in place).
        source_pdf: PDF whose pages the bboxes index (batch-local for the
            batch path; the whole doc for the soak path).
        assets_dir: Directory to write PNG crops into (created if missing).
        doc_hash: Stable doc id used in the asset filename.
        page_offset: Added to each chunk's local page number to form the
            absolute page used in the filename (0 for whole-doc runs).
        asset_ref_prefix: Relative prefix stored in ``asset_ref`` (the path is
            resolved against the output dir downstream).
        zoom: Render scale factor.
        warn_threshold: Drift fraction above which the report gates
            ``QA_WARN_CROP_DRIFT``.

    Returns:
        A :class:`CropAuditReport` with one :class:`CropHealth` per rendered
        crop and the document-level drift gate. Only IMAGE/TABLE chunks that
        lack an ``asset_ref`` are processed; a chunk with no usable bbox falls
        back to a full-page render so it still satisfies QA-CHECK-05.
    """
    report = CropAuditReport(warn_threshold=warn_threshold)

    visual = [
        c
        for c in uir_chunks
        if getattr(c, "modality", None) in (Modality.IMAGE, Modality.TABLE)
        and not getattr(c, "asset_ref", None)
    ]
    if not visual:
        return report

    try:
        doc = fitz.open(str(source_pdf))
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        logger.warning("[V3-ASSET] could not open PDF for crop render: %s", exc)
        return report

    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    per_page_idx: Dict[int, int] = {}
    # B1: per-page caches so detected objects are computed once and distributed
    # across the page's visual chunks (one object is not cropped twice when the
    # page offers enough of them).
    geo_cache: Dict[Tuple[int, str], List["fitz.Rect"]] = {}
    geo_consumed: Dict[Tuple[int, str], set] = {}
    try:
        for c in visual:
            loc = getattr(c, "locator", None)
            local_page = loc.page_number if loc and loc.page_number else 1
            page_index = local_page - 1
            if page_index < 0 or page_index >= doc.page_count:
                continue
            page = doc[page_index]
            pw, ph = float(page.rect.width), float(page.rect.height)

            # VLM-supplied bbox -> page-coord clip (the rough, semantic placement).
            vlm_clip = None
            vlm_edge = False
            bbox = loc.bbox if loc else None
            if bbox and len(bbox) == 4:
                x0 = max(0.0, min(pw, bbox[0] / COORD_SCALE * pw))
                y0 = max(0.0, min(ph, bbox[1] / COORD_SCALE * ph))
                x1 = max(0.0, min(pw, bbox[2] / COORD_SCALE * pw))
                y1 = max(0.0, min(ph, bbox[3] / COORD_SCALE * ph))
                if (x1 - x0) > 2.0 and (y1 - y0) > 2.0:
                    vlm_clip = fitz.Rect(x0, y0, x1, y1)
                    # Edge-touch is only meaningful for a usable bbox; a
                    # degenerate one is reported as a full-page fallback instead.
                    vlm_edge = _bbox_touches_edge(bbox)

            # B1: prefer a deterministic geometric bbox when the page yields a
            # detectable object. The VLM is bad at coordinates, so a real
            # object rect beats a (possibly hallucinated) VLM bbox; trust VLM
            # coords only when no geometric source exists.
            cache_key = (page_index, "table" if c.modality == Modality.TABLE else "image")
            if cache_key not in geo_cache:
                geo_cache[cache_key] = _geometric_candidates(page, c.modality)
                geo_consumed[cache_key] = set()
            picked = _pick_geometric_clip(geo_cache[cache_key], geo_consumed[cache_key], vlm_clip)

            if picked is not None:
                geo_idx, clip = picked
                geo_consumed[cache_key].add(geo_idx)
                # Trusted source: not a clamp artifact.
                is_edge_clamped = False
                crop_source = "geometric"
            else:
                clip = vlm_clip
                is_edge_clamped = vlm_edge
                crop_source = "vlm" if clip is not None else "full_page"

            is_full_page_fallback = clip is None

            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
            except Exception as exc:  # pragma: no cover - defensive render guard
                logger.warning("[V3-ASSET] pixmap render failed (page %d): %s", local_page, exc)
                continue

            abs_page = local_page + page_offset
            mod = "table" if c.modality == Modality.TABLE else "image"
            idx = per_page_idx.get(abs_page, 0)
            per_page_idx[abs_page] = idx + 1
            fname = f"{doc_hash or 'doc'}_{abs_page:04d}_{mod}_{idx:03d}.png"
            out_path = assets_dir / fname

            try:
                png_bytes = pix.tobytes("png")
                mean_lum, std_lum = _luminance_from_png(png_bytes)
            except Exception as exc:  # pragma: no cover - defensive encode guard
                # MuPDF's PNG band-writer raises (code=4: Invalid bandwriter
                # header dimensions/setup) on a degenerate crop pixmap. The crop
                # is cosmetic, but the chunk's asset_ref is NOT: an IMAGE/TABLE
                # chunk with no asset_ref fails the QA-CHECK-05 contract in
                # from_uir, which discards the whole batch's text. Fall back to a
                # full-page render (an honest degraded asset) so the chunk still
                # gets a valid asset_ref instead of poisoning the batch.
                logger.warning(
                    "[V3-ASSET] crop PNG encode failed (page %d): %s; "
                    "falling back to full-page render",
                    local_page,
                    exc,
                )
                try:
                    png_bytes = page.get_pixmap(
                        matrix=fitz.Matrix(zoom, zoom), clip=None
                    ).tobytes("png")
                    mean_lum, std_lum = _luminance_from_png(png_bytes)
                except Exception as exc2:  # pragma: no cover - page unrenderable
                    logger.warning(
                        "[V3-ASSET] full-page fallback render also failed "
                        "(page %d): %s; skipping crop",
                        local_page,
                        exc2,
                    )
                    continue
                is_full_page_fallback = True
                is_edge_clamped = False
                crop_source = "full_page"
            is_low_information = _is_low_information(mean_lum, std_lum)

            # B2: crop-audit as a re-extraction trigger (fail-open). The audit
            # fingerprints above still RECORD that the VLM crop drifted (blank /
            # edge-clamped) - that detection signal is preserved for telemetry
            # and the doc-level drift gate. But a garbage crop must never be the
            # PERSISTED asset, so re-render the FULL PAGE and write that instead:
            # a degraded-but-honest asset always beats a whitespace crop.
            # Geometric crops are trusted coordinates and never re-triggered (B1
            # already preferred a deterministic bbox; B2 covers the residue where
            # no detectable object existed).
            reextracted = False
            if crop_source != "geometric" and (is_edge_clamped or is_low_information):
                try:
                    full_png = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=None).tobytes(
                        "png"
                    )
                except Exception as exc:  # pragma: no cover - defensive render guard
                    logger.warning(
                        "[V3-ASSET] B2 re-extraction render failed (page %d): %s",
                        local_page,
                        exc,
                    )
                else:
                    png_bytes = full_png
                    reextracted = True
                    logger.info(
                        "[V3-ASSET] B2 re-extraction: page %d crop drift-flagged "
                        "(%s); persisted a full-page render instead of the garbage crop",
                        local_page,
                        "blank" if is_low_information else "edge-clamped",
                    )

            try:
                out_path.write_bytes(png_bytes)
            except Exception as exc:  # pragma: no cover - defensive write guard
                logger.warning("[V3-ASSET] asset save failed (%s): %s", fname, exc)
                continue

            asset_ref = f"{asset_ref_prefix}/{fname}"
            c.asset_ref = asset_ref
            report.crops.append(
                CropHealth(
                    asset_ref=asset_ref,
                    page=abs_page,
                    modality=mod,
                    mean_luminance=mean_lum,
                    std_luminance=std_lum,
                    is_full_page_fallback=is_full_page_fallback,
                    is_edge_clamped=is_edge_clamped,
                    is_low_information=is_low_information,
                    crop_source=crop_source,
                    reextracted=reextracted,
                )
            )
    finally:
        doc.close()

    if report.rendered:
        logger.info(
            "[V3-ASSET] Rendered %d region crop(s); crop-audit=%s "
            "(drift=%d/%d=%.0f%%, fallback=%d)",
            report.rendered,
            report.gate_status,
            report.drift_flagged,
            report.rendered,
            report.drift_rate * 100,
            report.full_page_fallbacks,
        )
    return report
