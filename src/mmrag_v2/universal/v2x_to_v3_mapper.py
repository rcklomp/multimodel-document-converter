"""v2.X IngestionChunk -> v3.0 UIRChunk projection mapper.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md Phase A task A0 (per-doc
spike on ATZ_Elektronik_German).

This mapper projects an existing v2.X ingestion.jsonl chunk record
into a v3.0 UIRChunk dataclass instance. It is a STRUCTURAL projection
only — no re-extraction, no chunking change, no semantic transformation.
The intent is to demonstrate that the v3.0 UIR contract can carry v2.X
content without information loss before committing to the A2 chunker /
mapper / serializer rewrite.

This is NOT a production code path. It is an A0 spike helper that
will be subsumed by A2's full v3.0 ElementProcessor when that lands.

v2.X -> v3.0 field map:

    v2.X chunk[*]                         v3.0 UIRChunk field
    -----------------------------------   --------------------------
    chunk_id                              (not stored — identity gate
                                          handles chunk_id rewrites
                                          via the per-phase rewrite map)
    doc_id                                (not stored on UIRChunk — lives
                                          on the parent UniversalDocument;
                                          for A0 spike we keep it in the
                                          projection's identity key)
    modality (str)                        modality (Modality enum)
    content (str)                         content (str)
    chunk_type (str)                      structural_flags (Set[StructuralFlag])
                                          — chunk_type stays as ChunkType
                                          on the eventual IngestionChunk,
                                          derived one-way from Modality
                                          per Charter §7.1
    metadata.page_number (int)            locator.page_number
    metadata.spatial.bbox ([x1,y1,x2,y2]) locator.bbox (REQ-COORD-01
                                          [0,1000])
    metadata.spatial.page_width/height    (used to validate bbox is
                                          already normalized; v2.X
                                          stores bbox in 0-1000 already)
    metadata.extraction_method            extraction_method
    metadata.ocr_confidence               confidence.ocr_confidence
    metadata.hierarchy.parent_heading     parent_heading
    schema_version                        (drops to the metadata-only
                                          field on the eventual output;
                                          not part of UIRChunk per §8.2)
    asset_ref                             asset_ref
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .intermediate import (
    ConfidenceBreakdown,
    CoordinateFrame,
    Locator,
    LocatorType,
    Modality,
    UIRChunk,
)


logger = logging.getLogger(__name__)


# v2.X modality strings -> v3.0 Modality enum.
# v2.X has only TEXT / IMAGE / TABLE (matches old ElementType).
# CODE and FORM widening happens at A2 (chunker rewrite); the A0
# spike preserves the v2.X 3-value mapping.
_MODALITY_MAP = {
    "text": Modality.TEXT,
    "image": Modality.IMAGE,
    "table": Modality.TABLE,
}


# v2.X extraction_method strings preserved as-is (the v3.0 UIRChunk
# field is typed as str, not enum, so any v2.X string round-trips).
# Charter §3.2 specifies an indicative vocabulary
# ("docling_direct" | "ocr_tesseract" | "ocr_doctr" | "vlm_enrichment")
# but v2.X uses different identifiers ("hybrid_chunker", "ocr",
# "vlm"). The A0 mapper keeps v2.X strings verbatim so the identity
# gate sees the same value on both sides; the v2.X -> v3.0
# extraction-method-vocabulary normalization is an A2 concern.


def map_v2x_to_v3_uirchunk(v2x_chunk: Dict[str, Any]) -> UIRChunk:
    """Project a single v2.X IngestionChunk dict to a v3.0 UIRChunk.

    Raises ValueError if the v2.X record is missing fields the v3.0
    contract requires (modality, content, page_number, bbox).
    """
    # --- modality ---
    v2_modality = v2x_chunk.get("modality")
    if v2_modality not in _MODALITY_MAP:
        raise ValueError(
            f"v2.X chunk modality {v2_modality!r} has no v3.0 Modality "
            f"mapping; valid: {sorted(_MODALITY_MAP)}"
        )
    modality = _MODALITY_MAP[v2_modality]

    # --- content ---
    content = v2x_chunk.get("content")
    if content is None:
        raise ValueError(
            f"v2.X chunk {v2x_chunk.get('chunk_id')!r} missing content"
        )

    # --- locator ---
    metadata = v2x_chunk.get("metadata") or {}
    page_number = metadata.get("page_number")
    spatial = metadata.get("spatial") or {}
    bbox_raw = spatial.get("bbox") if isinstance(spatial, dict) else None

    if bbox_raw is None or page_number is None:
        # Allow image-modality chunks without bbox to fall back to a
        # FLOW_OFFSET locator at the page level. v2.X image chunks
        # occasionally omit a precise bbox.
        if modality == Modality.IMAGE and page_number is not None:
            locator = Locator(
                type=LocatorType.FLOW_OFFSET,
                page_number=page_number,
                coordinate_frame=CoordinateFrame.UNKNOWN,
                path=f"page:{page_number}:image",
            )
        else:
            raise ValueError(
                f"v2.X chunk {v2x_chunk.get('chunk_id')!r} missing "
                "page_number or bbox required for BBOX locator"
            )
    else:
        # v2.X already stores bbox in [0,1000] normalized integers
        # (AGENTS.md REQ-COORD-01). Defensive int() coercion in case
        # of stray floats from older pipelines.
        bbox_int = [int(round(float(c))) for c in bbox_raw]
        locator = Locator(
            type=LocatorType.BBOX,
            bbox=bbox_int,
            page_number=int(page_number),
            coordinate_frame=CoordinateFrame.PDF_PAGE_PORTRAIT,
        )

    # --- confidence ---
    ocr_conf = metadata.get("ocr_confidence")
    confidence = ConfidenceBreakdown(
        ocr_confidence=float(ocr_conf) if ocr_conf is not None else None,
    )

    # --- structural / hierarchy ---
    hierarchy = metadata.get("hierarchy") or {}
    parent_heading = hierarchy.get("parent_heading") if isinstance(hierarchy, dict) else None

    # --- extraction provenance ---
    extraction_method = metadata.get("extraction_method") or "hybrid_chunker"
    # v2.X doesn't carry an explicit engine version; the v3.0 contract
    # requires the field, so we stamp "v2.X" so it's identifiable in
    # downstream traces as "this was migrated from v2.X via the A0 mapper"
    # rather than "this came out of the v3.0 ElementProcessor".
    extraction_engine_version = "docling-2.86.0-via-v2x-mapper"

    asset_ref = v2x_chunk.get("asset_ref")

    return UIRChunk(
        modality=modality,
        content=content,
        locator=locator,
        confidence=confidence,
        extraction_method=extraction_method,
        extraction_engine_version=extraction_engine_version,
        parent_heading=parent_heading,
        asset_ref=asset_ref,
        # source_element_ids: v2.X chunk_id is the closest analog; record
        # it so the migration is auditable (every UIRChunk knows which
        # v2.X chunk it came from).
        source_element_ids=[v2x_chunk["chunk_id"]] if v2x_chunk.get("chunk_id") else [],
    )


def map_v2x_corpus_to_v3(
    v2x_records: List[Dict[str, Any]],
) -> List[UIRChunk]:
    """Project an iterable of v2.X chunk records to v3.0 UIRChunks.

    Skips records that have no chunk_id (the ingestion.jsonl
    convention puts a metadata record at the head of the file).
    Errors on individual chunks are re-raised with the record's
    chunk_id for traceability.
    """
    out: List[UIRChunk] = []
    for record in v2x_records:
        if not record.get("chunk_id"):
            continue
        try:
            out.append(map_v2x_to_v3_uirchunk(record))
        except ValueError as exc:
            raise ValueError(
                f"v2.X -> v3.0 mapping failed for "
                f"chunk_id={record.get('chunk_id')!r}: {exc}"
            ) from exc
    return out


def uirchunk_to_identity_projection(
    chunk: UIRChunk,
    doc_id: str,
) -> Dict[str, Any]:
    """Project a UIRChunk into a dict the v3_identity_gate can ingest.

    The v3_identity_gate expects v2.X-shape chunk dicts; this function
    bridges by producing the identity-relevant subset (per Charter §8.2).
    """
    payload: Dict[str, Any] = {
        "doc_id": doc_id,
        "content": chunk.content,
        "modality": chunk.modality.value,
        "structural_flags": sorted(f.value for f in chunk.structural_flags),
    }
    if chunk.locator.type == LocatorType.BBOX:
        payload["page_number"] = chunk.locator.page_number
        payload["bbox"] = chunk.locator.bbox
    elif chunk.locator.page_number is not None:
        payload["page_number"] = chunk.locator.page_number
    if chunk.parent_heading is not None:
        payload["parent_heading"] = chunk.parent_heading
    # confidence_breakdown — wrap as v2.X-shape dict so identity gate
    # applies its ±0.01 tolerance rule consistently.
    cb = {}
    for field_name in (
        "layout_confidence",
        "text_extraction_confidence",
        "ocr_confidence",
        "classification_confidence",
    ):
        v = getattr(chunk.confidence, field_name)
        if v is not None:
            cb[field_name] = v
    if cb:
        payload["confidence_breakdown"] = cb
    return payload
