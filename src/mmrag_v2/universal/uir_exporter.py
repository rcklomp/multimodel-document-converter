"""V3.0 UIR JSONL exporter — Phase A task A2 (shim path).

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md Phase A scope-negotiation
option (b) UIR-shim fallback.

This module ships the v3.0 UIR contract surface (UIRChunk emission to
JSONL) without rewriting the v2.16 chunker / mapper / processor /
batch_processor hot paths. The shim works by reading an existing v2.X
ingestion.jsonl, projecting each chunk through `v2x_to_v3_mapper`
(proven lossless by A0 PASS at identity ratio 1.0000 on
ATZ_Elektronik_German), and writing a parallel v3.0 UIR JSONL.

Production v2.X output is unchanged → zero blast radius for existing
consumers (Qdrant ingestion, RAG app). New v3.0 consumers (Phase B
sanitization, Phase C visual retrieval, Phase D modality-aware judges)
read from the UIR JSONL produced here. The full chunker rewrite
(processor.py + batch_processor.py touching ~1,755 LOC) is rebooked
to v3.0.2 under a dedicated cycle plan; see
`docs/PHASE_A_SCOPE_NEGOTIATION.md` for the operator-invoked decision.

Acceptance: identity-half gate passes at ratio ≥0.95 (Charter §3.2)
when run on the round-trip baseline-vs-UIR-JSONL projection of a real
v2.16 corpus. By construction of the mapper this should be 1.0000.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .intermediate import (
    ConfidenceBreakdown,
    CoordinateFrame,
    Locator,
    LocatorType,
    Modality,
    StructuralFlag,
    UIRChunk,
)
from .v2x_to_v3_mapper import (
    map_v2x_corpus_to_v3,
    normalize_ocr_confidence,
    uirchunk_to_identity_projection,
)

logger = logging.getLogger(__name__)


# Stamped on every emitted UIR JSONL header so consumers know which
# code path produced the file. v2.X tooling stamps v2.X schema_version
# on its records; this is the v3.0-shim analogue.
SCHEMA_VERSION_V3_SHIM = "3.0.0-shim"
EXPORT_SOURCE_V2X_SHIM = "v2x_to_v3_mapper-shim"


@dataclass
class ExportReport:
    """Result of exporting one v2.X ingestion.jsonl to v3.0 UIR JSONL."""

    source_path: str
    output_path: str
    doc_id: str
    v2x_chunk_count: int
    uir_chunk_count: int
    mapper_errors: List[str] = field(default_factory=list)
    identity_ratio: Optional[float] = None  # None when --no-verify
    identity_matched: Optional[int] = None
    identity_differing: Optional[int] = None
    identity_missing: Optional[int] = None
    identity_new: Optional[int] = None

    @property
    def identity_passes(self) -> bool:
        """Per Charter §3.2 identity-half row 1: ≥95% match."""
        if self.identity_ratio is None:
            return False
        return self.identity_ratio >= 0.95


def serialize_uirchunk_to_jsonl_dict(
    chunk: UIRChunk,
    *,
    doc_id: str,
    source_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Render a UIRChunk into a JSON-serializable dict for the UIR JSONL.

    Enum values are exported as their `.value` strings (Modality,
    LocatorType, CoordinateFrame, StructuralFlag). Sets are sorted for
    deterministic output. The `uir_version` and `extraction_engine_version`
    fields ride along so downstream consumers can detect mismatched
    contract versions.

    Doc-level identity fields (`doc_id`, `source_file`) live on the
    parent UniversalDocument in the v3.0 contract; the shim attaches
    them at JSONL row level for compatibility with consumers that
    expect a flat record per line.
    """
    payload: Dict[str, Any] = {
        "doc_id": doc_id,
        "modality": chunk.modality.value,
        "content": chunk.content,
        "locator": {
            "type": chunk.locator.type.value,
            "page_number": chunk.locator.page_number,
            "bbox": list(chunk.locator.bbox) if chunk.locator.bbox else None,
            "coordinate_frame": chunk.locator.coordinate_frame.value,
            "path": chunk.locator.path,
        },
        "confidence": {
            "layout_confidence": chunk.confidence.layout_confidence,
            "text_extraction_confidence": chunk.confidence.text_extraction_confidence,
            "ocr_confidence": chunk.confidence.ocr_confidence,
            "classification_confidence": chunk.confidence.classification_confidence,
            "applicable": sorted(chunk.confidence.applicable),
        },
        "extraction_method": chunk.extraction_method,
        "extraction_engine_version": chunk.extraction_engine_version,
        "extraction_warnings": [
            {
                "code": w.code,
                "severity": w.severity,
                "message": w.message,
                "source_element_id": w.source_element_id,
            }
            for w in chunk.extraction_warnings
        ],
        "structural_flags": sorted(f.value for f in chunk.structural_flags),
        "source_element_ids": list(chunk.source_element_ids),
        "asset_ref": chunk.asset_ref,
        "lang": chunk.lang,
        "reading_order": chunk.reading_order,
        "original_vlm_type": chunk.original_vlm_type,
        "parent_element_id": chunk.parent_element_id,
        "parent_heading": chunk.parent_heading,
        "continuation_group_id": chunk.continuation_group_id,
        "uir_version": chunk.uir_version,
        "sanitization_status": chunk.sanitization_status,
        "schema_version": SCHEMA_VERSION_V3_SHIM,
    }
    if source_file is not None:
        payload["source_file"] = source_file
    return payload


def _load_v2x_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load v2.X JSONL into (chunk_records, header_metadata_record_or_None).

    The v2.X JSONL convention puts an optional metadata record (no
    `chunk_id` field) at the head of the file. The shim preserves the
    same convention by emitting its own UIR header record.
    """
    chunks: List[Dict[str, Any]] = []
    header: Optional[Dict[str, Any]] = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("chunk_id"):
                chunks.append(rec)
            elif header is None:
                # First non-chunk record is the metadata header.
                header = rec
    return chunks, header


def _infer_doc_id(chunks: List[Dict[str, Any]], header: Optional[Dict[str, Any]]) -> str:
    """Recover the canonical doc_id from JSONL records.

    v2.X chunks carry doc_id on every record; the metadata header may
    carry it under `doc_id` or `document.doc_id`. If neither is present
    raises ValueError — the v3 contract requires doc_id for the parent
    UniversalDocument identity.
    """
    for ch in chunks:
        doc_id = ch.get("doc_id")
        if doc_id:
            return str(doc_id)
    if header is not None:
        if header.get("doc_id"):
            return str(header["doc_id"])
        doc = header.get("document")
        if isinstance(doc, dict) and doc.get("doc_id"):
            return str(doc["doc_id"])
    raise ValueError("Cannot infer doc_id from v2.X JSONL: no chunk or header carries one")


def _build_uir_header(
    *,
    doc_id: str,
    source_path: Path,
    v2x_chunk_count: int,
) -> Dict[str, Any]:
    """Construct the first-line metadata record for the UIR JSONL output."""
    return {
        "schema_version": SCHEMA_VERSION_V3_SHIM,
        "export_source": EXPORT_SOURCE_V2X_SHIM,
        "uir_version": "3.0",
        "doc_id": doc_id,
        "v2x_source_path": str(source_path),
        "v2x_chunk_count": v2x_chunk_count,
        # Charter §Phase A scope-negotiation: this header tells consumers
        # the file was produced by the shim path, not the full v3.0
        # ElementProcessor rewrite. v3.0.2 will emit a different header
        # (export_source="elementprocessor-v3") when the rewrite lands.
    }


def export_v2x_jsonl_to_uir(
    *,
    input_path: Path,
    output_path: Path,
    doc_id: Optional[str] = None,
    verify_identity: bool = True,
) -> ExportReport:
    """Read v2.X ingestion.jsonl, project to UIRChunks, write UIR JSONL.

    Identity-gate verification (Charter §3.2 identity half ≥0.95) runs
    by default; set `verify_identity=False` to skip (useful in batch
    pipelines that verify separately).

    Raises ValueError on:
      - Mapper errors during v2.X → UIR projection (any single chunk
        failing to project)
      - identity_ratio < 0.95 when verify_identity=True
        (the shim contract is "lossless" — a sub-threshold result means
        the v2.X JSONL has a chunk shape the mapper can't handle, which
        is a contract bug to fix in v2x_to_v3_mapper, not a deferral)
    """
    if not input_path.exists():
        raise FileNotFoundError(f"v2.X ingestion.jsonl not found: {input_path}")

    chunks, header = _load_v2x_jsonl(input_path)
    if not chunks:
        raise ValueError(f"v2.X JSONL contains no chunk records: {input_path}")

    if doc_id is None:
        doc_id = _infer_doc_id(chunks, header)

    mapper_errors: List[str] = []
    try:
        uirchunks = map_v2x_corpus_to_v3(chunks)
    except ValueError as exc:
        mapper_errors.append(str(exc))
        raise

    source_file = None
    if header is not None:
        source_file = header.get("source_file") or header.get("source_path")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        out.write(
            json.dumps(
                _build_uir_header(
                    doc_id=doc_id,
                    source_path=input_path,
                    v2x_chunk_count=len(chunks),
                )
            )
            + "\n"
        )
        for chunk in uirchunks:
            payload = serialize_uirchunk_to_jsonl_dict(
                chunk,
                doc_id=doc_id,
                source_file=source_file,
            )
            out.write(json.dumps(payload) + "\n")

    report = ExportReport(
        source_path=str(input_path),
        output_path=str(output_path),
        doc_id=doc_id,
        v2x_chunk_count=len(chunks),
        uir_chunk_count=len(uirchunks),
        mapper_errors=mapper_errors,
    )

    if verify_identity:
        from ..v3_identity_gate import compare_for_identity

        baseline = [_baseline_identity_projection(c, doc_id) for c in chunks]
        candidate = [uirchunk_to_identity_projection(c, doc_id) for c in uirchunks]
        gate = compare_for_identity(
            baseline_chunks=baseline,
            candidate_chunks=candidate,
        )
        report.identity_ratio = gate.identity_ratio
        report.identity_matched = gate.matched
        report.identity_differing = len(gate.differing_baseline_ids)
        report.identity_missing = len(gate.missing_baseline_ids)
        report.identity_new = len(gate.new_candidate_ids)

        if not gate.passes(threshold=0.95):
            raise ValueError(
                f"UIR-shim export failed identity-half gate: "
                f"ratio {gate.identity_ratio:.4f} < 0.95 threshold "
                f"(baseline={len(baseline)}, candidate={len(candidate)}, "
                f"matched={gate.matched}). The shim is supposed to be "
                f"lossless — a sub-threshold result means v2x_to_v3_mapper "
                f"cannot handle some chunk shape in this corpus and the "
                f"mapper is the bug, not the gate."
            )

    return report


def _baseline_identity_projection(
    chunk: Dict[str, Any],
    doc_id: str,
) -> Dict[str, Any]:
    """Same identity projection v3_a0_atz_spike.baseline_projection uses.

    Kept in sync with `v2x_to_v3_mapper.uirchunk_to_identity_projection`
    so the gate compares apples-to-apples. If you change one, change
    both; the A0 report flagged this as work for A1+ (consolidate into
    one canonical projection in v3_identity_gate). The shim ships with
    the dual-projection in place for now; consolidation is a v3.0.1
    follow-up.
    """
    metadata = chunk.get("metadata") or {}
    spatial = metadata.get("spatial") or {}
    hierarchy = metadata.get("hierarchy") or {}
    payload: Dict[str, Any] = {
        "doc_id": doc_id,
        "content": chunk.get("content", ""),
        "modality": chunk.get("modality"),
        "structural_flags": [],
    }
    page_number = metadata.get("page_number")
    if page_number is not None:
        payload["page_number"] = int(page_number)
    bbox = spatial.get("bbox") if isinstance(spatial, dict) else None
    if bbox is not None:
        payload["bbox"] = [int(round(float(c))) for c in bbox]
    parent_heading = hierarchy.get("parent_heading") if isinstance(hierarchy, dict) else None
    if parent_heading is not None:
        payload["parent_heading"] = parent_heading
    # Use the same categorical-aware normalizer the v3 mapper uses so the
    # gate compares like for like; otherwise legacy "high"/"medium"/"low"
    # values would crash the baseline projection while the mapper has
    # already coerced them to floats.
    ocr_conf = normalize_ocr_confidence(metadata.get("ocr_confidence"))
    if ocr_conf is not None:
        payload["confidence_breakdown"] = {"ocr_confidence": ocr_conf}
    return payload


def parse_uir_jsonl_record(record: Dict[str, Any]) -> UIRChunk:
    """Reverse of `serialize_uirchunk_to_jsonl_dict`: parse one row back.

    Used by tests + v3 consumers that want to reconstruct UIRChunk
    objects from the shim's output. Header records (no `modality` field)
    must be filtered out by the caller before this is invoked.
    """
    loc = record["locator"]
    locator = Locator(
        type=LocatorType(loc["type"]),
        page_number=loc.get("page_number"),
        bbox=list(loc["bbox"]) if loc.get("bbox") else None,
        coordinate_frame=CoordinateFrame(loc["coordinate_frame"]),
        path=loc.get("path"),
    )
    conf = record["confidence"]
    confidence = ConfidenceBreakdown(
        layout_confidence=conf.get("layout_confidence"),
        text_extraction_confidence=conf.get("text_extraction_confidence"),
        ocr_confidence=conf.get("ocr_confidence"),
        classification_confidence=conf.get("classification_confidence"),
        applicable=set(conf.get("applicable", [])),
    )
    structural_flags = {StructuralFlag(v) for v in record.get("structural_flags", [])}
    return UIRChunk(
        modality=Modality(record["modality"]),
        content=record["content"],
        locator=locator,
        confidence=confidence,
        extraction_method=record["extraction_method"],
        extraction_engine_version=record["extraction_engine_version"],
        structural_flags=structural_flags,
        source_element_ids=list(record.get("source_element_ids", [])),
        asset_ref=record.get("asset_ref"),
        lang=record.get("lang"),
        reading_order=record.get("reading_order"),
        original_vlm_type=record.get("original_vlm_type"),
        parent_element_id=record.get("parent_element_id"),
        parent_heading=record.get("parent_heading"),
        continuation_group_id=record.get("continuation_group_id"),
        uir_version=record.get("uir_version", "3.0"),
        sanitization_status=record.get("sanitization_status", "not_applied"),
    )


def report_to_dict(report: ExportReport) -> Dict[str, Any]:
    """Convenience for CLI / test serialization."""
    return asdict(report)
