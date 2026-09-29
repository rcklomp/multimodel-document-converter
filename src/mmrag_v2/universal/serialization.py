"""JSON (de)serialization of the UIR ``UniversalDocument``.

PLAN_QUALITY_REMEDIATION_V1 WP-0.5. Extraction through a paid, non-deterministic VLM is
not reproducible (two live IRJET extractions were 84% identical chunk-for-chunk), so a
chunker change cannot be A/B-tested on a re-run. Dumping the extracted UIR once lets any
chunker version be replayed deterministically on the SAME extraction.

Scope and limits:
* Pure data: no network, no eval. Enums are stored by value; a numpy image (``raw_image``)
  is NOT preserved (only a ``has_raw_image`` flag), because it is heavy and no chunker
  reads it.
* Values inside ``metadata`` dictionaries are coerced to JSON types; tuples come back as
  lists. A round trip is therefore exact for documents whose metadata already holds JSON
  types, which is what the extraction engines emit.
* The dump is opt-in (``MMRAG_DUMP_UIR=<dir>``); with the variable unset nothing is written.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .intermediate import (
    BoundingBox,
    DocumentMetadata,
    Element,
    ElementType,
    ExtractionMethod,
    PageClassification,
    UniversalDocument,
    UniversalPage,
)

UIR_DUMP_SCHEMA = 1


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(v) for v in value), key=repr)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    item = getattr(value, "item", None)  # numpy scalar
    if callable(item):
        try:
            return item()
        except Exception:  # noqa: BLE001 - keep the dump usable
            pass
    return repr(value)


def _dt(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value) if value else None


def uir_to_dict(doc: UniversalDocument) -> Dict[str, Any]:
    pages = []
    for page in doc.pages:
        elements = []
        for e in page.elements:
            elements.append(
                {
                    "type": e.type.value,
                    "content": e.content,
                    "bbox": e.bbox.to_list() if e.bbox else None,
                    "confidence": e.confidence,
                    "extraction_method": e.extraction_method.value,
                    "element_index": e.element_index,
                    "source_label": e.source_label,
                    "metadata": _jsonable(e.metadata),
                    "has_raw_image": e.raw_image is not None,
                }
            )
        pages.append(
            {
                "page_number": page.page_number,
                "elements": elements,
                "classification": page.classification.value,
                "dimensions": list(page.dimensions),
                "text_density": page.text_density,
                "avg_confidence": page.avg_confidence,
            }
        )
    md = doc.metadata
    return {
        "schema": UIR_DUMP_SCHEMA,
        "doc_id": doc.doc_id,
        "source_file": doc.source_file,
        "file_type": doc.file_type,
        "total_pages": doc.total_pages,
        "created_at": doc.created_at.isoformat(),
        "metadata": {
            "title": md.title,
            "author": md.author,
            "creation_date": md.creation_date.isoformat() if md.creation_date else None,
            "modification_date": md.modification_date.isoformat() if md.modification_date else None,
            "page_count": md.page_count,
            "file_size_bytes": md.file_size_bytes,
            "has_text_layer": md.has_text_layer,
            "has_images": md.has_images,
            "language": md.language,
            "extra": _jsonable(md.extra),
        },
        "pages": pages,
    }


def uir_from_dict(data: Dict[str, Any]) -> UniversalDocument:
    if data.get("schema") != UIR_DUMP_SCHEMA:
        raise ValueError(f"unsupported UIR dump schema: {data.get('schema')!r}")
    pages = []
    for p in data["pages"]:
        elements = [
            Element(
                type=ElementType(e["type"]),
                content=e["content"],
                bbox=BoundingBox(*e["bbox"]) if e["bbox"] is not None else None,
                confidence=e["confidence"],
                raw_image=None,
                extraction_method=ExtractionMethod(e["extraction_method"]),
                element_index=e["element_index"],
                source_label=e["source_label"],
                metadata=dict(e["metadata"]),
            )
            for e in p["elements"]
        ]
        pages.append(
            UniversalPage(
                page_number=p["page_number"],
                elements=elements,
                classification=PageClassification(p["classification"]),
                dimensions=tuple(p["dimensions"]),  # type: ignore[arg-type]
                raw_image=None,
                text_density=p["text_density"],
                avg_confidence=p["avg_confidence"],
            )
        )
    m = data["metadata"]
    metadata = DocumentMetadata(
        title=m["title"],
        author=m["author"],
        creation_date=_dt(m["creation_date"]),
        modification_date=_dt(m["modification_date"]),
        page_count=m["page_count"],
        file_size_bytes=m["file_size_bytes"],
        has_text_layer=m["has_text_layer"],
        has_images=m["has_images"],
        language=m["language"],
        extra=dict(m["extra"]),
    )
    return UniversalDocument(
        doc_id=data["doc_id"],
        source_file=data["source_file"],
        file_type=data["file_type"],
        pages=pages,
        metadata=metadata,
        total_pages=data["total_pages"],
        created_at=datetime.fromisoformat(data["created_at"]),
    )


def dump_uir(
    doc: UniversalDocument,
    path: Union[str, Path],
    chunker_inputs: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``doc`` (and the chunker's other inputs, for a faithful replay) as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"uir": uir_to_dict(doc), "chunker_inputs": _jsonable(chunker_inputs or {})}
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return out


def load_uir(path: Union[str, Path]) -> "tuple[UniversalDocument, Dict[str, Any]]":
    """Load a dump written by :func:`dump_uir`: ``(document, chunker_inputs)``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    inputs = dict(payload.get("chunker_inputs") or {})
    toc = inputs.get("toc_headings")
    if isinstance(toc, dict):  # JSON object keys are strings; the chunker keys pages by int
        inputs["toc_headings"] = {
            (int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): v
            for k, v in toc.items()
        }
    return uir_from_dict(payload["uir"]), inputs
