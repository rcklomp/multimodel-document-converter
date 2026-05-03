"""Shared PDF conversion policy for Docling-backed extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Tuple

from ..schema.ingestion_schema import CHUNK_FACTORY_METADATA_KEYS


INTELLIGENCE_METADATA_KEYS = (
    "profile_type",
    "profile_sensitivity",
    "min_image_dims",
    "confidence_threshold",
    "document_domain",
    "document_modality",
    "has_flat_text_corruption",
    "has_encoding_corruption",
    "geometry_error_rate",
    "total_pages",
    "image_density",
    "avg_text_per_page",
    "needs_code_enrichment",
)

_VALID_ASSET_POLICIES = frozenset({"drop", "keep", "quarantine"})
_VALID_RECOVERY_POLICIES = frozenset({"quarantine", "keep", "recover"})
_VALID_READING_ORDER_STRATEGIES = frozenset(
    {"docling_native", "y_sort", "y_sort_with_dropcap"}
)


@dataclass(frozen=True)
class PdfConversionPlan:
    """Single source of truth for PDF extraction policy.

    Policy fields (Milestone 2: Plan Control Plane):

    - extraction_route: determines how Docling extracts PDF content.
        "native_digital"       — standard digital PDF, full Docling pipeline
        "scanned_book"         — scanned book, element-by-element chunking
        "scanned_degraded"     — degraded scanned book; explicit-override only
        "image_heavy_magazine"  — magazine with high image density
        "technical_manual"     — technical manual with structural content
    - hybrid_chunker_enabled: when True (default), use HybridChunker for text.
    - max_chunker_input_chars: total-text guard to skip HybridChunker.
    - max_chunker_per_element_chars: per-element guard for pathological elements.
    - allow_page_level_visuals: removes "full_page_image" from picture deny list.
    - asset_validation_policy: "drop" | "keep" | "quarantine".
    - corruption_recovery_policy: "quarantine" | "keep" | "recover".
    - drop_blank_assets: legacy @property derived from asset_validation_policy.
    - quarantine_corrupted_chunks: legacy @property derived from recovery policy.
    """

    images_scale: float = 2.0
    generate_page_images: bool = True
    generate_picture_images: bool = True
    generate_table_images: bool = False
    sort_by_reading_order: bool = True
    do_table_structure: bool = True
    do_cell_matching: bool = False
    table_former_mode: str = "ACCURATE"
    do_picture_classification: bool = True
    picture_classification_deny: Tuple[str, ...] = (
        "full_page_image",
        "page_thumbnail",
    )
    do_ocr: bool = True
    ocr_engine: str = "easyocr"
    needs_code_enrichment: bool = False
    code_enrichment_reason: str = ""
    code_enrichment_score: float = 0.0
    do_formula_enrichment: bool = False
    force_table_vlm: bool = False
    has_encoding_corruption: bool = False
    has_flat_text_corruption: bool = False
    geometry_error_rate: float = 0.0
    document_modality: str = ""
    profile_type: str = "unknown"
    profile_sensitivity: float = 0.5
    min_image_dims: str = ""
    confidence_threshold: float = 0.5
    document_domain: str = ""
    total_pages: int = 0
    image_density: float = 0.0
    avg_text_per_page: float = 0.0
    # Policy fields (Milestone 2: Plan Control Plane)
    extraction_route: str = "native_digital"
    hybrid_chunker_enabled: bool = True
    max_chunker_input_chars: int = 500_000
    max_chunker_per_element_chars: int = 100_000
    allow_page_level_visuals: bool = False
    asset_validation_policy: Literal["drop", "keep", "quarantine"] = "drop"
    corruption_recovery_policy: Literal["quarantine", "keep", "recover"] = "quarantine"
    # Post-Docling reading-order strategy (PLAN_DOCLING_POSTPROCESSOR.md Phase 1+2):
    #   "docling_native"        - keep Docling's emission order (default)
    #   "y_sort"                - re-sort each page by (-bbox.t, bbox.l)
    #   "y_sort_with_dropcap"   - y_sort plus drop-cap promotion (Phase 2)
    reading_order_strategy: Literal[
        "docling_native", "y_sort", "y_sort_with_dropcap"
    ] = "docling_native"
    # Phase 3: when True, the chunker swaps in MmragChunkingSerializerProvider
    # so picture items with only classification-label annotations and no caption
    # contribute no text (kills the "Other" / "Icon" / "Table" label leak).
    suppress_layout_label_text: bool = False
    # Phase 4: OCR gating - threshold for OCR triggering on bitmap-heavy pages.
    # Docling's native default is 0.05 (5%); we raise to 0.75 to avoid OCR'ing
    # photographic cover artwork on born-digital documents. Profiles that have
    # full-bleed photo pages (digital literature, magazines) raise it further to
    # ~0.92 so only nearly-pure-bitmap pages OCR.
    bitmap_area_threshold: float = 0.75
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.asset_validation_policy not in _VALID_ASSET_POLICIES:
            raise ValueError(
                f"asset_validation_policy must be one of {sorted(_VALID_ASSET_POLICIES)}, "
                f"got {self.asset_validation_policy!r}"
            )
        if self.corruption_recovery_policy not in _VALID_RECOVERY_POLICIES:
            raise ValueError(
                f"corruption_recovery_policy must be one of {sorted(_VALID_RECOVERY_POLICIES)}, "
                f"got {self.corruption_recovery_policy!r}"
            )
        if self.reading_order_strategy not in _VALID_READING_ORDER_STRATEGIES:
            raise ValueError(
                f"reading_order_strategy must be one of "
                f"{sorted(_VALID_READING_ORDER_STRATEGIES)}, "
                f"got {self.reading_order_strategy!r}"
            )

    # --- Legacy bool bridges (read as attributes, derived from policy) ---

    @property
    def drop_blank_assets(self) -> bool:
        """Derived from asset_validation_policy for backward compatibility."""
        return self.asset_validation_policy == "drop"

    @property
    def quarantine_corrupted_chunks(self) -> bool:
        """Derived from corruption_recovery_policy for backward compatibility.

        Both "quarantine" and "recover" gate corrupted chunks from export;
        "recover" additionally runs a more aggressive recovery pass (TODO:
        implement aggressive recovery mode in CorruptionInterceptor).
        """
        return self.corruption_recovery_policy in {"quarantine", "recover"}

    def to_intelligence_metadata(self) -> Dict[str, Any]:
        """Return the legacy full metadata dict used at object boundaries."""
        return {
            "profile_type": self.profile_type,
            "profile_sensitivity": self.profile_sensitivity,
            "min_image_dims": self.min_image_dims,
            "confidence_threshold": self.confidence_threshold,
            "document_domain": self.document_domain,
            "document_modality": self.document_modality,
            "has_flat_text_corruption": self.has_flat_text_corruption,
            "has_encoding_corruption": self.has_encoding_corruption,
            "geometry_error_rate": self.geometry_error_rate,
            "total_pages": self.total_pages,
            "image_density": self.image_density,
            "avg_text_per_page": self.avg_text_per_page,
            "needs_code_enrichment": self.needs_code_enrichment,
        }

    def chunk_factory_metadata(self) -> Dict[str, Any]:
        """Return only metadata keys accepted by chunk factory functions."""
        metadata = self.to_intelligence_metadata()
        return {
            key: value
            for key, value in metadata.items()
            if key in CHUNK_FACTORY_METADATA_KEYS and value is not None
        }


def build_pdf_conversion_plan(
    *,
    enable_ocr: bool = True,
    ocr_engine: str = "easyocr",
    force_table_vlm: bool = False,
    needs_code_enrichment: bool = False,
    code_enrichment_reason: str = "",
    code_enrichment_score: float = 0.0,
    has_encoding_corruption: bool = False,
    has_flat_text_corruption: bool = False,
    geometry_error_rate: float = 0.0,
    document_modality: str = "",
    profile_type: str = "unknown",
    profile_sensitivity: float = 0.5,
    min_image_dims: str = "",
    confidence_threshold: float = 0.5,
    document_domain: str = "",
    total_pages: int = 0,
    image_density: float = 0.0,
    avg_text_per_page: float = 0.0,
    extraction_route: str = "",
    hybrid_chunker_enabled: bool = True,
    allow_page_level_visuals: bool = False,
    max_chunker_input_chars: int = 500_000,
    max_chunker_per_element_chars: int = 100_000,
    reading_order_strategy: str = "",
    suppress_layout_label_text: bool = False,
    bitmap_area_threshold: float = 0.75,
    extra_metadata: Dict[str, Any] | None = None,
) -> PdfConversionPlan:
    """Build a resolved PDF conversion plan from diagnostics and config.

    Extraction route selection (auto-detected when not explicitly set):
    - scanned modality → "scanned_book" (hybrid_chunker_enabled=False)
    - profile_type == "digital_magazine" AND image_density > 2.0/page →
      "image_heavy_magazine" (allow_page_level_visuals=True)
    - profile_type == "technical_manual" AND not scanned → "technical_manual"
    - default → "native_digital"

    "scanned_degraded" is a valid explicit-override value but is never
    auto-selected; the profile_type field carries the "scanned_degraded"
    signal for downstream refiner-threshold logic (see
    _is_scanned_degraded_profile in batch_processor.py).

    The image_density threshold of 2.0 images/page was chosen because typical
    digital magazines average <1 image/page, while layout-heavy publications
    (PCWorld, Combat Aircraft) consistently exceed 2/page with decorative
    inline graphics that signal visual-first content.
    """
    normalized_modality = (document_modality or "").strip().lower()
    is_scanned = normalized_modality in {"scanned_clean", "scanned_degraded"}

    # Determine extraction route if not explicitly set
    route_computed = hybrid_chunker_enabled
    allow_page_level_visuals_computed = False

    if not extraction_route:
        if is_scanned:
            extraction_route = "scanned_book"
            route_computed = False
        elif profile_type == "digital_magazine" and image_density > 2.0:
            extraction_route = "image_heavy_magazine"
            allow_page_level_visuals_computed = True
        elif profile_type == "technical_manual" and not is_scanned:
            extraction_route = "technical_manual"
        else:
            extraction_route = "native_digital"

    # Derive allow_page_level_visuals from the final route (whether explicit
    # or computed). Callers who pass an explicit route still get correct flags.
    if allow_page_level_visuals:
        allow_page_level_visuals_computed = True
    elif extraction_route == "image_heavy_magazine":
        allow_page_level_visuals_computed = True

    # `digital_literature` opts into the full post-Docling pipeline by default:
    # y-sort + drop-cap promotion (Phase 1+2), label-leak filter (Phase 3),
    # and the higher bitmap-area threshold (Phase 4). Explicit caller-supplied
    # values override the auto-enable.
    is_digital_literature = (
        profile_type == "digital_literature" and not is_scanned
    )

    resolved_reading_order = reading_order_strategy or (
        "y_sort_with_dropcap" if is_digital_literature else "docling_native"
    )
    resolved_suppress_label = bool(suppress_layout_label_text) or is_digital_literature

    # Profiles with full-bleed photographic pages need a higher bitmap-area
    # threshold so cover/photo pages are not OCR'd into garbage. Callers can
    # override by passing an explicit non-default `bitmap_area_threshold`.
    resolved_bitmap_threshold = float(bitmap_area_threshold)
    if bitmap_area_threshold == 0.75 and not is_scanned:
        if (
            extraction_route in {"image_heavy_magazine"}
            or is_digital_literature
            or profile_type == "digital_magazine"
        ):
            resolved_bitmap_threshold = 0.92

    return PdfConversionPlan(
        generate_table_images=bool(force_table_vlm),
        do_picture_classification=not is_scanned,
        do_ocr=bool(enable_ocr),
        ocr_engine=ocr_engine or "easyocr",
        needs_code_enrichment=bool(needs_code_enrichment),
        code_enrichment_reason=code_enrichment_reason or "",
        code_enrichment_score=float(code_enrichment_score or 0.0),
        do_formula_enrichment=False if needs_code_enrichment else False,
        force_table_vlm=bool(force_table_vlm),
        has_encoding_corruption=bool(has_encoding_corruption),
        has_flat_text_corruption=bool(has_flat_text_corruption),
        geometry_error_rate=float(geometry_error_rate or 0.0),
        document_modality=document_modality or "",
        profile_type=profile_type or "unknown",
        profile_sensitivity=float(profile_sensitivity or 0.0),
        min_image_dims=min_image_dims or "",
        confidence_threshold=float(confidence_threshold or 0.0),
        document_domain=document_domain or "",
        total_pages=int(total_pages or 0),
        image_density=float(image_density or 0.0),
        avg_text_per_page=float(avg_text_per_page or 0.0),
        extraction_route=extraction_route,
        hybrid_chunker_enabled=bool(route_computed),
        max_chunker_input_chars=int(max_chunker_input_chars),
        max_chunker_per_element_chars=int(max_chunker_per_element_chars),
        allow_page_level_visuals=bool(allow_page_level_visuals_computed),
        reading_order_strategy=resolved_reading_order,
        suppress_layout_label_text=resolved_suppress_label,
        bitmap_area_threshold=resolved_bitmap_threshold,
        extra_metadata=dict(extra_metadata or {}),
    )