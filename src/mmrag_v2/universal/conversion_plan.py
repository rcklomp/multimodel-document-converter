"""V3.0 format-agnostic ConversionPlan parent class.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.2 — Stage 1 UIR refactor.

This module introduces the format-agnostic `ConversionPlan` parent class
that the v3.0 pipeline consumes uniformly. Format-specific plans (PDF,
EPUB, future DOCX/HTML) subclass it and add their own fields, but the
v3.0 pipeline contract only sees the parent.

Foundation-session scope (2026-05-26):
    Additive only. The existing `PdfConversionPlan` at
    `src/mmrag_v2/engines/pdf_plan.py` remains the v2.16 production
    construction site for Docling. Refactoring `PdfConversionPlan` to
    inherit from this parent happens in Phase A task A1; it is NOT
    done in the foundation session because that refactor needs to
    flow through every call site and the test suite to land cleanly.

Charter contract (§3.2):
    - `engine_options: Dict[str, Any]` carries an opaque blob for
      engine-specific flags (do_code_enrichment, do_picture_classification,
      OCR engine choice, etc.) so that pinned Docling toggles do not
      require a new typed plan field per option.
    - `render_dpi` validation range `[72, 600]` enforced in __post_init__.
      ColPali patch count varies with render_dpi × page dimensions; the
      visual collection is pinned to a single render_dpi value per
      Charter §7.11 visual-index stability contract. Changing render_dpi
      requires a full visual re-embed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


RENDER_DPI_MIN = 72
RENDER_DPI_MAX = 600
DEFAULT_RENDER_DPI = 200


@dataclass(frozen=True)
class ConversionPlan:
    """Format-agnostic extraction plan. Charter §3.2.

    Subclasses (PdfConversionPlan, EpubConversionPlan, ...) add
    format-specific fields. The v3.0 pipeline consumes a `ConversionPlan`
    uniformly via the UIR contract.
    """

    source_path: str
    file_type: str  # "pdf" | "epub" | "html"
    doc_id: str  # First 12 chars of MD5 hash (per UniversalDocument.compute_doc_id)
    profile_type: str  # From ProfileClassifier
    extraction_strategy: str  # "digital_native" | "ocr_nuclear" | "ocr_forced"
    reading_order_strategy: str  # "docling_native" | "y_sort" | "y_sort_with_dropcap"
    modality_flags: Dict[str, bool] = field(default_factory=dict)
    # Keys typically include: "is_scanned", "has_encoding_corruption", ...
    batch_size: int = 10
    # Charter §3.2 (Draft 0.5): validation range [72, 600] enforced below.
    # ColPali patch count varies with render_dpi × page dimensions; the
    # visual collection is pinned to a single render_dpi value per Charter
    # §7.11. C-spike measures at both 200 and 300 to confirm PASS conditions
    # are not DPI-lucky.
    render_dpi: int = DEFAULT_RENDER_DPI
    lang_hint: Optional[str] = None  # ISO 639-1, when known a priori
    engine_options: Dict[str, Any] = field(default_factory=dict)
    # Opaque per-engine blob; the v3.0 pipeline does not introspect it.
    # Engine adapters read the keys they understand and ignore the rest.

    def __post_init__(self) -> None:
        if not (RENDER_DPI_MIN <= self.render_dpi <= RENDER_DPI_MAX):
            raise ValueError(
                f"ConversionPlan.render_dpi must be in "
                f"[{RENDER_DPI_MIN}, {RENDER_DPI_MAX}]; got {self.render_dpi}"
            )
        if self.batch_size < 1:
            raise ValueError(
                f"ConversionPlan.batch_size must be >=1; got {self.batch_size}"
            )
        # AGENTS.md §1.4: keep PDF batch size at <=10 pages. The parent
        # class does not enforce the upper bound (would require knowing
        # the engine); PdfConversionPlan's __post_init__ should enforce
        # the PDF-specific ≤10 invariant when it gains an override.
        if not self.file_type:
            raise ValueError("ConversionPlan.file_type must not be empty")
        if not self.doc_id:
            raise ValueError("ConversionPlan.doc_id must not be empty")
