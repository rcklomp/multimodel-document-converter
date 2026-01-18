"""
Docling V2 Mapper - Transform Docling Documents to IngestionChunks
===================================================================
ENGINE_USE: Docling v2.66.0 (Native Layout Analysis)

This module provides the DoclingToV2Mapper class that transforms Docling's
DoclingDocument objects into validated IngestionChunk objects compliant
with the MM-Converter-V2 schema.

The mapper serves as the bridge between Docling's internal representation
and our "Gold Standard" output format for downstream RAG systems.

REQ Compliance:
- REQ-SCHEMA-01: All output conforms to IngestionChunk Pydantic model
- REQ-COORD-01: All bounding boxes normalized to 0.0-1.0 scale
- REQ-STATE: Hierarchical breadcrumb tracking via ContextStateV2
- REQ-MM-02: Asset naming [DocHash]_[Page]_[Type]_[Index].png
- REQ-CHUNK-03: Content truncated to 400 chars max

SRS Section 6: Output Schema
"Each document element MUST be mapped to an IngestionChunk with
validated fields and proper hierarchical context."

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-28
Updated: 2025-12-30 (Full Implementation)
"""

from __future__ import annotations

import hashlib
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, TYPE_CHECKING

from PIL import Image

from .schema.ingestion_schema import (
    ChunkType,
    FileType,
    HierarchyMetadata,
    IngestionChunk,
    Modality,
    COORD_SCALE,
    create_image_chunk,
    create_table_chunk,
    create_text_chunk,
)
from .state.context_state import ContextStateV2, create_context_state, is_valid_heading
from .utils.coordinate_normalization import ensure_normalized, normalize_bbox

if TYPE_CHECKING:
    from docling.datamodel.document import ConversionResult
    from docling_core.types.doc.document import DoclingDocument

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# REQ-MM-01: 10px padding on all image crops
CROP_PADDING_PX: int = 10

# REQ-MM-02: Asset naming pattern
ASSET_PATTERN: str = "{doc_hash}_{page:03d}_{element_type}_{index:02d}.png"

# REQ-CHUNK-03: Maximum content length
MAX_CONTENT_CHARS: int = 400

# Minimum image dimensions to filter noise
MIN_IMAGE_WIDTH_PX: int = 50
MIN_IMAGE_HEIGHT_PX: int = 50

# Standard PDF page dimensions (Letter size in points)
DEFAULT_PAGE_WIDTH: float = 612.0
DEFAULT_PAGE_HEIGHT: float = 792.0

# Heading level detection patterns
HEADING_PATTERNS = {
    "title": 1,
    "section_header": 2,
    "heading": 2,
}

# Content denoising pattern (OCR artifacts)
NOISE_PATTERN = re.compile(r"^[\s\-·•\.\d\*]+$")


# ============================================================================
# DOCLING ELEMENT TYPES
# ============================================================================


class DoclingElementType:
    """Docling element type constants."""

    TEXT = "text"
    PARAGRAPH = "paragraph"
    TITLE = "title"
    SECTION_HEADER = "section_header"
    LIST_ITEM = "list_item"
    PICTURE = "picture"
    FIGURE = "figure"
    TABLE = "table"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"


# ============================================================================
# MAPPER CLASS
# ============================================================================


class DoclingToV2Mapper:
    """
    Maps Docling DoclingDocument elements to V2 IngestionChunks.

    This mapper:
    1. Iterates through Docling document elements
    2. Maintains hierarchical state (ContextStateV2) for breadcrumbs
    3. Normalizes all coordinates to 0.0-1.0 scale
    4. Creates properly validated IngestionChunk objects
    5. Extracts and saves image/table assets

    Usage:
        mapper = DoclingToV2Mapper(
            doc_hash="abc123",
            source_file="document.pdf",
            output_dir=Path("./output"),
        )

        for chunk in mapper.map_document(docling_result):
            # chunk is validated IngestionChunk
            process_chunk(chunk)

    Attributes:
        doc_hash: MD5 hash prefix for document identification
        source_file: Original source filename
        file_type: Document file type
        output_dir: Directory for extracted assets
        state: ContextStateV2 for breadcrumb tracking
    """

    def __init__(
        self,
        doc_hash: str,
        source_file: str,
        output_dir: Path,
        file_type: FileType = FileType.PDF,
        initial_state: Optional[ContextStateV2] = None,
        page_offset: int = 0,
        min_image_width: int = MIN_IMAGE_WIDTH_PX,
        min_image_height: int = MIN_IMAGE_HEIGHT_PX,
        intelligence_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the DoclingToV2Mapper.

        Args:
            doc_hash: Document hash for identification and asset naming
            source_file: Original source filename
            output_dir: Directory for output assets
            file_type: Source file type
            initial_state: Initial ContextStateV2 (for batch continuity)
            page_offset: Page offset for batch processing
            min_image_width: Minimum image width to extract
            min_image_height: Minimum image height to extract
            intelligence_metadata: V2.4 intelligence stack metadata for observability
        """
        self.doc_hash = doc_hash
        self.source_file = source_file
        self.file_type = file_type
        self.output_dir = Path(output_dir)
        self.page_offset = page_offset
        self.min_image_width = min_image_width
        self.min_image_height = min_image_height

        # BUG-006 FIX: Store intelligence metadata for chunk creation
        self.intelligence_metadata = intelligence_metadata or {}

        # Create assets directory
        self.assets_dir = self.output_dir / "assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or inherit state
        if initial_state is not None:
            self.state = initial_state.get_state_copy()
        else:
            self.state = create_context_state(
                doc_id=doc_hash,
                source_file=source_file,
            )

        # Element counters for asset naming
        self._element_indices: Dict[str, int] = {
            "figure": 0,
            "table": 0,
        }

        # Text buffer for context (REQ-MM-03)
        self._text_buffer: List[str] = []  # Previous text snippets (for prev_text_snippet)
        self._next_text_buffer: List[str] = []  # Lookahead for next_text_snippet

        # Page dimensions cache
        self._page_dims: Dict[int, Tuple[float, float]] = {}

        logger.info(
            f"DoclingToV2Mapper initialized: "
            f"doc_hash={doc_hash}, source={source_file}, "
            f"page_offset={page_offset}"
        )

    def map_document(
        self,
        result: "ConversionResult",
    ) -> Generator[IngestionChunk, None, None]:
        """
        Map a Docling ConversionResult to IngestionChunks.

        REQ-MM-03: Uses two-pass approach to populate semantic_context
        with both prev_text_snippet AND next_text_snippet.

        Args:
            result: Docling ConversionResult from DocumentConverter

        Yields:
            Validated IngestionChunk objects
        """
        doc = result.document

        # Cache page dimensions
        self._cache_page_dimensions(doc)

        # Cache page images for asset extraction
        page_images = self._extract_page_images(doc)

        # REQ-MM-03: Two-pass approach for semantic context
        # Pass 1: Collect all elements and their text content for lookahead
        all_elements: List[Tuple[Any, int]] = []  # (element, index)
        text_contents: List[str] = []  # Text content for each element (or empty)

        for idx, item_tuple in enumerate(doc.iterate_items()):
            element, _ = item_tuple
            all_elements.append((element, idx))

            # Extract text for context building
            text = getattr(element, "text", "") or ""
            text_contents.append(text.strip())

        # Pass 2: Process elements with lookahead for next_text
        for idx, (element, _) in enumerate(all_elements):
            # Build next_text_snippet from subsequent text elements
            next_text_parts = []
            for lookahead_idx in range(idx + 1, min(idx + 4, len(text_contents))):
                if text_contents[lookahead_idx]:
                    next_text_parts.append(text_contents[lookahead_idx])
                    if len(" ".join(next_text_parts)) >= 300:
                        break

            self._next_text_buffer = next_text_parts

            for chunk in self._map_element(element, page_images):
                yield chunk

        logger.info(
            f"Mapping complete: figures={self._element_indices['figure']}, "
            f"tables={self._element_indices['table']}"
        )

    def get_final_state(self) -> ContextStateV2:
        """Get the final state after mapping."""
        return self.state.get_state_copy()

    def _cache_page_dimensions(self, doc: "DoclingDocument") -> None:
        """Cache page dimensions from document."""
        if hasattr(doc, "pages") and doc.pages:
            pages = doc.pages.values() if isinstance(doc.pages, dict) else doc.pages
            for page in pages:
                pg_no = getattr(page, "page_no", 1) or 1
                width = getattr(page, "width", DEFAULT_PAGE_WIDTH) or DEFAULT_PAGE_WIDTH
                height = getattr(page, "height", DEFAULT_PAGE_HEIGHT) or DEFAULT_PAGE_HEIGHT
                self._page_dims[pg_no] = (width, height)

    def _extract_page_images(
        self,
        doc: "DoclingDocument",
    ) -> Dict[int, Image.Image]:
        """Extract page images for asset cropping."""
        page_images: Dict[int, Image.Image] = {}

        if hasattr(doc, "pages") and doc.pages:
            pages = doc.pages.values() if isinstance(doc.pages, dict) else doc.pages
            for page in pages:
                pg_no = getattr(page, "page_no", 1) or 1
                if hasattr(page, "image") and page.image:
                    if hasattr(page.image, "pil_image") and page.image.pil_image:
                        page_images[pg_no] = page.image.pil_image
                    elif isinstance(page.image, Image.Image):
                        page_images[pg_no] = page.image

        return page_images

    def _map_element(
        self,
        element: Any,
        page_images: Dict[int, Image.Image],
    ) -> Generator[IngestionChunk, None, None]:
        """
        Map a single Docling element to IngestionChunk(s).

        Args:
            element: Docling document element
            page_images: Dictionary of page images for cropping

        Yields:
            IngestionChunk objects
        """
        # Extract label
        label = self._get_element_label(element)
        label_lower = label.lower()

        # Extract text content
        text = getattr(element, "text", "") or ""

        # Extract page number and bbox
        page_no, bbox = self._extract_provenance(element)
        page_no += self.page_offset

        # Update state
        self.state.update_page(page_no)

        # Process headings for state updates
        if self._is_heading_label(label):
            heading_level = self._get_heading_level(label)
            if is_valid_heading(text.strip()):
                self.state.update_on_heading(text.strip(), heading_level)

        # Create hierarchy metadata
        # REQ-HIER-03: breadcrumb_path MUST contain at minimum [source_filename, "Page X"]
        breadcrumbs = self.state.get_breadcrumb_path()

        # Ensure minimum breadcrumb depth per REQ-HIER-03
        source_name = Path(self.source_file).stem if self.source_file else "Document"
        page_marker = f"Page {page_no}"

        if not breadcrumbs:
            # No hierarchy detected - use minimum fallback
            breadcrumbs = [source_name, page_marker]
        elif len(breadcrumbs) == 1:
            # Only document name - add page marker
            breadcrumbs = [breadcrumbs[0], page_marker]
        elif page_marker not in breadcrumbs:
            # Ensure page marker is present for deep navigation
            # Insert after document name if hierarchy exists
            if len(breadcrumbs) >= 1:
                breadcrumbs = [breadcrumbs[0], page_marker] + breadcrumbs[1:]

        hierarchy = HierarchyMetadata(
            parent_heading=self.state.get_parent_heading(),
            breadcrumb_path=breadcrumbs,
            level=len(breadcrumbs) if breadcrumbs else None,
        )

        # Route to appropriate handler based on element type
        if self._is_image_label(label_lower):
            yield from self._map_image_element(
                element=element,
                page_no=page_no,
                bbox=bbox,
                hierarchy=hierarchy,
                page_images=page_images,
            )

        elif "table" in label_lower:
            yield from self._map_table_element(
                element=element,
                text=text,
                page_no=page_no,
                bbox=bbox,
                hierarchy=hierarchy,
                page_images=page_images,
            )

        elif text.strip():
            yield from self._map_text_element(
                text=text.strip(),
                label=label,
                page_no=page_no,
                hierarchy=hierarchy,
            )

    def _map_image_element(
        self,
        element: Any,
        page_no: int,
        bbox: Optional[List[int]],
        hierarchy: HierarchyMetadata,
        page_images: Dict[int, Image.Image],
    ) -> Generator[IngestionChunk, None, None]:
        """Map an image/figure element."""
        self._element_indices["figure"] += 1

        # Generate asset filename
        asset_name = ASSET_PATTERN.format(
            doc_hash=self.doc_hash,
            page=page_no,
            element_type="figure",
            index=self._element_indices["figure"],
        )
        asset_path = f"assets/{asset_name}"

        # Save the asset
        batch_page_no = page_no - self.page_offset
        saved_image = self._save_asset(
            element=element,
            asset_path=asset_path,
            bbox=bbox,
            page_images=page_images,
            page_no=batch_page_no,
        )

        # Check image size
        if saved_image is None:
            return

        width, height = saved_image.size
        if width < self.min_image_width or height < self.min_image_height:
            logger.debug(
                f"Filtered small image: {width}x{height}px "
                f"(min: {self.min_image_width}x{self.min_image_height})"
            )
            return

        # REQ-MM-03: Get semantic context (prev + next text snippets)
        prev_text = " ".join(self._text_buffer[-3:]) if self._text_buffer else None
        if prev_text:
            prev_text = prev_text[-300:]

        # Get next_text from lookahead buffer (populated in map_document two-pass)
        next_text = " ".join(self._next_text_buffer) if self._next_text_buffer else None
        if next_text:
            next_text = next_text[:300]

        # Generate description
        visual_description = self._generate_description(
            page_no=page_no,
            hierarchy=hierarchy,
            prev_text=prev_text,
        )

        # REQ-COORD-01: bbox is REQUIRED for image modality
        # Provide fallback full-page bbox if not available
        image_bbox: List[int] = bbox if bbox is not None else [0, 0, COORD_SCALE, COORD_SCALE]

        # REQ-COORD-02: Get page dimensions for UI overlay support
        page_w, page_h = self._page_dims.get(page_no, (DEFAULT_PAGE_WIDTH, DEFAULT_PAGE_HEIGHT))

        yield create_image_chunk(
            doc_id=self.doc_hash,
            content=visual_description,
            source_file=self.source_file,
            file_type=self.file_type,
            page_number=page_no,
            asset_path=asset_path,
            bbox=image_bbox,
            hierarchy=hierarchy,
            prev_text=prev_text,
            next_text=next_text,  # REQ-MM-03: Now includes next_text_snippet
            visual_description=visual_description,
            width_px=width,
            height_px=height,
            page_width=int(page_w),
            page_height=int(page_h),
            **self.intelligence_metadata,  # BUG-006 FIX: Propagate intelligence metadata
        )

    def _map_table_element(
        self,
        element: Any,
        text: str,
        page_no: int,
        bbox: Optional[List[int]],
        hierarchy: HierarchyMetadata,
        page_images: Dict[int, Image.Image],
    ) -> Generator[IngestionChunk, None, None]:
        """Map a table element."""
        self._element_indices["table"] += 1

        # Generate asset filename
        asset_name = ASSET_PATTERN.format(
            doc_hash=self.doc_hash,
            page=page_no,
            element_type="table",
            index=self._element_indices["table"],
        )
        asset_path = f"assets/{asset_name}"

        # Save table image
        batch_page_no = page_no - self.page_offset
        self._save_asset(
            element=element,
            asset_path=asset_path,
            bbox=bbox,
            page_images=page_images,
            page_no=batch_page_no,
        )

        # Table content
        content = text if text else f"[Table on page {page_no}]"

        # FIX #3: REQ-MM-03 - Get semantic context (prev + next text snippets) SAME AS IMAGES
        prev_text = " ".join(self._text_buffer[-3:]) if self._text_buffer else None
        if prev_text:
            prev_text = prev_text[-300:]

        # Get next_text from lookahead buffer (populated in map_document two-pass)
        next_text = " ".join(self._next_text_buffer) if self._next_text_buffer else None
        if next_text:
            next_text = next_text[:300]

        # REQ-COORD-01: bbox is REQUIRED for table modality
        # Provide fallback full-page bbox if not available
        table_bbox: List[int] = bbox if bbox is not None else [0, 0, COORD_SCALE, COORD_SCALE]

        # REQ-COORD-02: Get page dimensions for UI overlay support
        page_w, page_h = self._page_dims.get(page_no, (DEFAULT_PAGE_WIDTH, DEFAULT_PAGE_HEIGHT))

        yield create_table_chunk(
            doc_id=self.doc_hash,
            content=content,
            source_file=self.source_file,
            file_type=self.file_type,
            page_number=page_no,
            bbox=table_bbox,
            hierarchy=hierarchy,
            asset_path=asset_path,
            prev_text=prev_text,  # FIX #3: Add prev_text context parity with images
            next_text=next_text,  # FIX #3: Add next_text context parity with images
            page_width=int(page_w),
            page_height=int(page_h),
            **self.intelligence_metadata,  # BUG-006 FIX: Propagate intelligence metadata
        )

    def _map_text_element(
        self,
        text: str,
        label: str,
        page_no: int,
        hierarchy: HierarchyMetadata,
    ) -> Generator[IngestionChunk, None, None]:
        """Map a text element."""
        # Filter noise
        if self._is_noise_content(text):
            return

        # Update text buffer
        self._text_buffer.append(text)
        if len(self._text_buffer) > 10:
            self._text_buffer.pop(0)

        # Determine chunk type
        chunk_type = self._label_to_chunk_type(label)

        # Chunk long text
        chunks = self._chunk_text(text)

        for chunk_text in chunks:
            if not self._is_noise_content(chunk_text):
                yield create_text_chunk(
                    doc_id=self.doc_hash,
                    content=chunk_text,
                    source_file=self.source_file,
                    file_type=self.file_type,
                    page_number=page_no,
                    hierarchy=hierarchy,
                    chunk_type=chunk_type,
                    **self.intelligence_metadata,  # BUG-006 FIX: Propagate intelligence metadata
                )

    def _get_element_label(self, element: Any) -> str:
        """Extract label from element."""
        label_obj = getattr(element, "label", None)
        if label_obj is not None:
            return str(label_obj.value) if hasattr(label_obj, "value") else str(label_obj)
        return "text"

    def _extract_provenance(
        self,
        element: Any,
    ) -> Tuple[int, Optional[List[int]]]:
        """
        Extract page number and bounding box from element provenance.

        REQ-COORD-01: This method MUST extract bbox coordinates from Docling's
        prov (provenance) data. A null bbox indicates a failure that should be
        logged and investigated.

        Returns:
            Tuple of (page_number, normalized_bbox as [l, t, r, b] integers 0-1000)
        """
        page_no = 1
        bbox: Optional[List[float]] = None
        extraction_method = "none"

        # Get element label for logging
        label = self._get_element_label(element)

        if hasattr(element, "prov") and element.prov:
            prov_list = element.prov if isinstance(element.prov, list) else [element.prov]

            for prov in prov_list:
                # Page number extraction
                if hasattr(prov, "page_no") and prov.page_no is not None:
                    page_no = prov.page_no
                elif hasattr(prov, "page") and prov.page is not None:
                    page_no = prov.page

                # Bounding box extraction - try multiple approaches
                if bbox is None and hasattr(prov, "bbox") and prov.bbox:
                    bbox_obj = prov.bbox

                    # Approach 1: Named attributes (l, t, r, b)
                    if hasattr(bbox_obj, "l") and hasattr(bbox_obj, "r"):
                        try:
                            x0 = min(float(bbox_obj.l), float(bbox_obj.r))
                            x1 = max(float(bbox_obj.l), float(bbox_obj.r))
                            y0 = min(float(bbox_obj.t), float(bbox_obj.b))
                            y1 = max(float(bbox_obj.t), float(bbox_obj.b))
                            if x1 > x0 and y1 > y0:
                                bbox = [x0, y0, x1, y1]
                                extraction_method = "prov.bbox.ltrb"
                        except (TypeError, ValueError) as e:
                            logger.debug(f"ltrb extraction failed: {e}")

                    # Approach 2: as_tuple() method
                    if bbox is None and hasattr(bbox_obj, "as_tuple"):
                        try:
                            raw = bbox_obj.as_tuple()
                            if len(raw) >= 4:
                                x0 = min(float(raw[0]), float(raw[2]))
                                x1 = max(float(raw[0]), float(raw[2]))
                                y0 = min(float(raw[1]), float(raw[3]))
                                y1 = max(float(raw[1]), float(raw[3]))
                                if x1 > x0 and y1 > y0:
                                    bbox = [x0, y0, x1, y1]
                                    extraction_method = "prov.bbox.as_tuple"
                        except (TypeError, ValueError) as e:
                            logger.debug(f"as_tuple extraction failed: {e}")

                    # Approach 3: Direct indexing (for list-like bbox)
                    if bbox is None:
                        try:
                            if hasattr(bbox_obj, "__getitem__") and len(bbox_obj) >= 4:
                                x0 = min(float(bbox_obj[0]), float(bbox_obj[2]))
                                x1 = max(float(bbox_obj[0]), float(bbox_obj[2]))
                                y0 = min(float(bbox_obj[1]), float(bbox_obj[3]))
                                y1 = max(float(bbox_obj[1]), float(bbox_obj[3]))
                                if x1 > x0 and y1 > y0:
                                    bbox = [x0, y0, x1, y1]
                                    extraction_method = "prov.bbox.indexing"
                        except (TypeError, ValueError, IndexError) as e:
                            logger.debug(f"Direct indexing failed: {e}")

                    # Approach 4: Dict-like access
                    if bbox is None and hasattr(bbox_obj, "get"):
                        try:
                            x0 = bbox_obj.get("x0") or bbox_obj.get("left") or bbox_obj.get("l")
                            y0 = bbox_obj.get("y0") or bbox_obj.get("top") or bbox_obj.get("t")
                            x1 = bbox_obj.get("x1") or bbox_obj.get("right") or bbox_obj.get("r")
                            y1 = bbox_obj.get("y1") or bbox_obj.get("bottom") or bbox_obj.get("b")
                            if all(v is not None for v in [x0, y0, x1, y1]):
                                x0, x1 = min(float(x0), float(x1)), max(float(x0), float(x1))
                                y0, y1 = min(float(y0), float(y1)), max(float(y0), float(y1))
                                if x1 > x0 and y1 > y0:
                                    bbox = [x0, y0, x1, y1]
                                    extraction_method = "prov.bbox.dict"
                        except (TypeError, ValueError) as e:
                            logger.debug(f"Dict access failed: {e}")

                if bbox is not None:
                    break  # Got bbox from this prov entry

        # REQ-COORD-01: Log missing bbox as a warning for investigation
        if bbox is None:
            # Check if element has any self-referential bbox
            if hasattr(element, "bbox") and element.bbox:
                try:
                    elem_bbox = element.bbox
                    if hasattr(elem_bbox, "l"):
                        x0 = min(float(elem_bbox.l), float(elem_bbox.r))
                        x1 = max(float(elem_bbox.l), float(elem_bbox.r))
                        y0 = min(float(elem_bbox.t), float(elem_bbox.b))
                        y1 = max(float(elem_bbox.t), float(elem_bbox.b))
                        if x1 > x0 and y1 > y0:
                            bbox = [x0, y0, x1, y1]
                            extraction_method = "element.bbox.ltrb"
                except (TypeError, ValueError) as e:
                    logger.debug(f"element.bbox extraction failed: {e}")

            if bbox is None:
                # This is a genuine missing bbox - log it
                text_preview = ""
                if hasattr(element, "text") and element.text:
                    text_preview = (
                        element.text[:50] + "..." if len(element.text) > 50 else element.text
                    )
                logger.warning(
                    f"REQ-COORD-01 WARNING: No bbox for {label} element on page {page_no}. "
                    f"Text preview: '{text_preview}'. "
                    f"This may indicate Docling provenance data is incomplete."
                )
        else:
            logger.debug(f"Extracted bbox via {extraction_method} for {label}: {bbox}")

        # Normalize bbox to 0-1000 integer scale
        normalized_bbox: Optional[List[int]] = None
        if bbox:
            page_w, page_h = self._page_dims.get(page_no, (DEFAULT_PAGE_WIDTH, DEFAULT_PAGE_HEIGHT))
            bbox = self._apply_padding(bbox, page_w, page_h)
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                try:
                    normalized_bbox = ensure_normalized(
                        bbox, page_w, page_h, f"element_page{page_no}"
                    )
                except ValueError as e:
                    logger.warning(f"Failed to normalize bbox for {label}: {e}")
                    normalized_bbox = None

        return page_no, normalized_bbox

    def _apply_padding(
        self,
        bbox: List[float],
        page_width: float,
        page_height: float,
    ) -> List[float]:
        """Apply REQ-MM-01 10px padding to bounding box."""
        x0, y0, x1, y1 = bbox
        x0 = max(0.0, x0 - CROP_PADDING_PX)
        y0 = max(0.0, y0 - CROP_PADDING_PX)
        x1 = min(page_width, x1 + CROP_PADDING_PX)
        y1 = min(page_height, y1 + CROP_PADDING_PX)
        return [x0, y0, x1, y1]

    def _save_asset(
        self,
        element: Any,
        asset_path: str,
        bbox: Optional[Union[List[int], List[float]]],
        page_images: Dict[int, Image.Image],
        page_no: int,
    ) -> Optional[Image.Image]:
        """Save image/table asset to disk."""
        full_path = self.assets_dir / Path(asset_path).name
        saved_image: Optional[Image.Image] = None

        try:
            # Try to get image from element directly
            if hasattr(element, "image") and element.image:
                img_data = element.image

                if hasattr(img_data, "pil_image") and img_data.pil_image is not None:
                    saved_image = img_data.pil_image
                elif isinstance(img_data, bytes):
                    saved_image = Image.open(BytesIO(img_data))
                elif isinstance(img_data, Image.Image):
                    saved_image = img_data

                if saved_image is not None:
                    saved_image.save(str(full_path), "PNG")
                    logger.debug(f"Saved asset from element.image: {full_path}")
                    return saved_image

            # Try to crop from page image
            if bbox and page_no in page_images:
                page_img = page_images[page_no]
                scale = 2.0  # REQ-PDF-04: High-fidelity rendering

                crop_box = (
                    int(bbox[0] * scale),
                    int(bbox[1] * scale),
                    int(bbox[2] * scale),
                    int(bbox[3] * scale),
                )

                saved_image = page_img.crop(crop_box)
                saved_image.save(str(full_path), "PNG")
                logger.debug(f"Saved cropped asset: {full_path}")
                return saved_image

            logger.warning(f"Could not save asset: {asset_path}")
            return None

        except Exception as e:
            logger.error(f"Failed to save asset {asset_path}: {e}")
            return None

    def _is_heading_label(self, label: str) -> bool:
        """Check if label indicates a heading."""
        label_lower = label.lower()
        return any(h in label_lower for h in ["title", "section", "heading"])

    def _get_heading_level(self, label: str) -> int:
        """Get heading level from label."""
        label_lower = label.lower()

        if "title" in label_lower:
            return 1

        # Check for level number
        for i in range(1, 7):
            if str(i) in label_lower or f"level{i}" in label_lower:
                return i

        return 2  # Default for headings

    def _is_image_label(self, label_lower: str) -> bool:
        """Check if label indicates an image."""
        return any(t in label_lower for t in ["picture", "figure", "image", "background"])

    def _is_noise_content(self, text: str) -> bool:
        """Check if content is noise."""
        if not text or not text.strip():
            return True

        text = text.strip()

        if NOISE_PATTERN.match(text):
            return True

        if not any(c.isalpha() for c in text):
            return True

        if len(text) < 2:
            return True

        return False

    def _label_to_chunk_type(self, label: str) -> ChunkType:
        """Convert Docling label to ChunkType."""
        label_lower = label.lower()

        if "title" in label_lower:
            return ChunkType.TITLE
        elif "heading" in label_lower or "section" in label_lower:
            return ChunkType.HEADING
        elif "list" in label_lower:
            return ChunkType.LIST_ITEM
        elif "caption" in label_lower:
            return ChunkType.CAPTION
        elif "footnote" in label_lower:
            return ChunkType.FOOTNOTE
        elif "quote" in label_lower:
            return ChunkType.QUOTE
        elif "code" in label_lower:
            return ChunkType.CODE

        return ChunkType.PARAGRAPH

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= MAX_CONTENT_CHARS:
            return [text]

        chunks = []
        overlap = 60

        # Simple sentence-aware chunking
        sentences = re.split(r"(?<=[.!?])\s+", text)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > MAX_CONTENT_CHARS and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_text = (
                    current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                )
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text[:MAX_CONTENT_CHARS]]

    def _generate_description(
        self,
        page_no: int,
        hierarchy: HierarchyMetadata,
        prev_text: Optional[str] = None,
    ) -> str:
        """Generate fallback description for image."""
        parts = [f"[Figure on page {page_no}]"]

        if hierarchy.breadcrumb_path:
            path = " > ".join(hierarchy.breadcrumb_path)
            parts.append(f"Context: {path}")

        if prev_text and len(prev_text) > 20:
            ctx = prev_text[:100] + "..." if len(prev_text) > 100 else prev_text
            parts.append(f"Near: {ctx}")

        description = " | ".join(parts)
        return description[:MAX_CONTENT_CHARS]


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_mapper(
    doc_hash: str,
    source_file: str,
    output_dir: Path,
    file_type: FileType = FileType.PDF,
    initial_state: Optional[ContextStateV2] = None,
    page_offset: int = 0,
) -> DoclingToV2Mapper:
    """
    Factory function to create a DoclingToV2Mapper.

    Args:
        doc_hash: Document hash for identification
        source_file: Original source filename
        output_dir: Directory for output assets
        file_type: Source file type
        initial_state: Initial state for batch continuity
        page_offset: Page offset for batch processing

    Returns:
        Configured DoclingToV2Mapper instance
    """
    return DoclingToV2Mapper(
        doc_hash=doc_hash,
        source_file=source_file,
        output_dir=output_dir,
        file_type=file_type,
        initial_state=initial_state,
        page_offset=page_offset,
    )
