"""
Format Router - Detect and Route to Appropriate Engine
=======================================================

This module provides format detection and routing to the appropriate
extraction engine. It uses multiple signals (magic bytes, extension,
content sampling) to reliably identify document formats.

Usage:
    from mmrag_v2.universal.router import FormatRouter

    router = FormatRouter()
    engine = router.get_engine("document.pdf")
    uir = engine.convert("document.pdf")

Author: Claude (Architect)
Date: January 2026
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from ..engines.base import FormatEngine

logger = logging.getLogger(__name__)


# ============================================================================
# MAGIC BYTES SIGNATURES
# ============================================================================

# File format magic byte signatures
MAGIC_SIGNATURES: Dict[bytes, str] = {
    b"%PDF": "pdf",
    b"PK\x03\x04": "zip",  # ZIP-based formats (EPUB, DOCX, PPTX, XLSX)
    b"<!DOCTYPE html": "html",
    b"<html": "html",
    b"<?xml": "xml",
}

# ZIP-based format detection (check internal files)
ZIP_FORMAT_MARKERS: Dict[str, str] = {
    "mimetype": "epub",  # EPUB has 'mimetype' file with 'application/epub+zip'
    "[Content_Types].xml": "office",  # Office formats have this
    "word/document.xml": "docx",
    "ppt/presentation.xml": "pptx",
    "xl/workbook.xml": "xlsx",
}

# File extension to format mapping
EXTENSION_MAP: Dict[str, str] = {
    ".pdf": "pdf",
    ".epub": "epub",
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".docx": "docx",
    ".doc": "docx",  # Legacy, may need conversion
    ".pptx": "pptx",
    ".ppt": "pptx",  # Legacy, may need conversion
    ".xlsx": "xlsx",
    ".xls": "xlsx",  # Legacy, may need conversion
    ".txt": "text",
    ".md": "markdown",
    ".json": "json",
}


# ============================================================================
# FORMAT ROUTER
# ============================================================================


class FormatRouter:
    """
    Routes documents to appropriate extraction engines.

    The router uses multiple signals to detect format:
    1. Magic bytes (file header)
    2. File extension
    3. MIME type
    4. Internal structure (for ZIP-based formats)

    Attributes:
        engines: Registry of available format engines

    Usage:
        router = FormatRouter()
        router.register_engine(PDFEngine())
        router.register_engine(EpubEngine())

        engine = router.get_engine("document.pdf")
        uir = engine.convert("document.pdf")
    """

    def __init__(self) -> None:
        """Initialize the format router."""
        self._engines: Dict[str, "FormatEngine"] = {}
        self._engine_classes: Dict[str, Type["FormatEngine"]] = {}
        logger.info("FormatRouter initialized")

    def register_engine(self, engine: "FormatEngine") -> None:
        """
        Register a format engine.

        Args:
            engine: FormatEngine instance to register
        """
        for ext in engine.supported_extensions:
            self._engines[ext.lower()] = engine
            logger.debug(f"Registered engine {engine.__class__.__name__} for {ext}")

        logger.info(
            f"Registered {engine.__class__.__name__} "
            f"(formats: {', '.join(engine.supported_extensions)})"
        )

    def register_engine_class(
        self,
        format_type: str,
        engine_class: Type["FormatEngine"],
    ) -> None:
        """
        Register an engine class for lazy instantiation.

        Args:
            format_type: Format type (e.g., "pdf", "epub")
            engine_class: Engine class to instantiate when needed
        """
        self._engine_classes[format_type.lower()] = engine_class
        logger.debug(f"Registered engine class {engine_class.__name__} for {format_type}")

    def detect_format(self, file_path: Path) -> str:
        """
        Detect the format of a file.

        Uses multiple signals:
        1. Magic bytes from file header
        2. File extension
        3. MIME type detection
        4. ZIP internal structure (for EPUB, Office formats)

        Args:
            file_path: Path to the file

        Returns:
            Format string (e.g., "pdf", "epub", "html")

        Raises:
            ValueError: If format cannot be determined
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Signal 1: Magic bytes
        format_from_magic = self._detect_from_magic(file_path)

        # Signal 2: File extension
        format_from_ext = EXTENSION_MAP.get(file_path.suffix.lower(), "unknown")

        # Signal 3: MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        format_from_mime = self._mime_to_format(mime_type) if mime_type else "unknown"

        # Combine signals (prefer magic bytes, then extension, then MIME)
        if format_from_magic and format_from_magic != "unknown":
            detected = format_from_magic
        elif format_from_ext != "unknown":
            detected = format_from_ext
        elif format_from_mime != "unknown":
            detected = format_from_mime
        else:
            raise ValueError(
                f"Cannot determine format of {file_path.name}. "
                f"Magic: {format_from_magic}, Ext: {format_from_ext}, MIME: {format_from_mime}"
            )

        # Special handling for ZIP-based formats
        if detected == "zip":
            detected = self._detect_zip_format(file_path)

        logger.info(
            f"Format detected: {detected} "
            f"(magic={format_from_magic}, ext={format_from_ext}, mime={format_from_mime})"
        )

        return detected

    def _detect_from_magic(self, file_path: Path) -> str:
        """Detect format from magic bytes."""
        try:
            with open(file_path, "rb") as f:
                header = f.read(32)

            for signature, format_type in MAGIC_SIGNATURES.items():
                if header.startswith(signature):
                    return format_type

            # Check for HTML without doctype
            if b"<html" in header.lower():
                return "html"

            return "unknown"
        except Exception as e:
            logger.warning(f"Magic byte detection failed: {e}")
            return "unknown"

    def _detect_zip_format(self, file_path: Path) -> str:
        """
        Detect specific format for ZIP-based files (EPUB, Office).

        Examines internal structure to differentiate:
        - EPUB: Has 'mimetype' file
        - DOCX: Has 'word/document.xml'
        - PPTX: Has 'ppt/presentation.xml'
        - XLSX: Has 'xl/workbook.xml'
        """
        try:
            import zipfile

            with zipfile.ZipFile(file_path, "r") as zf:
                namelist = zf.namelist()

                # Check for EPUB
                if "mimetype" in namelist:
                    try:
                        mimetype = zf.read("mimetype").decode("utf-8").strip()
                        if "epub" in mimetype.lower():
                            return "epub"
                    except Exception:
                        pass

                # Check for Office formats
                if "[Content_Types].xml" in namelist:
                    if any(f.startswith("word/") for f in namelist):
                        return "docx"
                    elif any(f.startswith("ppt/") for f in namelist):
                        return "pptx"
                    elif any(f.startswith("xl/") for f in namelist):
                        return "xlsx"

                return "zip"  # Unknown ZIP format

        except Exception as e:
            logger.warning(f"ZIP format detection failed: {e}")
            return "zip"

    def _mime_to_format(self, mime_type: str) -> str:
        """Convert MIME type to format string."""
        mime_map = {
            "application/pdf": "pdf",
            "application/epub+zip": "epub",
            "text/html": "html",
            "application/xhtml+xml": "html",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
            "text/plain": "text",
            "text/markdown": "markdown",
            "application/json": "json",
        }
        return mime_map.get(mime_type, "unknown")

    def get_engine(self, file_path: Path) -> "FormatEngine":
        """
        Get the appropriate engine for a file.

        Args:
            file_path: Path to the file

        Returns:
            FormatEngine instance for the detected format

        Raises:
            ValueError: If no engine registered for format
        """
        file_path = Path(file_path)
        format_type = self.detect_format(file_path)

        # Check registered engines
        if format_type in self._engines:
            return self._engines[format_type]

        # Check by extension
        ext = file_path.suffix.lower()
        if ext in self._engines:
            return self._engines[ext]

        # Check engine classes for lazy instantiation
        if format_type in self._engine_classes:
            engine = self._engine_classes[format_type]()
            self.register_engine(engine)
            return engine

        raise ValueError(
            f"No engine registered for format '{format_type}' " f"(file: {file_path.name})"
        )

    def can_handle(self, file_path: Path) -> bool:
        """
        Check if any registered engine can handle this file.

        Args:
            file_path: Path to the file

        Returns:
            True if file can be processed, False otherwise
        """
        try:
            format_type = self.detect_format(file_path)
            return (
                format_type in self._engines
                or format_type in self._engine_classes
                or file_path.suffix.lower() in self._engines
            )
        except Exception:
            return False

    @property
    def supported_formats(self) -> List[str]:
        """List of all supported formats."""
        formats = set(self._engines.keys())
        formats.update(self._engine_classes.keys())
        return sorted(formats)

    @property
    def supported_extensions(self) -> List[str]:
        """List of all supported file extensions."""
        extensions = set()
        for engine in self._engines.values():
            extensions.update(engine.supported_extensions)
        return sorted(extensions)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_router() -> FormatRouter:
    """
    Create a FormatRouter with default engines registered.

    Returns:
        Configured FormatRouter instance
    """
    router = FormatRouter()

    # Register engines lazily to avoid import issues
    # Engines will be instantiated when first needed
    try:
        from ..engines.pdf_engine import PDFEngine

        router.register_engine_class("pdf", PDFEngine)
    except ImportError:
        logger.warning("PDFEngine not available")

    try:
        from ..engines.epub_engine import EpubEngine

        router.register_engine_class("epub", EpubEngine)
    except ImportError:
        logger.debug("EpubEngine not available (optional)")

    try:
        from ..engines.html_engine import HTMLEngine

        router.register_engine_class("html", HTMLEngine)
    except ImportError:
        logger.debug("HTMLEngine not available (optional)")

    return router


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Global router instance (lazy initialization)
_router: Optional[FormatRouter] = None


def get_router() -> FormatRouter:
    """
    Get the global FormatRouter instance.

    Creates the router on first call (lazy initialization).

    Returns:
        Global FormatRouter instance
    """
    global _router
    if _router is None:
        _router = create_router()
    return _router
