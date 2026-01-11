"""
FormatEngine Abstract Base Class
=================================

This module defines the contract that all format engines must implement.
Engines convert documents from their native format to the Universal
Intermediate Representation (UIR).

Engine Contract:
    1. detect(file_path) -> bool: Check if engine can handle file
    2. convert(file_path) -> UniversalDocument: Extract to UIR
    3. supported_extensions: List of supported file extensions

Usage:
    class PDFEngine(FormatEngine):
        @property
        def supported_extensions(self) -> List[str]:
            return [".pdf"]

        def detect(self, file_path: Path) -> bool:
            return file_path.suffix.lower() == ".pdf"

        def convert(self, file_path: Path) -> UniversalDocument:
            # Implementation...
            return universal_document

Author: Claude (Architect)
Date: January 2026
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..universal.intermediate import UniversalDocument

logger = logging.getLogger(__name__)


class FormatEngine(ABC):
    """
    Abstract base class for format-specific extraction engines.

    All format engines inherit from this class and implement the
    required methods to convert documents to UIR format.

    Attributes:
        name: Human-readable engine name
        version: Engine version string

    Abstract Methods:
        detect: Check if engine can handle a file
        convert: Convert file to UniversalDocument

    Abstract Properties:
        supported_extensions: List of file extensions this engine handles
    """

    def __init__(self, name: str = "BaseEngine", version: str = "1.0.0") -> None:
        """
        Initialize the format engine.

        Args:
            name: Human-readable engine name
            version: Engine version string
        """
        self.name = name
        self.version = version
        self._initialized = False
        logger.debug(f"FormatEngine initialized: {name} v{version}")

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """
        List of file extensions this engine can handle.

        Returns:
            List of extensions including the dot (e.g., [".pdf", ".PDF"])
        """
        pass

    @abstractmethod
    def detect(self, file_path: Path) -> bool:
        """
        Check if this engine can handle the given file.

        This method should perform quick checks (extension, magic bytes)
        without fully parsing the file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this engine can handle the file, False otherwise
        """
        pass

    @abstractmethod
    def convert(self, file_path: Path) -> "UniversalDocument":
        """
        Convert a file to Universal Intermediate Representation.

        This is the main extraction method. It should:
        1. Parse the document structure
        2. Extract all content (text, images, tables)
        3. Classify page quality (digital vs scanned)
        4. Return a UniversalDocument with all elements

        Args:
            file_path: Path to the document file

        Returns:
            UniversalDocument with extracted content

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
            RuntimeError: If extraction fails
        """
        pass

    def initialize(self) -> None:
        """
        Perform lazy initialization of engine resources.

        Called automatically before first conversion. Override to
        load models, initialize libraries, etc.
        """
        if not self._initialized:
            self._do_initialize()
            self._initialized = True
            logger.info(f"Engine {self.name} initialized")

    def _do_initialize(self) -> None:
        """
        Internal initialization hook for subclasses.

        Override this method to perform actual initialization.
        """
        pass

    def cleanup(self) -> None:
        """
        Clean up engine resources.

        Called when engine is no longer needed. Override to
        release resources, close connections, etc.
        """
        self._initialized = False
        logger.debug(f"Engine {self.name} cleaned up")

    def can_handle(self, file_path: Path) -> bool:
        """
        Check if this engine can handle a file (convenience method).

        Combines extension check with detect() method.

        Args:
            file_path: Path to check

        Returns:
            True if file can be processed
        """
        file_path = Path(file_path)

        # Quick extension check first
        if file_path.suffix.lower() not in [ext.lower() for ext in self.supported_extensions]:
            return False

        # Then detailed detection
        return self.detect(file_path)

    def validate_file(self, file_path: Path) -> None:
        """
        Validate that a file exists and can be read.

        Args:
            file_path: Path to validate

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file can't be read
            ValueError: If file is empty
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        if file_path.stat().st_size == 0:
            raise ValueError(f"File is empty: {file_path}")

        # Try to open file to check permissions
        try:
            with open(file_path, "rb") as f:
                f.read(1)
        except PermissionError:
            raise PermissionError(f"Cannot read file: {file_path}")

    def __repr__(self) -> str:
        """String representation of engine."""
        return f"{self.__class__.__name__}(name={self.name}, version={self.version})"

    def __enter__(self) -> "FormatEngine":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.cleanup()


class BaseTextEngine(FormatEngine):
    """
    Base class for text-based format engines.

    Provides common utilities for text extraction:
    - Character encoding detection
    - Text normalization
    - Whitespace handling
    """

    def __init__(self, name: str = "TextEngine", version: str = "1.0.0") -> None:
        super().__init__(name=name, version=version)
        self.default_encoding = "utf-8"

    def detect_encoding(self, file_path: Path) -> str:
        """
        Detect file encoding.

        Args:
            file_path: Path to file

        Returns:
            Detected encoding string (e.g., "utf-8")
        """
        try:
            import chardet

            with open(file_path, "rb") as f:
                raw = f.read(10000)  # Read first 10KB

            result = chardet.detect(raw)
            encoding = result.get("encoding", self.default_encoding)
            confidence = result.get("confidence", 0)

            if confidence < 0.5:
                logger.warning(f"Low confidence encoding detection: {encoding} ({confidence:.2%})")
                return self.default_encoding

            return encoding or self.default_encoding

        except ImportError:
            logger.debug("chardet not available, using default encoding")
            return self.default_encoding
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}")
            return self.default_encoding

    def normalize_text(self, text: str) -> str:
        """
        Normalize text content.

        - Replace multiple spaces with single space
        - Normalize line endings
        - Strip leading/trailing whitespace

        Args:
            text: Raw text

        Returns:
            Normalized text
        """
        import re

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Replace multiple spaces (but not newlines) with single space
        text = re.sub(r"[^\S\n]+", " ", text)

        # Replace multiple blank lines with single blank line
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip lines
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()


class BaseBinaryEngine(FormatEngine):
    """
    Base class for binary format engines.

    Provides common utilities for binary formats:
    - Magic byte detection
    - Binary header parsing
    - Chunk reading
    """

    def __init__(
        self,
        name: str = "BinaryEngine",
        version: str = "1.0.0",
        magic_bytes: Optional[bytes] = None,
    ) -> None:
        super().__init__(name=name, version=version)
        self.magic_bytes = magic_bytes

    def check_magic_bytes(self, file_path: Path) -> bool:
        """
        Check if file starts with expected magic bytes.

        Args:
            file_path: Path to file

        Returns:
            True if magic bytes match
        """
        if self.magic_bytes is None:
            return True

        try:
            with open(file_path, "rb") as f:
                header = f.read(len(self.magic_bytes))
            return header == self.magic_bytes
        except Exception as e:
            logger.warning(f"Magic byte check failed: {e}")
            return False

    def read_chunks(self, file_path: Path, chunk_size: int = 8192):
        """
        Generator to read file in chunks.

        Args:
            file_path: Path to file
            chunk_size: Bytes per chunk

        Yields:
            Byte chunks
        """
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
