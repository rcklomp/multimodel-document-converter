"""
Image Hash Registry - Perceptual Hash Deduplication for MM-Converter-V2
========================================================================
ENGINE_USE: imagehash library for perceptual hashing

This module provides utilities for detecting and filtering duplicate images
using perceptual hashing (pHash). This prevents redundant VLM calls and
duplicate entries in the output.

REQ Compliance:
- REQ-DEDUP-01: Perceptual hash-based duplicate detection
- REQ-DEDUP-02: Configurable similarity threshold
- REQ-DEDUP-03: First occurrence wins (keeps original, rejects duplicates)

SRS Section 7.2: Deduplication
"The system MUST detect and filter duplicate images using perceptual
hashing to prevent redundant processing and output."

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-29
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Default hamming distance threshold for considering images as duplicates
# Lower = stricter (fewer matches), Higher = looser (more matches)
DEFAULT_THRESHOLD: int = 10

# Minimum threshold (very strict)
MIN_THRESHOLD: int = 0

# Maximum threshold (very loose)
MAX_THRESHOLD: int = 32


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ImageRecord:
    """
    Record of a registered image.
    
    Attributes:
        phash: Perceptual hash string
        page_number: Page where image was found
        asset_path: Path to the asset file
        width: Image width in pixels
        height: Image height in pixels
    """
    phash: str
    page_number: int
    asset_path: str
    width: int = 0
    height: int = 0


@dataclass
class DuplicateInfo:
    """
    Information about a duplicate check result.
    
    Attributes:
        is_duplicate: Whether the image is a duplicate
        original_record: The original image record if duplicate
        hamming_distance: Distance from original (0 = identical)
    """
    is_duplicate: bool
    original_record: Optional[ImageRecord] = None
    hamming_distance: int = 0


# ============================================================================
# PERCEPTUAL HASH FUNCTIONS
# ============================================================================

def compute_phash(image: Image.Image) -> str:
    """
    Compute perceptual hash of an image.
    
    Uses the imagehash library for robust perceptual hashing that is
    invariant to minor changes in scale, color, and compression.
    
    Args:
        image: PIL Image to hash
        
    Returns:
        Hex string representation of the perceptual hash
    """
    try:
        import imagehash
        
        # Convert to RGB if needed (handles RGBA, P mode, etc.)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        
        # Compute perceptual hash
        phash = imagehash.phash(image)
        return str(phash)
        
    except ImportError:
        # Fallback: simple average hash if imagehash not available
        logger.warning("imagehash not installed, using fallback hash")
        return _fallback_hash(image)
    except Exception as e:
        logger.warning(f"pHash computation failed: {e}")
        return _fallback_hash(image)


def _fallback_hash(image: Image.Image) -> str:
    """
    Fallback hash using simple averaging.
    
    This is less robust than proper pHash but works without dependencies.
    """
    # Resize to 8x8 and convert to grayscale
    img = image.copy()
    img = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    
    # Get pixel data
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    
    # Create hash based on whether each pixel is above average
    bits = "".join("1" if p > avg else "0" for p in pixels)
    
    # Convert to hex
    return format(int(bits, 2), "016x")


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Calculate hamming distance between two hash strings.
    
    Args:
        hash1: First hash (hex string)
        hash2: Second hash (hex string)
        
    Returns:
        Number of differing bits
    """
    try:
        import imagehash
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2
    except ImportError:
        # Fallback: manual hamming distance
        return _fallback_hamming(hash1, hash2)
    except Exception:
        return _fallback_hamming(hash1, hash2)


def _fallback_hamming(hash1: str, hash2: str) -> int:
    """Fallback hamming distance calculation."""
    try:
        # Convert hex to binary and compare
        bin1 = bin(int(hash1, 16))[2:].zfill(64)
        bin2 = bin(int(hash2, 16))[2:].zfill(64)
        return sum(b1 != b2 for b1, b2 in zip(bin1, bin2))
    except Exception:
        # If all else fails, assume different
        return MAX_THRESHOLD + 1


# ============================================================================
# IMAGE HASH REGISTRY
# ============================================================================

class ImageHashRegistry:
    """
    Registry for tracking and deduplicating images using perceptual hashing.
    
    REQ-DEDUP-01: Maintains a registry of all processed images and their
    perceptual hashes to detect duplicates.
    
    Usage:
        registry = ImageHashRegistry(threshold=10)
        
        for image in images:
            dup_info = registry.check_and_register(image, page, path)
            if dup_info.is_duplicate:
                skip_image()
            else:
                process_image()
    
    Attributes:
        threshold: Hamming distance threshold for duplicate detection
        records: Dictionary mapping pHash to ImageRecord
    """
    
    def __init__(
        self,
        threshold: int = DEFAULT_THRESHOLD,
    ) -> None:
        """
        Initialize the image hash registry.
        
        Args:
            threshold: Hamming distance threshold (0-32)
                      Lower = stricter matching
                      Default: 10 (allows minor variations)
        """
        self.threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, threshold))
        self._records: Dict[str, ImageRecord] = {}
        self._hash_list: List[Tuple[str, ImageRecord]] = []  # For distance search
        
        logger.info(f"ImageHashRegistry initialized: threshold={self.threshold}")
    
    def check_and_register(
        self,
        image: Image.Image,
        page_number: int,
        asset_path: str,
    ) -> DuplicateInfo:
        """
        Check if image is a duplicate and register if not.
        
        REQ-DEDUP-03: First occurrence wins - if this image is a duplicate
        of an existing one, the original is kept and this one is rejected.
        
        Args:
            image: PIL Image to check
            page_number: Page number where image was found
            asset_path: Path to the asset file
            
        Returns:
            DuplicateInfo indicating if this is a duplicate
        """
        # Compute perceptual hash
        phash = compute_phash(image)
        
        # Check for exact match first
        if phash in self._records:
            original = self._records[phash]
            logger.debug(
                f"Exact duplicate found: {asset_path} matches {original.asset_path}"
            )
            return DuplicateInfo(
                is_duplicate=True,
                original_record=original,
                hamming_distance=0,
            )
        
        # Check for near-duplicates
        for existing_hash, existing_record in self._hash_list:
            distance = hamming_distance(phash, existing_hash)
            if distance <= self.threshold:
                logger.debug(
                    f"Near-duplicate found (distance={distance}): "
                    f"{asset_path} matches {existing_record.asset_path}"
                )
                return DuplicateInfo(
                    is_duplicate=True,
                    original_record=existing_record,
                    hamming_distance=distance,
                )
        
        # Not a duplicate - register this image
        record = ImageRecord(
            phash=phash,
            page_number=page_number,
            asset_path=asset_path,
            width=image.width,
            height=image.height,
        )
        self._records[phash] = record
        self._hash_list.append((phash, record))
        
        logger.debug(f"Registered new image: {asset_path} (hash={phash[:8]}...)")
        
        return DuplicateInfo(
            is_duplicate=False,
            original_record=None,
            hamming_distance=0,
        )
    
    def is_duplicate(self, image: Image.Image) -> bool:
        """
        Simple check if image is a duplicate (without registering).
        
        Args:
            image: PIL Image to check
            
        Returns:
            True if this image is a duplicate of a registered one
        """
        phash = compute_phash(image)
        
        # Check exact match
        if phash in self._records:
            return True
        
        # Check near-duplicates
        for existing_hash, _ in self._hash_list:
            if hamming_distance(phash, existing_hash) <= self.threshold:
                return True
        
        return False
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        return {
            "total_registered": len(self._records),
            "threshold": self.threshold,
        }
    
    def clear(self) -> None:
        """Clear all registered images."""
        self._records.clear()
        self._hash_list.clear()
        logger.debug("ImageHashRegistry cleared")


# ============================================================================
# PAGE 1 VALIDATOR (Special Case)
# ============================================================================

class Page1Validator:
    """
    Special validator for Page 1 images (often contains cover/masthead).
    
    Page 1 typically has unique editorial images that should not be
    deduplicated against other pages.
    
    This validator keeps a separate registry for Page 1 and validates
    that any Page 1 duplicate claims are actually legitimate.
    """
    
    def __init__(self) -> None:
        """Initialize the Page 1 validator."""
        self._page1_hashes: List[str] = []
    
    def register_page1_image(self, image: Image.Image) -> str:
        """
        Register an image from Page 1.
        
        Args:
            image: PIL Image from Page 1
            
        Returns:
            The perceptual hash of the image
        """
        phash = compute_phash(image)
        self._page1_hashes.append(phash)
        return phash
    
    def is_valid_page1_duplicate(
        self,
        image: Image.Image,
        claimed_page: int,
    ) -> bool:
        """
        Validate if claiming this image is a Page 1 duplicate is legitimate.
        
        Args:
            image: PIL Image to check
            claimed_page: The page claiming this is a duplicate
            
        Returns:
            True if this is legitimately a duplicate of a Page 1 image
        """
        if claimed_page == 1:
            return False  # Page 1 can't be duplicate of itself
        
        phash = compute_phash(image)
        
        for page1_hash in self._page1_hashes:
            if hamming_distance(phash, page1_hash) <= DEFAULT_THRESHOLD:
                return True
        
        return False


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_image_hash_registry(
    threshold: int = DEFAULT_THRESHOLD,
) -> ImageHashRegistry:
    """
    Factory function to create an ImageHashRegistry.
    
    Args:
        threshold: Hamming distance threshold (0-32)
        
    Returns:
        Configured ImageHashRegistry instance
    """
    return ImageHashRegistry(threshold=threshold)


def create_page1_validator() -> Page1Validator:
    """
    Factory function to create a Page1Validator.
    
    Returns:
        Page1Validator instance
    """
    return Page1Validator()
