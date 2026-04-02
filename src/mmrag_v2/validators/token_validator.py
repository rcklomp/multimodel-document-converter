"""
Token Validator - QA-CHECK-01 Data Integrity Guard
===================================================
SPEC: Gap #4 - Final Token Count Post-Validation

This module implements the critical "data integrity" validator that ensures
no information is lost during text chunking, especially when using Dynamic
Semantic Overlap (DSO).

**Core Problem (Why QA-CHECK-01 is essential):**

Without this validator, we can never be sure if the chunker accidentally
"ate" (lost) some text. Because DSO adds controlled overlap, the sum of
chunk tokens will ALWAYS be higher than the source text tokens. This creates
an impossible audit scenario without a formal validation station.

**The Science:**
- `total_source_tokens = count_tokens(source_text)` using tiktoken cl100k_base
- `sum_chunk_tokens = sum(count_tokens(chunk.content) for chunk in chunks)`
- Because overlap exists: `sum_chunk_tokens >= total_source_tokens` (always)
- The check: `(sum_chunk_tokens - overlap_allowance) / total_source_tokens`
  must be within tolerance (10% default)

**REQ Compliance:**
- QA-CHECK-01: Token balance verification with overlap awareness
- REQ-CHUNK-03: DSO overlap < 25% per chunk (overlap_allowance = estimated DSO tokens)
- REQ-ERR-01: Per-file error handling (no crash on validation failure)
- REQ-ERR-03: Startup logging of "Using Docling v2.66.0" at INFO

**Strict Mode (--strict-qa):**
When enabled, validation failure throws Exception and stops document processing.
This prevents bad documents from being silently ingested.

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-30
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Dict, Any

try:
    import tiktoken
except ImportError:
    raise ImportError(
        "tiktoken is required for token validation. Install with: pip install tiktoken"
    )

if TYPE_CHECKING:
    from ..schema.ingestion_schema import IngestionChunk
    from .quality_filter_tracker import QualityFilterTracker, QualityFilterSummary

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

ENCODING_NAME: str = "cl100k_base"  # OpenAI's standard encoding (GPT-3.5, GPT-4)
DEFAULT_TOLERANCE: float = 0.10  # 10% variance tolerance
DEFAULT_OVERLAP_ALLOWANCE: float = 0.15  # 15% estimated overlap from DSO


# ============================================================================
# VALIDATION RESULT
# ============================================================================


@dataclass
class TokenValidationResult:
    """Result of token balance validation with filtering awareness."""

    is_valid: bool
    source_token_count: int
    chunk_token_count: int
    variance_percent: float
    overlap_allowance_tokens: int
    tolerance_percent: float
    error_message: Optional[str] = None

    # Filtering analytics (new in v2.4)
    filtered_token_count: int = 0
    filtered_ratio_percent: float = 0.0
    noise_allowance_percent: float = 0.0
    is_filtered_ratio_valid: bool = True
    adjusted_source_token_count: int = 0


# ============================================================================
# TOKEN COUNTER (SINGLETON)
# ============================================================================


class _TokenCounter:
    """
    Singleton token counter using tiktoken cl100k_base.

    Motivation for singleton:
    - tiktoken.encoding_for_model() is expensive (loads BSON file)
    - Reusing the same encoder avoids repeated disk I/O
    - Thread-safe at the encoding.encode() level
    """

    _instance: Optional[_TokenCounter] = None
    _encoder: Optional[tiktoken.Encoding] = None

    def __new__(cls) -> _TokenCounter:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_encoder()
        return cls._instance

    def _initialize_encoder(self) -> None:
        """Initialize tiktoken encoder (one-time cost)."""
        try:
            self._encoder = tiktoken.get_encoding(ENCODING_NAME)
            logger.debug(f"Initialized tiktoken encoder: {ENCODING_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize tiktoken encoder: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using cl100k_base encoding.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens
        """
        if not text:
            return 0
        if self._encoder is None:
            logger.error("Token encoder not initialized")
            return 0
        try:
            tokens = self._encoder.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            return 0


# ============================================================================
# VALIDATOR
# ============================================================================


class TokenValidator:
    """
    QA-CHECK-01 validator for token balance verification.

    This is the "Guardian of Data Integrity" - it ensures that chunking
    operations don't lose information, even when DSO introduces overlap.

    **CRITICAL ARCHITECTURAL NOTES:**

    1. DENOISED SOURCE REQUIREMENT:
       The source_text parameter MUST be the denoised version (after removal of ads,
       navigation, mastheads, etc.). The validator compares clean text against clean
       chunks. If you pass raw PDF text containing ads, validation will incorrectly fail.

    2. TEXT-ONLY VALIDATION:
       This validator audits TEXT CHUNKS against TEXT SOURCE. It intentionally does NOT
       validate image/table content because:
       - Images are atomic semantic units (IRON-01) that cannot be "chunked"
       - visual_description is VLM-generated metadata, not source text that was lost
       - Image integrity is validated separately via asset_ref existence checks
       For future multi-modal balance validation, see Gap #5.

    3. OVERLAP AWARENESS:
       The validator understands that Dynamic Semantic Overlap (DSO) adds controlled
       redundancy. Expected formula:
       (sum_chunk_tokens - overlap_allowance) / source_tokens should be ~0 ± tolerance
    """

    def __init__(self, tolerance: float = DEFAULT_TOLERANCE):
        """
        Initialize validator.

        Args:
            tolerance: Allowed variance as decimal (0.10 = 10%, default=10%)
        """
        self.tolerance = tolerance
        self._counter = _TokenCounter()

        logger.info(f"TokenValidator initialized: tolerance={tolerance*100:.1f}%")

    def validate_token_balance(
        self,
        chunks: List["IngestionChunk"],
        source_text: str,
        overlap_ratio: float = DEFAULT_OVERLAP_ALLOWANCE,
        quality_filter_tracker: Optional["QualityFilterTracker"] = None,
        profile_type: str = "unknown",
        noise_allowance: Optional[float] = None,
    ) -> TokenValidationResult:
        """
        Validate that chunk tokens match source tokens (with overlap allowance and filtering awareness).

        **New Algorithm with Filtering Awareness:**
        1. Count tokens in source_text using tiktoken cl100k_base
        2. Sum tokens across all chunks using tiktoken
        3. Get filtered tokens from QualityFilterTracker (if available)
        4. Calculate effective source tokens: source_tokens - filtered_tokens
        5. Calculate filtered ratio and check against noise allowance (profile-specific)
        6. Calculate variance: (chunk_tokens - overlap_allowance - effective_source_tokens) / effective_source_tokens
        7. Check if variance is within tolerance AND filtered ratio is within noise allowance

        Args:
            chunks: List of IngestionChunk objects
            source_text: Full source text (original document text)
            overlap_ratio: Expected overlap as decimal (0.15 = 15% overlap)
            quality_filter_tracker: Optional QualityFilterTracker with filtered token analytics
            profile_type: Document profile type (e.g., "academic_whitepaper")
            noise_allowance: Maximum allowed filtered ratio (overrides profile-based default)

        Returns:
            TokenValidationResult with filtering analytics
        """
        # Step 1: Count source tokens
        source_token_count = self._counter.count_tokens(source_text)

        if source_token_count == 0:
            logger.warning("Source text has zero tokens (empty document?)")
            return TokenValidationResult(
                is_valid=True,  # Empty docs are valid (vacuous truth)
                source_token_count=0,
                chunk_token_count=0,
                variance_percent=0.0,
                overlap_allowance_tokens=0,
                tolerance_percent=self.tolerance,
                error_message="Empty source text",
            )

        # Step 2: Sum chunk tokens
        chunk_token_count = 0
        for chunk in chunks:
            chunk_tokens = self._counter.count_tokens(chunk.content)
            chunk_token_count += chunk_tokens

        # Step 3: Calculate overlap allowance in tokens
        # Do not penalize further when we already have fewer chunk tokens than source
        overlap_allowance_tokens = int(source_token_count * overlap_ratio)
        if chunk_token_count < source_token_count:
            overlap_allowance_tokens = 0

        # Step 4: Handle filtered tokens and noise allowance
        filtered_token_count = 0
        filtered_ratio = 0.0
        is_filtered_ratio_valid = True
        adjusted_source_token_count = source_token_count

        # Get noise allowance for profile if not provided
        if noise_allowance is None:
            noise_allowance = self._get_noise_allowance_for_profile(profile_type)

        # If we have a quality filter tracker, get filtered token analytics
        if quality_filter_tracker is not None:
            filtered_summary = quality_filter_tracker.get_summary()
            filtered_token_count = filtered_summary.total_filtered_tokens

            if source_token_count > 0:
                filtered_ratio = filtered_token_count / source_token_count
                # Check if filtered ratio exceeds noise allowance for this profile
                is_filtered_ratio_valid = filtered_ratio <= noise_allowance

            # Calculate adjusted source tokens (source minus intentionally filtered tokens)
            adjusted_source_token_count = max(0, source_token_count - filtered_token_count)

        # Step 5: Calculate variance with adjusted source tokens
        # Note: We subtract filtered tokens from source because they were intentionally removed
        effective_chunk_count = chunk_token_count - overlap_allowance_tokens
        variance_tokens = effective_chunk_count - adjusted_source_token_count
        variance_percent = (
            (variance_tokens / adjusted_source_token_count)
            if adjusted_source_token_count > 0
            else 0.0
        )

        # Step 6: Check tolerance (variance and filtered ratio)
        # Negative variance (missing tokens) can be acceptable up to noise_allowance for the profile.
        if variance_percent < 0:
            allowed_negative = max(self.tolerance, noise_allowance)
            is_variance_valid = abs(variance_percent) <= allowed_negative
        else:
            is_variance_valid = variance_percent <= self.tolerance

        is_valid = is_variance_valid and is_filtered_ratio_valid

        # Create result with filtering analytics
        result = TokenValidationResult(
            is_valid=is_valid,
            source_token_count=source_token_count,
            chunk_token_count=chunk_token_count,
            variance_percent=variance_percent * 100,
            overlap_allowance_tokens=overlap_allowance_tokens,
            tolerance_percent=self.tolerance * 100,
            filtered_token_count=filtered_token_count,
            filtered_ratio_percent=filtered_ratio * 100,
            noise_allowance_percent=noise_allowance * 100,
            is_filtered_ratio_valid=is_filtered_ratio_valid,
            adjusted_source_token_count=adjusted_source_token_count,
        )

        if not is_valid:
            if not is_variance_valid:
                result.error_message = (
                    f"Token balance validation failed: "
                    f"variance {variance_percent*100:.1f}% exceeds tolerance {self.tolerance*100:.1f}%"
                )
            elif not is_filtered_ratio_valid:
                result.error_message = (
                    f"Filtered ratio exceeds noise allowance: "
                    f"filtered {filtered_ratio*100:.1f}% > allowance {noise_allowance*100:.1f}% "
                    f"for profile '{profile_type}'"
                )

        return result

    @staticmethod
    def _get_noise_allowance_for_profile(profile_type: str) -> float:
        """
        Get noise allowance percentage (as decimal) for a given profile type.

        These allowances represent the maximum acceptable ratio of filtered tokens
        to source tokens for each document profile.

        Returns:
            Noise allowance as decimal (e.g., 0.25 = 25%)
        """
        noise_allowances = {
            "academic_whitepaper": 0.25,  # 25% - high bibliography/footnote noise
            "technical_manual": 0.15,  # 15% - moderate noise
            "digital_magazine": 0.10,  # 10% - low noise
            "scanned_degraded": 0.05,  # 5%  - minimal filtering for OCR
            "scanned": 0.05,  # 5%  - minimal filtering for OCR
            "unknown": 0.10,  # 10% - default
        }
        return noise_allowances.get(profile_type, 0.10)  # Default 10%

    def log_validation_result(
        self,
        result: TokenValidationResult,
        doc_name: str = "Unknown",
    ) -> None:
        """
        Log validation result with appropriate level, including filtering analytics.

        Args:
            result: TokenValidationResult object with filtering analytics
            doc_name: Document name for context in log
        """
        # Base metrics
        base_metrics = (
            f"Source={result.source_token_count} tokens, "
            f"Chunks={result.chunk_token_count} tokens, "
            f"Overlap={result.overlap_allowance_tokens} tokens, "
            f"Variance={result.variance_percent:+.1f}%"
        )

        # Filtering analytics (if available)
        filtering_metrics = ""
        if result.filtered_token_count > 0:
            filtering_metrics = (
                f", Filtered={result.filtered_token_count} tokens "
                f"({result.filtered_ratio_percent:.1f}% of source, "
                f"allowance={result.noise_allowance_percent:.1f}%)"
            )

        metrics = base_metrics + filtering_metrics

        if result.is_valid:
            if result.filtered_token_count > 0:
                logger.info(
                    f"[QA-CHECK-01] ✓ Token balance verified with filtering ({doc_name}): {metrics}"
                )
            else:
                logger.info(f"[QA-CHECK-01] ✓ Token balance verified ({doc_name}): {metrics}")
        else:
            if not result.is_filtered_ratio_valid:
                logger.critical(
                    f"[QA-CHECK-01] ✗ CRITICAL WARNING - Filtered ratio exceeded ({doc_name}): {metrics} "
                    f"(filtered {result.filtered_ratio_percent:.1f}% > allowance {result.noise_allowance_percent:.1f}%). "
                    f"Too much content filtered for profile."
                )
            else:
                logger.critical(
                    f"[QA-CHECK-01] ✗ CRITICAL WARNING ({doc_name}): {metrics} "
                    f"(exceeds {result.tolerance_percent:.1f}% tolerance). "
                    f"Possible data loss during chunking."
                )


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_token_validator(tolerance: float = DEFAULT_TOLERANCE) -> TokenValidator:
    """
    Factory function to create a TokenValidator.

    Args:
        tolerance: Allowed variance as decimal (0.10 = 10%)

    Returns:
        Configured TokenValidator instance
    """
    return TokenValidator(tolerance=tolerance)
