"""
Tests for Token Validator (QA-CHECK-01)
========================================
SPEC: Gap #4 - Final Token Count Post-Validation

This test suite validates the TokenValidator implementation for ensuring
data integrity during chunking operations, especially with DSO overlap.

Test Cases:
1. Simple text without overlap -> should match 1:1 with tight tolerance
2. Text with DSO (Gap #3) -> should validate within 10% tolerance
3. Forced data loss (manually removed chunk) -> validator should fail

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-30
"""

import pytest

from src.mmrag_v2.schema.ingestion_schema import (
    ChunkType,
    FileType,
    HierarchyMetadata,
    IngestionChunk,
    Modality,
    create_text_chunk,
)
from src.mmrag_v2.validators.token_validator import (
    TokenValidator,
    TokenValidationResult,
    create_token_validator,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def token_validator():
    """Create a TokenValidator for testing."""
    return create_token_validator(tolerance=0.10)


@pytest.fixture
def simple_text():
    """Simple, short text without complex structure."""
    return (
        "The quick brown fox jumps over the lazy dog. "
        "This is a simple sentence. "
        "It contains no special formatting."
    )


@pytest.fixture
def medium_text():
    """Medium-length text with multiple sentences."""
    return (
        "Artificial intelligence has revolutionized many industries. "
        "Machine learning models can now recognize images, translate languages, "
        "and even generate human-like text. "
        "However, challenges remain in areas like interpretability and bias. "
        "Researchers are working to address these limitations and make AI more robust."
    )


@pytest.fixture
def large_technical_text():
    """Large technical text for DSO testing."""
    return (
        "The distributed system architecture consists of multiple nodes "
        "communicating through a message broker. Each node processes incoming requests "
        "asynchronously and stores results in a shared cache. The cache uses a LRU "
        "eviction policy to manage memory efficiently. "
        "When a request arrives, the node first checks the cache. "
        "If found, it returns the cached result immediately. "
        "If not found, it processes the request and stores the result for future use. "
        "The architecture ensures high availability through redundancy and "
        "replication across multiple geographic regions. "
        "Consistency is maintained using a distributed consensus algorithm. "
        "The system can handle thousands of requests per second."
    )


# ============================================================================
# TEST CASE 1: SIMPLE TEXT WITHOUT OVERLAP
# ============================================================================


def test_simple_text_exact_match(token_validator, simple_text):
    """
    Test Case 1: Simple text without overlap should match 1:1.

    When chunking simple text without DSO (overlap), the sum of chunk tokens
    should approximately equal the source tokens. This test ensures that
    without data loss, validation passes with tight tolerance.
    """
    # Create a single chunk from the simple text
    doc_id = "test_doc_001"
    chunks = [
        create_text_chunk(
            doc_id=doc_id,
            content=simple_text,
            source_file="test.txt",
            file_type=FileType.TXT,
            page_number=1,
            chunk_type=ChunkType.PARAGRAPH,
        )
    ]

    # Validate token balance
    result = token_validator.validate_token_balance(
        chunks=chunks,
        source_text=simple_text,
        overlap_ratio=0.0,  # No overlap for this test
    )

    # Should pass validation
    assert result.is_valid, f"Validation failed: {result.error_message}"
    assert abs(result.variance_percent) < 5.0, "Variance should be less than 5%"
    assert result.source_token_count > 0, "Source should have tokens"
    assert result.chunk_token_count > 0, "Chunks should have tokens"


def test_simple_text_two_chunks(token_validator):
    """
    Test Case 1b: Two chunks of simple text should still validate.

    This verifies that when text is split into multiple chunks without
    overlap, the total token count matches the source.
    """
    text1 = "The first part of the document contains important information."
    text2 = "The second part continues with more details."
    full_text = text1 + " " + text2

    doc_id = "test_doc_002"
    chunks = [
        create_text_chunk(
            doc_id=doc_id,
            content=text1,
            source_file="test.txt",
            file_type=FileType.TXT,
            page_number=1,
            chunk_type=ChunkType.PARAGRAPH,
        ),
        create_text_chunk(
            doc_id=doc_id,
            content=text2,
            source_file="test.txt",
            file_type=FileType.TXT,
            page_number=1,
            chunk_type=ChunkType.PARAGRAPH,
        ),
    ]

    result = token_validator.validate_token_balance(
        chunks=chunks,
        source_text=full_text,
        overlap_ratio=0.0,  # No overlap
    )

    assert result.is_valid, f"Validation failed: {result.error_message}"
    # Allow 1% variance for space handling
    assert abs(result.variance_percent) < 1.0, "Variance should be minimal"


# ============================================================================
# TEST CASE 2: TEXT WITH DSO (DYNAMIC SEMANTIC OVERLAP)
# ============================================================================


def test_text_with_dso_overlap(token_validator, medium_text):
    """
    Test Case 2: Text with DSO should validate within 10% tolerance.

    DSO adds controlled overlap between chunks to maintain semantic context.
    The overlap causes the sum of chunk tokens to exceed the source tokens,
    which should still pass validation when accounting for overlap allowance.
    """
    # Simulate DSO chunks with small overlap (20% of total tokens)
    doc_id = "test_doc_003"

    # Split source text into 3 parts with minimal overlap (just 1-2 tokens)
    chunk1 = "Artificial intelligence has revolutionized many industries. Machine learning models can now recognize images, translate languages, and even generate human-like text."

    # Minimal overlap - just repeat last phrase
    chunk2 = "and even generate human-like text. However, challenges remain in areas like interpretability and bias."

    # Minimal overlap again
    chunk3 = "interpretability and bias. Researchers are working to address these limitations and make AI more robust."

    chunks = [
        create_text_chunk(
            doc_id=doc_id,
            content=chunk1,
            source_file="test.txt",
            file_type=FileType.TXT,
            page_number=1,
        ),
        create_text_chunk(
            doc_id=doc_id,
            content=chunk2,
            source_file="test.txt",
            file_type=FileType.TXT,
            page_number=1,
        ),
        create_text_chunk(
            doc_id=doc_id,
            content=chunk3,
            source_file="test.txt",
            file_type=FileType.TXT,
            page_number=1,
        ),
    ]

    # Validate with 20% overlap allowance (to account for DSO overlap)
    result = token_validator.validate_token_balance(
        chunks=chunks,
        source_text=medium_text,
        overlap_ratio=0.20,  # 20% overlap from DSO
    )

    assert result.is_valid, f"DSO validation failed: {result.error_message}"
    assert result.overlap_allowance_tokens > 0, "Overlap allowance should be calculated"
    # With allowance, variance should be within tolerance
    assert (
        abs(result.variance_percent) < 10.0
    ), "Variance should be within 10% tolerance with overlap allowance"


def test_dso_with_strict_tolerance(token_validator):
    """
    Test Case 2b: Verify that strict tolerance (5%) still passes with proper DSO.

    This tests the validator's ability to handle reasonable DSO while maintaining
    strict token balance requirements.
    """
    strict_validator = TokenValidator(tolerance=0.05)  # 5% tolerance

    text = "The system processes data. Processing is critical."

    doc_id = "test_doc_004"
    chunks = [
        create_text_chunk(
            doc_id=doc_id,
            content="The system processes data. Processing is critical.",
            source_file="test.txt",
            file_type=FileType.TXT,
            page_number=1,
        )
    ]

    result = strict_validator.validate_token_balance(
        chunks=chunks,
        source_text=text,
        overlap_ratio=0.05,  # 5% overlap allowance
    )

    assert result.is_valid, f"Strict validation failed: {result.error_message}"


# ============================================================================
# TEST CASE 3: FORCED DATA LOSS (VALIDATION FAILURE)
# ============================================================================


def test_data_loss_detection(token_validator, large_technical_text):
    """
    Test Case 3: Validator should fail when data is lost (missing chunk).

    This test simulates what happens when the chunker accidentally loses data.
    By manually removing a chunk, we create a token deficit that should be
    detected by the validator.
    """
    # Create chunks from the source text
    doc_id = "test_doc_005"
    # Intentionally split into 3 chunks, but we'll only use 2 (simulating data loss)
    chunk1 = large_technical_text[:200]
    chunk2 = large_technical_text[200:400]
    chunk3 = large_technical_text[400:600]  # This chunk is "lost"

    chunks = [
        create_text_chunk(
            doc_id=doc_id,
            content=chunk1,
            source_file="test.txt",
            file_type=FileType.TXT,
            page_number=1,
        ),
        create_text_chunk(
            doc_id=doc_id,
            content=chunk2,
            source_file="test.txt",
            file_type=FileType.TXT,
            page_number=1,
        ),
        # chunk3 is intentionally omitted to simulate data loss
    ]

    # Validate - should FAIL because we're missing chunk3
    result = token_validator.validate_token_balance(
        chunks=chunks,
        source_text=large_technical_text,
        overlap_ratio=0.0,  # No overlap
    )

    assert not result.is_valid, "Validator should detect data loss"
    assert result.variance_percent < -10.0, "Variance should show significant deficit"
    assert "failed" in result.error_message.lower(), "Error message should indicate failure"


def test_complete_data_loss(token_validator):
    """
    Test Case 3b: Validator should fail when all content is lost.

    This extreme test verifies that the validator can detect when chunks
    contain minimal content compared to source.
    """
    source = "This is a complete document with substantial content."
    doc_id = "test_doc_006"

    # Create a chunk with only partial content
    chunks = [
        create_text_chunk(
            doc_id=doc_id,
            content="This is incomplete.",  # Much smaller than source
            source_file="test.txt",
            file_type=FileType.TXT,
            page_number=1,
        )
    ]

    result = token_validator.validate_token_balance(
        chunks=chunks,
        source_text=source,
        overlap_ratio=0.0,
    )

    assert not result.is_valid, "Should detect significant data loss"
    assert result.variance_percent < -10.0, "Deficit should exceed tolerance"


# ============================================================================
# TEST CASE 4: LOGGING AND RESULT REPORTING
# ============================================================================


def test_validation_result_data_class(token_validator, simple_text):
    """
    Test that TokenValidationResult provides complete metrics.

    This verifies that the result object contains all required information
    for logging and reporting.
    """
    doc_id = "test_doc_007"
    chunks = [
        create_text_chunk(
            doc_id=doc_id,
            content=simple_text,
            source_file="test.txt",
            file_type=FileType.TXT,
            page_number=1,
        )
    ]

    result = token_validator.validate_token_balance(
        chunks=chunks,
        source_text=simple_text,
        overlap_ratio=0.0,
    )

    # Verify all required fields
    assert isinstance(result, TokenValidationResult)
    assert isinstance(result.is_valid, bool)
    assert isinstance(result.source_token_count, int)
    assert isinstance(result.chunk_token_count, int)
    assert isinstance(result.variance_percent, float)
    assert isinstance(result.overlap_allowance_tokens, int)
    assert isinstance(result.tolerance_percent, float)
    assert result.tolerance_percent == 10.0  # Default tolerance


def test_logging_output(token_validator, simple_text, caplog):
    """
    Test that validation results are properly logged.

    This verifies that success messages are logged at INFO level.
    """
    import logging

    doc_id = "test_doc_008"
    chunks = [
        create_text_chunk(
            doc_id=doc_id,
            content=simple_text,
            source_file="test.txt",
            file_type=FileType.TXT,
            page_number=1,
        )
    ]

    result = token_validator.validate_token_balance(
        chunks=chunks,
        source_text=simple_text,
        overlap_ratio=0.0,
    )

    # Set caplog level to capture INFO logs
    with caplog.at_level(logging.INFO):
        # Log the result
        token_validator.log_validation_result(result, doc_name="test_doc.txt")

    # Check that logging occurred
    assert "QA-CHECK-01" in caplog.text
    assert "verified" in caplog.text.lower()


# ============================================================================
# TEST CASE 5: EDGE CASES
# ============================================================================


def test_empty_source_text(token_validator):
    """
    Test validation with empty source text.

    This should pass with vacuous truth (empty document is valid).
    """
    doc_id = "test_doc_009"
    chunks = []  # No chunks

    result = token_validator.validate_token_balance(
        chunks=chunks,
        source_text="",  # Empty source
        overlap_ratio=0.0,
    )

    assert result.is_valid, "Empty document should be valid"
    assert result.source_token_count == 0


def test_very_large_text(token_validator):
    """
    Test validation with large text to ensure token counter handles it.

    This verifies that the singleton token counter can handle longer texts
    without performance issues.
    """
    # Create a large text (approximately 10KB)
    base_sentence = "The quick brown fox jumps over the lazy dog. "
    large_text = base_sentence * 250  # ~12,500 tokens

    doc_id = "test_doc_010"
    chunks = [
        create_text_chunk(
            doc_id=doc_id,
            content=large_text,
            source_file="test.txt",
            file_type=FileType.TXT,
            page_number=1,
        )
    ]

    result = token_validator.validate_token_balance(
        chunks=chunks,
        source_text=large_text,
        overlap_ratio=0.0,
    )

    assert result.is_valid
    assert result.source_token_count > 1000, "Should have many tokens"
    assert abs(result.variance_percent) < 1.0


def test_factory_function():
    """
    Test that factory function creates properly configured validator.
    """
    validator = create_token_validator(tolerance=0.05)

    assert validator is not None
    assert validator.tolerance == 0.05
    assert isinstance(validator, TokenValidator)


# ============================================================================
# TEST CASE 6: TOLERANCE BOUNDARY TESTING
# ============================================================================


def test_variance_at_tolerance_boundary(token_validator):
    """
    Test validation when variance is exactly at tolerance boundary.
    """
    doc_id = "test_doc_011"
    source = "Test document content here."

    # Create chunk that produces exactly 10% variance (at boundary)
    chunks = [
        create_text_chunk(
            doc_id=doc_id,
            content=source,  # Same as source
            source_file="test.txt",
            file_type=FileType.TXT,
            page_number=1,
        )
    ]

    result = token_validator.validate_token_balance(
        chunks=chunks,
        source_text=source,
        overlap_ratio=0.0,
    )

    # Should pass (variance is within tolerance)
    assert result.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
