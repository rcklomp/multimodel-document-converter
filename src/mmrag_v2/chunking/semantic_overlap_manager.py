"""
Dynamic Semantic Overlap (DSO) Manager
======================================
REQ-CHUNK-03: Intelligent chunk overlap based on semantic similarity.

This module implements the Dynamic Semantic Overlap (DSO) algorithm:
1. Extract last 3 sentences from Chunk A
2. Extract first 3 sentences from Chunk B
3. Compute cosine similarity using sentence-transformers
4. If sim > 0.85: overlap = base_overlap * 1.5
5. Otherwise: overlap = base_overlap
6. Constraint: overlap < 25% of total chunk size

Features:
- Lazy loading of embedding model (Singleton pattern)
- Memory-efficient caching on Apple Silicon
- Sentence segmentation for proper semantic units
- Configurable thresholds

Author: Claude (Senior Architect)
Date: 2026-01-02
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ============================================================================
# SINGLETON EMBEDDING MODEL MANAGER
# ============================================================================


class EmbeddingModelManager:
    """
    Lazy-loading Singleton for sentence-transformers embedding model.

    This ensures the model is loaded exactly ONCE in memory, even when
    processing multiple documents or in batch mode. On Apple Silicon,
    this is critical to avoid OOM errors.

    The model `sentence-transformers/all-MiniLM-L6-v2` is extremely
    lightweight (~22MB) and runs efficiently.

    Device Selection (Apple Silicon Optimization):
    - Attempts MPS (Metal Performance Shaders) first if available
    - Falls back to CPU if MPS not available
    - For batch processing (1000+ docs), MPS provides significant acceleration
    """

    _instance: Optional[EmbeddingModelManager] = None
    _model = None
    _is_initialized: bool = False

    @staticmethod
    def _detect_device() -> str:
        """
        Detect optimal device for embeddings.

        Returns: "mps" if Apple Silicon GPU available, "cpu" otherwise
        """
        try:
            import torch

            if torch.backends.mps.is_available():
                logger.info("✓ Apple Silicon MPS detected. Using GPU for embeddings.")
                return "mps"
        except Exception as e:
            logger.debug(f"MPS detection failed: {e}")

        logger.info("Using CPU for embeddings")
        return "cpu"

    def __new__(cls) -> EmbeddingModelManager:
        """Implement Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(EmbeddingModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize (only once per process)."""
        # Prevent re-initialization on subsequent __new__ calls
        if not EmbeddingModelManager._is_initialized:
            EmbeddingModelManager._is_initialized = True
            self._load_model()

    def _load_model(self) -> None:
        """Lazy-load the embedding model on first use."""
        if EmbeddingModelManager._model is not None:
            logger.debug("Embedding model already loaded (Singleton)")
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")

            # REQ-CHUNK-03: Use specified model version
            model_name = "sentence-transformers/all-MiniLM-L6-v2"

            # Detect optimal device (MPS preferred on Apple Silicon)
            device = self._detect_device()

            EmbeddingModelManager._model = SentenceTransformer(model_name, device=device)

            logger.info(f"✓ Embedding model loaded: {model_name} on device={device}")

        except ImportError as e:
            logger.error(
                f"Failed to import sentence_transformers: {e}\n"
                f"Install with: pip install sentence-transformers>=3.0.0"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def get_model(self):
        """Get the loaded embedding model."""
        if EmbeddingModelManager._model is None:
            self._load_model()
        return EmbeddingModelManager._model

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If model failed to load
        """
        model = self.get_model()
        if model is None:
            raise RuntimeError(
                "Embedding model not loaded. Check installation of sentence-transformers."
            )
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


# ============================================================================
# SENTENCE SEGMENTATION
# ============================================================================


class SentenceSegmenter:
    """Extract sentence boundaries for DSO algorithm using NLTK punkt tokenizer."""

    _nltk_available: Optional[bool] = None

    @staticmethod
    def _check_nltk_available() -> bool:
        """Check if NLTK sent_tokenize is available (lazy check, cached)."""
        if SentenceSegmenter._nltk_available is None:
            try:
                import nltk  # type: ignore[import-not-found]

                # Verify punkt data is available by trying a test tokenize
                nltk.sent_tokenize("Test sentence.")
                SentenceSegmenter._nltk_available = True
                logger.debug("✓ NLTK sent_tokenize available")
            except ImportError:
                logger.warning("NLTK not installed. Falling back to regex-based segmentation.")
                SentenceSegmenter._nltk_available = False
            except LookupError:
                # Punkt data not downloaded
                logger.warning(
                    "NLTK punkt data not found. Run: python -m nltk.downloader punkt. "
                    "Falling back to regex-based segmentation."
                )
                SentenceSegmenter._nltk_available = False
            except Exception as e:
                logger.warning(f"Failed to initialize NLTK: {e}. Using regex fallback.")
                SentenceSegmenter._nltk_available = False

        return SentenceSegmenter._nltk_available

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences at proper boundaries using NLTK Punkt.

        NLTK Punkt is a robust sentence tokenizer that handles:
        - Standard sentence endings (., !, ?)
        - Abbreviations (Dr., U.S., Fig., etc.)
        - Decimal numbers (p. 42)
        - Newlines as sentence boundaries

        Falls back to regex if NLTK unavailable.

        Args:
            text: Input text

        Returns:
            List of sentence strings (stripped)
        """
        if not text or not text.strip():
            return []

        # Try NLTK first (production mode)
        if SentenceSegmenter._check_nltk_available():
            try:
                import nltk  # type: ignore[import-not-found]

                sentences = nltk.sent_tokenize(text)
                return [s.strip() for s in sentences if s.strip()]
            except Exception as e:
                logger.warning(f"NLTK tokenization failed: {e}. Using regex fallback.")

        # Fallback: regex-based segmentation (simple, less robust)
        return SentenceSegmenter._regex_split_sentences(text)

    @staticmethod
    def _regex_split_sentences(text: str) -> List[str]:
        """
        Fallback regex-based sentence segmentation.

        Less robust than NLTK but works without external dependencies.
        """
        # Replace newlines with sentence breaks
        text = text.replace("\n", " . ")

        # Simple regex split on common sentence endings
        sentences = re.split(r"[.!?]\s+", text)
        return [s.strip() for s in sentences if s.strip()]


# ============================================================================
# DYNAMIC SEMANTIC OVERLAP (DSO) ALGORITHM
# ============================================================================


class DSOCalculator:
    """
    Calculate dynamic semantic overlap between chunks.

    REQ-CHUNK-04: DSO Algorithm
    - Extract last 3 sentences from Chunk A
    - Extract first 3 sentences from Chunk B
    - Compute cosine similarity
    - Apply multiplier: if sim > 0.85: overlap * 1.5, else overlap

    REQ-CHUNK-05: Constraint - overlap < 25% of chunk size
    """

    # Thresholds (from SRS v2.3)
    SIMILARITY_THRESHOLD: float = 0.85
    OVERLAP_MULTIPLIER: float = 1.5
    MAX_OVERLAP_RATIO: float = 0.25  # 25% of chunk size

    def __init__(self, enable_dso: bool = False) -> None:
        """
        Initialize DSO calculator.

        Args:
            enable_dso: Whether to enable DSO. If False, uses base overlap only.
        """
        self.enable_dso = enable_dso
        self._model_manager: Optional[EmbeddingModelManager] = None

        if enable_dso:
            try:
                self._model_manager = EmbeddingModelManager()
                logger.info("✓ Dynamic Semantic Overlap (DSO) enabled")
            except Exception as e:
                logger.error(f"Failed to enable DSO: {e}. Falling back to static overlap.")
                self.enable_dso = False

    def calculate_overlap(
        self,
        chunk_a: str,
        chunk_b: str,
        base_overlap_chars: int,
    ) -> int:
        """
        Calculate overlap length between two chunks based on semantic similarity.

        Algorithm:
        1. Extract last 3 sentences from chunk_a
        2. Extract first 3 sentences from chunk_b
        3. Compute cosine similarity
        4. If sim > 0.85: use base_overlap * 1.5
        5. Otherwise: use base_overlap
        6. Cap at 25% of chunk_b size

        Args:
            chunk_a: First chunk (we take last sentences)
            chunk_b: Second chunk (we take first sentences)
            base_overlap_chars: Base overlap in characters

        Returns:
            Actual overlap length in characters
        """
        if not self.enable_dso or self._model_manager is None:
            # Fallback to static overlap
            return min(base_overlap_chars, int(len(chunk_b) * self.MAX_OVERLAP_RATIO))

        try:
            # Extract sentences
            sentences_a = SentenceSegmenter.split_into_sentences(chunk_a)
            sentences_b = SentenceSegmenter.split_into_sentences(chunk_b)

            # Take last 3 from A, first 3 from B
            tail_a = sentences_a[-3:] if sentences_a else []
            head_b = sentences_b[:3] if sentences_b else []

            # If either is empty, fall back to base overlap
            if not tail_a or not head_b:
                logger.debug("DSO fallback: insufficient sentences for similarity calc")
                return min(base_overlap_chars, int(len(chunk_b) * self.MAX_OVERLAP_RATIO))

            # Compute embeddings
            tail_text = " ".join(tail_a)
            head_text = " ".join(head_b)

            embeddings = self._model_manager.encode([tail_text, head_text])

            # Cosine similarity (normalized vectors)
            import numpy as np

            tail_emb = np.array(embeddings[0])
            head_emb = np.array(embeddings[1])

            # Normalize vectors
            tail_emb = tail_emb / (np.linalg.norm(tail_emb) + 1e-8)
            head_emb = head_emb / (np.linalg.norm(head_emb) + 1e-8)

            # Cosine similarity
            similarity = float(np.dot(tail_emb, head_emb))

            logger.debug(f"DSO similarity: {similarity:.3f} | A→B overlap decision")

            # Apply multiplier rule
            if similarity > self.SIMILARITY_THRESHOLD:
                overlap_chars = int(base_overlap_chars * self.OVERLAP_MULTIPLIER)
                logger.debug(
                    f"High semantic similarity ({similarity:.3f} > {self.SIMILARITY_THRESHOLD}) "
                    f"→ overlap increased: {base_overlap_chars} → {overlap_chars}"
                )
            else:
                overlap_chars = base_overlap_chars
                logger.debug(
                    f"Low/medium similarity ({similarity:.3f} ≤ {self.SIMILARITY_THRESHOLD}) "
                    f"→ using base overlap: {base_overlap_chars}"
                )

            # REQ-CHUNK-05: Cap at 25% of chunk_b size
            max_overlap = int(len(chunk_b) * self.MAX_OVERLAP_RATIO)
            final_overlap = min(overlap_chars, max_overlap)

            if final_overlap < overlap_chars:
                logger.debug(
                    f"Overlap capped at 25% rule: {overlap_chars} → {final_overlap} "
                    f"(max {max_overlap} for chunk of {len(chunk_b)} chars)"
                )

            return final_overlap

        except Exception as e:
            logger.warning(f"DSO calculation failed: {e}. Using base overlap.")
            return min(base_overlap_chars, int(len(chunk_b) * self.MAX_OVERLAP_RATIO))


# ============================================================================
# TOKEN VALIDATION (QA-CHECK-01)
# ============================================================================


class TokenValidator:
    """
    Validate token counts across document chunks.

    QA-CHECK-01: Verify `sum(chunk_tokens) ~= total_document_tokens` (tolerance 10%)
    """

    def __init__(self) -> None:
        """Initialize token counter."""
        self._tokenizer = None

    def _load_tokenizer(self):
        """Lazy-load tiktoken tokenizer."""
        if self._tokenizer is not None:
            return

        try:
            import tiktoken

            self._tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info("✓ Token validator initialized (cl100k_base)")
        except ImportError:
            logger.warning(
                "tiktoken not installed. Token validation disabled. "
                "Install with: pip install tiktoken"
            )
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using cl100k_base encoding.

        Falls back to character-based estimation if tiktoken unavailable.

        Args:
            text: Input text

        Returns:
            Token count (approximate if tiktoken unavailable)
        """
        self._load_tokenizer()

        if self._tokenizer is None:
            # Fallback: ~4 characters per token on average
            return max(1, len(text) // 4)

        try:
            tokens = self._tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Token counting failed: {e}. Using estimate.")
            return max(1, len(text) // 4)

    def validate_chunk_tokens(
        self,
        chunks: List[str],
        total_document_tokens: int,
        tolerance: float = 0.10,
    ) -> Tuple[bool, str]:
        """
        Validate that sum of chunk tokens matches document total.

        QA-CHECK-01: tolerance = 10%

        Args:
            chunks: List of chunk texts
            total_document_tokens: Total tokens in original document
            tolerance: Acceptable variance ratio (default 0.10 = 10%)

        Returns:
            Tuple of (is_valid, message)
        """
        chunk_tokens = sum(self.count_tokens(c) for c in chunks)

        if total_document_tokens == 0:
            return True, "Empty document (0 tokens)"

        variance = abs(chunk_tokens - total_document_tokens) / total_document_tokens

        is_valid = variance <= tolerance

        status = "✓ PASS" if is_valid else "✗ FAIL"
        message = (
            f"{status} | QA-CHECK-01 Token Validation | "
            f"Chunks: {chunk_tokens} tokens | "
            f"Document: {total_document_tokens} tokens | "
            f"Variance: {variance*100:.1f}% (tolerance: {tolerance*100:.1f}%)"
        )

        if is_valid:
            logger.info(message)
        else:
            logger.warning(message)

        return is_valid, message
