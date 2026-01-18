"""
Semantic Text Refiner - Hybrid OCR Artifact Repair Engine (v18.2)
==================================================================
ENGINE_USE: Hybrid LLM + Heuristic Post-Processing

This module implements a 3-stage "sandwich" architecture for fixing OCR
artifacts while maintaining 100% technical integrity:

Stage A: NoiseScanner (Diagnostic Triage)
    - Calculates corruption_score based on OCR artifact patterns
    - Only sends high-corruption text to LLM (efficiency)

Stage B: ContextualRefiner (LLM Layer)
    - Uses visual_description as Named Entity Anchor
    - Contextual grounding with semantic neighbors
    - Deterministic (temp=0, strict prompt)

Stage C: ConsistencyValidator (Integrity Guardrail)
    - Levenshtein edit-distance budget enforcement
    - 0% edit tolerance for protected tokens (ECU codes, P/N, etc.)
    - Rejects over-edited LLM outputs

REQ Compliance:
- REQ-REFINE-01: Original content NEVER overwritten
- REQ-REFINE-02: Protected tokens have 0% edit budget
- REQ-REFINE-03: Vision anchoring for entity disambiguation
- REQ-REFINE-04: Graceful degradation on LLM failures

Author: Claude 4.5 Opus (Architect)
Date: 2026-01-13
Version: 18.2 (Legendary Edition)
"""

from __future__ import annotations

import logging
import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import requests

from .schema.ingestion_schema import SemanticContext

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# DEFAULT EDIT BUDGET - SINGLE SOURCE OF TRUTH
# This is the maximum allowed edit ratio for LLM refinements (35% = balanced)
DEFAULT_MAX_EDIT_RATIO: float = 0.35

# Stage A: Corruption Pattern Weights (multi-language: EN/DE/NL)
CORRUPTION_PATTERNS: Dict[str, Tuple[str, float]] = {
    "broken_words": (r"\b\w{1,2}\s+\w{1,2}\s+\w{1,2}\b", 0.3),  # "G u l f"
    "floating_hyphens": (r"\s-\s|\s-$|^-\s", 0.2),  # " - "
    "excessive_kerning": (r"(\w)\s{2,}(\w)", 0.25),  # "air  craft"
    "ocr_artifacts": (r"[|lI1]{3,}|[O0]{3,}", 0.3),  # "|||" or "OOO"
    "ligature_breaks": (r"fi\s|fl\s|ff\s", 0.15),  # "fi " → "fi"
    "german_umlaut_errors": (r'[aou]\s*[¨"]', 0.2),  # "a¨" → "ä"
    "dutch_ij_break": (r"i\s+j\b", 0.2),  # "i j" → "ij"
    "misplaced_punctuation": (r"\s+[.,;:!?]", 0.1),  # " ."
    "character_substitution": (r"[O0]{1}[lI1]{1}|[lI1]{1}[O0]{1}", 0.2),  # "O1" vs "01"
}

# Minimum hits per pattern to count (prevents false positives on short text)
MIN_PATTERN_HITS: int = 2

# Stage B: LLM Prompts (Vision Anchor Protocol)
REFINER_SYSTEM_PROMPT = """You are a precision text recovery tool. Your ONLY task is to fix OCR artifacts (spaces in words, broken characters). Do NOT change sentence structure. Do NOT improve style. Do NOT add words. If the text is clear, return it exactly as is.

MINIMAL EDIT STRICTURE (CRITICAL):
- Fix ONLY: broken words ("G u l f" → "Gulf"), floating hyphens (" - " → "-"), ligature breaks ("fi " → "fi")
- PRESERVE: sentence structure, word order, punctuation placement, capitalization patterns
- DO NOT: rephrase, improve grammar, add/remove words, change vocabulary
- If text is readable: return it EXACTLY as provided, even with minor issues

PROTECTED TOKEN PATTERNS (0% EDIT TOLERANCE):
- Part numbers: ABC-123, P/N: XXX
- ECU codes: ECU-XXX
- Serial numbers: SN: XXX
- URLs: http://... or https://...
- Dates: YYYY-MM-DD

VISION ANCHOR: If visual_description contains entity names (brands, models), treat them as GROUND TRUTH for correcting OCR errors.

OUTPUT FORMAT (CRITICAL):
Start your response directly with the recovered text. No introductions like "Here is..." or "Cleaned text:". No explanations. No commentary. Just the text itself."""

REFINER_USER_PROMPT = """Context from surrounding text:
{context}

Visual description (GROUND TRUTH for entity names):
{visual_description}

Original OCR text to refine:
{original_text}

Corrected text:"""

# Stage C: Protected Token Patterns (0% Edit Budget)
PROTECTED_PATTERNS: List[str] = [
    r"\b[A-Z]{2,}-\d{2,}[A-Z0-9-]*\b",  # ABC-123, ECU-001-A
    r"\b[A-Z]{2,}\d{2,}[A-Z0-9-]*\b",  # ABC123, ECU001A
    r"P/N:?\s*\S+",  # P/N: 12345 or P/N:ABC-123
    r"ECU-\S+",  # ECU-001-A
    r"SN:?\s*\S+",  # SN: 12345 or SN:ABC123
    r"ID:?\s*\S+",  # ID: 12345
    r"https?://[^\s)\\]\\}>;,]+",  # URLs (exclude trailing punctuation)
    r"\d{4}-\d{2}-\d{2}",  # Dates: 2025-01-13
]

# LLM Provider Defaults
DEFAULT_TIMEOUT: Tuple[float, float] = (5.0, 30.0)  # (connect, read)
DEFAULT_MAX_TOKENS: int = 150
DEFAULT_TEMPERATURE: float = 0.0  # Deterministic


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class RefinerConfig:
    """Configuration for Semantic Text Refiner."""

    # Stage A: Diagnostic Triage
    min_refine_threshold: float = 0.15  # Skip LLM if corruption_score < this
    language_hint: Optional[str] = None  # "en", "de", "nl", or None for auto-detect
    technical_density_trigger: float = 0.08  # Protected-token density to lower threshold
    technical_density_threshold_delta: float = 0.05  # Threshold reduction when dense

    # Stage B: LLM Layer
    llm_provider: str = "ollama"  # ollama | openai | anthropic
    llm_model: Optional[str] = None  # Auto-detect for Ollama
    llm_base_url: Optional[str] = None  # Custom API base URL
    llm_api_key: Optional[str] = None  # API key for cloud providers
    llm_timeout: Tuple[float, float] = DEFAULT_TIMEOUT
    llm_max_tokens: int = DEFAULT_MAX_TOKENS
    llm_temperature: float = DEFAULT_TEMPERATURE

    # Stage C: Integrity Guardrail - USE CENTRAL CONSTANT
    max_edit_budget: float = DEFAULT_MAX_EDIT_RATIO  # Max character edit ratio (35% default)
    protected_token_patterns: List[str] = field(default_factory=lambda: PROTECTED_PATTERNS.copy())


@dataclass
class RefinementResult:
    """Result of refinement process."""

    original_text: str
    refined_text: str
    refinement_applied: bool
    corruption_score: float
    edit_distance: int
    edit_ratio: float
    protected_tokens_preserved: bool
    rejection_reason: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of consistency validation."""

    is_valid: bool
    edit_distance: int
    edit_ratio: float
    protected_tokens_preserved: bool
    rejection_reason: Optional[str] = None


# ============================================================================
# STAGE A: NOISE SCANNER (Diagnostic Triage)
# ============================================================================


class NoiseScanner:
    """
    Stage A: Diagnostic triage for OCR corruption detection.

    Calculates a weighted corruption_score based on multi-language patterns.
    Only text with score > threshold is sent to LLM for efficiency.
    """

    def __init__(
        self,
        patterns: Optional[Dict[str, Tuple[str, float]]] = None,
        min_hits: int = MIN_PATTERN_HITS,
    ):
        """
        Initialize NoiseScanner.

        Args:
            patterns: Custom corruption patterns (pattern_regex, weight)
            min_hits: Minimum pattern matches to count (prevents false positives)
        """
        self.patterns = patterns or CORRUPTION_PATTERNS
        self.min_hits = min_hits
        self._compiled_patterns: Dict[str, Tuple[re.Pattern, float]] = {}

        # Compile regex patterns
        for name, (pattern, weight) in self.patterns.items():
            try:
                self._compiled_patterns[name] = (re.compile(pattern), weight)
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{name}': {e}")

    def calculate_corruption_score(
        self,
        text: str,
        language_hint: Optional[str] = None,
    ) -> float:
        """
        Calculate weighted corruption score (0.0-1.0).

        Args:
            text: Text to analyze
            language_hint: Language hint ("en", "de", "nl") for pattern selection

        Returns:
            Corruption score between 0.0 (clean) and 1.0 (severely corrupted)
        """
        if not text or len(text) < 10:
            return 0.0

        total_score = 0.0
        detected_patterns: List[str] = []

        for name, (compiled_pattern, weight) in self._compiled_patterns.items():
            # Language-specific filtering
            if language_hint:
                if language_hint == "de" and "german" not in name:
                    continue
                elif language_hint == "nl" and "dutch" not in name:
                    continue
                elif language_hint == "en" and ("german" in name or "dutch" in name):
                    continue

            matches = compiled_pattern.findall(text)
            if len(matches) >= self.min_hits:
                # Weight by number of matches, capped at 1.0
                pattern_score = min(len(matches) * weight, weight * 3)
                total_score += pattern_score
                detected_patterns.append(name)

        # Normalize score to 0.0-1.0 range
        normalized_score = min(total_score, 1.0)

        if detected_patterns:
            logger.debug(
                f"[NoiseScanner] Corruption score: {normalized_score:.2f}, "
                f"patterns: {', '.join(detected_patterns)}"
            )

        return normalized_score


def _levenshtein_distance(text_a: str, text_b: str) -> int:
    """
    Compute Levenshtein distance with lazy import and difflib fallback.

    Avoids hard dependency on python-Levenshtein for environments without a compiler.
    """
    try:
        from Levenshtein import distance as levenshtein_distance  # type: ignore

        return levenshtein_distance(text_a, text_b)
    except Exception:
        matcher = difflib.SequenceMatcher(None, text_a, text_b)
        ratio = matcher.ratio()
        max_len = max(len(text_a), len(text_b))
        return int(round((1.0 - ratio) * max_len))


# ============================================================================
# STAGE B: CONTEXTUAL REFINER (LLM Layer)
# ============================================================================


class ContextualRefiner:
    """
    Stage B: LLM-powered text repair with contextual grounding.

    Uses visual_description as Named Entity Anchor and semantic_context
    for disambiguation. Deterministic with temp=0.
    """

    def __init__(self, config: RefinerConfig):
        """Initialize ContextualRefiner with config."""
        self.config = config
        self.provider = config.llm_provider.lower()
        self.model = config.llm_model
        self.base_url = config.llm_base_url
        self.api_key = config.llm_api_key

        # Auto-detect Ollama model if not specified
        if self.provider == "ollama" and not self.model:
            self.model = self._detect_ollama_model()

        logger.info(
            f"[ContextualRefiner] Initialized: provider={self.provider}, "
            f"model={self.model}, temp={config.llm_temperature}"
        )

    def _detect_ollama_model(self) -> str:
        """Auto-detect available Ollama model."""
        base_url = self.base_url or "http://localhost:11434"
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=(5.0, 10.0))
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    model_name = models[0].get("name", "llama2")
                    logger.info(f"[ContextualRefiner] Auto-detected model: {model_name}")
                    return model_name
        except Exception as e:
            logger.warning(f"[ContextualRefiner] Model detection failed: {e}")

        return "llama2"  # Fallback

    def refine_with_context(
        self,
        raw_text: str,
        visual_description: Optional[str] = None,
        semantic_context: Optional[SemanticContext] = None,
    ) -> Optional[str]:
        """
        Repair OCR artifacts using contextual grounding.

        Args:
            raw_text: Original OCR text
            visual_description: VLM description (Vision Anchor)
            semantic_context: Neighboring text for context

        Returns:
            Refined text or None on failure
        """
        # Build context string
        context_parts: List[str] = []
        if semantic_context:
            if semantic_context.prev_text_snippet:
                context_parts.append(f"Previous: {semantic_context.prev_text_snippet}")
            if semantic_context.next_text_snippet:
                context_parts.append(f"Next: {semantic_context.next_text_snippet}")
            if semantic_context.parent_heading:
                context_parts.append(f"Heading: {semantic_context.parent_heading}")
            if semantic_context.breadcrumb_path:
                breadcrumb_str = " > ".join(semantic_context.breadcrumb_path)
                context_parts.append(f"Breadcrumbs: {breadcrumb_str}")

        context_str = " | ".join(context_parts) if context_parts else "No context"
        vision_str = visual_description or "No visual description"

        # Build prompt
        user_prompt = REFINER_USER_PROMPT.format(
            context=context_str,
            visual_description=vision_str,
            original_text=raw_text,
        )

        # Call LLM based on provider
        try:
            if self.provider == "ollama":
                return self._call_ollama(user_prompt)
            elif self.provider == "openai":
                return self._call_openai(user_prompt)
            elif self.provider == "anthropic":
                return self._call_anthropic(user_prompt)
            else:
                logger.error(f"[ContextualRefiner] Unknown provider: {self.provider}")
                return None
        except Exception as e:
            logger.error(f"[ContextualRefiner] LLM call failed: {e}")
            return None

    def _call_ollama(self, user_prompt: str) -> Optional[str]:
        """Call Ollama API."""
        base_url = self.base_url or "http://localhost:11434"
        endpoint = f"{base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": f"{REFINER_SYSTEM_PROMPT}\n\n{user_prompt}",
            "stream": False,
            "options": {
                "temperature": self.config.llm_temperature,
                "num_predict": self.config.llm_max_tokens,
            },
        }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=self.config.llm_timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.Timeout:
            logger.error("[ContextualRefiner] Ollama timeout")
            return None
        except Exception as e:
            logger.error(f"[ContextualRefiner] Ollama error: {e}")
            return None

    def _call_openai(self, user_prompt: str) -> Optional[str]:
        """Call OpenAI API."""
        if not self.api_key:
            logger.error("[ContextualRefiner] OpenAI API key missing")
            return None

        base_url = (self.base_url or "https://api.openai.com").rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        endpoint = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model or "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": REFINER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.llm_temperature,
            "max_tokens": self.config.llm_max_tokens,
        }

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.config.llm_timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"[ContextualRefiner] OpenAI error: {e}")
            return None

    def _call_anthropic(self, user_prompt: str) -> Optional[str]:
        """Call Anthropic API."""
        if not self.api_key:
            logger.error("[ContextualRefiner] Anthropic API key missing")
            return None

        endpoint = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model or "claude-3-5-haiku-20241022",
            "max_tokens": self.config.llm_max_tokens,
            "system": REFINER_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.config.llm_timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result["content"][0]["text"].strip()
        except Exception as e:
            logger.error(f"[ContextualRefiner] Anthropic error: {e}")
            return None


# ============================================================================
# STAGE C: CONSISTENCY VALIDATOR (Integrity Guardrail)
# ============================================================================


class ConsistencyValidator:
    """
    Stage C: Edit-budget guardrail using Levenshtein distance.

    Enforces 0% edit tolerance for protected tokens (ECU codes, P/N, etc.)
    and global edit ratio limit to prevent LLM over-editing.
    """

    def __init__(
        self,
        max_edit_ratio: float = DEFAULT_MAX_EDIT_RATIO,
        protected_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize ConsistencyValidator.

        Args:
            max_edit_ratio: Maximum allowed edit ratio (default from DEFAULT_MAX_EDIT_RATIO)
            protected_patterns: Regex patterns for protected tokens
        """
        self.max_edit_ratio = max_edit_ratio
        self.protected_patterns = protected_patterns or PROTECTED_PATTERNS

        # TRANSPARENCY LOGGING: Show exact budget being used
        logger.info(
            f"[ConsistencyValidator] Initialized with edit budget: {self.max_edit_ratio:.2%} "
            f"({self.max_edit_ratio:.2f})"
        )

        # Compile protected patterns
        self._compiled_protected: List[re.Pattern] = []
        for pattern in self.protected_patterns:
            try:
                self._compiled_protected.append(re.compile(pattern))
            except re.error as e:
                logger.warning(f"Invalid protected pattern: {pattern} - {e}")

    def extract_protected_tokens(self, text: str) -> Set[str]:
        """Extract all protected tokens from text."""
        protected_tokens: Set[str] = set()

        for pattern in self._compiled_protected:
            matches = pattern.findall(text)
            protected_tokens.update(matches)

        return protected_tokens

    def count_protected_token_hits(self, text: str) -> int:
        """Count protected token occurrences (non-unique)."""
        total_hits = 0
        for pattern in self._compiled_protected:
            total_hits += len(pattern.findall(text))
        return total_hits

    def validate(self, original: str, refined: str) -> ValidationResult:
        """
        Validate refinement against edit budget constraints.

        Returns ValidationResult with:
        - is_valid: True if refinement passes all checks
        - edit_distance: Levenshtein distance
        - edit_ratio: edit_distance / len(original)
        - protected_tokens_preserved: True if all protected tokens unchanged
        - rejection_reason: Explanation if invalid
        """
        # Calculate Levenshtein distance
        edit_dist = _levenshtein_distance(original, refined)
        edit_ratio = edit_dist / len(original) if len(original) > 0 else 0.0

        # Check global edit budget
        if edit_ratio > self.max_edit_ratio:
            return ValidationResult(
                is_valid=False,
                edit_distance=edit_dist,
                edit_ratio=edit_ratio,
                protected_tokens_preserved=False,
                rejection_reason=f"Edit ratio {edit_ratio:.2%} exceeds budget {self.max_edit_ratio:.2%}",
            )

        # Check protected tokens (0% edit tolerance)
        original_protected = self.extract_protected_tokens(original)
        refined_protected = self.extract_protected_tokens(refined)

        if original_protected != refined_protected:
            missing = original_protected - refined_protected
            added = refined_protected - original_protected

            details = []
            if missing:
                details.append(f"Missing: {missing}")
            if added:
                details.append(f"Added: {added}")

            return ValidationResult(
                is_valid=False,
                edit_distance=edit_dist,
                edit_ratio=edit_ratio,
                protected_tokens_preserved=False,
                rejection_reason=f"Protected tokens changed: {'; '.join(details)}",
            )

        # All checks passed
        return ValidationResult(
            is_valid=True,
            edit_distance=edit_dist,
            edit_ratio=edit_ratio,
            protected_tokens_preserved=True,
            rejection_reason=None,
        )


# ============================================================================
# SEMANTIC REFINER (Orchestration)
# ============================================================================


class SemanticRefiner:
    """
    Orchestrates the 3-stage refinement pipeline:
    Stage A → Stage B → Stage C

    Provides safe fallback on any stage failure.
    """

    def __init__(self, config: RefinerConfig):
        """Initialize SemanticRefiner with config."""
        self.config = config

        # Initialize stages
        self.scanner = NoiseScanner()
        self.refiner = ContextualRefiner(config)
        self.validator = ConsistencyValidator(
            max_edit_ratio=config.max_edit_budget,
            protected_patterns=config.protected_token_patterns,
        )

        logger.info(
            f"[SemanticRefiner] Initialized with provider={config.llm_provider}, "
            f"threshold={config.min_refine_threshold}, "
            f"max_edit={config.max_edit_budget}"
        )

    def process(
        self,
        raw_text: str,
        visual_description: Optional[str] = None,
        semantic_context: Optional[SemanticContext] = None,
    ) -> RefinementResult:
        """
        Process text through 3-stage pipeline.

        Args:
            raw_text: Original OCR text
            visual_description: VLM description (Vision Anchor)
            semantic_context: Neighboring text for context

        Returns:
            RefinementResult with refinement status and metadata
        """
        # Stage A: Diagnostic Triage
        corruption_score = self.scanner.calculate_corruption_score(
            raw_text,
            language_hint=self.config.language_hint,
        )

        word_count = len(re.findall(r"\b\w+\b", raw_text))
        token_hits = self.validator.count_protected_token_hits(raw_text)
        technical_density = token_hits / word_count if word_count > 0 else 0.0

        effective_threshold = self.config.min_refine_threshold
        if technical_density >= self.config.technical_density_trigger:
            effective_threshold = max(
                0.0,
                self.config.min_refine_threshold - self.config.technical_density_threshold_delta,
            )
            logger.debug(
                f"[SemanticRefiner] Technical density {technical_density:.2f} "
                f"lowers threshold to {effective_threshold:.2f}"
            )

        # Bypass LLM if corruption is below threshold
        if corruption_score < effective_threshold:
            logger.debug(
                f"[SemanticRefiner] Bypassing LLM: corruption_score "
                f"{corruption_score:.2f} < threshold {effective_threshold:.2f}"
            )
            return RefinementResult(
                original_text=raw_text,
                refined_text=raw_text,
                refinement_applied=False,
                corruption_score=corruption_score,
                edit_distance=0,
                edit_ratio=0.0,
                protected_tokens_preserved=True,
                rejection_reason="Below threshold",
            )

        # Stage B: LLM Refinement
        refined_text = self.refiner.refine_with_context(
            raw_text,
            visual_description=visual_description,
            semantic_context=semantic_context,
        )

        # Handle LLM failure
        if refined_text is None or not refined_text.strip():
            logger.warning("[SemanticRefiner] LLM returned empty result, using original")
            return RefinementResult(
                original_text=raw_text,
                refined_text=raw_text,
                refinement_applied=False,
                corruption_score=corruption_score,
                edit_distance=0,
                edit_ratio=0.0,
                protected_tokens_preserved=True,
                rejection_reason="LLM failure",
                provider=self.config.llm_provider,
                model=self.config.llm_model,
            )

        # Stage C: Consistency Validation
        validation = self.validator.validate(raw_text, refined_text)

        if not validation.is_valid:
            logger.warning(f"[SemanticRefiner] Refinement rejected: {validation.rejection_reason}")
            return RefinementResult(
                original_text=raw_text,
                refined_text=raw_text,  # Fallback to original
                refinement_applied=False,
                corruption_score=corruption_score,
                edit_distance=validation.edit_distance,
                edit_ratio=validation.edit_ratio,
                protected_tokens_preserved=validation.protected_tokens_preserved,
                rejection_reason=validation.rejection_reason,
                provider=self.config.llm_provider,
                model=self.config.llm_model,
            )

        # All stages passed - accept refinement
        logger.info(
            f"[SemanticRefiner] Refinement accepted: "
            f"corruption={corruption_score:.2f}, "
            f"edit_ratio={validation.edit_ratio:.2%}"
        )

        return RefinementResult(
            original_text=raw_text,
            refined_text=refined_text,
            refinement_applied=True,
            corruption_score=corruption_score,
            edit_distance=validation.edit_distance,
            edit_ratio=validation.edit_ratio,
            protected_tokens_preserved=validation.protected_tokens_preserved,
            rejection_reason=None,
            provider=self.config.llm_provider,
            model=self.config.llm_model,
        )


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_refiner(
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    threshold: float = 0.15,
    max_edit: float = 0.15,
    **kwargs,
) -> SemanticRefiner:
    """
    Factory function to create SemanticRefiner.

    Args:
        provider: LLM provider ("ollama", "openai", "anthropic")
        model: Model name (optional for Ollama - auto-detects)
        api_key: API key for cloud providers
        threshold: Min corruption threshold (0.0-1.0)
        max_edit: Max edit ratio (0.0-1.0)
        **kwargs: Additional config parameters

    Returns:
        Configured SemanticRefiner instance
    """
    config = RefinerConfig(
        min_refine_threshold=threshold,
        llm_provider=provider,
        llm_model=model,
        llm_base_url=base_url,
        llm_api_key=api_key,
        max_edit_budget=max_edit,
        **kwargs,
    )

    return SemanticRefiner(config)
