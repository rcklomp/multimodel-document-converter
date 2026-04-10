"""
Vision Manager - VLM Integration for Image Enrichment
=======================================================
ENGINE_USE: Multi-provider VLM support (Ollama, OpenAI, Anthropic)

This module provides a unified interface for Vision Language Model (VLM)
based image enrichment. It supports multiple providers and includes
caching to avoid redundant API calls.

REQ Compliance:
- REQ-VLM-01: Support for Ollama (local), OpenAI, Anthropic
- REQ-VLM-02: Image caching to prevent duplicate calls
- REQ-CHUNK-03: Descriptions truncated to 400 chars

SRS Section 8: Vision Enrichment
"The system SHOULD provide VLM-based descriptions for images using
configurable providers with intelligent caching."

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-29
"""

from __future__ import annotations

import base64
import gc
import hashlib
import io
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING, Tuple

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from ..state.context_state import ContextStateV2
    from ..orchestration.strategy_profiles import ProfileParameters

# Import the new visual-only prompt system
from .vision_prompts import build_visual_prompt, clean_vlm_response, validate_vlm_response

logger = logging.getLogger(__name__)


# ============================================================================
# IMAGE TYPE HEURISTIC
# ============================================================================


def _classify_image_type(image: Image.Image) -> tuple[bool, bool]:
    """Light heuristic to classify an image as diagram or photograph.

    Returns (is_diagram, is_photograph).  Both False means unknown/generic.

    Signals used (no ML model, ~2 ms per image):
    - Unique color count on a 128×128 thumbnail (diagrams use few colors)
    - White/near-white pixel fraction (diagrams have white backgrounds)
    - Mean saturation in HSV space (photos are more saturated)
    """
    try:
        thumb = image.convert("RGB").resize((128, 128), Image.BILINEAR)
        arr = np.asarray(thumb, dtype=np.uint8)

        # --- Unique color count (quantised to 8-bit cubes) ---
        quantised = (arr >> 4).reshape(-1, 3)
        unique_colors = len(set(map(tuple, quantised.tolist())))

        # --- White pixel fraction ---
        luminance = arr.mean(axis=2)
        white_frac = float((luminance > 240).sum()) / (128 * 128)

        # --- Mean saturation (HSV) ---
        hsv = np.asarray(thumb.convert("HSV"), dtype=np.float32)
        mean_sat = float(hsv[:, :, 1].mean()) / 255.0

        # Decision logic
        # Diagrams: few colors, large white area, low saturation
        if unique_colors < 120 and white_frac > 0.30 and mean_sat < 0.25:
            return (True, False)
        # Photographs: many colors, high saturation, little white
        if unique_colors > 400 and mean_sat > 0.20 and white_frac < 0.40:
            return (False, True)
        return (False, False)
    except Exception:
        return (False, False)


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_DESCRIPTION_CHARS: int = 400
# FIX: Increased timeout for large vision models and high-DPI scans
DEFAULT_TIMEOUT: int = 180  # Seconds - increased for llama3.2-vision (10.7B) and 300 DPI scans
DEFAULT_OPENAI_MODEL: str = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL: str = "claude-3-haiku-20240307"
# Ollama max tokens - prevents truncation
OLLAMA_NUM_PREDICT: int = 500  # Increased from default ~128


# ============================================================================
# PROMPTS
# ============================================================================

# ============================================================================
# GEMINI AUDIT FIX: CONTEXTUAL SEMANTIC ENRICHMENT + DYNAMIC PROMPT INJECTION
# ============================================================================
# Changed from pixel-only to CONTEXTUAL analysis.
# The VLM now receives prev_text to anchor descriptions semantically.
# Format: OBJ: [subject] DESC: [visual + contextual relevance]
#
# GEMINI AUDIT FIX #2: Dynamic Prompt Injection
# Prompts are now template-based with classification and detected features.
# GEMINI AUDIT FIX #3: Confidence Scoring
# VLM output now includes reasoning and confidence for hallucination detection.

ENRICHMENT_PROMPT_BASE = """IMAGE ANALYSIS REPORT
OBJECTIVE: Describe this image AND its relevance to the surrounding document context.

REQUIRED FORMAT (STRICT JSON):
{{
  "object": "[1-3 words identifying the primary subject]",
  "description": "[Visual attributes + relevance to document context]",
  "confidence": [0.0-1.0 how certain you are about this classification],
  "reasoning": "[Brief explanation of why you classified it this way]"
}}

RULES:
1. Identify what you SEE in the image (objects, colors, shapes)
2. Connect it to the document context provided below
3. Maximum 400 characters for description
4. Be HONEST about confidence - if unsure, set confidence < 0.7
5. If you cannot identify the content clearly, say so in reasoning

{context_section}

{diagnostic_hints}

Analyze this image now. Respond ONLY with valid JSON."""

ENRICHMENT_PROMPT_NO_CONTEXT = """IMAGE ANALYSIS REPORT
OBJECTIVE: Technical inventory of visual elements.

REQUIRED FORMAT (STRICT JSON):
{{
  "object": "[1-3 words identifying the primary subject]",
  "description": "[Visual attributes only - colors, shapes, composition, visible elements]",
  "confidence": [0.0-1.0 how certain you are about this classification],
  "reasoning": "[Brief explanation of why you classified it this way]"
}}

RULES:
- Describe ONLY what you see in the image pixels
- Focus on: objects, colors, shapes, text visible IN the image
- Maximum 400 characters for description
- Be HONEST about confidence - if unsure, set confidence < 0.7

{diagnostic_hints}

Analyze this image now. Respond ONLY with valid JSON."""

# Template for scanned document hints (injected dynamically)
SCAN_ARTIFACT_HINTS = """
IMPORTANT - SCANNED DOCUMENT CONTEXT:
- This image comes from a SCANNED document
- IGNORE: paper texture, grain, stains, foxing, discoloration
- IGNORE: scan artifacts, dust specks, fold marks
- FOCUS ON: the actual printed/drawn content only
- Do NOT describe paper quality or scan issues as content
"""

# Template for historical document hints
HISTORICAL_DOCUMENT_HINTS = """
HISTORICAL DOCUMENT CONTEXT:
- This appears to be a historical/vintage document
- Expect aged typography, printing imperfections, dated imagery
- Focus on the CONTENT being depicted, not the age indicators
"""

# ============================================================================
# REQ-OCR-02: EXTRACTIVE PROMPT FOR SHADOW-FIRST MODE
# ============================================================================
# This prompt is EXTRACTIVE not DESCRIPTIVE. It tells the VLM to:
# 1. EXTRACT all visible text (brand names, titles, headers)
# 2. LIST detected keywords prominently in the output
# 3. NOT describe visual qualities (colors, shapes, etc.)
#
# This is used when processing full-page scans where OCR provides hints
# and the VLM acts as the "judge" to validate and prioritize keywords.

EXTRACTIVE_PROMPT_OCR_HYBRID = """EXTRACTION TASK: SCANNED DOCUMENT PAGE ANALYSIS

You are a HIGH-PRECISION OCR VALIDATOR and TECHNICAL CATALOGER.
This is a FULL-PAGE scan from a technical catalog or magazine.

=== YOUR TASK ===
1. EXTRACT every visible text: brand names, model numbers, serial numbers, titles, headers
2. If you see brand names like "Browning", "Sako", "Springfield", "Winchester" - they MUST appear in your output
3. The OCR engine has detected some text (see hints below). VALIDATE these detections.
4. If OCR found a word and you can see it in the image - INCLUDE IT IN YOUR RESPONSE.

{ocr_hints_section}

=== REQUIRED OUTPUT FORMAT ===
HEADER: [Main page title or section header - EXACT TEXT as shown]
KEYWORDS: [Comma-separated list of brand names, model numbers you can see]
DESCRIPTION: [Brief technical description of the page content]

=== RULES ===
- If OCR says "Browning" and you see text that looks like "Browning" → Your HEADER/KEYWORDS MUST include "Browning"
- DO NOT replace specific brand names with generic terms like "rifle" or "firearm"
- DO NOT describe visual qualities (colors, lighting, composition)
- FOCUS ON: What text can you read? What is this page about? What brands/models are shown?
- If you cannot read any text clearly, say "HEADER: [Unreadable]" - do NOT fabricate

PAGE: {page_number}

Respond NOW with HEADER:, KEYWORDS:, and DESCRIPTION:"""

# ============================================================================
# GEMINI AUDIT FIX #1: STRICTER FULL-PAGE GUARD
# ============================================================================
# Added "advertisement" classification to prevent marketing content from
# polluting the RAG corpus. The VLM now detects sales-oriented language.

FULLPAGE_GUARD_PROMPT = """FULL-PAGE GUARD VERIFICATION (STRICT AD FILTER)

Analyze this full-page image and classify it:

QUESTION: Is this image GENUINE editorial content (article photo, technical diagram, infographic)
OR is it an ADVERTISEMENT, promotional content, or non-editorial element?

STRICT RULES:
1. If text contains sales language (buy, subscribe, special offer, price, order now) → "advertisement"
2. If >50% of page is promotional/marketing content with product photos → "advertisement"
3. If page shows branded product with pricing/purchasing info → "advertisement"
4. Only classify as "editorial" if content is INFORMATIONAL, not PROMOTIONAL

RESPONSE FORMAT (STRICT JSON):
{
  "classification": "editorial|advertisement|ui_navigation|page_scan",
  "confidence": 0.0-1.0,
  "has_sales_language": true|false,
  "reason": "Brief explanation (max 50 chars)"
}

Classifications:
- "editorial": GENUINE article photo, diagram, chart, illustration, infographic (NO sales intent)
- "advertisement": Promotional content, product marketing, subscription offers, sponsored content
- "ui_navigation": Website/app navigation, menus, headers, interface elements  
- "page_scan": Screenshot or scan of document page with mostly text

IMPORTANT: When in doubt between editorial and advertisement, classify as "advertisement".
Respond ONLY with valid JSON. No other text."""


# ============================================================================
# CACHE
# ============================================================================


class VisionCache:
    """
    Simple disk-based cache for VLM descriptions.

    Uses image content hash as key to avoid redundant VLM calls
    for identical or duplicate images.

    Model-aware: if the configured model differs from what was used to
    populate the cache, the stale cache is discarded and VLM is called fresh.
    """

    # Reserved key used to store metadata inside the flat cache dict.
    _META_KEY = "_model"

    def __init__(self, cache_dir: Path, model_name: str = "") -> None:
        """Initialize cache with directory.

        Args:
            cache_dir: Directory for cache file storage.
            model_name: Name of the VLM model being used.  When non-empty,
                the cache is invalidated if the stored model differs from this
                value, ensuring model changes always trigger fresh VLM calls.
        """
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / ".vision_cache.json"
        self._cache: Dict[str, str] = {}
        self._model_name = model_name
        self._load()

    def _load(self) -> None:
        """Load cache from disk, invalidating it when the VLM model changed."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    data: Dict[str, str] = json.load(f)

                # Extract metadata embedded in the flat cache dict.
                cached_model = data.pop(self._META_KEY, None)

                if self._model_name and cached_model and cached_model != self._model_name:
                    # Model changed → stale cache; discard and call VLM fresh.
                    logger.info(
                        f"[VISION-CACHE] Model changed: '{cached_model}' → '{self._model_name}'. "
                        f"Discarding {len(data)} stale cached descriptions so fresh VLM calls are made."
                    )
                    self._cache = {}
                else:
                    self._cache = data
                    if self._cache:
                        logger.info(
                            f"[VISION-CACHE] Loaded {len(self._cache)} cached descriptions "
                            f"(model: {cached_model or 'unknown'}). "
                            f"VLM will NOT be called for these images. "
                            f"Use --no-cache to force fresh VLM calls."
                        )
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}

    def _save(self) -> None:
        """Save cache to disk, embedding the model name for future validation."""
        try:
            data = dict(self._cache)
            if self._model_name:
                data[self._META_KEY] = self._model_name
            with open(self.cache_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _compute_hash(self, image: Image.Image) -> str:
        """Compute content hash of image."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        content = buffer.getvalue()
        return hashlib.md5(content).hexdigest()

    def get(self, image: Image.Image) -> Optional[str]:
        """Get cached description for image."""
        img_hash = self._compute_hash(image)
        return self._cache.get(img_hash)

    def set(self, image: Image.Image, description: str) -> None:
        """Cache description for image."""
        img_hash = self._compute_hash(image)
        self._cache[img_hash] = description

    def flush(self) -> None:
        """Flush cache to disk."""
        self._save()

    def size(self) -> int:
        """Get number of cached items."""
        return len(self._cache)


# ============================================================================
# VISION PROVIDERS (Abstract Base)
# ============================================================================


class VisionProvider(ABC):
    """Abstract base class for vision providers."""

    @abstractmethod
    def describe_image(
        self,
        image: Image.Image,
        context: str,
        max_chars: Optional[int] = MAX_DESCRIPTION_CHARS,
    ) -> str:
        """Generate description for image."""
        pass


# ============================================================================
# OLLAMA PROVIDER
# ============================================================================


class OllamaProvider(VisionProvider):
    """
    Ollama-based local VLM provider.

    Uses auto-detected or user-specified multimodal models running locally.
    If no model is specified, auto-detects the loaded model from Ollama.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        host: str = "http://localhost:11434",
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize Ollama provider.

        Args:
            model: Ollama model name (optional - auto-detects if not specified)
            host: Ollama server URL (default: localhost:11434)
            timeout: Request timeout in seconds
        """
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._request_seq = 0

        # Auto-detect model if not specified
        if not model:
            model = self._detect_loaded_model()

        self.model = model
        logger.info(f"OllamaProvider: model={self.model}, host={host}")

    def _next_request_id(self) -> int:
        """Generate a monotonic request id for provider call tracing."""
        self._request_seq += 1
        return self._request_seq

    def _detect_loaded_model(self) -> str:
        """
        Auto-detect which VLM model is currently loaded/running in Ollama.

        Returns:
            Name of the detected model

        Raises:
            RuntimeError: If no models are available in Ollama
        """
        import requests

        # First, check for currently running models via /api/ps
        try:
            response = requests.get(
                f"{self.host}/api/ps",
                timeout=(5.0, 10.0),
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                if models:
                    model_name = models[0].get("name", models[0].get("model"))
                    logger.info(f"[AUTO-DETECT] Found running model: {model_name}")
                    return model_name
        except requests.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.host}. Is 'ollama serve' running?"
            )
        except Exception as e:
            logger.warning(f"Failed to check running models: {e}")

        # If no running model, check available models via /api/tags
        try:
            response = requests.get(
                f"{self.host}/api/tags",
                timeout=(5.0, 10.0),
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                if models:
                    model_name = models[0].get("name", models[0].get("model"))
                    logger.info(f"[AUTO-DETECT] Using first available model: {model_name}")
                    return model_name
        except Exception as e:
            logger.warning(f"Failed to list available models: {e}")

        raise RuntimeError(
            "No VLM models found in Ollama. "
            "Please load a vision model first (e.g., 'ollama pull llava:latest')"
        )

    def describe_image(
        self,
        image: Image.Image,
        context: str,
        max_chars: Optional[int] = MAX_DESCRIPTION_CHARS,
    ) -> str:
        """Generate description using Ollama."""
        import requests

        limit = (
            max_chars
            if isinstance(max_chars, int) and max_chars > 0
            else MAX_DESCRIPTION_CHARS
        )

        # Convert image to base64
        buffer = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Log image and payload size for debugging
        img_size_kb = len(img_base64) / 1024
        prompt_size = len(context)
        logger.debug(
            f"[OLLAMA] Sending request: model={self.model}, "
            f"image={img_size_kb:.1f}KB, prompt={prompt_size} chars, timeout={self.timeout}s"
        )

        # Build request with increased num_predict to prevent truncation
        # FIX: Ollama default num_predict is ~128 tokens which causes truncation
        payload = {
            "model": self.model,
            "prompt": context,
            "images": [img_base64],
            "stream": False,
            "options": {
                "num_predict": OLLAMA_NUM_PREDICT,  # 500 tokens - prevents truncation
            },
            "keep_alive": "30m",  # Houd het model 30 minuten vast na de laatste aanroep
        }
        if limit > MAX_DESCRIPTION_CHARS:
            payload["options"]["num_predict"] = max(
                OLLAMA_NUM_PREDICT, min(3000, limit // 2)
            )
        request_id = self._next_request_id()
        t0 = time.perf_counter()

        try:
            logger.info(
                f"[VLM-REQ {request_id}] provider=ollama model={self.model} "
                f"timeout={self.timeout}s image_kb={img_size_kb:.1f}"
            )
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=(10, self.timeout),
            )
            response.raise_for_status()
            result = response.json()
            description = result.get("response", "").strip()
            elapsed = time.perf_counter() - t0
            logger.info(
                f"[VLM-RES {request_id}] provider=ollama elapsed={elapsed:.2f}s "
                f"chars={len(description)}"
            )
            logger.debug(f"[OLLAMA] Response received: {len(description)} chars")
            return description[:limit]

        except requests.exceptions.Timeout as e:
            elapsed = time.perf_counter() - t0
            logger.error(
                f"[VLM-TIMEOUT {request_id}] provider=ollama elapsed={elapsed:.2f}s "
                f"timeout={self.timeout}s. "
                f"Image: {img_size_kb:.1f}KB, Prompt: {prompt_size} chars. "
                f"Try increasing --vlm-timeout to 180 or 300 seconds."
            )
            raise RuntimeError(
                f"Ollama timeout after {self.timeout}s - image may be too large"
            ) from e

        except requests.exceptions.ConnectionError as e:
            elapsed = time.perf_counter() - t0
            logger.error(
                f"[VLM-ERROR {request_id}] provider=ollama elapsed={elapsed:.2f}s "
                f"Cannot connect to Ollama at {self.host}. "
                f"Is 'ollama serve' running? Error: {e}"
            )
            raise RuntimeError(f"Ollama connection failed at {self.host}") from e

        except requests.exceptions.HTTPError as e:
            elapsed = time.perf_counter() - t0
            status_code = e.response.status_code if e.response else "unknown"
            response_text = e.response.text[:200] if e.response else "no response"
            logger.error(
                f"[VLM-ERROR {request_id}] provider=ollama elapsed={elapsed:.2f}s "
                f"HTTP {status_code}. Response: {response_text}"
            )
            raise RuntimeError(f"Ollama HTTP error {status_code}: {response_text}") from e

        except Exception as e:
            elapsed = time.perf_counter() - t0
            error_type = type(e).__name__
            logger.error(
                f"[VLM-ERROR {request_id}] provider=ollama elapsed={elapsed:.2f}s "
                f"{error_type}: {e}. "
                f"Image: {img_size_kb:.1f}KB, Prompt: {prompt_size} chars"
            )
            raise


# ============================================================================
# OPENAI PROVIDER
# ============================================================================


class OpenAIProvider(VisionProvider):
    """
    OpenAI GPT-4 Vision provider.

    Uses gpt-4o-mini or gpt-4o for image description.
    Supports custom base_url for LM Studio / Local API compatibility.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (or dummy key for local APIs)
            model: Model name (default: gpt-4o-mini)
            timeout: Request timeout in seconds
            base_url: Custom API base URL (e.g., http://192.168.10.11:1234/v1 for LM Studio)
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = base_url
        self._request_seq = 0
        self._resolved_model: Optional[str] = None

    @staticmethod
    def _post_with_deadline(url, *, headers, json, connect_timeout=10, read_timeout=180, deadline=200):
        """requests.post with a hard thread-based deadline.

        The `requests` timeout only detects socket-level silence. A server
        that keeps the TCP connection alive but never sends the response body
        will hang forever. This wrapper runs the request in a daemon thread
        and raises requests.Timeout if the deadline is exceeded.
        """
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                requests.post, url, headers=headers, json=json,
                timeout=(connect_timeout, read_timeout),
            )
            try:
                return future.result(timeout=deadline)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise requests.Timeout(
                    f"Hard deadline {deadline}s exceeded (connect={connect_timeout}, read={read_timeout})"
                )
        self._model_cache: Optional[Tuple[float, list[str]]] = None
        self._model_cache_ttl_sec: float = 120.0

        if base_url:
            logger.info(f"OpenAIProvider: model={model}, base_url={base_url}")
        else:
            logger.info(f"OpenAIProvider: model={model}")

    def _next_request_id(self) -> int:
        """Generate a monotonic request id for provider call tracing."""
        self._request_seq += 1
        return self._request_seq

    def _normalized_base_url(self) -> Optional[str]:
        """Normalize custom base_url to include /v1 once for OpenAI-compatible APIs."""
        if not self.base_url:
            return None
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    @staticmethod
    def _canonical_model_name(name: str) -> str:
        """Canonicalize model names for fuzzy matching."""
        return re.sub(r"[^a-z0-9]+", "", (name or "").lower())

    @staticmethod
    def _model_tokens(name: str) -> set[str]:
        """Tokenize model names into lexical hints for fuzzy matching."""
        return {t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if t}

    def _list_local_models(self, headers: Dict[str, str], refresh: bool = False) -> list[str]:
        """Fetch model IDs from /v1/models with a short TTL cache."""
        import requests

        api_base = self._normalized_base_url()
        if not api_base:
            return []
        now = time.time()
        if (
            not refresh
            and self._model_cache is not None
            and (now - self._model_cache[0]) < self._model_cache_ttl_sec
        ):
            return list(self._model_cache[1])

        models_url = f"{api_base}/models"
        try:
            response = requests.get(models_url, headers=headers, timeout=min(20, self.timeout))
            response.raise_for_status()
            payload = response.json() or {}
            rows = payload.get("data", [])
            model_ids = []
            for row in rows:
                if isinstance(row, dict):
                    model_id = str(row.get("id", "")).strip()
                    if model_id:
                        model_ids.append(model_id)
            self._model_cache = (now, model_ids)
            return model_ids
        except Exception as e:
            logger.debug(f"[VLM-MODELS] Failed to fetch local model list: {e}")
            return []

    def _resolve_local_model_id(
        self,
        requested_model: str,
        headers: Dict[str, str],
        refresh: bool = False,
    ) -> str:
        """
        Resolve HuggingFace-style aliases to the actual model ID exposed by local OpenAI APIs.
        Example: 'numind/NuMarkdown-8B-Thinking' -> 'numarkdown-8b-thinking-mlxs'
        """
        if not self.base_url:
            return requested_model
        if self._resolved_model and not refresh:
            return self._resolved_model

        available = self._list_local_models(headers=headers, refresh=refresh)
        if not available:
            return requested_model
        if requested_model in available:
            self._resolved_model = requested_model
            return requested_model

        req_lower = requested_model.lower()
        for model_id in available:
            if model_id.lower() == req_lower:
                self._resolved_model = model_id
                return model_id

        requested_tail = requested_model.split("/")[-1]
        tail_lower = requested_tail.lower()
        for model_id in available:
            if model_id.lower() == tail_lower:
                self._resolved_model = model_id
                return model_id

        req_canon = self._canonical_model_name(requested_model)
        tail_canon = self._canonical_model_name(requested_tail)
        req_tokens = self._model_tokens(requested_model) | self._model_tokens(requested_tail)

        best_score = -1
        best_model = requested_model
        for model_id in available:
            model_canon = self._canonical_model_name(model_id)
            model_tokens = self._model_tokens(model_id)
            score = 0
            if model_canon == req_canon or model_canon == tail_canon:
                score += 100
            if tail_canon and tail_canon in model_canon:
                score += 60
            if req_canon and req_canon in model_canon:
                score += 60
            if model_canon.startswith(tail_canon) and tail_canon:
                score += 20
            if model_canon.startswith(req_canon) and req_canon:
                score += 20
            overlap = req_tokens & model_tokens
            score += 8 * len(overlap)
            if score > best_score:
                best_score = score
                best_model = model_id

        if best_score >= 60 and best_model != requested_model:
            self._resolved_model = best_model
            logger.warning(
                f"[VLM-MODEL-ALIAS] Requested model '{requested_model}' not found on local endpoint. "
                f"Using '{best_model}'."
            )
            return best_model
        return requested_model

    @staticmethod
    def _extract_openai_content(result: Dict[str, Any]) -> str:
        """Extract assistant text from OpenAI-style responses with string or segment-list content."""
        choices = result.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            return "\n".join(parts).strip()
        return ""

    def describe_image(
        self,
        image: Image.Image,
        context: str,
        max_chars: Optional[int] = MAX_DESCRIPTION_CHARS,
    ) -> str:
        """Generate description using OpenAI or OpenAI-compatible API (LM Studio, etc.)."""
        import requests

        limit = (
            max_chars
            if isinstance(max_chars, int) and max_chars > 0
            else MAX_DESCRIPTION_CHARS
        )
        max_tokens = 200
        if limit > MAX_DESCRIPTION_CHARS:
            max_tokens = min(2000, max(600, limit // 3))

        # Convert image to base64
        buffer = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        model_for_request = self._resolve_local_model_id(self.model, headers=headers)
        payload = {
            "model": model_for_request,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }

        # Use custom base_url if provided, otherwise default to OpenAI
        local_base = self._normalized_base_url()
        api_url = (
            f"{local_base}/chat/completions"
            if local_base
            else "https://api.openai.com/v1/chat/completions"
        )
        request_id = self._next_request_id()
        t0 = time.perf_counter()

        try:
            logger.info(
                f"[VLM-REQ {request_id}] provider=openai model={payload['model']} "
                f"timeout={self.timeout}s endpoint={api_url}"
            )
            response = self._post_with_deadline(
                api_url,
                headers=headers,
                json=payload,
                connect_timeout=10,
                read_timeout=self.timeout,
                deadline=self.timeout + 20,
            )
            response.raise_for_status()
            result = response.json()
            description = self._extract_openai_content(result)
            elapsed = time.perf_counter() - t0
            logger.info(
                f"[VLM-RES {request_id}] provider=openai elapsed={elapsed:.2f}s "
                f"chars={len(description)}"
            )
            return description[:limit]
        except requests.Timeout:
            elapsed = time.perf_counter() - t0
            logger.error(
                f"[VLM-TIMEOUT {request_id}] provider=openai elapsed={elapsed:.2f}s "
                f"timeout={self.timeout}s endpoint={api_url}"
            )
            raise
        except requests.HTTPError as e:
            elapsed = time.perf_counter() - t0
            status = e.response.status_code if e.response is not None else "unknown"
            body = ""
            if e.response is not None:
                try:
                    body = (e.response.text or "")[:1200]
                except Exception:
                    body = ""
            is_model_not_found = "model_not_found" in body or "Invalid model identifier" in body
            if status == 400 and local_base and is_model_not_found:
                resolved = self._resolve_local_model_id(
                    self.model,
                    headers=headers,
                    refresh=True,
                )
                if resolved != payload["model"]:
                    retry_payload = dict(payload)
                    retry_payload["model"] = resolved
                    logger.warning(
                        f"[VLM-RETRY {request_id}] provider=openai model_not_found, retrying with model={resolved}"
                    )
                    retry_response = self._post_with_deadline(
                        api_url,
                        headers=headers,
                        json=retry_payload,
                        connect_timeout=10,
                        read_timeout=self.timeout,
                        deadline=self.timeout + 20,
                    )
                    retry_response.raise_for_status()
                    retry_result = retry_response.json()
                    description = self._extract_openai_content(retry_result)
                    logger.info(
                        f"[VLM-RES {request_id}] provider=openai elapsed={time.perf_counter() - t0:.2f}s "
                        f"chars={len(description)}"
                    )
                    return description[:limit]

            logger.error(
                f"[VLM-ERROR {request_id}] provider=openai elapsed={elapsed:.2f}s "
                f"url={api_url} status={status} error={e} body={body}"
            )
            raise
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error(
                f"[VLM-ERROR {request_id}] provider=openai elapsed={elapsed:.2f}s "
                f"url={api_url} error={e}"
            )
            raise


# ============================================================================
# ANTHROPIC PROVIDER
# ============================================================================


class AnthropicProvider(VisionProvider):
    """
    Anthropic Claude Vision provider.

    Uses Claude 3 Haiku or Sonnet for image description.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize Anthropic provider."""
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        logger.info(f"AnthropicProvider: model={model}")

    def describe_image(
        self,
        image: Image.Image,
        context: str,
        max_chars: Optional[int] = MAX_DESCRIPTION_CHARS,
    ) -> str:
        """Generate description using Anthropic."""
        import requests

        limit = (
            max_chars
            if isinstance(max_chars, int) and max_chars > 0
            else MAX_DESCRIPTION_CHARS
        )
        max_tokens = 200
        if limit > MAX_DESCRIPTION_CHARS:
            max_tokens = min(2000, max(600, limit // 3))

        # Convert image to base64
        buffer = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_base64,
                            },
                        },
                        {"type": "text", "text": context},
                    ],
                }
            ],
        }

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=(10, self.timeout),
            )
            response.raise_for_status()
            result = response.json()
            description = result["content"][0]["text"].strip()
            return description[:limit]
        except Exception as e:
            logger.error(f"Anthropic request failed: {e}")
            raise


# ============================================================================
# VISION MANAGER
# ============================================================================


class VisionManager:
    """
    Unified manager for VLM-based image enrichment.

    REQ-VLM-01: Supports multiple providers (Ollama, OpenAI, Anthropic)
    REQ-VLM-02: Includes caching to prevent duplicate API calls

    Usage:
        manager = VisionManager(provider="ollama", cache_dir=Path("./cache"))
        description = manager.enrich_image(image, state, page_number)
    """

    def __init__(
        self,
        provider: VisionProvider,
        cache_dir: Optional[Path] = None,
        model_name: str = "",
    ) -> None:
        """
        Initialize VisionManager.

        Args:
            provider: Configured VisionProvider instance
            cache_dir: Directory for caching (None = no caching)
            model_name: Model name used for cache invalidation when the model
                changes between runs.
        """
        self._provider = provider
        self._cache: Optional[VisionCache] = None
        self._processed_count = 0
        self.document_domain: str = ""  # Set by caller to influence prompt selection
        # Cache OCR hint engines by (min_confidence, languages) to avoid
        # repeatedly reloading EasyOCR models for every image.
        self._ocr_hint_engines: Dict[Tuple[float, Tuple[str, ...]], Any] = {}

        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache = VisionCache(cache_dir, model_name=model_name)

    def enrich_image(
        self,
        image: Image.Image,
        state: "ContextStateV2",
        page_number: int,
        anchor_text: Optional[str] = None,
        ocr_confidence: Optional[float] = None,
    ) -> str:
        """
        Generate VLM description for an image using VISUAL-ONLY prompts.

        SIGNAL-TO-NOISE OPTIMIZATION:
        ==============================
        This method now uses the new visual-only prompt system that enforces:
        1. NO textual meta-language (banned: "This image shows...", "The page contains...")
        2. VISUAL DESCRIPTORS ONLY (what you SEE, not what you read)
        3. Confidence threshold to prevent VLM overlap with high-quality OCR

        CONFIDENCE THRESHOLD LOGIC:
        ===========================
        If ocr_confidence > 0.8, the VLM is instructed to SKIP text description
        and focus ONLY on non-textual visual elements (diagrams, photos, layouts).

        Args:
            image: PIL Image to describe
            state: Current document context state (used for breadcrumbs)
            page_number: Page number for context
            anchor_text: Surrounding text (USED in prompt for semantic anchoring)
            ocr_confidence: OCR confidence score (0.0-1.0) - if > 0.8, VLM skips text

        Returns:
            Generated description (max 400 chars) - CLEAN TEXT ONLY, visual descriptors
        """
        # Check cache first
        if self._cache:
            cached = self._cache.get(image)
            if cached:
                logger.debug("Using cached description")
                return cached

        # ================================================================
        # CONFIDENCE THRESHOLD: Skip VLM if high-quality OCR already exists
        # ================================================================
        if ocr_confidence and ocr_confidence > 0.8:
            logger.info(
                f"[CONFIDENCE-THRESHOLD] OCR confidence {ocr_confidence:.2f} > 0.8 - "
                f"VLM will focus on visual-only elements (no text description)"
            )

        # ================================================================
        # BUILD VISUAL-ONLY PROMPT (no breadcrumb/text/page injection)
        # ================================================================
        # AGENTS.md / Source Sanctity:
        # - OCR extracts text; the VLM must describe visuals only.
        # - We intentionally do NOT pass breadcrumbs, page number, or surrounding text
        #   because it encourages meta-language ("page", "document", "surrounding text")
        #   and speculative statements that degrade retrieval quality.
        is_diagram, is_photograph = _classify_image_type(image)
        # Literature domain: book illustrations look like diagrams (few colors,
        # white background) but should NOT get the technical diagram prompt.
        if self.document_domain == "literature":
            is_diagram = False
        prompt = build_visual_prompt(
            context_section=None,
            diagnostic_hints=None,
            is_scan=False,  # Caller may route scan-specific prompting elsewhere
            is_diagram=is_diagram,
            is_photograph=is_photograph,
            ocr_confidence=ocr_confidence,
        )

        logger.debug(
            f"[VISUAL-ONLY-VLM] Using visual-only prompt, "
            f"OCR confidence: {ocr_confidence if ocr_confidence else 'N/A'}"
        )

        # Get description from provider (with validation + one strict retry)
        try:
            raw_response = self._provider.describe_image(image, prompt)
            self._processed_count += 1

            validation = validate_vlm_response(raw_response)
            if not validation.is_valid:
                logger.warning(
                    f"[VISUAL-ONLY-VLM] Rejected VLM response on page {page_number}: "
                    f"{', '.join(validation.issues) if validation.issues else 'invalid'}; retrying once"
                )
                raw_response = self._provider.describe_image(image, prompt)
                self._processed_count += 1
                validation = validate_vlm_response(raw_response)

            if not validation.is_valid:
                logger.warning(
                    f"[VISUAL-ONLY-VLM] Rejected VLM response on page {page_number} after retry: "
                    f"{', '.join(validation.issues) if validation.issues else 'invalid'}"
                )
                # If the model keeps trying to "read" or reference text, degrade gracefully:
                # emit a stable layout descriptor instead of dropping to an unhelpful placeholder.
                if validation.text_reading_detected:
                    description = "Dense typographic layout; no distinct non-text visuals."
                else:
                    description = "[VLM_FAILED: response invalid]"

                if self._cache:
                    self._cache.set(image, description)
                return description

            description = validation.cleaned_response

            # Truncate to max length
            if len(description) > MAX_DESCRIPTION_CHARS:
                description = description[:MAX_DESCRIPTION_CHARS]

            # Cache the CLEAN result
            if self._cache:
                self._cache.set(image, description)

            logger.debug(f"[VISUAL-ONLY-VLM] Generated: {description[:60]}...")
            return description

        except Exception as e:
            logger.warning(f"VLM enrichment failed: {e}")
            # Return fallback - keep it visual-only (no page/document meta-language).
            return "[VLM_FAILED: call error]"

    def extract_table_markdown(
        self,
        image: Image.Image,
        page_number: Optional[int] = None,
    ) -> str:
        """
        Extract table content as markdown from an image.

        This is a fallback path for table chunks when native table extraction and OCR
        are insufficient.
        """
        if image is None:
            return ""

        prompt_primary = (
            "You are a technical documentation expert.\n"
            "Convert the table in this image into a clean Markdown table.\n"
            "Rules:\n"
            "1. Output ONLY Markdown table text (no commentary).\n"
            "2. Preserve headers and row values exactly when readable.\n"
            "3. If merged cells exist, repeat the merged value in each applicable row.\n"
            "4. If a cell is unreadable, write [unreadable] in that cell.\n"
            "5. Keep table orientation and column order as shown."
        )
        prompt_retry = (
            "Retry extraction: read the table image carefully and output ONLY one Markdown table.\n"
            "No prose, no bullets, no JSON, no code fences."
        )

        try:
            import re

            def _clean_to_markdown(raw: str) -> str:
                cleaned = (raw or "").strip()
                if not cleaned:
                    return ""
                # Strip chain-of-thought / reasoning tags emitted by thinking models.
                cleaned = re.sub(
                    r"(?is)<(think|analysis|reasoning)>.*?</\1>\s*",
                    "",
                    cleaned,
                ).strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", cleaned).strip()
                    cleaned = re.sub(r"\n?```$", "", cleaned).strip()
                fence_match = re.search(
                    r"```(?:markdown|md|text)?\s*([\s\S]*?)\s*```",
                    cleaned,
                    flags=re.IGNORECASE,
                )
                if fence_match:
                    cleaned = fence_match.group(1).strip()
                lowered = cleaned.lower()
                if "unable to read" in lowered or lowered in {"unreadable", "n/a", "none"}:
                    return ""
                if lowered in {"[not_a_table]", "not_a_table"}:
                    return ""

                # If markdown rows exist, keep only pipe rows to strip accidental commentary.
                if "|" in cleaned:
                    pipe_lines = [ln.strip() for ln in cleaned.splitlines() if "|" in ln and ln.strip()]
                    if len(pipe_lines) >= 2:
                        cleaned = "\n".join(pipe_lines)

                # Fallback: preserve raw lines in a one-column markdown table.
                if "|" not in cleaned:
                    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
                    if not lines:
                        return ""
                    md_lines = ["| Table Content |", "| --- |"]
                    for line in lines:
                        escaped_line = line.replace("|", "\\|")
                        md_lines.append(f"| {escaped_line} |")
                    cleaned = "\n".join(md_lines)

                # Ensure a header separator exists.
                lines = [ln for ln in cleaned.splitlines() if ln.strip()]
                if len(lines) >= 2 and not re.search(r"\|\s*-{2,}", lines[1]):
                    if "|" in lines[0]:
                        columns = max(1, lines[0].count("|") - 1)
                        separator = "| " + " | ".join(["---"] * columns) + " |"
                        lines.insert(1, separator)
                        cleaned = "\n".join(lines)

                if len(cleaned) > 8000:
                    cleaned = cleaned[:8000]
                return cleaned

            page_info = f" page {page_number}" if page_number is not None else ""
            # Two attempts: base prompt + retry prompt, with slightly larger resize on retry.
            attempts = (
                (prompt_primary, 1800),
                (prompt_retry, 2200),
            )
            for attempt_idx, (prompt, max_dim) in enumerate(attempts, start=1):
                table_image = self._resize_image_for_vlm(image, max_dimension=max_dim)
                raw_response = self._provider.describe_image(
                    table_image,
                    prompt,
                    max_chars=14000,
                )
                self._processed_count += 1
                cleaned = _clean_to_markdown(raw_response or "")
                if cleaned:
                    logger.info(
                        f"[TABLE-VLM]{page_info}: extracted markdown ({len(cleaned)} chars) "
                        f"attempt={attempt_idx}"
                    )
                    return cleaned
                logger.debug(
                    f"[TABLE-VLM]{page_info}: empty/unusable response on attempt {attempt_idx}"
                )
            return ""
        except Exception as e:
            page_info = f" page {page_number}" if page_number is not None else ""
            logger.debug(f"[TABLE-VLM]{page_info}: extraction failed: {e}")
            return ""

    def transcribe_code_block(
        self,
        image: Image.Image,
        page_number: Optional[int] = None,
    ) -> str:
        """
        Transcribe code from an image region with strict formatting constraints.

        This is an opt-in fallback path for severely degraded scanned code blocks.
        """
        if image is None:
            return ""

        prompt_primary = (
            "You are transcribing source code from an image.\n"
            "Output ONLY code text.\n"
            "Rules:\n"
            "1. Preserve indentation and line breaks exactly.\n"
            "2. Do not add commentary.\n"
            "3. Do not add Markdown fences.\n"
            "4. If a token is unreadable, write [unreadable_token]."
        )
        prompt_retry = (
            "Retry transcription carefully.\n"
            "Output ONLY raw code with exact indentation and line breaks.\n"
            "No explanations. No markdown fences."
        )

        def _clean_code(raw: str) -> str:
            cleaned = (raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if not cleaned:
                return ""

            # Remove reasoning traces from thinking models.
            cleaned = re.sub(r"(?is)<think>.*?</think>\s*", "", cleaned).strip()
            cleaned = re.sub(r"(?is)<analysis>.*?</analysis>\s*", "", cleaned).strip()
            cleaned = re.sub(r"(?is)<reasoning>.*?</reasoning>\s*", "", cleaned).strip()

            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", cleaned).strip()
                cleaned = re.sub(r"\n?```$", "", cleaned).strip()
            fenced = re.search(r"```(?:python|py|text)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
            if fenced:
                cleaned = fenced.group(1).strip()

            # Drop common chatty prefixes while keeping code lines untouched.
            cleaned = re.sub(
                r"(?im)^\s*(here(?:'s| is)\s+(?:the\s+)?(?:code|transcription)\s*:)\s*$",
                "",
                cleaned,
            ).strip()

            # Keep indentation intact, just trim trailing whitespace per line.
            lines = [ln.rstrip() for ln in cleaned.split("\n")]
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            cleaned = "\n".join(lines).strip()

            # If the model returns flat top-level statements without indentation,
            # normalize to REPL-style lines to preserve executable semantics.
            if lines:
                has_repl = any(re.match(r"^\s*(>>>|\.\.\.)\s", ln) for ln in lines)
                has_indent = any(ln.startswith(("    ", "\t")) for ln in lines if ln.strip())
                if not has_repl and not has_indent and 2 <= len(lines) <= 12:
                    stmt_like = re.compile(
                        r"^\s*(?:from\s+[A-Za-z_][\w\.]*\s+import\b|import\s+[A-Za-z_][\w\.]*(?:\s*,\s*[A-Za-z_][\w\.]*)*(?:\s+as\s+[A-Za-z_][\w]*)?|[A-Za-z_][\w\.]*(?:\[[^\]]+\])?\s*=|[A-Za-z_][\w\.]*\s*\(|if\b|elif\b|else:|for\b|while\b|try:|except\b|with\b|return\b|yield\b|raise\b|assert\b|pass\b|break\b|continue\b)"
                    )
                    codey_lines = sum(
                        1
                        for ln in lines
                        if stmt_like.search(ln) is not None or re.search(r"[=\(\)\[\]\{\}:#]", ln)
                    )
                    if codey_lines >= max(1, int(len(lines) * 0.6)):
                        lines = [f">>> {ln.lstrip()}" for ln in lines]
                        cleaned = "\n".join(lines).strip()

            if len(cleaned) > 12000:
                cleaned = cleaned[:12000].rstrip()
            return cleaned

        page_info = f" page {page_number}" if page_number is not None else ""
        attempts = (
            (prompt_primary, 2200),
            (prompt_retry, 2600),
        )
        try:
            for attempt_idx, (prompt, max_dim) in enumerate(attempts, start=1):
                code_image = self._resize_image_for_vlm(image, max_dimension=max_dim)
                raw_response = self._provider.describe_image(
                    code_image,
                    prompt,
                    max_chars=16000,
                )
                self._processed_count += 1
                cleaned = _clean_code(raw_response or "")
                if cleaned:
                    logger.info(
                        f"[CODE-VLM]{page_info}: transcribed ({len(cleaned)} chars) "
                        f"attempt={attempt_idx}"
                    )
                    return cleaned
                logger.debug(
                    f"[CODE-VLM]{page_info}: empty/unusable response on attempt {attempt_idx}"
                )
            return ""
        except Exception as e:
            logger.debug(f"[CODE-VLM]{page_info}: transcription failed: {e}")
            return ""

    def _extract_clean_description(
        self,
        raw_response: str,
        page_number: int,
        state: "ContextStateV2",
    ) -> str:
        """
        AUTO-PILOT FIX: Extract clean description from VLM response.

        This ensures RAW JSON is NEVER returned as content.
        Only the human-readable description text goes into the RAG corpus.

        FINAL POLISH: Handles markdown code blocks (```json ... ```) and
        returns ONLY clean prose text - no JSON, no markdown, no OBJ/DESC prefixes.

        Args:
            raw_response: Raw VLM output (may be JSON, markdown-wrapped JSON, or plain text)
            page_number: Page number for fallback context
            state: Context state for fallback breadcrumbs

        Returns:
            Clean description text ONLY (max 400 chars) - no JSON, no markdown
        """
        import re

        # ================================================================
        # STEP 1: Strip markdown code blocks if present
        # VLMs often wrap JSON in ```json ... ``` or just ``` ... ```
        # ================================================================
        cleaned_response = raw_response.strip()

        # Remove markdown code block wrappers
        # Pattern: ```json\n{...}\n``` or ```\n{...}\n```
        markdown_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        markdown_match = re.search(markdown_pattern, cleaned_response, re.IGNORECASE)
        if markdown_match:
            cleaned_response = markdown_match.group(1).strip()
            logger.debug("[CLEAN-CONTENT] Stripped markdown code block wrapper")

        # ================================================================
        # STEP 2: Try to parse as JSON and extract description
        # ================================================================
        # Look for JSON object - handles nested braces better
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        json_match = re.search(json_pattern, cleaned_response, re.DOTALL)

        if json_match:
            try:
                parsed = json.loads(json_match.group(0))

                if isinstance(parsed, dict):
                    # Extract ONLY the description - this is what goes in content
                    desc = str(parsed.get("description", "")).strip()
                    obj = str(parsed.get("object", "")).strip()

                    # Build clean prose description
                    if desc:
                        # If object is meaningful, prepend it naturally
                        if obj and obj.lower() not in ("unknown", "visual", "image"):
                            # Natural sentence: "Firearm parts diagram showing..."
                            if not desc.lower().startswith(obj.lower()):
                                clean = f"{obj}: {desc}"
                            else:
                                clean = desc
                        else:
                            clean = desc
                    elif obj:
                        clean = f"Image of {obj}"
                    else:
                        clean = "[VLM_FAILED: parse error]"

                    # Remove any remaining JSON artifacts that might have slipped through
                    clean = self._strip_json_artifacts(clean)

                    logger.debug(f"[CLEAN-CONTENT] Parsed JSON → clean text: {clean[:60]}...")
                    return clean[:MAX_DESCRIPTION_CHARS]

            except json.JSONDecodeError as e:
                logger.debug(f"[CLEAN-CONTENT] JSON parse failed: {e}")

        # ================================================================
        # STEP 3: Fallback - extract description from malformed JSON
        # ================================================================
        if "{" in cleaned_response or '"description"' in cleaned_response:
            logger.warning("[CLEAN-CONTENT] Attempting extraction from malformed JSON")

            # Try to extract description field even from broken JSON
            # Handle escaped quotes and multi-line descriptions
            desc_patterns = [
                r'"description"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"',  # Standard
                r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"',  # With escapes
                r"description[\"']?\s*:\s*[\"']([^\"']+)[\"']",  # Loose format
            ]

            for pattern in desc_patterns:
                desc_match = re.search(pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                if desc_match:
                    desc = desc_match.group(1).strip()
                    # Unescape any escaped characters
                    desc = desc.replace('\\"', '"').replace("\\n", " ").replace("\\t", " ")
                    desc = self._strip_json_artifacts(desc)

                    if desc and len(desc) > 5:  # Must have meaningful content
                        logger.debug(f"[CLEAN-CONTENT] Extracted from malformed: {desc[:60]}...")
                        return desc[:MAX_DESCRIPTION_CHARS]

            # Try object field as last resort
            obj_match = re.search(r'"object"\s*:\s*"([^"]+)"', cleaned_response)
            if obj_match:
                obj = obj_match.group(1).strip()
                clean = f"Image of {obj}"
                return clean[:MAX_DESCRIPTION_CHARS]

            # Strip all JSON syntax and use what's left
            stripped = self._strip_json_artifacts(cleaned_response)
            if stripped and len(stripped) > 10:
                return stripped[:MAX_DESCRIPTION_CHARS]

        # ================================================================
        # STEP 4: Plain text response - clean up legacy OBJ:/DESC: format
        # ================================================================
        # Response is in legacy format like "OBJ: Brand logo\nDESC: The image shows..."
        clean = self._clean_legacy_format(cleaned_response)

        # If still has artifacts, strip them
        if "{" in clean or '"' in clean:
            clean = self._strip_json_artifacts(clean)

        if not clean or len(clean) < 5:
            clean = "[VLM_FAILED: parse error]"

        return clean[:MAX_DESCRIPTION_CHARS]

    @staticmethod
    def _strip_json_artifacts(text: str) -> str:
        """
        Remove any JSON syntax artifacts and legacy format prefixes from text.

        This ensures the final output is clean prose without:
        - Curly braces { }
        - Square brackets [ ]
        - Quotation marks at start/end
        - Field names like "description:", "object:", etc.
        - Markdown backticks
        - Legacy OBJ: / DESC: prefixes

        Args:
            text: Text that may contain JSON artifacts or legacy prefixes

        Returns:
            Clean prose text
        """
        import re

        if not text:
            return ""

        # Remove common JSON/markdown artifacts
        result = text.strip()

        # Remove leading/trailing quotes
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        if result.startswith("'") and result.endswith("'"):
            result = result[1:-1]

        # Remove field name prefixes like "description: " or "object: "
        result = re.sub(
            r"^(description|object|reasoning|confidence)\s*:\s*", "", result, flags=re.IGNORECASE
        )

        # Remove curly braces and their contents if they look like JSON
        result = re.sub(r"\{[^{}]*\}", "", result)

        # Remove square brackets content
        result = re.sub(r"\[[^\[\]]*\]", "", result)

        # Remove stray JSON characters
        result = re.sub(r'[{}\[\]"]', "", result)

        # Remove markdown backticks
        result = re.sub(r"`+", "", result)

        # Remove common JSON field patterns that might remain
        result = re.sub(r"^\s*(true|false|null)\s*$", "", result, flags=re.IGNORECASE)
        result = re.sub(r"^\s*\d+\.?\d*\s*$", "", result)  # Remove lone numbers

        # Collapse multiple spaces
        result = " ".join(result.split())

        return result.strip()

    @staticmethod
    def _clean_legacy_format(text: str) -> str:
        """
        Clean legacy OBJ: / DESC: format from VLM output.

        This handles the case where VLM returns:
        "OBJ: Brand logo\nDESC: The image shows..."

        And converts it to:
        "Brand logo: The image shows..."

        Or just the description if object is generic.

        Args:
            text: Text that may contain OBJ:/DESC: format

        Returns:
            Clean prose text without legacy prefixes
        """
        import re

        if not text:
            return ""

        result = text.strip()

        # Pattern 1: "OBJ: something\nDESC: description" or "OBJ: something DESC: description"
        # Extract both parts
        obj_desc_pattern = r"^OBJ:\s*([^\n]+?)[\n\s]+DESC:\s*(.+)$"
        match = re.match(obj_desc_pattern, result, re.IGNORECASE | re.DOTALL)

        if match:
            obj = match.group(1).strip()
            desc = match.group(2).strip()

            # If object is meaningful, prepend it naturally
            generic_objects = ("unknown", "visual", "image", "picture", "photo", "photograph")
            if obj.lower() not in generic_objects and obj:
                # Don't duplicate if description already starts with the object
                if not desc.lower().startswith(obj.lower()):
                    return f"{obj}: {desc}"
            return desc

        # Pattern 2: Just "DESC: description" without OBJ
        desc_only_pattern = r"^DESC:\s*(.+)$"
        match = re.match(desc_only_pattern, result, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

        # Pattern 3: Just "OBJ: something" without DESC
        obj_only_pattern = r"^OBJ:\s*(.+)$"
        match = re.match(obj_only_pattern, result, re.IGNORECASE | re.DOTALL)
        if match:
            obj = match.group(1).strip()
            # Remove any trailing DESC: if present but empty
            obj = re.sub(r"\s*DESC:\s*$", "", obj, flags=re.IGNORECASE)
            return f"Image of {obj}"

        # No legacy format found, return as-is
        return result

    def enrich_image_with_diagnostics(
        self,
        image: Image.Image,
        state: "ContextStateV2",
        page_number: int,
        anchor_text: Optional[str] = None,
        diagnostic_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        GEMINI AUDIT FIX #2 & #3: Enhanced image enrichment with diagnostic context.

        This method provides:
        1. Dynamic prompt injection based on document classification
        2. Confidence scoring for hallucination detection
        3. Reasoning field to validate VLM output quality

        Args:
            image: PIL Image to describe
            state: Current document context state
            page_number: Page number for context
            anchor_text: Surrounding text for semantic anchoring
            diagnostic_context: Context from DocumentDiagnosticEngine containing:
                - classification: Document modality (scanned, digital, etc.)
                - detected_features: List of detected document features
                - is_scan: Whether document is likely a scan
                - scan_hints: Hints for handling scan artifacts
                - confidence_level: Overall diagnostic confidence

        Returns:
            Dict with keys:
            - description: Generated description (max 400 chars)
            - object: Primary subject (1-3 words)
            - confidence: VLM confidence score (0.0-1.0)
            - reasoning: VLM's reasoning for its classification
            - is_low_confidence: True if confidence < 0.7 (hallucination risk)
            - raw_response: Original VLM response
        """
        # Check cache first
        if self._cache:
            cached = self._cache.get(image)
            if cached:
                logger.debug("Using cached description")
                # Return cached with default confidence
                return {
                    "description": cached,
                    "object": "cached",
                    "confidence": 1.0,  # Assume cached entries are valid
                    "reasoning": "cached response",
                    "is_low_confidence": False,
                    "raw_response": cached,
                }

        # ================================================================
        # GEMINI AUDIT FIX #2: Build dynamic prompt with diagnostic hints
        # ================================================================
        diagnostic_hints = ""

        if diagnostic_context:
            hints_parts = []

            # Add scan artifact hints
            if diagnostic_context.get("is_scan"):
                hints_parts.append(SCAN_ARTIFACT_HINTS)

            # Add scan hints from diagnostic
            scan_hints = diagnostic_context.get("scan_hints", [])
            if scan_hints:
                hints_parts.append(
                    "ADDITIONAL CONTEXT:\n" + "\n".join(f"- {h}" for h in scan_hints)
                )

            # Add classification context
            classification = diagnostic_context.get("classification", "unknown")
            domain = diagnostic_context.get("content_domain", "unknown")
            hints_parts.append(f"\nDOCUMENT TYPE: {classification}")
            hints_parts.append(f"CONTENT DOMAIN: {domain}")

            # Add confidence warning
            if diagnostic_context.get("confidence_level") == "low":
                hints_parts.append(
                    "\n⚠️ LOW DIAGNOSTIC CONFIDENCE - Be extra careful in your analysis."
                )

            diagnostic_hints = "\n".join(hints_parts)
            logger.debug(
                f"[DIAGNOSTIC-VLM] Injecting {len(diagnostic_hints)} chars of diagnostic context"
            )

        # Build context section
        context_section = ""
        if anchor_text or state.breadcrumbs:
            context_parts = []

            if state.breadcrumbs:
                breadcrumb_str = " > ".join(state.breadcrumbs)
                context_parts.append(f"DOCUMENT SECTION: {breadcrumb_str}")

            if anchor_text:
                text_context = anchor_text[:200] if len(anchor_text) > 200 else anchor_text
                context_parts.append(f'SURROUNDING TEXT: "{text_context}"')

            context_parts.append(f"PAGE: {page_number}")
            context_section = "DOCUMENT CONTEXT:\n" + "\n".join(context_parts)

        # Select and format prompt
        if context_section:
            prompt = ENRICHMENT_PROMPT_BASE.format(
                context_section=context_section,
                diagnostic_hints=diagnostic_hints,
            )
        else:
            prompt = ENRICHMENT_PROMPT_NO_CONTEXT.format(
                diagnostic_hints=diagnostic_hints,
            )

        # Get VLM response
        try:
            raw_response = self._provider.describe_image(image, prompt)
            self._processed_count += 1

            # ================================================================
            # GEMINI AUDIT FIX #3: Parse confidence and reasoning
            # ================================================================
            parsed = self._parse_enrichment_json(raw_response)

            if parsed is None:
                # Fallback: treat entire response as description
                logger.warning("[CONFIDENCE-CHECK] Failed to parse JSON, using raw response")
                description = raw_response[:MAX_DESCRIPTION_CHARS]
                result = {
                    "description": description,
                    "object": "unknown",
                    "confidence": 0.5,  # Medium confidence for unparsed
                    "reasoning": "JSON parse failed - using raw response",
                    "is_low_confidence": True,
                    "raw_response": raw_response,
                }
            else:
                confidence = parsed.get("confidence", 0.5)
                is_low_confidence = confidence < 0.7

                # Log confidence warning
                if is_low_confidence:
                    logger.warning(
                        f"[CONFIDENCE-CHECK] Low confidence ({confidence:.2f}) for page {page_number}: "
                        f"{parsed.get('reasoning', 'no reason')[:50]}"
                    )

                description = parsed.get("description", "")[:MAX_DESCRIPTION_CHARS]

                # Format description in legacy OBJ: DESC: format for compatibility
                obj = parsed.get("object", "unknown")
                formatted_desc = f"OBJ: {obj} DESC: {description}"

                result = {
                    "description": formatted_desc,
                    "object": obj,
                    "confidence": confidence,
                    "reasoning": parsed.get("reasoning", ""),
                    "is_low_confidence": is_low_confidence,
                    "raw_response": raw_response,
                }

            # Cache the formatted description
            if self._cache:
                self._cache.set(image, result["description"])

            logger.debug(
                f"[DIAGNOSTIC-VLM] Generated: obj={result['object']}, "
                f"confidence={result['confidence']:.2f}, "
                f"low_conf={result['is_low_confidence']}"
            )

            return result

        except Exception as e:
            logger.warning(f"VLM enrichment failed: {e}")
            fallback = "OBJ: Unknown DESC: [VLM analysis failed]"

            return {
                "description": fallback,
                "object": "unknown",
                "confidence": 0.0,
                "reasoning": f"VLM error: {str(e)[:50]}",
                "is_low_confidence": True,
                "raw_response": "",
            }

    @staticmethod
    def _parse_enrichment_json(response: str) -> Optional[Dict[str, Any]]:
        """
        GEMINI AUDIT FIX #3: Parse JSON response with confidence scoring.

        Args:
            response: Raw VLM response

        Returns:
            Dict with keys: object, description, confidence, reasoning
            Returns None if parsing fails
        """
        import re

        # Extract JSON object from response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if not json_match:
            logger.debug(f"[ENRICHMENT-PARSE] No JSON found in response: {response[:100]}")
            return None

        json_str = json_match.group(0)

        try:
            parsed = json.loads(json_str)

            if not isinstance(parsed, dict):
                return None

            # Extract and validate fields
            obj = str(parsed.get("object", "unknown"))[:50]
            description = str(parsed.get("description", ""))[:MAX_DESCRIPTION_CHARS]
            reasoning = str(parsed.get("reasoning", ""))[:100]

            # Parse confidence
            try:
                confidence = float(parsed.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1
            except (ValueError, TypeError):
                confidence = 0.5

            return {
                "object": obj,
                "description": description,
                "confidence": confidence,
                "reasoning": reasoning,
            }

        except json.JSONDecodeError as e:
            logger.debug(f"[ENRICHMENT-PARSE] JSON decode error: {e}")
            return None

    def enrich_image_with_ocr_hints(
        self,
        image: Image.Image,
        state: "ContextStateV2",
        page_number: int,
        anchor_text: Optional[str] = None,
        profile_params: Optional["ProfileParameters"] = None,
    ) -> str:
        """
        REQ-OCR-01: Non-destructive hybrid VLM+OCR enrichment.

        This method provides OCR hints to the VLM for scanned/degraded documents.
        The OCR output is NEVER used directly - it's passed as context to help
        the VLM identify brand names, model numbers, and titles.

        SAFETY MECHANISM: VLM as Judge
        ===============================
        - OCR often produces noise like "I1I!1l" for degraded scans
        - The VLM sees both the IMAGE and the OCR hints
        - The VLM decides whether to trust the OCR (e.g., "Sako", "Browning")
        - This prevents OCR noise from polluting the RAG corpus

        PAGE ISOLATION (REQ-OCR-ISOLATION):
        ===================================
        - Hints are LOCAL to this method call (no cross-page hint accumulation)
        - OCR engine runtime may be reused for performance/memory stability

        Args:
            image: PIL Image to describe
            state: Current document context state
            page_number: Page number for context
            anchor_text: Surrounding text for semantic anchoring
            profile_params: Profile parameters (enables OCR hints if profile.enable_ocr_hints)

        Returns:
            Generated description (max 400 chars) - CLEAN TEXT ONLY
        """
        # ================================================================
        # REQ-OCR-ISOLATION: EXPLICIT PAGE BOUNDARY - CLEAR ALL HINTS
        # ================================================================
        # This log line confirms that we're starting fresh for this page.
        # NO hints from previous pages should exist at this point.
        logger.info(f"[OCR-CLEANUP] Clearing hints for new page {page_number}")

        # Check if OCR hints are enabled via profile
        use_ocr_hints = False
        if profile_params and profile_params.enable_ocr_hints:
            use_ocr_hints = True
            logger.info(f"[OCR-HINT-VLM] OCR hints ENABLED for page {page_number}")

        # Check cache first
        if self._cache:
            cached = self._cache.get(image)
            if cached:
                logger.debug("Using cached description")
                return cached

        # ================================================================
        # OCR-HINT BYPASS: visual-only prompts intentionally avoid OCR hint
        # injection to enforce modality boundaries (Source Sanctity).
        # ================================================================
        ocr_hint_section = ""
        if use_ocr_hints:
            logger.info(
                f"[OCR-HINT-VLM] Page {page_number}: profile requested OCR hints, "
                "but visual-only prompt isolation is active; skipping OCR hint extraction."
            )

        # ================================================================
        # BUILD VISUAL-ONLY PROMPT (scan-aware, no OCR/breadcrumb/page injection)
        # ================================================================
        # AGENTS.md / Source Sanctity:
        # - OCR extracts text; VLM must not transcribe or summarize text.
        # - Passing OCR hints / surrounding text / breadcrumbs encourages meta-language
        #   and "reading" behavior, so we do not inject any of that into the prompt.
        is_scan = bool(profile_params and profile_params.inject_scan_hints)
        diagnostic_hints = SCAN_ARTIFACT_HINTS if is_scan else None

        _is_diagram, _is_photograph = _classify_image_type(image)
        prompt = build_visual_prompt(
            context_section=None,
            diagnostic_hints=diagnostic_hints,
            is_scan=is_scan,
            is_diagram=_is_diagram,
            is_photograph=_is_photograph,
            ocr_confidence=None,
        )

        # Log image dimensions for debugging large image issues
        img_width, img_height = image.size
        logger.info(
            f"[OCR-HINT-VLM] Page {page_number}: Image size {img_width}x{img_height}px, "
            f"prompt length: {len(prompt)} chars, OCR hints: {'Yes' if ocr_hint_section else 'No'}"
        )

        # Get description from provider (with validation + one strict retry)
        try:
            raw_response = self._provider.describe_image(image, prompt)
            self._processed_count += 1

            validation = validate_vlm_response(raw_response)
            if not validation.is_valid:
                logger.warning(
                    f"[VISUAL-ONLY-VLM] Rejected VLM response on page {page_number}: "
                    f"{', '.join(validation.issues) if validation.issues else 'invalid'}; retrying once"
                )
                raw_response = self._provider.describe_image(image, prompt)
                self._processed_count += 1
                validation = validate_vlm_response(raw_response)

            if not validation.is_valid:
                logger.warning(
                    f"[VISUAL-ONLY-VLM] Rejected VLM response on page {page_number} after retry: "
                    f"{', '.join(validation.issues) if validation.issues else 'invalid'}"
                )
                if validation.text_reading_detected:
                    description = "Dense typographic layout; no distinct non-text visuals."
                else:
                    description = "[VLM_FAILED: response invalid]"

                if self._cache:
                    self._cache.set(image, description)
                return description

            description = validation.cleaned_response
            if len(description) > MAX_DESCRIPTION_CHARS:
                description = description[:MAX_DESCRIPTION_CHARS]

            # Cache the result
            if self._cache:
                self._cache.set(image, description)

            logger.info(f"[VISUAL-ONLY-VLM] Page {page_number} SUCCESS: {description[:80]}...")
            return description

        except Exception as e:
            # ================================================================
            # ENHANCED ERROR LOGGING - Show the REAL error cause
            # ================================================================
            import traceback

            error_type = type(e).__name__
            error_msg = str(e)

            # Check for specific error types to give better diagnostics
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                logger.error(
                    f"[VLM-ERROR] Page {page_number}: TIMEOUT after waiting for VLM response. "
                    f"Image: {img_width}x{img_height}px, Prompt: {len(prompt)} chars. "
                    f"Consider increasing --vlm-timeout or reducing image DPI."
                )
            elif "connection" in error_msg.lower():
                logger.error(
                    f"[VLM-ERROR] Page {page_number}: CONNECTION FAILED to VLM provider. "
                    f"Is Ollama running? Check 'ollama serve' status. Error: {error_msg}"
                )
            elif "400" in error_msg or "bad request" in error_msg.lower():
                logger.error(
                    f"[VLM-ERROR] Page {page_number}: BAD REQUEST (400) - Invalid payload. "
                    f"Prompt may contain invalid characters. Prompt length: {len(prompt)} chars. "
                    f"First 200 chars of prompt: {prompt[:200]}"
                )
            elif "413" in error_msg or "too large" in error_msg.lower():
                logger.error(
                    f"[VLM-ERROR] Page {page_number}: PAYLOAD TOO LARGE (413). "
                    f"Image {img_width}x{img_height}px may be too big. Consider resizing."
                )
            elif "500" in error_msg or "internal server" in error_msg.lower():
                logger.error(
                    f"[VLM-ERROR] Page {page_number}: VLM SERVER ERROR (500). "
                    f"The VLM provider crashed. This may be due to GPU memory issues."
                )
            else:
                # Generic error with full details
                logger.error(
                    f"[VLM-ERROR] Page {page_number}: {error_type}: {error_msg}\n"
                    f"Image: {img_width}x{img_height}px, Prompt: {len(prompt)} chars\n"
                    f"Traceback: {traceback.format_exc()}"
                )

            # Also print to console for immediate visibility
            print(
                f"    ❌ [VLM-ERROR] Page {page_number}: {error_type}: {error_msg[:100]}",
                flush=True,
            )

            return "OBJ: Unknown DESC: [VLM analysis failed]"

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "processed_count": self._processed_count,
            "cache_size": self._cache.size() if self._cache else 0,
        }

    def flush_cache(self) -> None:
        """Flush cache to disk."""
        if self._cache:
            self._cache.flush()
            logger.debug("Vision cache flushed")
        if self._ocr_hint_engines:
            for ocr_engine in self._ocr_hint_engines.values():
                try:
                    cleanup = getattr(ocr_engine, "cleanup", None)
                    if callable(cleanup):
                        cleanup()
                except Exception as e:
                    logger.debug(f"OCR hint engine cleanup skipped: {e}")
            self._ocr_hint_engines.clear()
            gc.collect()
            logger.debug("OCR hint engine cache cleared")

    def verify_shadow_integrity(
        self,
        image: Image.Image,
        breadcrumbs: list[str],
    ) -> Dict[str, Any]:
        """
        REQ-MM-10: Full-Page Guard VLM Verification.

        Verify that a full-page shadow asset is editorial (not UI/navigation).

        Args:
            image: PIL Image to verify
            breadcrumbs: Document breadcrumbs for context

        Returns:
            Dict with keys:
            - classification: "editorial|ui_navigation|page_scan"
            - confidence: float (0.0-1.0)
            - reason: str
            - valid: bool (True if editorial, False if should discard)

        On VLM timeout or JSON parse error, returns:
            {"valid": False, "classification": "error", "reason": "..."}
        """
        # Pre-processing: Resize image to max 1024px on longest side (REQ-MM-10)
        resized = self._resize_image_for_vlm(image, max_dimension=1024)

        # Call VLM with Full-Page Guard prompt
        try:
            raw_response = self._provider.describe_image(resized, FULLPAGE_GUARD_PROMPT)
            logger.info(f"[VLM-GUARD] Raw response: {raw_response[:100]}")

            # Parse strict JSON response
            response_json = self._parse_fullpage_guard_json(raw_response)

            if response_json is None:
                logger.warning(f"[VLM-GUARD] Failed to parse JSON response. Defaulting to DISCARD.")
                return {
                    "valid": False,
                    "classification": "error",
                    "confidence": 0.0,
                    "reason": "JSON parse failed - default DISCARD",
                }

            classification = response_json.get("classification", "").lower()
            confidence = float(response_json.get("confidence", 0.0))
            reason = response_json.get("reason", "")

            # REQ-MM-11: Determine if asset is valid (editorial only)
            is_valid = classification == "editorial"

            # Log decision
            logger.info(
                f"[VLM-GUARD] Classification={classification} "
                f"Confidence={confidence:.2f} "
                f"Valid={is_valid}"
            )

            return {
                "valid": is_valid,
                "classification": classification,
                "confidence": confidence,
                "reason": reason,
            }

        except Exception as e:
            logger.error(f"[VLM-GUARD] VLM verification failed: {e}. Defaulting to DISCARD.")
            # FALLBACK: Default to DISCARD for safety (REQ-MM-10)
            return {
                "valid": False,
                "classification": "error",
                "confidence": 0.0,
                "reason": f"VLM error: {str(e)[:30]}",
            }

    @staticmethod
    def _resize_image_for_vlm(image: Image.Image, max_dimension: int = 1024) -> Image.Image:
        """
        Pre-process image by resizing to max dimension for VLM efficiency.

        REQ-MM-10: Resize to max 1024px to save bandwidth and VLM tokens.

        Args:
            image: PIL Image to resize
            max_dimension: Maximum width/height in pixels (default: 1024)

        Returns:
            Resized PIL Image
        """
        width, height = image.size

        if width > max_dimension or height > max_dimension:
            # Calculate scaling factor based on longest side
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)

            resized = image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS,
            )
            logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            return resized

        return image

    @staticmethod
    def _parse_fullpage_guard_json(response: str) -> Optional[Dict[str, Any]]:
        """
        Parse strict JSON response from Full-Page Guard VLM.

        REQ-MM-10: Enforce STRICT JSON format.

        Args:
            response: Raw text response from VLM

        Returns:
            Parsed dict with keys: classification, confidence, reason
            Returns None if JSON parsing fails
        """
        # Try to find JSON block in response (may have extra text)
        import re

        # Look for JSON object {...}
        json_match = re.search(r"\{[^{}]*\}", response)
        if not json_match:
            logger.warning(f"No JSON found in response: {response[:100]}")
            return None

        json_str = json_match.group(0)

        try:
            parsed = json.loads(json_str)

            # Validate required fields
            if not isinstance(parsed, dict):
                logger.warning(f"Parsed JSON is not a dict: {parsed}")
                return None

            if "classification" not in parsed or "confidence" not in parsed:
                logger.warning(f"Missing required fields in JSON: {parsed}")
                return None

            # Validate classification enum (GEMINI FIX: Added "advertisement")
            classification = str(parsed.get("classification", "")).lower()
            valid_classifications = ("editorial", "advertisement", "ui_navigation", "page_scan")
            if classification not in valid_classifications:
                logger.warning(
                    f"Invalid classification '{classification}'. " f"Valid: {valid_classifications}"
                )
                return None

            # Validate confidence is numeric
            try:
                confidence = float(parsed.get("confidence", 0.0))
                if not (0.0 <= confidence <= 1.0):
                    logger.warning(f"Confidence out of range: {confidence}")
                    confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                logger.warning(f"Could not parse confidence as float")
                confidence = 0.0

            return {
                "classification": classification,
                "confidence": confidence,
                "reason": str(parsed.get("reason", "")),
            }

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            return None

    def analyze_page_preview(
        self,
        pdf_path: str | Path,
        page_number: int,
        dpi: int = 150,
        dynamic_dpi: bool = True,
    ) -> Dict[str, Any]:
        """
        REQ-VLM-02: Analyze full-page preview for low-recall trigger.

        Renders a page in-memory (IRON-03: no disk export) and analyzes it
        with VLM to detect missing editorial visuals.

        ARCHITECTURAL NOTE: Dynamic DPI Scaling
        ========================================
        By default (dynamic_dpi=True), this method implements adaptive DPI:
        - First attempt: 150 DPI (fast, low bandwidth)
        - If inconclusive: 300 DPI re-render (for complex infographics)

        This ensures VLM can see fine details (small text in infographics)
        without wasting bandwidth on simple text-heavy pages.

        Args:
            pdf_path: Path to PDF file
            page_number: 1-indexed page number to analyze
            dpi: Initial rendering DPI (default: 150 per REQ-VLM-02)
            dynamic_dpi: Enable adaptive DPI scaling (default: True)

        Returns:
            Dict with keys:
            - missing_visuals: bool (True if editorial visuals detected)
            - count: int (estimated count of missing visuals)
            - description: str (detailed analysis)
            - raw_response: str (raw VLM response)
            - dpi_used: int (actual DPI used for rendering)
            - adaptive_scaling: bool (True if re-rendered at higher DPI)

        Raises:
            RuntimeError: If page rendering fails or page doesn't exist
        """

        import fitz
        from pathlib import Path

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise RuntimeError(f"PDF not found: {pdf_path}")

        # In-memory rendering (IRON-03 compliance: no file export)
        page_image = None
        try:
            doc = fitz.open(str(pdf_path))
            if page_number < 1 or page_number > len(doc):
                raise RuntimeError(f"Page {page_number} out of range (1-{len(doc)})")

            # Load page (1-indexed to 0-indexed)
            page = doc.load_page(page_number - 1)

            # Render at 150 DPI (default for low-recall analysis)
            # DPI = 72 * zoom_factor, so 150 DPI ≈ zoom=2.08
            zoom_factor = dpi / 72.0
            mat = fitz.Matrix(zoom_factor, zoom_factor)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Convert to PIL Image (in-memory only)
            if pix.n - pix.alpha < 4:  # Gray or RGB
                img_mode = "RGB" if pix.n - pix.alpha == 3 else "L"
            else:  # RGBA
                img_mode = "RGBA"

            img_data = pix.tobytes("ppm")
            page_image = Image.open(io.BytesIO(img_data)).convert("RGB")

            logger.info(
                f"[LOW-RECALL] Rendered page {page_number} at {dpi}DPI "
                f"({page_image.width}x{page_image.height}px)"
            )

        except Exception as e:
            logger.error(f"[LOW-RECALL] Failed to render page {page_number}: {e}")
            raise RuntimeError(f"Page rendering failed: {e}")
        finally:
            if "doc" in locals():
                doc.close()

        # VLM analysis of rendered page
        if not page_image:
            raise RuntimeError("Failed to create page image")

        # Define low-recall detection prompt
        low_recall_prompt = """NIEDOSTĘPNE WIZUALNE ANALIZY STRON

Przeanalizuj tę stronę i zdecyduj, czy zawiera ważne wizualne 
elementy redakcyjne (zdjęcia, infografiki, wykresy, mapy), 
które mogły nie być wyodrębnione z dokumentu.

ODPOWIEDŹ (JSON):
{
  "missing_visuals": true|false,
  "count": liczba (szacunkowa),
  "description": "Krótka analiza (max 100 znaków)"
}

Klasyfikacja:
- true: Strona zawiera zdjęcia, ilustracje, diagramy, które nie zostały przechwycone
- false: Strona zawiera głównie tekst, już wyekstrahowane elementy wizualne, lub zawartość UI

WAŻNE: Odpowiedz TYLKO poprawnym JSON. Brak innego tekstu."""

        try:
            raw_response = self._provider.describe_image(page_image, low_recall_prompt)
            logger.info(f"[LOW-RECALL] VLM Response: {raw_response[:100]}")

            # Parse JSON response
            response_json = self._parse_low_recall_json(raw_response)

            if response_json is None:
                logger.warning(
                    f"[LOW-RECALL] Failed to parse JSON. Defaulting to safe (no missing visuals)."
                )
                return {
                    "missing_visuals": False,
                    "count": 0,
                    "description": "JSON parse failed - conservative default",
                    "raw_response": raw_response,
                }

            return {
                "missing_visuals": response_json.get("missing_visuals", False),
                "count": response_json.get("count", 0),
                "description": response_json.get("description", ""),
                "raw_response": raw_response,
            }

        except Exception as e:
            logger.error(f"[LOW-RECALL] VLM analysis failed: {e}")
            raise RuntimeError(f"VLM analysis failed: {e}")

    @staticmethod
    def _parse_low_recall_json(response: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON response from low-recall VLM analysis.

        Args:
            response: Raw VLM response

        Returns:
            Dict with keys: missing_visuals (bool), count (int), description (str)
            Returns None if parsing fails
        """
        import re

        # Extract JSON object
        json_match = re.search(r"\{[^{}]*\}", response)
        if not json_match:
            logger.warning(f"[LOW-RECALL] No JSON found in response")
            return None

        json_str = json_match.group(0)

        try:
            parsed = json.loads(json_str)

            if not isinstance(parsed, dict):
                logger.warning(f"[LOW-RECALL] Parsed JSON is not a dict")
                return None

            missing_visuals = bool(parsed.get("missing_visuals", False))
            count = int(parsed.get("count", 0))
            description = str(parsed.get("description", ""))

            return {
                "missing_visuals": missing_visuals,
                "count": max(0, count),  # Ensure non-negative
                "description": description[:100],  # Truncate to 100 chars
            }

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"[LOW-RECALL] JSON parse error: {e}")
            return None


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_vision_manager(
    provider: str = "ollama",
    api_key: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    model: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    base_url: Optional[str] = None,
) -> VisionManager:
    """
    Factory function to create a VisionManager.

    Args:
        provider: Provider name ("ollama", "openai", "anthropic", "haiku")
        api_key: API key for cloud providers
        cache_dir: Directory for caching
        model: Model name (optional for Ollama - auto-detects if not specified)
        timeout: Request timeout in seconds
        base_url: Custom API base URL for OpenAI-compatible APIs (e.g., LM Studio).
                  Can also be set via OPENAI_BASE_URL environment variable.

    Returns:
        Configured VisionManager instance

    Raises:
        ValueError: If provider is unsupported or API key missing for cloud providers

    Example:
        # Use with LM Studio on a remote machine:
        manager = create_vision_manager(
            provider="openai",
            api_key="lm-studio",  # LM Studio accepts any non-empty key
            model="your-local-model",
            base_url="http://192.168.10.11:1234/v1"
        )

        # Or set OPENAI_BASE_URL environment variable and call normally:
        # export OPENAI_BASE_URL="http://192.168.10.11:1234/v1"
        manager = create_vision_manager(provider="openai", api_key="lm-studio")
    """
    import os

    provider_lower = provider.lower()

    if provider_lower == "ollama":
        # Model is optional - OllamaProvider auto-detects if not specified
        vision_provider = OllamaProvider(
            model=model,  # None triggers auto-detection
            timeout=timeout,
        )

    elif provider_lower == "openai":
        if not api_key:
            raise ValueError("OpenAI provider requires API key")

        # Resolve base_url: prefer explicit parameter, fallback to env var
        effective_base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if effective_base_url:
            logger.info(
                f"[OPENAI-CONFIG] Using custom base_url: {effective_base_url} "
                f"(source: {'parameter' if base_url else 'OPENAI_BASE_URL env'})"
            )

        vision_provider = OpenAIProvider(
            api_key=api_key,
            model=model or DEFAULT_OPENAI_MODEL,
            timeout=timeout,
            base_url=effective_base_url,
        )

    elif provider_lower in ("anthropic", "haiku"):
        if not api_key:
            raise ValueError("Anthropic provider requires API key")
        vision_provider = AnthropicProvider(
            api_key=api_key,
            model=model or DEFAULT_ANTHROPIC_MODEL,
            timeout=timeout,
        )

    else:
        raise ValueError(f"Unsupported vision provider: {provider}")

    # Resolve the effective model name for cache invalidation tracking.
    # For Ollama without an explicit model, the name is auto-detected at call
    # time so we leave it blank to avoid spurious cache invalidations.
    effective_model_name: str = ""
    if provider_lower == "openai":
        effective_model_name = model or DEFAULT_OPENAI_MODEL
    elif provider_lower in ("anthropic", "haiku"):
        effective_model_name = model or DEFAULT_ANTHROPIC_MODEL
    elif provider_lower == "ollama" and model:
        effective_model_name = model

    return VisionManager(
        provider=vision_provider,
        cache_dir=cache_dir,
        model_name=effective_model_name,
    )
