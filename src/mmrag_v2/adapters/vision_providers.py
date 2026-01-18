"""
Vision Providers - Pluggable VLM Interface for Image Enrichment
================================================================
ENGINE_USE: Claude 4.5 Opus (Architect)

This module provides a pluggable Vision Provider architecture for enriching
image chunks with VLM-generated descriptions. Supports local (Ollama) and
cloud (OpenAI, Anthropic) providers with automatic fallback.

REQ Compliance:
- REQ-CHUNK-03: VLM descriptions MUST NOT exceed 400 characters
- REQ-MM-02: Asset naming preserved (handled by processor)
- REQ-STATE: Context passed to providers for domain-specific prompting

Providers:
- OllamaProvider: Local llava via localhost:11434
- OpenAIProvider: GPT-4o-mini via API
- AnthropicProvider: Claude 3.5 Haiku via API
- FallbackProvider: Breadcrumb + anchor text (no VLM, zero latency)

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-29
"""

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Optional

import requests
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# REQ-CHUNK-03: Max 400 characters for child chunks
MAX_DESCRIPTION_CHARS: int = 400

# ============================================================================
# VLM PROMPTS - "PIXEL-PRISON" PROTOCOL V3 (BRUTAL)
# ============================================================================
# V3: BRUTAL mode - strips ALL preambles, forces OBJ|DESC pipe format.
# The VLM output must be RAG-ready: no "The image shows", no "This is a".

VLM_SYSTEM_PROMPT: str = """VISUAL SENSOR. OUTPUT ONLY:
OBJ: [type] | DESC: [visual details]

FORBIDDEN PHRASES (auto-stripped):
- "The image shows"
- "This is a"
- "I can see"
- "appears to be"
- "photograph of"

RAW DATA ONLY. NO PROSE."""

VLM_PRIMARY_PROMPT: str = """RESPOND ONLY IN THIS FORMAT:
OBJ: [aircraft/vehicle/graph/diagram/photo/chart/table/circuit/person/scene] | DESC: [what you see]

RULES:
1. Start with "OBJ:" - NO OTHER TEXT BEFORE IT
2. Use pipe "|" separator
3. DESC must be visual details only (colors, shapes, objects visible)
4. NO introductory phrases ("The image shows", "This appears to be")

EXAMPLE:
OBJ: aircraft | DESC: B-2 stealth bomber in flight, dark grey delta wing, blue sky background

NOW DESCRIBE:"""

VLM_FALLBACK_PROMPT: str = "OBJ: [type] | DESC: [what you see]"

# BLIND MODE PROMPT - even stricter
VLM_BLIND_PROMPT: str = """OBJ: [type] | DESC: [visual elements only]"""

VLM_UNAVAILABLE_MSG: str = "OBJ: unknown | DESC: [VLM unavailable]"

# Preambles to strip from VLM output (case-insensitive)
VLM_PREAMBLE_PATTERNS: list = [
    "the image shows ",
    "the image displays ",
    "the image depicts ",
    "the image features ",
    "the image contains ",
    "the image is ",
    "the image appears to ",
    "this image shows ",
    "this image displays ",
    "this image depicts ",
    "this image is ",
    "this is a photograph of ",
    "this is a photo of ",
    "this is an image of ",
    "this is a picture of ",
    "this appears to be ",
    "this seems to be ",
    "i can see ",
    "i see ",
    "we can see ",
    "we see ",
    "it shows ",
    "it displays ",
    "it depicts ",
    "it appears to be ",
    "in this image, ",
    "in the image, ",
    "here we see ",
    "here is ",
    "there is ",
    "there are ",
    "looking at the image, ",
    "upon examination, ",
    "the picture shows ",
    "the picture displays ",
    "the photo shows ",
    "the photograph shows ",
]

# Timeouts for providers
# Increased to 120s to handle large magazine pages (2178x3076px)
CONNECT_TIMEOUT: float = 5.0
READ_TIMEOUT: float = 120.0  # Increased for large images and memory-constrained systems
DEFAULT_TIMEOUT: tuple = (CONNECT_TIMEOUT, READ_TIMEOUT)

# Retry configuration for exponential backoff
MAX_RETRIES: int = 3
INITIAL_BACKOFF: float = 2.0  # seconds


# ============================================================================
# EXCEPTIONS
# ============================================================================


class VisionProviderError(Exception):
    """Base exception for vision provider failures."""

    pass


class VisionProviderTimeoutError(VisionProviderError):
    """Raised when a vision provider times out."""

    pass


class VisionProviderConnectionError(VisionProviderError):
    """Raised when a vision provider is unreachable."""

    pass


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================


class VisionProvider(ABC):
    """
    Abstract base class for Vision Language Model providers.

    All providers must implement the describe_image method which takes a PIL Image
    and context string, returning a VLM-generated description.

    Implementations must:
    1. Handle their own timeouts and connection errors
    2. Raise VisionProviderError subclasses on failure
    3. Return descriptions truncated to MAX_DESCRIPTION_CHARS
    """

    @abstractmethod
    def describe_image(
        self,
        image: Image.Image,
        context: str,
        page_number: int = 1,
    ) -> str:
        """
        Generate a VLM description for an image.

        Args:
            image: PIL Image object to describe
            context: Contextual information (breadcrumbs, anchor text)
            page_number: Page number for reference

        Returns:
            VLM-generated description, max MAX_DESCRIPTION_CHARS characters

        Raises:
            VisionProviderError: On any failure (timeout, connection, API error)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name for logging."""
        pass

    def _strip_preambles(self, text: str) -> str:
        """
        Strip verbose VLM preambles from output.

        VLMs like llava often add "The image shows..." or "This is a photograph of..."
        which pollutes the RAG database. This removes them.

        Args:
            text: Raw VLM output

        Returns:
            Cleaned text without preamble pollution
        """
        if not text:
            return text

        cleaned = text.strip()
        text_lower = cleaned.lower()

        # Try each preamble pattern
        for pattern in VLM_PREAMBLE_PATTERNS:
            if text_lower.startswith(pattern):
                # Strip the preamble, preserve case of remaining text
                cleaned = cleaned[len(pattern) :].strip()
                text_lower = cleaned.lower()
                logger.debug(f"Stripped preamble: '{pattern}'")

        # Capitalize first letter if it became lowercase
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]

        return cleaned

    def _truncate_description(self, text: str) -> str:
        """
        Clean and truncate description to REQ-CHUNK-03 limit.

        First strips verbose preambles, then truncates if needed.
        Attempts to break at sentence boundary if possible.
        """
        # First strip preambles
        text = self._strip_preambles(text)

        if len(text) <= MAX_DESCRIPTION_CHARS:
            return text

        # Try to break at sentence boundary
        truncated = text[:MAX_DESCRIPTION_CHARS]

        # Look for last sentence ending
        for delim in [". ", "! ", "? "]:
            last_pos = truncated.rfind(delim)
            if last_pos > MAX_DESCRIPTION_CHARS * 0.6:  # At least 60% of content
                return truncated[: last_pos + 1].strip()

        # Fallback: break at word boundary
        last_space = truncated.rfind(" ")
        if last_space > MAX_DESCRIPTION_CHARS * 0.8:
            return truncated[:last_space].strip() + "..."

        return truncated.strip() + "..."

    def _image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _resize_for_vlm(
        self,
        image: Image.Image,
        max_dimension: int = 1024,
    ) -> Image.Image:
        """
        Resize image if it exceeds max_dimension to prevent VLM timeouts.

        Local VLMs like llava perform better with smaller images.
        This does not affect the original asset saved to disk.

        Args:
            image: PIL Image to potentially resize
            max_dimension: Maximum width or height in pixels

        Returns:
            Resized image if larger than max_dimension, otherwise original
        """
        width, height = image.size

        if width <= max_dimension and height <= max_dimension:
            return image

        # Calculate new size preserving aspect ratio
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))

        logger.debug(f"Resizing image from {width}x{height} to {new_width}x{new_height} for VLM")

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


# ============================================================================
# OLLAMA PROVIDER (LOCAL)
# ============================================================================


class OllamaProvider(VisionProvider):
    """
    Local vision provider using Ollama with auto-detected or user-specified VLM.

    Optimized for Apple Silicon. Connects to localhost:11434 by default.
    Implements timeout with automatic failure signaling for fallback.

    Uses /api/generate endpoint for vision models with images parameter.
    This is the correct endpoint for Ollama vision models.

    KEEPALIVE: Model is kept loaded for 30 minutes to prevent unloading during
    long Docling processing phases.

    AUTO-DETECTION: If no model is specified, the provider will query Ollama
    to find the first running/loaded model and use that.
    """

    DEFAULT_BASE_URL: str = "http://localhost:11434"
    KEEPALIVE_DURATION: str = "30m"  # Keep model loaded for 30 minutes

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: tuple = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initialize Ollama provider and preload model.

        Args:
            model: Ollama model name (optional - auto-detects if not specified)
            base_url: Ollama API base URL (default: localhost:11434)
            timeout: (connect_timeout, read_timeout) tuple in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Auto-detect model if not specified
        if not model:
            model = self._detect_loaded_model()

        self.model = model

        # Preload model with keepalive to prevent unloading during processing
        self._preload_model()

        logger.info(
            f"OllamaProvider initialized: model={self.model}, "
            f"url={base_url}, timeout={timeout}, keepalive={self.KEEPALIVE_DURATION}"
        )

    def _detect_loaded_model(self) -> str:
        """
        Auto-detect which VLM model is currently loaded/running in Ollama.

        Queries the Ollama API to find running models. If no model is running,
        falls back to checking available models and uses the first one found.

        Returns:
            Name of the detected model

        Raises:
            VisionProviderConnectionError: If Ollama is not running
            VisionProviderError: If no models are available in Ollama
        """
        # First, check for currently running models via /api/ps
        try:
            response = requests.get(
                f"{self.base_url}/api/ps",
                timeout=(5.0, 10.0),
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                if models:
                    # Return the first running model
                    model_name = models[0].get("name", models[0].get("model"))
                    logger.info(f"[AUTO-DETECT] Found running model: {model_name}")
                    return model_name
        except requests.ConnectionError:
            raise VisionProviderConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. " "Is 'ollama serve' running?"
            )
        except Exception as e:
            logger.warning(f"Failed to check running models: {e}")

        # If no running model, check available models via /api/tags
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=(5.0, 10.0),
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                if models:
                    # Return the first available model
                    model_name = models[0].get("name", models[0].get("model"))
                    logger.info(f"[AUTO-DETECT] Using first available model: {model_name}")
                    return model_name
        except Exception as e:
            logger.warning(f"Failed to list available models: {e}")

        # No models found
        raise VisionProviderError(
            "No VLM models found in Ollama. "
            "Please load a vision model first (e.g., 'ollama pull llava:latest')"
        )

    def _preload_model(self) -> None:
        """
        Preload model with extended keep_alive to prevent unloading.

        Ollama unloads models after 5 minutes by default. During long
        Docling processing phases, this can cause model reloads on each
        VLM call. This sends a warmup request with 30m keep_alive.
        """
        try:
            payload = {
                "model": self.model,
                "keep_alive": self.KEEPALIVE_DURATION,
            }
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(5.0, 10.0),  # Short timeout for preload
            )
            if response.status_code == 200:
                logger.info(
                    f"[KEEPALIVE] Model {self.model} preloaded "
                    f"with {self.KEEPALIVE_DURATION} keepalive"
                )
            else:
                logger.warning(f"[KEEPALIVE] Preload failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"[KEEPALIVE] Preload failed: {e}")

    @property
    def name(self) -> str:
        return f"Ollama/{self.model}"

    def describe_image(
        self,
        image: Image.Image,
        context: str,
        page_number: int = 1,
    ) -> str:
        """
        Generate description using Ollama vision model (llava).

        Uses /api/generate endpoint with images parameter.
        This is the correct API for Ollama vision models like llava.

        Args:
            image: PIL Image to describe
            context: Contextual breadcrumbs/anchor text
            page_number: Page number for reference

        Returns:
            VLM description, max 400 chars

        Raises:
            VisionProviderTimeoutError: If Ollama times out
            VisionProviderConnectionError: If Ollama is unreachable
            VisionProviderError: For other API errors
        """
        # Use /api/generate endpoint for vision models (NOT /api/chat!)
        endpoint = f"{self.base_url}/api/generate"

        # Resize large images to prevent timeouts with local VLMs
        # This only affects the VLM input, not the saved asset
        resized_image = self._resize_for_vlm(image, max_dimension=1024)

        # Convert image to base64
        image_b64 = self._image_to_base64(resized_image)

        # ATTEMPT 1: Primary simplified prompt (llava-compatible)
        description = self._call_ollama(endpoint, VLM_PRIMARY_PROMPT, image_b64)

        if description:
            logger.debug(f"Ollama response (primary): {description[:100]}...")
            return self._truncate_description(description)

        # ATTEMPT 2: Retry with even shorter fallback prompt
        logger.warning("Primary prompt returned empty, retrying with fallback prompt")
        description = self._call_ollama(endpoint, VLM_FALLBACK_PROMPT, image_b64)

        if description:
            logger.debug(f"Ollama response (fallback): {description[:100]}...")
            return self._truncate_description(description)

        # Both attempts failed - return unavailable message
        logger.error("Both VLM attempts failed, returning unavailable message")
        return VLM_UNAVAILABLE_MSG

    def _call_ollama(
        self,
        endpoint: str,
        prompt: str,
        image_b64: str,
    ) -> str:
        """
        Make a single Ollama API call with the given prompt.

        Uses /api/generate with images at top level.
        This is the correct format for Ollama vision models like llava.

        Args:
            endpoint: Ollama API endpoint (/api/generate)
            prompt: Simple prompt string
            image_b64: Base64 encoded image

        Returns:
            Response content or empty string on failure
        """
        # Ollama /api/generate format for vision models
        # Images are at top level, NOT inside messages!
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 150,
            },
        }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            # /api/generate returns response in "response" field
            content = result.get("response", "").strip()

            if not content:
                logger.warning(f"Ollama empty response for prompt: {prompt[:50]}...")

            return content

        except requests.Timeout:
            logger.warning(f"Ollama timeout after {self.timeout}s")
            raise VisionProviderTimeoutError(f"Ollama timed out after {self.timeout[1]}s")
        except requests.ConnectionError as e:
            logger.warning(f"Ollama connection failed: {e}")
            raise VisionProviderConnectionError(f"Cannot connect to Ollama at {self.base_url}")
        except requests.HTTPError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise VisionProviderError(f"Ollama API error: {e}")
        except VisionProviderError:
            # Re-raise VisionProviderError without wrapping
            raise
        except Exception as e:
            logger.error(f"Ollama unexpected error: {e}")
            raise VisionProviderError(f"Unexpected Ollama error: {e}")


# ============================================================================
# OPENAI PROVIDER (CLOUD)
# ============================================================================


class OpenAIProvider(VisionProvider):
    """
    Cloud vision provider using OpenAI GPT-4o-mini.

    Requires OPENAI_API_KEY environment variable or explicit api_key.
    """

    DEFAULT_MODEL: str = "gpt-4o-mini"
    API_URL: str = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: tuple = DEFAULT_TIMEOUT,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4o-mini)
            timeout: Request timeout tuple
            base_url: Custom API base URL (e.g., http://192.168.10.11:1234/v1 for LM Studio)
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = base_url

        if base_url:
            logger.info(f"OpenAIProvider initialized: model={model}, base_url={base_url}")
        else:
            logger.info(f"OpenAIProvider initialized: model={model}")

    @property
    def name(self) -> str:
        return f"OpenAI/{self.model}"

    def describe_image(
        self,
        image: Image.Image,
        context: str,
        page_number: int = 1,
    ) -> str:
        """
        Generate description using OpenAI vision model.

        Args:
            image: PIL Image to describe
            context: Contextual breadcrumbs/anchor text
            page_number: Page number for reference

        Returns:
            VLM description, max 400 chars

        Raises:
            VisionProviderError: On API failure
        """
        image_b64 = self._image_to_base64(image)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": VLM_PRIMARY_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Context: {context}\n\nDescribe this image:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "low",  # Optimize for speed/cost
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 150,
            "temperature": 0.3,
        }

        # Use custom base_url if provided, otherwise default to OpenAI
        api_url = f"{self.base_url.rstrip('/')}/chat/completions" if self.base_url else self.API_URL

        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            description = result["choices"][0]["message"]["content"].strip()

            logger.debug(f"OpenAI response: {description[:100]}...")
            return self._truncate_description(description)

        except requests.Timeout:
            raise VisionProviderTimeoutError("OpenAI API timeout")
        except requests.HTTPError as e:
            raise VisionProviderError(f"OpenAI API error: {e}")
        except Exception as e:
            raise VisionProviderError(f"OpenAI error: {e}")


# ============================================================================
# ANTHROPIC PROVIDER (CLOUD)
# ============================================================================


class AnthropicProvider(VisionProvider):
    """
    Cloud vision provider using Anthropic Claude 3.5 Haiku.

    Requires ANTHROPIC_API_KEY environment variable or explicit api_key.
    """

    DEFAULT_MODEL: str = "claude-3-5-haiku-20241022"
    API_URL: str = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: tuple = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Model name (default: claude-3-5-haiku)
            timeout: Request timeout tuple
        """
        if not api_key:
            raise ValueError("Anthropic API key is required")

        self.api_key = api_key
        self.model = model
        self.timeout = timeout

        logger.info(f"AnthropicProvider initialized: model={model}")

    @property
    def name(self) -> str:
        return f"Anthropic/{self.model}"

    def describe_image(
        self,
        image: Image.Image,
        context: str,
        page_number: int = 1,
    ) -> str:
        """
        Generate description using Anthropic Claude vision.

        Args:
            image: PIL Image to describe
            context: Contextual breadcrumbs/anchor text
            page_number: Page number for reference

        Returns:
            VLM description, max 400 chars

        Raises:
            VisionProviderError: On API failure
        """
        image_b64 = self._image_to_base64(image)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": 150,
            "system": VLM_PRIMARY_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Context: {context}\n\nDescribe this image:",
                        },
                    ],
                },
            ],
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            description = result["content"][0]["text"].strip()

            logger.debug(f"Anthropic response: {description[:100]}...")
            return self._truncate_description(description)

        except requests.Timeout:
            raise VisionProviderTimeoutError("Anthropic API timeout")
        except requests.HTTPError as e:
            raise VisionProviderError(f"Anthropic API error: {e}")
        except Exception as e:
            raise VisionProviderError(f"Anthropic error: {e}")


# ============================================================================
# FALLBACK PROVIDER (NO VLM)
# ============================================================================


class FallbackProvider(VisionProvider):
    """
    Zero-latency fallback provider using breadcrumb + anchor text.

    SRS-compliant fallback when VLM is unavailable. Generates descriptions
    from ContextStateV2 hierarchy and surrounding text.

    Example output:
    "[Figure on page 5] Context: Chapter 3: Stealth Technology > 3.2 Radar Cross-Section"
    """

    def __init__(self) -> None:
        """Initialize fallback provider."""
        logger.info("FallbackProvider initialized (no VLM, breadcrumb-based)")

    @property
    def name(self) -> str:
        return "Fallback/Breadcrumb"

    def describe_image(
        self,
        image: Image.Image,
        context: str,
        page_number: int = 1,
    ) -> str:
        """
        Generate description from context without VLM.

        Args:
            image: PIL Image (not used, but required by interface)
            context: Contextual breadcrumbs/anchor text
            page_number: Page number for reference

        Returns:
            Context-based description, max 400 chars
        """
        # Clean up context
        context_clean = context.strip() if context else "Document"

        # Build description
        description = f"[Figure on page {page_number}] Context: {context_clean}"

        return self._truncate_description(description)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_vision_provider(
    provider_type: str,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> VisionProvider:
    """
    Factory function to create a vision provider.

    Args:
        provider_type: One of "ollama", "openai", "anthropic", "none"
        api_key: API key for cloud providers
        **kwargs: Additional provider-specific arguments
            - model: Optional for Ollama (auto-detects if not specified), optional for others

    Returns:
        Configured VisionProvider instance

    Raises:
        ValueError: If provider_type is unknown or api_key missing for cloud providers
    """
    provider_type = provider_type.lower().strip()

    if provider_type == "ollama":
        # Model is optional - OllamaProvider auto-detects if not specified
        return OllamaProvider(
            model=kwargs.get("model"),  # None triggers auto-detection
            base_url=kwargs.get("base_url", OllamaProvider.DEFAULT_BASE_URL),
            timeout=kwargs.get("timeout", DEFAULT_TIMEOUT),
        )

    elif provider_type == "openai":
        if not api_key:
            raise ValueError("OpenAI provider requires api_key")
        return OpenAIProvider(
            api_key=api_key,
            model=kwargs.get("model", OpenAIProvider.DEFAULT_MODEL),
            timeout=kwargs.get("timeout", DEFAULT_TIMEOUT),
            base_url=kwargs.get("base_url"),  # ✅ Now passes base_url parameter
        )

    elif provider_type in ("anthropic", "haiku"):
        if not api_key:
            raise ValueError("Anthropic provider requires api_key")
        return AnthropicProvider(
            api_key=api_key,
            model=kwargs.get("model", AnthropicProvider.DEFAULT_MODEL),
            timeout=kwargs.get("timeout", DEFAULT_TIMEOUT),
        )

    elif provider_type in ("none", "fallback"):
        return FallbackProvider()

    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Valid options: ollama, openai, anthropic, none"
        )
