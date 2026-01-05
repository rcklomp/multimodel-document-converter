"""
Adapters Module - Pluggable Provider Interfaces
================================================
ENGINE_USE: Claude 4.5 Opus (Architect)

This module provides pluggable adapter interfaces for external services,
including Vision Language Model providers for image enrichment.

Components:
- VisionProvider: Abstract interface for VLM providers
- OllamaProvider: Local llama3.2-vision integration
- OpenAIProvider: GPT-4o-mini integration
- AnthropicProvider: Claude 3.5 Haiku integration
- FallbackProvider: Zero-latency breadcrumb fallback

Author: Claude 4.5 Opus (Architect)
Date: 2025-12-29
"""
from __future__ import annotations

from .vision_providers import (
    AnthropicProvider,
    FallbackProvider,
    OllamaProvider,
    OpenAIProvider,
    VisionProvider,
    VisionProviderConnectionError,
    VisionProviderError,
    VisionProviderTimeoutError,
    create_vision_provider,
)

__all__ = [
    # Base classes
    "VisionProvider",
    # Providers
    "OllamaProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "FallbackProvider",
    # Exceptions
    "VisionProviderError",
    "VisionProviderTimeoutError",
    "VisionProviderConnectionError",
    # Factory
    "create_vision_provider",
]
