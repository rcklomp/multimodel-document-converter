"""
Orchestration module for MM-Converter-V2.
Contains SmartConfigProvider, StrategyOrchestrator, ShadowExtractor, DocumentDiagnosticEngine,
and Strategy Profiles for polymorphic document processing.

GEMINI AUDIT: Added DocumentDiagnosticEngine for pre-processing intelligence.
AUTO-PILOT: Added Strategy Profiles for polymorphic document-type separation.
"""

from .smart_config import (
    DocumentProfile,
    DocumentType,
    SmartConfigProvider,
)
from .strategy_orchestrator import (
    ExtractionStrategy,
    StrategyOrchestrator,
)
from .shadow_extractor import (
    ShadowAsset,
    ShadowScanResult,
    ShadowExtractor,
    create_shadow_extractor,
)
from .document_diagnostic import (
    DocumentDiagnosticEngine,
    DiagnosticReport,
    PhysicalCheckResult,
    ConfidenceProfile,
    PageDiagnostic,
    DocumentModality,
    DocumentEra,
    ContentDomain,
    create_diagnostic_engine,
)
from .strategy_profiles import (
    BaseProfile,
    ProfileType,
    ProfileParameters,
    VLMPromptConfig,
    VLMFreedom,
    DigitalMagazineProfile,
    ScannedDegradedProfile,
    ProfileManager,
)

__all__ = [
    # Smart config
    "DocumentProfile",
    "DocumentType",
    "SmartConfigProvider",
    # Strategy orchestrator
    "ExtractionStrategy",
    "StrategyOrchestrator",
    # Shadow extractor
    "ShadowAsset",
    "ShadowScanResult",
    "ShadowExtractor",
    "create_shadow_extractor",
    # Document diagnostic (GEMINI AUDIT FIX)
    "DocumentDiagnosticEngine",
    "DiagnosticReport",
    "PhysicalCheckResult",
    "ConfidenceProfile",
    "PageDiagnostic",
    "DocumentModality",
    "DocumentEra",
    "ContentDomain",
    "create_diagnostic_engine",
    # Strategy profiles (AUTO-PILOT)
    "BaseProfile",
    "ProfileType",
    "ProfileParameters",
    "VLMPromptConfig",
    "VLMFreedom",
    "DigitalMagazineProfile",
    "ScannedDegradedProfile",
    "ProfileManager",
]
