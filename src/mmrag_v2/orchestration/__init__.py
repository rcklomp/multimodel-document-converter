"""
Orchestration module for MM-Converter-V2.
Contains SmartConfigProvider, StrategyOrchestrator, DocumentDiagnosticEngine,
and Strategy Profiles for polymorphic document processing.

V3.0.0: Shadow extraction REMOVED per ARCHITECTURE.md (NEVER "shadow").
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
    # Document diagnostic
    "DocumentDiagnosticEngine",
    "DiagnosticReport",
    "PhysicalCheckResult",
    "ConfidenceProfile",
    "PageDiagnostic",
    "DocumentModality",
    "DocumentEra",
    "ContentDomain",
    "create_diagnostic_engine",
    # Strategy profiles
    "BaseProfile",
    "ProfileType",
    "ProfileParameters",
    "VLMPromptConfig",
    "VLMFreedom",
    "DigitalMagazineProfile",
    "ScannedDegradedProfile",
    "ProfileManager",
]
