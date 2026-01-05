"""
State module for MM-Converter-V2.
Contains ContextStateV2 for hierarchical breadcrumb tracking.
"""
from .context_state import (
    ContextStateV2,
    create_context_state,
    is_valid_heading,
)

__all__ = [
    "ContextStateV2",
    "create_context_state",
    "is_valid_heading",
]
