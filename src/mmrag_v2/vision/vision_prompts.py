"""
Vision Prompts - VLM Prompt Engineering for Retrieval Optimization
===================================================================
ENGINE_USE: Signal-to-Noise Optimization for RAG Systems

This module contains optimized VLM prompts that enforce the "Visual Only" rule
to prevent textual meta-language from polluting the search index.

MASTER PROMPT PRINCIPLE: VISUAL ONLY
=====================================
The VLM must NEVER summarize or transcribe text that is already captured by OCR.
Instead, it must focus EXCLUSIVELY on visual descriptors that add unique value:
- Object identification (what IS in the image)
- Visual composition (layout, arrangement, spatial relationships)
- Visual attributes (colors, shapes, patterns, styles)
- Technical diagrams (component labels, exploded views, schematics)

ANTI-PATTERN (Banned):
- "This image shows a page about..."
- "The text provides an introduction to..."
- "The document discusses..."
- Any meta-commentary about the page/document itself

CORRECT PATTERN:
- "Exploded diagram of bolt action rifle mechanism with 12 labeled components"
- "Technical schematic showing trigger assembly spring positions"
- "Vintage advertisement featuring hunting rifles arranged in grid layout"

REQ Compliance:
- REQ-VLM-SIGNAL: VLM descriptions must add unique visual information
- REQ-VLM-NOISE: No textual meta-language in descriptions
- REQ-RETRIEVAL-01: OCR text is primary, VLM is supplementary

Author: Senior Document Intelligence Engineer
Date: 2026-01-05
"""

from typing import Optional

# ============================================================================
# VISUAL-ONLY PROMPT (Strict Mode)
# ============================================================================

VISUAL_ONLY_PROMPT = """VISUAL CONTENT ANALYSIS - STRICT MODE

You are a VISUAL CATALOGER. Your ONLY job is to describe what you SEE in the image pixels.

CRITICAL RULES:
1. Describe ONLY visual elements: objects, colors, shapes, layouts, diagrams
2. NEVER use meta-language like "This image shows..." or "The page contains..."
3. NEVER transcribe or summarize text - OCR already captured it
4. NEVER mention "document", "page", "text", "article", or "book"
5. Start DIRECTLY with the subject: "Exploded diagram...", "Technical schematic...", "Vintage advertisement..."

ALLOWED DESCRIPTORS:
✓ "Exploded view diagram of bolt action assembly"
✓ "Technical schematic showing spring positions"
✓ "Vintage advertisement layout with product grid"
✓ "Detailed technical drawing with numbered components"
✓ "Black and white photograph of workshop machinery"

BANNED PHRASES (NEVER USE):
✗ "This image shows..."
✗ "The page contains..."
✗ "The document discusses..."
✗ "Text visible in the image..."
✗ "A photograph of a page from a book..."

OUTPUT FORMAT:
Provide a single sentence (max 60 words) describing the VISUAL CONTENT ONLY.
{diagnostic_hints}

Analyze now - respond with VISUAL DESCRIPTION ONLY."""

# ============================================================================
# CONTEXTUAL VISUAL-ONLY PROMPT (With Document Context)
# ============================================================================

CONTEXTUAL_VISUAL_PROMPT = """VISUAL CONTENT ANALYSIS - CONTEXTUAL MODE

You are a VISUAL CATALOGER working within a document processing pipeline.

DOCUMENT CONTEXT:
{context_section}

YOUR TASK:
Describe the VISUAL CONTENT of this image in relation to the document context above.

CRITICAL RULES:
1. Focus on VISUAL elements: diagrams, photographs, illustrations, layouts
2. Connect visual content to document context (e.g., "Diagram illustrating the bolt mechanism mentioned in surrounding text")
3. NEVER use generic meta-language like "This image shows a page..."
4. NEVER transcribe text - OCR already captured it
5. Start with the visual subject: "Diagram...", "Photograph...", "Schematic...", "Advertisement..."

ALLOWED PATTERNS:
✓ "Exploded diagram of Mauser 98 bolt assembly referenced in text"
✓ "Technical photograph showing disassembly steps for trigger group"
✓ "Vintage advertisement featuring Browning rifles mentioned in catalog section"

BANNED PATTERNS:
✗ "This image is a photograph of a text document..."
✗ "The page shows text discussing..."
✗ "An image showing a page with instructions..."

OUTPUT FORMAT:
Single sentence (max 60 words) - VISUAL CONTENT with contextual relevance.
{diagnostic_hints}

Analyze now."""

# ============================================================================
# DIAGRAM-SPECIFIC PROMPT (For Technical Illustrations)
# ============================================================================

DIAGRAM_PROMPT = """TECHNICAL DIAGRAM ANALYSIS

You are analyzing a TECHNICAL DIAGRAM or SCHEMATIC.

TASK: Identify the subject and visual structure.

OUTPUT STRUCTURE:
"[Type] of [Subject] showing [Key Visual Features]"

EXAMPLES:
✓ "Exploded view diagram of bolt action mechanism with 15 labeled components"
✓ "Cross-sectional schematic of trigger assembly showing spring tensions"
✓ "Assembly diagram for rifle stock with numbered installation sequence"
✓ "Technical illustration of magazine feeding system in cutaway view"

RULES:
- Start with diagram type: exploded view, cross-section, schematic, assembly diagram
- Identify the mechanical subject clearly
- Mention visual features: labeled parts, numbered sequence, cutaway, arrows
- NO meta-language about "the image" or "the page"

{diagnostic_hints}

Analyze this diagram now."""

# ============================================================================
# PHOTOGRAPH-SPECIFIC PROMPT (For Editorial Photos)
# ============================================================================

PHOTOGRAPH_PROMPT = """PHOTOGRAPH ANALYSIS

You are analyzing an EDITORIAL PHOTOGRAPH (not a diagram or text page).

TASK: Describe what is photographed and visual composition.

OUTPUT STRUCTURE:
"[Subject] in [Setting/Context] - [Visual Details]"

EXAMPLES:
✓ "Workshop craftsman operating drill press with vintage machinery visible"
✓ "Close-up of rifle barrel threading with measurement tools"
✓ "Historical firearms collection arranged on display table"
✓ "Author portrait in gun workshop surrounded by tools and equipment"

RULES:
- Identify the photographed subject immediately
- Describe setting/context if relevant
- Mention visual composition: close-up, arrangement, lighting, era
- NO text transcription or document meta-language

{diagnostic_hints}

Analyze this photograph now."""

# ============================================================================
# SCANNED DOCUMENT HINTS (Template)
# ============================================================================

SCAN_ARTIFACT_HINTS = """
IMPORTANT - SCANNED DOCUMENT CONTEXT:
- This image comes from a SCANNED document
- IGNORE: paper texture, grain, stains, foxing, discoloration
- IGNORE: scan artifacts, dust specks, fold marks
- FOCUS ON: the actual printed/drawn content only
- Do NOT describe paper quality or scan issues as content
"""

# ============================================================================
# OCR-CONFLICT PREVENTION PROMPT
# ============================================================================

OCR_CONFLICT_PREVENTION = """
OCR CONFLICT PREVENTION:
The surrounding text has ALREADY been captured by OCR with high confidence.
DO NOT describe text content - describe ONLY non-textual visual elements:
- Diagrams, illustrations, photographs
- Visual layouts, compositions, arrangements
- Graphical elements that OCR cannot capture

If this image contains ONLY text (no diagrams/photos), respond with:
"Text-only page - visual content already captured by OCR"
"""


# ============================================================================
# PROMPT BUILDER FUNCTION
# ============================================================================


def build_visual_prompt(
    context_section: Optional[str] = None,
    diagnostic_hints: Optional[str] = None,
    is_scan: bool = False,
    is_diagram: bool = False,
    is_photograph: bool = False,
    ocr_confidence: Optional[float] = None,
) -> str:
    """
    Build optimized VLM prompt based on content type and context.

    Args:
        context_section: Document context (breadcrumbs, surrounding text)
        diagnostic_hints: Additional diagnostic hints from document analysis
        is_scan: Whether this is a scanned document (adds artifact hints)
        is_diagram: Whether this appears to be a technical diagram
        is_photograph: Whether this appears to be a photograph
        ocr_confidence: OCR confidence score for surrounding text (0.0-1.0)

    Returns:
        Formatted prompt string optimized for visual-only descriptions
    """
    # Build diagnostic hints section
    hints_parts = []

    if diagnostic_hints:
        hints_parts.append(diagnostic_hints)

    if is_scan:
        hints_parts.append(SCAN_ARTIFACT_HINTS)

    # Add OCR conflict prevention if high-confidence OCR exists
    if ocr_confidence and ocr_confidence > 0.8:
        hints_parts.append(OCR_CONFLICT_PREVENTION)

    hints_str = "\n".join(hints_parts) if hints_parts else ""

    # Select appropriate prompt template
    if is_diagram:
        base_prompt = DIAGRAM_PROMPT
    elif is_photograph:
        base_prompt = PHOTOGRAPH_PROMPT
    elif context_section:
        base_prompt = CONTEXTUAL_VISUAL_PROMPT
    else:
        base_prompt = VISUAL_ONLY_PROMPT

    # Format prompt with context and hints
    if context_section and "{context_section}" in base_prompt:
        prompt = base_prompt.format(context_section=context_section, diagnostic_hints=hints_str)
    else:
        prompt = base_prompt.format(diagnostic_hints=hints_str)

    return prompt


# ============================================================================
# LEGACY RESPONSE CLEANER
# ============================================================================


def clean_vlm_response(raw_response: str) -> str:
    """
    Clean VLM response to remove any remaining meta-language.

    This is a safety net in case the VLM still produces banned phrases.

    Args:
        raw_response: Raw VLM output

    Returns:
        Cleaned response with meta-language removed
    """
    import re

    # Remove common meta-language patterns
    response = raw_response.strip()

    # Remove leading meta-phrases
    meta_patterns = [
        r"^This image shows?\s+",
        r"^The image (?:appears to be|is|depicts|displays|shows)\s+",
        r"^The page contains?\s+",
        r"^The document (?:shows|discusses|presents)\s+",
        r"^A photograph of\s+",
        r"^An? illustration of\s+",
        r"^(?:This|The) (?:is|appears to be)\s+",
    ]

    for pattern in meta_patterns:
        response = re.sub(pattern, "", response, flags=re.IGNORECASE)

    # Remove trailing document references
    response = re.sub(
        r"\s+(?:in|on|from) (?:the|this) (?:page|document|image|photograph)\.?$",
        "",
        response,
        flags=re.IGNORECASE,
    )

    # Capitalize first letter after cleaning
    if response:
        response = response[0].upper() + response[1:]

    return response.strip()
