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

VISUAL_ONLY_PROMPT = """VISUAL CONTENT ANALYSIS - VISUAL-ONLY MODE

You are a VISUAL CATALOGER. Your ONLY job is to describe what you SEE in the image pixels.

CRITICAL RULES:
1. Describe ONLY visual elements: objects, colors, shapes, layouts, diagrams
2. NEVER use meta-language like "This image shows..." or "The page contains..."
3. NEVER transcribe readable text - OCR already captured it
4. NEVER quote or paraphrase readable text
5. You MAY describe typographic/layout structure at a high level (e.g., "two-column layout", "numbered list", "dense typographic layout") without quoting any words
6. Start DIRECTLY with the subject: "Exploded diagram...", "Technical schematic...", "Vintage advertisement..."

ALLOWED DESCRIPTORS:
✓ "Exploded view diagram of bolt action assembly"
✓ "Technical schematic showing spring positions"
✓ "Vintage advertisement layout with product grid"
✓ "Detailed technical drawing with numbered components"
✓ "Black and white photograph of workshop machinery"
✓ "Dense typographic layout with two columns and a numbered list"

BANNED PHRASES (NEVER USE):
✗ "This image shows..."
✗ "The page contains..."
✗ "The document discusses..."
✗ "The text says/reads..."
✗ Quoting any readable words (no quotes, no copied headings)
✗ "A photograph of a page from a book/document..."

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

If this image contains ONLY typographic content (no diagrams/photos), respond with:
"Dense typographic layout; no distinct non-text visuals."
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

    # Select appropriate prompt template.
    #
    # IMPORTANT (AGENTS.md / Source Sanctity):
    # We do not use CONTEXTUAL_VISUAL_PROMPT by default because it encourages
    # meta-language ("surrounding text", "document section") and speculative
    # claims. Context, if provided, must be used only for disambiguation and
    # must NEVER be referenced in the output.
    if is_diagram:
        base_prompt = DIAGRAM_PROMPT
    elif is_photograph:
        base_prompt = PHOTOGRAPH_PROMPT
    else:
        base_prompt = VISUAL_ONLY_PROMPT

    # Format prompt with context and hints
    if context_section and "{context_section}" in base_prompt:
        prompt = base_prompt.format(context_section=context_section, diagnostic_hints=hints_str)
    else:
        prompt = base_prompt.format(diagnostic_hints=hints_str)

    return prompt


# ============================================================================
# LEGACY ALIAS (for backward compatibility with tests)
# ============================================================================

# STRICTER_VISUAL_PROMPT is the old name for VISUAL_ONLY_PROMPT
# Keep as alias for existing tests
STRICTER_VISUAL_PROMPT = VISUAL_ONLY_PROMPT


# ============================================================================
# VLM TEXT DETECTION VALIDATOR (REQ-VLM-SIGNAL)
# ============================================================================


def detect_text_reading(response: str) -> bool:
    """
    Detect if VLM response contains text transcription instead of visual description.

    Returns True if text reading is detected (response should be rejected).
    Returns False if response is a valid visual description (acceptable).

    Args:
        response: VLM response text

    Returns:
        True if text transcription detected, False otherwise
    """
    import re

    if not response:
        return False

    response_lower = response.lower()

    # We allow high-level typographic/layout descriptions (e.g., "two-column layout",
    # "numbered list", "dense typographic layout"), but we reject anything that
    # looks like quoting/reading/paraphrasing specific words.

    # Pattern 0: Meta-language that violates visual-only constraints.
    # Keep this phrase-based (not word-based) to avoid false positives such as
    # "PDF document layout" or "text-only page".
    meta_terms = [
        # Context leakage
        "surrounding text",
        "the surrounding text",
        "document section",
        "in the text",
        "mentioned in the text",
        "as described in the text",
        # Explicit document/page meta (banned by VISUAL_ONLY_PROMPT)
        "this document",
        "the document",
        "this page",
        "the page",
        "on page",
        "a page from",
        "pages from",
        "the document discusses",
        "this document discusses",
        "the page discusses",
        "this page discusses",
        "the page contains",
        "this page contains",
    ]
    if any(t in response_lower for t in meta_terms):
        return True

    # Pattern 1: "The text says/reads..." patterns
    if re.search(
        r"\b(text|caption|label|title|heading)\s+(?:that\s+)?(says?|reads?|indicates?|states?)\b",
        response_lower,
    ):
        return True

    # Pattern 1b: "text that reads ..." / "label that reads ..."
    if re.search(r"\b(text|caption|label|title|heading)\s+that\s+reads?\b", response_lower):
        return True

    # Pattern 2: "written on" patterns
    if "written on" in response_lower:
        return True

    # Pattern 3: "text visible" patterns
    if re.search(r"\btext\s+visible\b", response_lower):
        return True

    # Pattern 4: Excessive quotes (more than 2 pairs = transcription)
    quote_count = response.count('"') + response.count("'")
    if quote_count > 4:  # More than 2 pairs
        return True

    # Pattern 5: Long quoted strings (>20 chars) indicate transcription
    quoted_strings = re.findall(r'["\'](.{20,})["\']', response)
    if quoted_strings:
        return True

    # Pattern 5b: Looks like copied headings (long ALLCAPS line) - often OCR/VLM reading.
    # Keep small abbreviations allowed (handled below).
    if re.search(r"^[A-Z0-9 ,./\\-]{25,}$", response.strip()):
        return True

    # Pattern 6: Multiple ALL CAPS words (>2) may indicate transcription
    # But allow common technical abbreviations
    caps_words = re.findall(r"\b[A-Z]{2,}\b", response)
    # Filter out common abbreviations
    technical_abbrevs = {"PDF", "OCR", "RGB", "DPI", "USA", "UK", "EU", "NATO", "USAF", "RAF"}
    actual_caps = [w for w in caps_words if w not in technical_abbrevs]
    if len(actual_caps) > 2:
        return True

    return False


def validate_vlm_response(response: str) -> "VLMValidationResult":
    """
    Validate VLM response for quality and rule compliance.

    Args:
        response: Raw VLM response

    Returns:
        VLMValidationResult with validation status and cleaned response
    """
    from dataclasses import dataclass
    from typing import List

    @dataclass
    class VLMValidationResult:
        is_valid: bool
        text_reading_detected: bool
        cleaned_response: str
        issues: List[str]

    issues = []

    # Check 1: Empty response
    if not response or not response.strip():
        return VLMValidationResult(
            is_valid=False,
            text_reading_detected=False,
            cleaned_response="",
            issues=["Empty response"],
        )

    # Check 2: Clean first (so we can strip common meta-language before validation)
    cleaned = clean_vlm_response(response)
    if not cleaned or not cleaned.strip():
        return VLMValidationResult(
            is_valid=False,
            text_reading_detected=False,
            cleaned_response="",
            issues=["Empty after cleaning"],
        )

    # Check 3: Generic fallback responses (after cleaning)
    fallback_phrases = [
        "unable to describe",
        "cannot describe",
        "no description available",
        "description unavailable",
    ]
    response_lower = cleaned.lower()
    if any(phrase in response_lower for phrase in fallback_phrases):
        issues.append("Generic fallback response detected")
        return VLMValidationResult(
            is_valid=False, text_reading_detected=False, cleaned_response="", issues=issues
        )

    # Check 4: Text transcription / context leakage detection (on cleaned text)
    text_reading = detect_text_reading(cleaned)
    if text_reading:
        issues.append("Text transcription detected")
        return VLMValidationResult(
            is_valid=False, text_reading_detected=True, cleaned_response="", issues=issues
        )

    # Check 5: Response too short after cleaning
    if len(cleaned) < 10:
        issues.append("Response too short after cleaning")
        return VLMValidationResult(
            is_valid=False, text_reading_detected=False, cleaned_response=cleaned, issues=issues
        )

    # All checks passed
    return VLMValidationResult(
        is_valid=True, text_reading_detected=False, cleaned_response=cleaned, issues=[]
    )


# ============================================================================
# LEGACY RESPONSE CLEANER
# ============================================================================


def clean_vlm_response(raw_response: str) -> str:
    """
    Clean VLM response to remove any remaining meta-language and VLM internal monologue.

    This is a safety net in case the VLM still produces banned phrases or internal
    thinking blocks (like <think>...</think> from DeepSeek-VL2, Qwen, etc.).
    VLM descriptions should NEVER start with "An image of...", "Illustration of...",
    "A photo of...", etc. They should describe the content directly.

    Args:
        raw_response: Raw VLM output

    Returns:
        Cleaned response with meta-language removed, max 400 chars (SRS-compliant)
    """
    import re

    if not raw_response:
        return ""

    # =========================================================================
    # PHASE 1: Remove VLM internal monologue (<think>...</think> blocks)
    # =========================================================================
    # Verwijder de interne monoloog (<think>...</think>)
    response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)
    # Verwijder een eventueel afgebroken <think> blok
    response = re.sub(r"<think>.*", "", response, flags=re.DOTALL)

    # Strip witruimte en haal onnodige aanhalingstekens weg
    response = response.strip().replace('"', "").replace("'", "")

    # =========================================================================
    # PHASE 2: Remove common meta-language patterns
    # =========================================================================

    # Remove leading meta-phrases (ordered from most specific to least specific)
    # These patterns match WITH or WITHOUT leading article (A/An/The)
    meta_patterns = [
        # "This image shows...", "The image depicts..."
        r"^(?:This|The) image (?:shows?|appears to be|is|depicts|displays)\s+",
        # "The page contains...", "This page shows..."
        r"^(?:This|The) page (?:contains?|shows?|displays?)\s+",
        # "The document shows...", "This document discusses..."
        r"^(?:This|The) document (?:shows?|discusses?|presents?)\s+",
        # "An image of...", "Image of...", "The image of..."
        r"^(?:An? |The )?[Ii]mage of\s+",
        # "A photograph of...", "Photograph of...", "The photograph of..."
        r"^(?:An? |The )?[Pp]hotograph of\s+",
        # "A photo of...", "Photo of...", "The photo of..."
        r"^(?:An? |The )?[Pp]hoto of\s+",
        # "An illustration of...", "Illustration of...", "The illustration of..."
        r"^(?:An? |The )?[Ii]llustration of\s+",
        # "A picture of...", "Picture of...", "The picture of..."
        r"^(?:An? |The )?[Pp]icture of\s+",
        # "A drawing of...", "Drawing of...", "The drawing of..."
        r"^(?:An? |The )?[Dd]rawing of\s+",
        # "A depiction of...", "Depiction of..."
        r"^(?:An? |The )?[Dd]epiction of\s+",
        # "A rendering of...", "Rendering of..."
        r"^(?:An? |The )?[Rr]endering of\s+",
        # "A graphic of...", "Graphic of..."
        r"^(?:An? |The )?[Gg]raphic of\s+",
        # "A figure showing...", "Figure showing..."
        r"^(?:An? |The )?[Ff]igure (?:showing|depicting|illustrating|of)\s+",
        # "A visual of...", "Visual of..."
        r"^(?:An? |The )?[Vv]isual (?:representation )?of\s+",
        # Generic "This is...", "This appears to be..."
        r"^(?:This|It) (?:is|appears to be)\s+",
        # "Here is...", "Here we see..."
        r"^Here (?:is|we see)\s+",
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

    # Remove trailing meta-commentary
    response = re.sub(
        r",?\s+(?:as shown|as seen|as depicted|as illustrated) (?:in|on) (?:the|this) (?:image|page|figure)\.?$",
        "",
        response,
        flags=re.IGNORECASE,
    )

    # Remove generic "source" clauses that add retrieval noise.
    # Keep this narrow to avoid deleting real content.
    response = re.sub(
        r",?\s+(?:from|in) (?:a|an|the) (?:technical\s+)?(?:manual|instructional guide|guide|book|document)\.?$",
        "",
        response,
        flags=re.IGNORECASE,
    )
    response = re.sub(
        r",?\s+appears to be from (?:a|an|the) (?:technical\s+)?(?:manual|instructional guide|guide|book|document)\.?$",
        "",
        response,
        flags=re.IGNORECASE,
    )

    # Capitalize first letter after cleaning
    if response:
        response = response[0].upper() + response[1:]

    # =========================================================================
    # PHASE 3: Apply hard character limit (SRS-compliant 400 chars)
    # =========================================================================
    return response.strip()[:400]
