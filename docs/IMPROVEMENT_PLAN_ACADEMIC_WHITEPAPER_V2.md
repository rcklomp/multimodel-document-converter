# Improvement Plan: Academic Whitepaper Processing V2

**Date**: January 9, 2026  
**Status**: Planning Phase  
**Context**: Post-table-fix quality improvements

## Current State

✅ **RESOLVED**: QA-CHECK-05 violations - Tables now have proper asset_ref  
✅ **SUCCESS**: 131 chunks extracted from AIOS whitepaper (previously 0)  
✅ **SUCCESS**: Proper modality separation (text/image/table)

## Identified Quality Issues

### Issue 1: Text Fragmentatie (OCR Line-Breaking)

**Problem**:
```json
"content": "...using AIOS can achieve up to 2 . 1 × faster execution..."
```

Should be: `2.1×` (not `2 . 1 ×`)

**Root Cause**:
- OCR engines (EasyOCR/Doctr) extracting line-by-line
- Not joining fragmented technical values (numbers with units)
- Space normalization too aggressive

**Impact**: 
- RAG retrieval quality degradation
- LLM misinterpretation of numerical data
- Search fragmentation ("2.1" != "2" + "1")

**Solution Strategy**:
Post-process OCR output with **Safe-Join** technical-value pattern recognition:

```python
def safe_join_technical_values(text: str) -> str:
    """Join fragmented technical values without over-correcting."""
    # Decimals: "2 . 1" → "2.1"
    text = re.sub(r'(\d+)\s+\.\s+(\d+)', r'\1.\2', text)
    
    # Multiplication: "2 . 1 ×" → "2.1×"
    text = re.sub(r'(\d+\.?\d*)\s+×', r'\1×', text)
    
    # Percentages: "10 %" → "10%"
    text = re.sub(r'(\d+\.?\d*)\s+%', r'\1%', text)
    
    # Units (GHz, MHz, KB, MB, etc.)
    text = re.sub(r'(\d+\.?\d*)\s+(GHz|MHz|KB|MB|GB|TB|ms|μs)', r'\1\2', text)
    
    # Mathematical symbols: "± 2" → "±2"
    text = re.sub(r'([±≈])\s+(\d)', r'\1\2', text)
    
    return text
```

**Critical Note**: Do NOT join all spaces between numbers - preserve intentional spacing in mathematical sequences (e.g., "1 2 3 4" should stay as is).

---

### Issue 2: Hierarchie Paradox (Missing Section Structure)

**Problem**:
```json
"breadcrumb_path": ["AIOS LLM Agent Operating System", "Page 3"]
"parent_heading": null
```

**Expected** (for academic papers):
```json
"breadcrumb_path": ["AIOS", "3. AIOS KERNEL", "3.2 LLM Core(s)"]
"parent_heading": "3. AIOS KERNEL"
```

**Root Cause**:
- `ContextState` not capturing section headings
- Layout-aware processor not detecting heading hierarchy
- No semantic heading detection (ABSTRACT, INTRODUCTION, etc.)

**Impact**:
- Lost document structure in RAG
- Poor semantic search (can't filter by section)
- No contextual chunk grouping

**Solution Strategy**:
1. **Heading Detection Layer** (USE DOCLING METADATA FIRST!):
   ```python
   def detect_heading(element, text: str) -> Tuple[bool, int]:
       """Detect if text is a heading using Docling metadata + patterns."""
       
       # PRIORITY 1: Use Docling's label metadata
       if hasattr(element, 'label'):
           label = str(element.label).lower()
           if 'heading' in label or 'title' in label:
               # Docling often provides heading level
               level = extract_heading_level(label)
               return True, level
       
       # PRIORITY 2: Font size heuristics (from Docling bbox)
       # Larger font = heading (compare to page average)
       if is_large_font(element):
           return True, infer_level_from_font_size(element)
       
       # PRIORITY 3: Pattern matching (fallback)
       # - All caps + short (< 60 chars) = likely heading
       # - Numbered section (e.g., "1. INTRODUCTION")
       if len(text) < 60 and text.isupper():
           return True, infer_level_from_pattern(text)
       
       return False, 0
   ```

2. **Academic Pattern Recognition** (as fallback):
   ```python
   SECTION_PATTERNS = [
       r"^\s*\d+\.?\s+[A-Z][A-Z\s]+$",  # "1. INTRODUCTION"
       r"^\s*\d+\.\d+\.?\s+[A-Z]",      # "1.1 Background"
       r"^(ABSTRACT|INTRODUCTION|CONCLUSION|REFERENCES|ACKNOWLEDGMENTS)$"
   ]
   ```

3. **ContextState Heading Stack**:
   ```python
   # In ContextState:
   self.heading_stack: List[Tuple[str, int]] = []  # (heading_text, level)
   
   def update_heading(self, text: str, level: int):
       """Update heading stack, popping higher levels."""
       # Remove all headings at same or deeper level
       while self.heading_stack and self.heading_stack[-1][1] >= level:
           self.heading_stack.pop()
       # Add new heading
       self.heading_stack.append((text, level))
   
   def get_breadcrumb_path(self) -> List[str]:
       """Build breadcrumb from document title + heading stack."""
       breadcrumb = [self.doc_title]
       breadcrumb.extend([h[0] for h in self.heading_stack])
       return breadcrumb
   ```

**Critical Insight**: Docling already provides `Label.HEADING` or font metadata - USE IT! Don't rely solely on regex patterns.

---

### Issue 3: Empty Content Artifacts

**Problem** (Page 7):
```json
{"content": "", "bbox": [29, 13, 40, 20], ...}
{"content": "", "bbox": [31, 15, 43, 23], ...}
{"content": "", "bbox": [14, 13, 42, 21], ...}
```

Multiple chunks with empty strings cluttering the JSONL.

**Root Cause**:
- Layout artifacts (page numbers, decorations, margins)
- Bbox threshold (30x30px minimum) catching layout elements
- No content validation before chunk creation

**Impact**:
- Vector index pollution
- Wasted storage and embedding costs
- Noise in RAG retrieval

**Solution Strategy** (ASSET-AWARE FILTERING):
```python
def should_skip_chunk(content: str, bbox: List[int], asset_ref: Optional[dict]) -> bool:
    """Filter out artifacts before finalizing chunk."""
    
    # CRITICAL: Keep chunks with assets even if content is empty!
    # Images without captions still have visual information in asset_ref
    if asset_ref is not None:
        return False  # NEVER skip chunks with assets
    
    # 1. Empty or whitespace-only (and no asset)
    if not content or not content.strip():
        return True
    
    # 2. Too short (< 3 chars - likely artifacts)
    if len(content.strip()) < 3:
        return True
    
    # 3. Only special characters (decorations)
    if re.match(r'^[\s\-_=•]+$', content):
        return True
    
    # 4. Suspicious bbox (very small area)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height
    if area < 100:  # Normalized units (0.01% of page)
        return True
    
    return False
```

**Critical Rule**: A chunk with `asset_ref` (image/table) must NEVER be filtered, even with empty `content`. The asset itself contains the information.

---

## Implementation Priority

### Phase 1: Quick Wins (Immediate)
1. ✅ Asset-aware empty chunk filtering (batch_processor before JSONL write)
2. ✅ Safe-join number fragmentation (post-process OCR text)
3. ✅ Look-ahead buffer for symmetric overlap (fill next_text_snippet)

### Phase 2: Structural Intelligence (Next)
4. ⏳ Docling-metadata-first heading detection
5. ⏳ ContextState heading stack implementation  
6. ⏳ Breadcrumb auto-population from heading stack

### Phase 3: Polish (Final)
7. ⏳ Academic-specific pattern library
8. ⏳ Citation/reference detection
9. ⏳ Figure/table caption linking

---

## Issue 4: Symmetric Overlap Gap (DISCOVERED)

**Problem**:
Many chunks have `next_text_snippet: null` in the current output.

**Root Cause**:
Chunks are written to JSONL immediately after creation - no "look-ahead" to grab next chunk's text.

**Impact**:
- Asymmetric context (only prev_text, not next_text)
- Suboptimal RAG retrieval (missing forward context)
- REQ-MM-03 violation (semantic context should be bidirectional)

**Solution - Look-Ahead Buffer**:
```python
class ChunkBuffer:
    """1-chunk buffer to enable symmetric overlap."""
    def __init__(self):
        self.pending: Optional[IngestionChunk] = None
    
    def add_chunk(self, chunk: IngestionChunk) -> Optional[IngestionChunk]:
        """Add chunk to buffer, return previous chunk with next_text filled."""
        if self.pending is None:
            # First chunk - hold it
            self.pending = chunk
            return None
        
        # Fill previous chunk's next_text_snippet from current chunk
        if self.pending.semantic_context is None:
            self.pending.semantic_context = SemanticContext()
        
        self.pending.semantic_context.next_text_snippet = chunk.content[:300]
        
        # Swap: return old chunk, hold new chunk
        to_write = self.pending
        self.pending = chunk
        return to_write
    
    def flush(self) -> Optional[IngestionChunk]:
        """Flush remaining chunk (no next_text available)."""
        result = self.pending
        self.pending = None
        return result
```

**Integration Point**: In `batch_processor.py`, before writing to JSONL:
```python
buffer = ChunkBuffer()
for chunk in all_chunks:
    ready_chunk = buffer.add_chunk(chunk)
    if ready_chunk:
        write_to_jsonl(ready_chunk)

# Flush final chunk
final_chunk = buffer.flush()
if final_chunk:
    write_to_jsonl(final_chunk)
```

---

## Testing Strategy

### Test Documents
- ✅ AIOS whitepaper (current test case)
- ⏳ arXiv papers with clear section structure
- ⏳ Academic conference papers (ACL, NeurIPS format)

### Success Criteria
1. **Text Quality**: No fragmented numbers in technical content
2. **Hierarchy**: ≥80% of chunks have meaningful breadcrumb_path (not just "Page X")
3. **Cleanliness**: Zero empty chunks in final JSONL
4. **Consistency**: Section headings correctly propagate to child chunks

---

## Code Locations

### Files to Modify
1. **`src/mmrag_v2/batch_processor.py`**:
   - Add `_should_skip_chunk()` validation before JSONL write
   - Add `_post_process_ocr_text()` for number joining

2. **`src/mmrag_v2/ocr/layout_aware_processor.py`**:
   - Add `_detect_heading()` method
   - Integrate with ContextState for heading updates
   - Pass heading info to ProcessedChunk

3. **`src/mmrag_v2/state/context_state.py`**:
   - Add `heading_stack: List[Tuple[str, int]]` field
   - Add `update_heading(text: str, level: int)` method
   - Add `get_breadcrumb_from_headings()` → List[str]

4. **`src/mmrag_v2/schema/ingestion_schema.py`**:
   - No changes needed (schema supports hierarchy already)

---

## Next Steps

**Immediate Action**:
```bash
# Test current output quality
grep '"content": ""' output/AIOS_Test/ingestion.jsonl | wc -l

# Test number fragmentation
grep -E '\d\s+\.\s+\d' output/AIOS_Test/ingestion.jsonl
```

**Implementation Order**:
1. Start with empty chunk filtering (easiest win)
2. Add OCR text post-processing (number joining)
3. Implement heading detection pipeline
4. Test with AIOS whitepaper
5. Validate breadcrumb quality improvement

---

## Notes

- The table asset fix proved the validation system works
- These improvements don't require schema changes
- Focus on data quality, not architecture changes
- Academic whitepapers have predictable structure - leverage it!
