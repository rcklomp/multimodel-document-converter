# MMRAG V2.0: Anti-Overfitting Architecture
## Multi-Dimensional Document Classification System

**Author:** Claude 4.5 Opus (System Architect)  
**Date:** 2026-01-10  
**Version:** 2.0 (Anti-Overfitting Release)

---

## 🎯 Executive Summary

This document describes the **V2.0 Anti-Overfitting Architecture** for MMRAG document processing. The system replaces hardcoded, document-specific rules with an intelligent, feature-based classification matrix that prevents "greedy optimization" while maintaining accuracy across diverse document types.

### The Problem: Greedy Optimization

**Before V2.0:**
```python
# ANTI-PATTERN: Document-specific overfitting
if document.name == "Harry Potter":
    use_literature_profile()
elif document.name == "Combat Aircraft":
    use_magazine_profile()
# ... more hardcoded rules
```

**The Trap:**
- Fix for Harry Potter → Breaks Combat Aircraft
- Fix for Combat Aircraft → Breaks AIOS whitepaper
- Endless cycle of document-specific patches

### The Solution: Multi-Dimensional Classification

**V2.0 Architecture:**
```python
# PATTERN: Generic feature-based classification
features = {
    "text_density": 1200,      # chars/page
    "image_density": 0.15,     # images/page
    "median_dim": 45,          # pixels (KEY discriminator!)
    "page_count": 150,
    "domain": "editorial"
}

profile = classifier.classify(features)  # → ScannedLiterature
```

**CRITICAL ARCHITECTURAL RULE: MODALITY IS PRIMARY SPLITSING**

```
                        ┌─────────────────────────────────────────┐
                        │           Document Analysis             │
                        └────────────────┬────────────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────────────┐
                        │   is_scan == True OR is_scan == False?  │
                        │         (HARD BINARY DECISION)          │
                        └────────────────┬────────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
              ▼                          │                          ▼
    ┌─────────────────┐                  │              ┌─────────────────┐
    │   DIGITAL       │                  │              │   SCANNED       │
    │   PROFILES      │                  │              │   PROFILES      │
    ├─────────────────┤                  │              ├─────────────────┤
    │ academic_white  │                  │              │ scanned_clean   │
    │ digital_magazine│                  │              │ scanned_degraded│
    │ technical_manual│◄─────────────────┼──────────────►technical_manual │
    └─────────────────┘                  │              │ scanned_lit     │
                                         │              │ scanned_magazine│
                                         │              └─────────────────┘
                                         │
                        ┌────────────────┴────────────────────────┐
                        │ technical_manual kan BEIDE zijn!        │
                        │ (legacy scans OR modern digital PDFs)   │
                        └─────────────────────────────────────────┘
```

**WHY THIS MATTERS:**
- Digital documents have **native text** → 100% perfect text extraction, no OCR noise
- Digital documents have **embedded assets** → Direct extraction, no DPI-based cutting
- Scanned documents need **OCR** → Introduces noise, needs different handling
- Scanned documents need **image processing** → Background extraction, DPI matters

**HARD REJECTION RULE:**
- If `is_scan=False` → ALL `scanned_*` profiles return `score=0.0, confidence=0.0`
- If `is_scan=True` → ALL `digital_*` profiles return `score=0.0, confidence=0.0`
- `technical_manual` is the ONLY exception (valid for both modalities)

**Key Insight: Visual Density Heuristic (SECONDARY to Modality)**

After the modality split, `median_dim` (median image dimensions) is the **secondary discriminator**:

| Document Type | Median Dim | Image Density | Text Density | Scan? |
|--------------|------------|---------------|--------------|-------|
| **Scanned Literature** (Harry Potter) | 25-100px | LOW (0.1-0.3) | LOW (500-1500) | ✅ |
| **Digital Magazine** (Combat Aircraft) | 200-800px | HIGH (0.5+) | LOW (500-1500) | ❌ |
| **Academic Whitepaper** (AIOS) | 50-200px | MODERATE (0.1-0.3) | HIGH (3000+) | ❌ |
| **Scanned Magazine** (hypothetical scan) | 200-800px | HIGH (0.5+) | LOW (500-1500) | ✅ |
| **Technical Manual** (Firearms.pdf) | 50-150px | MEDIUM (0.1-0.5) | MEDIUM (1000-2500) | ✅/❌ |

### V2.1 Enhancement: Two Critical New Profiles

**Problem Identified (2026-01-10):**
1. **Scanned magazines** (Combat Aircraft als scan) werden geclassificeerd als `scanned_clean` met te conservatieve parameters (50x50 min, 150 DPI)
2. **Technical manuals** (Firearms.pdf) verloren kleine maar kritieke details (pinnetjes, schroefjes < 50px) met standaard `scanned_clean`

**Solution: Two New Specialized Profiles**

| Profile | Min Dimensions | Sensitivity | DPI | Use Case |
|---------|---------------|-------------|-----|----------|
| `ScannedMagazineProfile` | **100x100px** | **0.7** | **200** | Scanned editorial content with large photos |
| `TechnicalManualProfile` | **30x30px** | **0.8** | **300** | Technical docs with small critical parts |

---

## 🏗️ Architecture Components

### 1. ProfileClassifier (New Component)

**Location:** `src/mmrag_v2/orchestration/profile_classifier.py`

**Responsibility:** Multi-dimensional feature-based profile selection using weighted scoring.

#### Feature Extraction

```python
@dataclass
class ClassificationFeatures:
    # Text features
    text_density: float           # chars/page
    has_extractable_text: bool
    
    # Image features  
    image_density: float          # images/page
    median_dim: int              # CRITICAL: median image size
    image_count: int
    
    # Document features
    page_count: int
    is_scan: bool
    scan_confidence: float
    
    # Domain features
    domain: str                  # academic, editorial, technical
    modality: str                # native_digital, scanned_clean, etc.
```

#### Weighted Scoring System

Each profile is scored on a 0.0-1.0 scale:

```python
def _score_academic_whitepaper(features) -> ProfileScore:
    score = 0.0
    
    # TEXT DENSITY: Primary signal (weight: 0.35)
    if features.text_density >= 3000:
        score += 0.35
    
    # DOMAIN: Strong signal (weight: 0.25)
    if features.domain in ("academic", "technical"):
        score += 0.25
    
    # IMAGE DENSITY: Should be LOW-MODERATE (weight: 0.20)
    if features.image_density <= 0.3:
        score += 0.20
    
    # MEDIAN DIM: Small-medium diagrams (weight: 0.15)
    if 50 <= features.median_dim <= 200:
        score += 0.15
    
    # SCAN CHECK: Digital source (weight: 0.05)
    if not features.is_scan:
        score += 0.05
    
    return ProfileScore(score=score, confidence=...)
```

#### Fallback Logic

```python
# Confidence check
if best_match.confidence < MIN_CONFIDENCE:
    logger.warning(f"Low confidence ({confidence:.2f}), falling back to safe default")
    return ProfileType.DIGITAL_MAGAZINE  # Safe fallback
```

---

### 2. ProfileManager (Refactored)

**Location:** `src/mmrag_v2/orchestration/strategy_profiles.py`

**Changes:**
- Added `doc_profile` parameter to `select_profile()`
- Uses `ProfileClassifier` when `doc_profile` is provided
- Falls back to legacy heuristic path for backward compatibility

```python
@classmethod
def select_profile(
    cls,
    diagnostic_report: Optional[DiagnosticReport] = None,
    force_profile: Optional[ProfileType] = None,
    doc_profile: Optional[Any] = None,  # NEW!
) -> BaseProfile:
    # Manual override takes precedence
    if force_profile is not None:
        return cls._profiles[force_profile]()
    
    # MULTI-DIMENSIONAL CLASSIFICATION PATH
    if doc_profile is not None:
        classifier = ProfileClassifier()
        selected_type = classifier.classify(doc_profile, diagnostic_report)
        return type_mapping[selected_type]()
    
    # FALLBACK: Legacy heuristic path (backward compatible)
    # ... existing logic for when doc_profile not provided
```

---

### 3. CLI Integration

**Location:** `src/mmrag_v2/cli.py`

**Changes:**
- Pass `smart_profile` to `ProfileManager.select_profile()`
- Display classifier reasoning in console output

```python
# Step 1: Analyze document profile (SmartConfigProvider)
analyzer = SmartConfigProvider()
smart_profile = analyzer.analyze(input_file, diagnostic_report=diagnostic_report)

# Step 2: Multi-dimensional classification
selected_profile = ProfileManager.select_profile(
    diagnostic_report=diagnostic_report,
    force_profile=None,
    doc_profile=smart_profile,  # NEW: Enable classifier
)
```

---

## 📊 The "Holy Trinity" Test Cases

The system is designed to maintain accuracy across three challenging documents:

### 1. Academic Whitepaper (AIOS)

**Characteristics:**
- HIGH text density (4500+ chars/page)
- SMALL-MEDIUM diagrams (50-200px)
- LOW-MODERATE image density (0.1-0.3)
- Academic domain
- Born-digital

**Expected Profile:** `AcademicWhitepaperProfile`

**Scoring:**
```
Academic Whitepaper: 0.950 (confidence: 1.00)
  + High text density (4500 >= 3000)
  + Academic domain
  + Appropriate image density (0.15)
  + Appropriate diagram size (120px)
  + Born-digital
```

---

### 2. Scanned Literature (Harry Potter)

**Characteristics:**
- LOW text density (800 chars/page, due to poor OCR)
- SMALL decorative illustrations (25-100px) **← KEY DISCRIMINATOR**
- LOW image density (0.15)
- 150+ pages
- Editorial domain
- Scanned document

**Expected Profile:** `ScannedLiteratureProfile`

**Scoring:**
```
Scanned Literature: 0.800 (confidence: 0.95)
  + Scanned document
  + Book-length (150 >= 50 pages)
  + Small decorative illustrations (45px < 100) ← CRITICAL
  + Editorial domain (fiction/literature)
  + Appropriate image density (0.15)
```

**Why Not Magazine?**
- Median dim (45px) is MUCH SMALLER than magazine photos (200+px)
- Low image density (0.15) vs magazine high density (0.5+)
- Large page count (150) vs magazine typical (20-50)

---

### 3. Digital Magazine (Combat Aircraft)

**Characteristics:**
- LOW-MEDIUM text density (1200 chars/page)
- LARGE editorial photos (300-600px) **← KEY DISCRIMINATOR**
- HIGH image density (0.8+ images/page)
- Editorial domain
- Born-digital

**Expected Profile:** `DigitalMagazineProfile`

**Scoring:**
```
Digital Magazine: 0.950 (confidence: 1.00)
  + High image density (0.85 >= 0.5)
  + Large editorial photos (450px >= 200) ← CRITICAL
  + Magazine-appropriate text density (1200)
  + Editorial domain
  + Born-digital
```

**Why Not Literature?**
- Median dim (450px) is MUCH LARGER than book illustrations (25-100px)
- High image density (0.85) vs book low density (0.1-0.3)
- Not a scan (is_scan=False)

---

### 4. Scanned Magazine (Combat Aircraft als Scan) **NEW in V2.1**

**The Problem:**
When Combat Aircraft is SCANNED, the classifier correctly detects `is_scan=True`. But the standard `scanned_clean` profile is too conservative:
- 50x50 min dimensions → Too coarse for magazine layout
- 150 DPI → Loses photo detail
- 0.6 sensitivity → Misses subtle editorial elements

**Characteristics:**
- SCAN = True (essential differentiator from digital_magazine)
- LARGE editorial photos (200-800px) **← Same as digital magazine!**
- HIGH image density (0.5+)
- Editorial domain
- Typically < 50 pages

**Expected Profile:** `ScannedMagazineProfile`

**Scoring:**
```
Scanned Magazine: 0.950 (confidence: 1.00)
  + Scanned document
  + High image density (0.85 >= 0.5) - magazine signature
  + Large editorial photos (450px >= 200) - magazine hallmark
  + Editorial domain - magazine content
  + Magazine-appropriate page count (23 < 50)
```

**Parameters Override:**
| Parameter | scanned_clean | scanned_magazine | Why? |
|-----------|---------------|------------------|------|
| Min Dimensions | 50x50px | **100x100px** | Filter icons but preserve photos |
| Sensitivity | 0.6 | **0.7** | Better photo recall |
| DPI | 150 | **200** | Higher photo quality |

---

### 5. Technical Manual (Firearms.pdf) **NEW in V2.1**

**The Problem:**
Technical manuals like Firearms.pdf contain SMALL but CRITICAL visual elements:
- Small parts diagrams: pins, screws, springs (often **< 50px**)
- Part numbers and callout labels
- Exploded view diagrams with fine detail

Standard `scanned_clean` with 50x50 min dimensions would **FILTER OUT** these crucial elements. This is **CATASTROPHIC** for RAG on technical documentation!

**Characteristics:**
- Technical domain (primary signal)
- MEDIUM image density (0.1-0.5)
- MEDIUM median_dim (50-200px for exploded views)
- Can be scan OR digital
- Typically 5-100 pages

**Expected Profile:** `TechnicalManualProfile`

**Scoring:**
```
Technical Manual: 0.900 (confidence: 1.00)
  + Technical domain - manual/documentation signature
  + Manual-appropriate image density (0.35) - diagrams but not photo-heavy
  + Technical diagram size (120px) - appropriate for exploded views
  + Scanned manual - common for legacy documentation
  + Manual-appropriate page count (45 pages)
```

**Parameters for Maximum Detail Capture:**
| Parameter | scanned_clean | technical_manual | Why? |
|-----------|---------------|------------------|------|
| Min Dimensions | 50x50px | **30x30px** | Catch pins, screws, springs |
| Sensitivity | 0.6 | **0.8** | HIGH recall for small parts |
| DPI | 150 | **300** | Maximum quality for fine detail |
| OCR Confidence | 0.5 | **0.4** | Catch faint part labels |

**Critical Insight:**
```
In a technical manual, a 35x35px image might be:
- A retaining pin diagram
- A spring illustration  
- A screw callout

In scanned_clean profile: FILTERED OUT (< 50px)
In technical_manual profile: PRESERVED AND DESCRIBED

For RAG on "How do I reassemble the firing mechanism?", 
those small parts are THE ANSWER.
```

---

## 🔬 Feature Importance Analysis

### Primary Discriminators (Ordered by Impact)

1. **`is_scan`** (Boolean)
   - **Impact:** Immediately eliminates wrong profile family
   - If `is_scan=True` → Scanned profiles only
   - If `is_scan=False` → Digital profiles only

2. **`median_dim`** (Integer, pixels)
   - **Impact:** Distinguishes literature from magazines
   - Small (25-100px) → Decorative book elements
   - Medium (100-200px) → Technical diagrams
   - Large (200+px) → Magazine editorial photos
   - **Weight:** 0.25-0.30 in scoring functions

3. **`text_density`** (Float, chars/page)
   - **Impact:** Distinguishes academic from editorial
   - High (3000+) → Academic/technical papers
   - Medium (1500-3000) → Reports, articles
   - Low (500-1500) → Magazines, scanned books
   - **Weight:** 0.20-0.35 in scoring functions

4. **`image_density`** (Float, images/page)
   - **Impact:** Distinguishes magazines from books
   - High (0.5+) → Magazines, presentations
   - Moderate (0.3-0.5) → Reports with charts
   - Low (0.1-0.3) → Academic papers, books
   - **Weight:** 0.20-0.30 in scoring functions

5. **`domain`** (String)
   - **Impact:** Semantic hint for classification
   - "academic" / "technical" → Academic whitepaper
   - "editorial" → Magazine or literature (use other features to decide)
   - **Weight:** 0.15-0.25 in scoring functions

6. **`page_count`** (Integer)
   - **Impact:** Distinguishes books from magazines/papers
   - Many (50+) → Books, long-form content
   - Moderate (10-50) → Academic papers, reports
   - Few (1-10) → Articles, short documents
   - **Weight:** 0.05-0.25 in scoring functions

---

## 🎛️ Tuning Guidelines

### Threshold Adjustments

Located in `ProfileClassifier`:

```python
class ProfileClassifier:
    # Text density thresholds (chars/page)
    TEXT_DENSITY_HIGH = 3000      # Academic papers
    TEXT_DENSITY_MEDIUM = 1500    # Reports, articles
    TEXT_DENSITY_LOW = 500        # Magazines, scanned books
    
    # Image density thresholds (images/page)
    IMAGE_DENSITY_HIGH = 0.5      # Magazines, presentations
    IMAGE_DENSITY_MEDIUM = 0.3    # Reports with charts
    IMAGE_DENSITY_LOW = 0.1       # Text-heavy documents
    
    # Median dimension thresholds (pixels)
    MEDIAN_DIM_LARGE = 200        # Full-page magazine photos
    MEDIAN_DIM_MEDIUM = 100       # Standard diagrams
    MEDIAN_DIM_SMALL = 50         # Small illustrations, icons
    
    # Page count thresholds
    PAGE_COUNT_BOOK = 50          # Books typically 50+ pages
    PAGE_COUNT_ARTICLE = 20       # Articles/papers 10-30 pages
    
    # Confidence threshold
    MIN_CONFIDENCE = 0.6          # Below this, use fallback
```

### Weight Adjustments

Modify scoring functions if certain features should have more/less influence:

```python
def _score_scanned_literature(features) -> ProfileScore:
    # Example: Increase median_dim importance
    if features.median_dim < MEDIAN_DIM_MEDIUM:
        score += 0.30  # Increased from 0.25
        # Now median_dim has 30% weight instead of 25%
```

---

## 🚀 Usage Examples

### Basic Classification

```python
from mmrag_v2.orchestration.profile_classifier import classify_document
from mmrag_v2.orchestration.smart_config import SmartConfigProvider
from mmrag_v2.orchestration.document_diagnostic import create_diagnostic_engine

# Analyze document
analyzer = SmartConfigProvider()
diagnostic_engine = create_diagnostic_engine()

doc_profile = analyzer.analyze("document.pdf")
diagnostic_report = diagnostic_engine.analyze("document.pdf")

# Classify using multi-dimensional features
profile_type = classify_document(doc_profile, diagnostic_report)
print(f"Selected profile: {profile_type.value}")
```

### With Logging (See Decision Process)

```python
import logging
logging.basicConfig(level=logging.INFO)

# Classifier logs all scores and reasoning:
# [CLASSIFIER] Features: text=4500ch/pg, imgs=0.15/pg, median=120px, pages=25, scan=False, domain=academic
# [CLASSIFIER] Profile scores:
#   academic_whitepaper: 0.950 (confidence: 1.00)
#     - High text density (4500 >= 3000)
#     - Academic domain
#     - Appropriate image density (0.15)
#   digital_magazine: 0.250 (confidence: 0.40)
#     - Low image density (0.15)
#     - Small images (120px) - not typical magazine photos
```

---

## 🧪 Testing Strategy

### Regression Tests

Create test cases for the "Holy Trinity":

```python
def test_aios_whitepaper_classification():
    """AIOS should classify as AcademicWhitepaperProfile."""
    profile = analyze_and_classify("data/raw/aios_paper.pdf")
    assert profile == ProfileType.ACADEMIC_WHITEPAPER

def test_harry_potter_classification():
    """Harry Potter should classify as ScannedLiteratureProfile."""
    profile = analyze_and_classify("data/raw/harry_potter.pdf")
    assert profile == ProfileType.SCANNED_LITERATURE

def test_combat_aircraft_classification():
    """Combat Aircraft should classify as DigitalMagazineProfile."""
    profile = analyze_and_classify("data/raw/combat_aircraft.pdf")
    assert profile == ProfileType.DIGITAL_MAGAZINE
```

### Feature Unit Tests

Test scoring functions in isolation:

```python
def test_scanned_literature_scoring():
    """Test literature profile scoring with book-like features."""
    features = ClassificationFeatures(
        text_density=800,
        image_density=0.15,
        median_dim=45,  # Small illustrations
        page_count=150,
        is_scan=True,
        domain="editorial",
        # ...
    )
    
    classifier = ProfileClassifier()
    score = classifier._score_scanned_literature(features)
    
    assert score.score >= 0.75  # Should score highly
    assert "Small decorative illustrations" in score.reasoning
```

---

## 📈 Benefits

### 1. **Eliminates Document-Specific Overfitting**
- No hardcoded checks for specific filenames or titles
- Generic features work across document types
- New documents automatically handled by existing logic

### 2. **Transparent Decision Making**
- All scores and reasoning logged
- Easy to debug misclassifications
- Clear feature importance for tuning

### 3. **Graceful Degradation**
- Confidence scores catch uncertain classifications
- Fallback to safe default (DigitalMagazineProfile)
- No catastrophic failures from edge cases

### 4. **Maintainable & Extensible**
- Add new profiles by implementing `_score_new_profile()`
- Adjust thresholds without touching core logic
- Clear separation of concerns

### 5. **Prevents Regression Cascades**
- Fix for one document doesn't break others
- Weighted scoring allows gradual tuning
- Test suite catches profile shifts

---

## 🎓 Architectural Principles

### 1. **Feature Engineering Over Rules**
```python
# BAD: Hardcoded rules
if "Potter" in filename and pages > 100:
    use_literature_profile()

# GOOD: Feature-based classification
if (is_scan and median_dim < 100 and page_count > 50 and domain == "editorial"):
    score_literature += 0.80
```

### 2. **Weighted Scoring Over Binary Decisions**
```python
# BAD: Binary classification
if text_density > 3000:
    return ACADEMIC
else:
    return MAGAZINE

# GOOD: Weighted scoring
academic_score = (
    0.35 if text_density > 3000 else 0.0 +
    0.25 if domain == "academic" else 0.0 +
    # ... more dimensions
)
```

### 3. **Explicit Confidence Over Silent Failures**
```python
# BAD: Silent fallback
try:
    profile = detect_profile()
except:
    profile = DigitalMagazineProfile()  # Why?

# GOOD: Explicit confidence check
best_match = max(scores, key=lambda s: s.score)
if best_match.confidence < MIN_CONFIDENCE:
    logger.warning(f"Low confidence ({confidence}), using safe fallback")
    return ProfileType.DIGITAL_MAGAZINE
```

### 4. **Separation of Concerns**
```python
# SmartConfigProvider: Extracts features (what IS)
# DocumentDiagnosticEngine: Analyzes characteristics (what MIGHT BE)
# ProfileClassifier: Classifies based on features (what SHOULD BE)
# ProfileManager: Instantiates the selected profile (what TO DO)
```

---

## 🔮 Future Enhancements

### 1. **Machine Learning Integration**
Replace hand-tuned weights with learned weights from training data:
```python
# Train on labeled dataset
classifier = MLProfileClassifier()
classifier.train(training_data)

# Use learned weights
profile = classifier.classify(features)
```

### 2. **Dynamic Threshold Adjustment**
Automatically adjust thresholds based on classification history:
```python
# Track classification accuracy
classifier.record_result(features, actual_profile, correct=True/False)

# Adjust thresholds to minimize error
classifier.optimize_thresholds()
```

### 3. **Profile Confidence Scores**
Not just "is this the right profile?" but "how confident are we?"
```python
result = classifier.classify_with_confidence(features)
# result = {
#     "profile": ProfileType.ACADEMIC_WHITEPAPER,
#     "confidence": 0.95,
#     "alternatives": [
#         (ProfileType.DIGITAL_MAGAZINE, 0.35),
#         (ProfileType.SCANNED_LITERATURE, 0.15),
#     ]
# }
```

---

## 📚 References

- **SRS Document:** `docs/SRS_Multimodal_Ingestion_V2.3.md`
- **SmartConfigProvider:** `src/mmrag_v2/orchestration/smart_config.py`
- **DocumentDiagnosticEngine:** `src/mmrag_v2/orchestration/document_diagnostic.py`
- **Strategy Profiles:** `src/mmrag_v2/orchestration/strategy_profiles.py`
- **ProfileClassifier:** `src/mmrag_v2/orchestration/profile_classifier.py` (NEW)

---

## ✅ Summary

The V2.0 Anti-Overfitting Architecture replaces hardcoded document-specific rules with an intelligent, multi-dimensional classification system. By using **generic visual and textual density features** (especially `median_dim`), the system:

1. ✅ Prevents "greedy optimization" that fixes one document but breaks another
2. ✅ Maintains accuracy across diverse document types (academic, literature, magazines)
3. ✅ Provides transparent, explainable classification decisions
4. ✅ Gracefully handles edge cases with confidence-based fallbacks
5. ✅ Enables easy tuning and extension without touching core logic

**The key insight:** A 45px median image dimension (Harry Potter's decorative illustrations) is fundamentally different from a 450px median dimension (Combat Aircraft's editorial photos), and this single metric—combined with other features—creates a robust, generalizable classification system.
