## Uitgebreide Analyse & Verbeterd Implementatieplan

### 🔍 Huidige Architectuur Diagnose

Na analyse van `processor.py`, `batch_processor.py`, en `ocr_hint_engine.py` is het probleem helder:

**Huidige Flow (Shadow-First voor Scans):**
```
Page Render (300 DPI) → OCR (hints only) → VLM → "A poster showing Browning rifle..."
                                                    ↓
                                           content = VLM beschrijving 😞
```

**Gewenste Flow:**
```
Page Render → Layout Detection → Text Regions → OCR → TEXT chunks (verbatim)
                              → Image Regions → VLM → IMAGE chunks (descriptions)
```

---

### 📊 Gemini's Plan Beoordeling

| Aspect | Gemini's Voorstel | Beoordeling |
|--------|------------------|-------------|
| OCR-First, VLM-Second | ✅ Correct principe | Goed |
| OCR → content field | ✅ Juist | Goed |
| VLM → visual_description | ✅ Juist | Goed |
| Layout awareness | ❌ Niet geadresseerd | **Kritiek gemis** |
| Image region detection | ❌ Niet geadresseerd | **Kritiek gemis** |
| Multi-column handling | ❌ Niet geadresseerd | Probleem voor magazines |
| Table detection | ❌ Niet geadresseerd | Probleem |

**Gemini's plan is 60% compleet** - het mist cruciale layout analyse.

---

### 🏗️ Verbeterd Architectuurplan: "Layout-First Hybrid Extraction"

#### Fase 1: Layout Segmentatie (NIEUW)

**Doel**: Splits elke pagina in discrete regio's (tekst, afbeelding, tabel)

**Technologie opties:**
1. **Doctr** (Document Text Recognition) - Open source, layout-aware OCR
2. **LayoutLMv3** - Microsoft's layout-aware model
3. **PaddleOCR** - Alibaba's layout+OCR combo
4. **Docling's native layout** - Reeds geïntegreerd!

**Aanbeveling**: Docling ondersteunt al layout analysis! We gebruiken het verkeerd. In plaats van "shadow-first" moeten we Docling's layout output COMBINEREN met OCR:

```python
# HUIDIGE (verkeerd)
if use_shadow_first:
    # Skip Docling entirely, render full page
    page_image = render_page(300_dpi)
    vlm_description = vlm.describe(page_image)  # ← Verliest tekst!
    
# NIEUW (correct)
def process_scanned_page(page):
    # 1. Layout detection (via Docling of LayoutParser)
    regions = detect_layout_regions(page)
    
    # 2. Process text regions met OCR
    for region in regions.text_boxes:
        ocr_text = ocr.extract(region.crop)
        yield TextChunk(content=ocr_text, bbox=region.bbox)
    
    # 3. Process image regions met VLM
    for region in regions.images:
        description = vlm.describe(region.crop)
        yield ImageChunk(content=description, bbox=region.bbox)
```

#### Fase 2: OCR als Primaire Tekst Bron

**Huidige situatie**: OCR wordt alleen gebruikt als "hints" voor de VLM
**Nieuwe situatie**: OCR output IS de content

```python
class ScanTextExtractor:
    """Extract ACTUAL text from scanned documents."""
    
    def extract_text_from_region(self, image: Image, bbox: List[float]) -> str:
        """
        Extract verbatim text from a text region.
        
        Returns:
            The actual OCR text - NOT a description, NOT a summary.
        """
        region_crop = image.crop(bbox)
        
        # EasyOCR met paragraph=True voor natuurlijke tekstflow
        results = self.reader.readtext(region_crop, paragraph=True)
        
        # Combineer tekst in reading order
        text_blocks = [r[1] for r in results if r[2] > MIN_CONFIDENCE]
        return "\n".join(text_blocks)
```

#### Fase 3: VLM alleen voor Visuele Elementen

**VLM krijgt ALLEEN image regions**, niet hele pagina's:

```python
def process_image_region(region_crop: Image, surrounding_text: str) -> str:
    """
    Describe a detected IMAGE region (not a full page).
    
    Args:
        region_crop: The cropped image (just the photo/diagram)
        surrounding_text: OCR-extracted text near this image (for context)
    """
    prompt = f"""
    Describe this IMAGE (not text).
    Context: This image appears near text about: {surrounding_text[:200]}
    
    Focus on:
    - What is depicted (object, person, diagram)
    - Visual details (colors, composition)
    - Relevance to surrounding text
    
    DO NOT describe text or read labels - only visual content.
    """
    return vlm.describe(region_crop, prompt)
```

#### Fase 4: Smart Chunking voor OCR Tekst

**Probleem**: OCR output is vaak één grote blob zonder structuur
**Oplossing**: Layout-aware chunking

```python
class LayoutAwareChunker:
    """Chunk OCR text using detected layout structure."""
    
    def chunk_by_layout(self, text_regions: List[TextRegion]) -> List[str]:
        """
        Create chunks that respect:
        1. Column boundaries (don't merge columns)
        2. Paragraph breaks
        3. Section headers
        """
        chunks = []
        current_chunk = ""
        
        for region in sorted(text_regions, key=reading_order):
            # Detect if this is a new logical section
            if self._is_section_break(region, prev_region):
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = region.text
            else:
                current_chunk += " " + region.text
                
            # Respect max chunk size
            if len(current_chunk) > MAX_CHUNK_SIZE:
                chunks.append(current_chunk)
                current_chunk = ""
        
        return chunks
```

---

### 📋 Implementatie Stappenplan

#### Stap 1: Layout Detection Module (NIEUW)
```
src/mmrag_v2/layout/
├── __init__.py
├── region_detector.py      # Detecteert text/image/table regio's
├── reading_order.py        # Bepaalt leesvolorde (links→rechts, boven→onder)
└── region_merger.py        # Merged aangrenzende tekst regio's
```

**Technologie**: Gebruik Docling's bestaande layout output OF integreer `layoutparser`

#### Stap 2: Scan Text Extractor (NIEUW)
```
src/mmrag_v2/ocr/
├── __init__.py
├── scan_text_extractor.py  # OCR → verbatim tekst
├── text_cleaner.py         # Post-processing (dehyphenation, etc.)
└── confidence_filter.py    # Filter laag-confidence OCR
```

#### Stap 3: Dual-Track Processor (REFACTOR)
```
src/mmrag_v2/processor.py
  └── _process_scanned_page()  # NIEUW: Vervangt shadow-first
       ├── detect_regions()
       ├── process_text_regions() → TEXT chunks
       └── process_image_regions() → IMAGE chunks
```

#### Stap 4: Schema Update (MOGELIJK)
```python
# Toevoegen aan ingestion_schema.py
class IngestionChunk:
    # Bestaande velden...
    
    # NIEUW: Extraction provenance
    extraction_source: Literal["docling", "ocr", "vlm"]
    ocr_confidence: Optional[float]  # Alleen voor OCR chunks
```

---

### 🎯 Verwachte Output na Implementatie

**VOOR (huidige output voor scan):**
```json
{
  "content": "A poster featuring a semi-automatic rifle from Browning company...",
  "modality": "shadow",
  "metadata": {"page_number": 21}
}
```

**NA (verbeterde output voor scan):**
```json
// TEXT chunk (OCR extracted)
{
  "content": "BROWNING BAR AUTOMATIC RIFLE\n\nThe Browning BAR is a semi-automatic rifle manufactured by Fabrique Nationale. Caliber: .30-06 Springfield. Barrel length: 24 inches. Magazine capacity: 4 rounds.",
  "modality": "text",
  "metadata": {
    "page_number": 21,
    "extraction_source": "ocr",
    "ocr_confidence": 0.87
  }
}

// IMAGE chunk (VLM described)
{
  "content": "Profile photograph of Browning BAR rifle showing walnut stock and blued barrel",
  "modality": "image",
  "metadata": {
    "page_number": 21,
    "extraction_source": "vlm"
  },
  "asset_ref": {"file_path": "assets/xxx_021_figure_01.png"}
}
```

---

### ⚠️ Risico's en Mitigaties

| Risico | Impact | Mitigatie |
|--------|--------|-----------|
| OCR kwaliteit op oude scans | Onleesbare tekst | Confidence filtering + VLM fallback |
| Layout detection errors | Verkeerde regio classificatie | Hybrid: VLM verify uncertain regions |
| Performance impact | Langzamere processing | Batch OCR, cache layout results |
| Multi-column merge fouten | Tekst uit verkeerde kolommen | Reading order algorithm |

---

### 📈 Prioritering

1. **P0 (Must Have)**: Layout detection + OCR text extraction
2. **P1 (Should Have)**: Reading order voor multi-column
3. **P2 (Nice to Have)**: OCR confidence in schema

---

### 🔄 Alternatief: Doctr Integratie

Als Docling's layout niet voldoet, overweeg **Doctr** (mindee's library):

```python
from doctr.models import ocr_predictor

model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn')
result = model([page_image])

# Doctr geeft BEIDE: layout + OCR in één pass
for page in result.pages:
    for block in page.blocks:
        for line in block.lines:
            print(line.render())  # Actual text
            print(line.geometry)  # Bbox
```

---

### Volgende Stap

Wil je dat ik:
1. **Start met implementatie** van de Layout Detection module?
2. **Proof of Concept** maak met Doctr/LayoutParser?
3. **Benchmark** run van EasyOCR vs Doctr vs PaddleOCR op je Firearms scan?

Als je akkoord bent met dit plan, toggle naar **Act Mode** en ik begin met de implementatie.