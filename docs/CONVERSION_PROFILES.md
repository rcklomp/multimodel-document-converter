# Conversion Profiles — Optimal Settings Per Document Class

The document classifier (`document_modality` + `profile_type`) determines the conversion strategy. Each class has different optimal settings for image extraction, OCR, heading detection, chunking, and post-processing.

## Document Classes

| Class | Example Documents | Characteristics |
|---|---|---|
| `native_digital` + `technical_manual` | Sekar MCP, Raieli AI Agents, Fluent Python | Clean text layer, embedded images, PDF bookmarks, structured TOC |
| `native_digital` + `academic_whitepaper` | AIOS, IRJET Solar PV | Two-column layout, figures, tables, references section |
| `image_heavy` + `digital_magazine` | Combat Aircraft, PCWorld | Complex multi-column layout, many embedded photos, ads, stylized text |
| `scanned_clean` | Harry Potter | Scanned pages, OCR needed, chapter headings detectable |
| `scanned_degraded` | Firearms | Poor scan quality, OCR unreliable, VLM transcription needed |

---

## Pipeline Settings Per Class

### 1. Image Extraction

| Setting | technical_manual | academic_whitepaper | digital_magazine | scanned_clean | scanned_degraded |
|---|---|---|---|---|---|
| **Strategy** | PyMuPDF embedded | PyMuPDF embedded | PyMuPDF embedded | Docling layout | Docling layout + VLM |
| **Why** | PDF has discrete image objects | Same | Same — layout model produces oversized captures | No embedded objects | No embedded objects |
| `generate_picture_images` | False (use PyMuPDF) | False | False | True | True |
| `do_picture_classification` | Not needed | Not needed | Not needed | True (filter full-page) | True |
| Shadow extraction | Disabled | Disabled | Disabled | Safety net | Safety net + VLM gate |
| Min image size | 50x50 | 50x50 | 100x100 | 50x50 | 50x50 |

### 2. OCR

| Setting | technical_manual | academic_whitepaper | digital_magazine | scanned_clean | scanned_degraded |
|---|---|---|---|---|---|
| **Strategy** | Disabled (native text) | Disabled | Disabled | Tesseract | EasyOCR + Doctr fallback |
| `do_ocr` | False | False | False | True | True |
| `force_full_page_ocr` | No | No | No | No | Yes |
| Refiner threshold | 0.15 (default) | 0.15 | 0.15 | 0.0 (all chunks refined) | 0.0 |

### 3. Heading Detection

| Setting | technical_manual | academic_whitepaper | digital_magazine | scanned_clean | scanned_degraded |
|---|---|---|---|---|---|
| **Primary source** | PDF bookmarks (TOC) | PDF bookmarks | Content-based TOC parse | HybridChunker + infer | HybridChunker + infer |
| **Fallback** | HybridChunker | HybridChunker | Forward-propagation | Forward-propagation | Forward-propagation |
| `_extract_toc_headings` | Yes (PyMuPDF bookmarks) | Yes | Yes (content fallback) | Yes if bookmarks exist | No (usually no bookmarks) |
| `_extract_toc_from_content` | Not needed | Not needed | Yes (magazine TOC page) | Not needed | Not needed |
| `_infer_headings_from_text` | No | No | No | Yes (all-caps detection) | Yes |
| `docling-hierarchical-pdf` | Optional (font-based) | Optional | No (OCR text has no font) | No | No |
| `is_valid_heading` filters | Caption/listing rejection | Caption/listing rejection | OCR junk rejection | Length/sentence checks | Length/sentence checks |

### 4. Chunking

| Setting | technical_manual | academic_whitepaper | digital_magazine | scanned_clean | scanned_degraded |
|---|---|---|---|---|---|
| HybridChunker `max_tokens` | 350 | 350 | 350 | 350 | 350 |
| Oversize breaker | 1500 chars | 1500 chars | 1500 chars | 1500 chars | 1500 chars |
| Code detection | Yes (`_apply_code_hygiene`) | Yes | No | No | No |
| Semantic overlap | Yes | Yes | Yes | Yes | Yes |

### 5. VLM Usage

| Setting | technical_manual | academic_whitepaper | digital_magazine | scanned_clean | scanned_degraded |
|---|---|---|---|---|---|
| Image descriptions | Optional (diagrams benefit) | Optional | Optional (photos) | Recommended | Required |
| Full-page VLM guard | No | No | No | Yes | Yes |
| VLM transcription | No | No | No | No | Yes (replaces OCR) |
| Refiner (text cleanup) | Optional | Optional | Optional | Recommended | Required |

### 6. Post-Processing

| Setting | technical_manual | academic_whitepaper | digital_magazine | scanned_clean | scanned_degraded |
|---|---|---|---|---|---|
| POS merger (orphan prepositions) | Yes | Yes | No | Yes | Yes |
| Mid-sentence merger | Yes | Yes | Yes | Yes | Yes |
| Dedup (overlap + subset) | Yes | Yes | Yes | Yes | Yes |
| Spaced heading collapse | No | No | No | Yes | Yes |
| Boilerplate detection | Yes | Yes | Yes | No | No |
| TOC heading propagation | Yes | Yes | Yes | If bookmarks | Forward-propagation |

---

## Implementation Status

| Feature | Current State | Target State |
|---|---|---|
| Image extraction routing (I10) | All use Docling layout model | PyMuPDF for digital, Docling for scanned |
| OCR routing | Partially implemented (digital guard) | Fully class-driven |
| Heading detection | TOC + content fallback + filters | Class-specific strategy selection |
| Chunking | Same for all | Consider class-specific tuning |
| VLM | Disabled due to API stability | Re-enable with timeout fix |
| Post-processing | Same for all | Consider class-specific passes |

## Priority Order for Implementation

1. **I10 — Image extraction routing** (highest impact — fixes magazine image quality)
2. **Heading strategy per class** (partially done — TOC bookmarks vs content parse)
3. **VLM re-enablement** (timeout fix done, needs stability test)
4. **Class-specific post-processing** (low priority — current universal approach works)
