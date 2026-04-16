# Conversion Profiles — Optimal Settings Per Document Class

The document classifier (`document_modality` + `profile_type`) determines the conversion strategy. Each class has different optimal settings for image extraction, OCR, heading detection, chunking, and post-processing.

## Document Classes

| Class | Example Documents | Characteristics |
|---|---|---|
| `native_digital` + `technical_manual` | Sekar MCP, Raieli AI Agents, Fluent Python | Clean text layer, embedded images, PDF bookmarks, structured TOC |
| `native_digital` + `academic_whitepaper` | AIOS, IRJET Solar PV | Two-column layout, figures, tables, references section |
| `image_heavy` + `digital_magazine` | Combat Aircraft, PCWorld | Complex multi-column layout, many embedded photos, ads, stylized text |
| `scanned` | Harry Potter | Scanned pages, OCR needed, chapter headings detectable |
| `scanned_degraded` | Firearms | Poor scan quality, OCR unreliable, VLM transcription needed |

---

## Pipeline Settings Per Class

### 1. Image Extraction

| Setting | technical_manual | academic_whitepaper | digital_magazine | scanned | scanned_degraded |
|---|---|---|---|---|---|
| **Strategy** | Docling layout + classification | Docling layout + classification | Docling layout + classification | Docling layout (no classification) | Docling layout (no classification) |
| **Why** | Consistent across all digital types; classification filters layout artifacts | Same | Same | Classifier hangs on large scanned books; not needed (images are illustrations) | Same |
| `generate_picture_images` | True | True | True | True | True |
| `do_picture_classification` | True (deny full_page_image, page_thumbnail) | True | True | False (disabled — hangs) | False |
| Shadow extraction | Disabled | Disabled | Disabled | Safety net | Safety net + VLM gate |
| Min image size | 50x50 | 50x50 | 100x100 | 50x50 | 50x50 |

### 2. OCR

| Setting | technical_manual | academic_whitepaper | digital_magazine | scanned | scanned_degraded |
|---|---|---|---|---|---|
| **Strategy** | Disabled (native text) | Disabled | Disabled | Tesseract | EasyOCR + Doctr fallback |
| `do_ocr` | False | False | False | True | True |
| `force_full_page_ocr` | No | No | No | No | Yes |
| Refiner threshold | 0.15 (default) | 0.15 | 0.15 | 0.0 (all chunks refined) | 0.0 |

### 3. Heading Detection

| Setting | technical_manual | academic_whitepaper | digital_magazine | scanned | scanned_degraded |
|---|---|---|---|---|---|
| **Primary source** | PDF bookmarks (TOC) | PDF bookmarks | Content-based TOC parse | HybridChunker + infer | HybridChunker + infer |
| **Fallback** | HybridChunker | HybridChunker | Forward-propagation | Forward-propagation | Forward-propagation |
| `_extract_toc_headings` | Yes (PyMuPDF bookmarks) | Yes | Yes (content fallback) | Yes if bookmarks exist | No (usually no bookmarks) |
| `_extract_toc_from_content` | Not needed | Not needed | Yes (magazine TOC page) | Not needed | Not needed |
| `_infer_headings_from_text` | No | No | No | Yes (all-caps detection) | Yes |
| `docling-hierarchical-pdf` | Optional (font-based) | Optional | No (OCR text has no font) | No | No |
| `is_valid_heading` filters | Caption/listing rejection | Caption/listing rejection | OCR junk rejection | Length/sentence checks | Length/sentence checks |

### 4. Chunking

| Setting | technical_manual | academic_whitepaper | digital_magazine | scanned | scanned_degraded |
|---|---|---|---|---|---|
| HybridChunker `max_tokens` | 350 | 350 | 350 | 350 | 350 |
| Oversize breaker | 1500 chars | 1500 chars | 1500 chars | 1500 chars | 1500 chars |
| Code detection | Yes (`_apply_code_hygiene`) | Yes | No | No | No |
| Semantic overlap | Yes | Yes | Yes | Yes | Yes |

### 5. VLM Usage

| Setting | technical_manual | academic_whitepaper | digital_magazine | scanned | scanned_degraded |
|---|---|---|---|---|---|
| Image descriptions | Optional (diagrams benefit) | Optional | Optional (photos) | Recommended | Required |
| Full-page VLM guard | No | No | No | Yes | Yes |
| VLM transcription | No | No | No | No | Yes (replaces OCR) |
| Refiner (text cleanup) | Optional | Optional | Optional | Recommended | Required |

### 6. Post-Processing

| Setting | technical_manual | academic_whitepaper | digital_magazine | scanned | scanned_degraded |
|---|---|---|---|---|---|
| POS merger (orphan prepositions) | Yes | Yes | No | Yes | Yes |
| Mid-sentence merger | Yes | Yes | Yes | Yes | Yes |
| Dedup (overlap + subset) | Yes | Yes | Yes | Yes | Yes |
| Spaced heading collapse | No | No | No | Yes | Yes |
| Boilerplate detection | Yes | Yes | Yes | No | No |
| TOC heading propagation | Yes | Yes | Yes | If bookmarks | Forward-propagation |

---

## Implementation Status (v2.7.0)

| Feature | Status | Notes |
|---|---|---|
| Image extraction routing (I10) | Done | All types use Docling layout + picture classification. PyMuPDF tested and reverted (unreliable for magazines/papers). Classification disabled for scanned docs. |
| OCR routing | Done | Structural integrity pre-flight tests drive pathway. Digital guard skips OCR cascade. |
| Heading detection | Done | TOC bookmarks (PyMuPDF `get_toc()`), content-based magazine TOC fallback, `is_valid_heading()` filters, `_sanitize_chunk_for_export()` final gate. |
| HybridChunker | Done | Docling HybridChunker active for all profiles (350 max_tokens). Replaced custom 30-pass chunking. |
| VLM | Available | Timeout fix applied to all 3 providers. Currently disabled for batch runs (`--vision-provider none`) due to Alibaba API instability. |
| Post-processing | Done | 4 multimodal validation layers: CorruptionInterceptor, POS Boundary Logic, Vision-Gated Hierarchy, Content-Type Classification. |
| Encoding corruption | Done | Heal-over strategy: keep HybridChunker, force refiner at threshold=0.0. |

## Open Items

1. **Magazine image quality** — composite page layouts (text+photo baked together) need rendered-region-crop architecture. Docling layout model is best-available but not ideal for separating editorial photos from page backgrounds.
2. **Class-specific chunking** — HybridChunker uses same `max_tokens=350` for all profiles. May benefit from per-profile tuning.
3. **`docling-hierarchical-pdf`** — installed, compatible with Docling 2.86.0, not yet integrated (font-based heading classification).
