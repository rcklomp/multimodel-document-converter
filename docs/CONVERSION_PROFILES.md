# Conversion Profiles — Optimal Settings Per Document Class

The document classifier (`document_modality` + `profile_type`) determines the conversion strategy. Each class has different optimal settings for image extraction, OCR, heading detection, chunking, and post-processing.

## Document Classes

| Class | Example Documents | Characteristics |
|---|---|---|
| `native_digital` + `technical_manual` | Sekar MCP, Raieli AI Agents, Fluent Python | Clean text layer, embedded images, PDF bookmarks, structured TOC |
| `native_digital` + `academic_whitepaper` | AIOS, IRJET Solar PV | Two-column layout, figures, tables, references section |
| `native_digital` + `digital_literature` | **Harry Potter and the Sorcerer's Stone** | Born-digital novels: clean text layer, narrative prose with dialogue, display-font drop caps, photographic cover artwork, dingbat scene-breaks. **Routed via the post-Docling sanity pass** — see §6 below. |
| `image_heavy` + `digital_magazine` | Combat Aircraft, PCWorld | Complex multi-column layout, many embedded photos, ads, stylized text |
| `scanned` | (no current corpus PDF; SCAN0013 is a small business form used as test probe) | Scanned pages, OCR needed, chapter headings detectable |
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

## 6b. Digital Literature (born-digital novels)

`digital_literature` is a `native_digital`-modality profile added 2026-05-03 to
catch born-digital novels (typesetting like AGaramondPro / Acrobat Distiller
output, e.g. Harry Potter and the Sorcerer's Stone). It inherits the
`technical_manual` defaults for image extraction, OCR (off), heading
detection, and chunking, with these differences:

| Setting | digital_literature | Source |
|---|---|---|
| `reading_order_strategy` | `y_sort_with_dropcap` (default for the profile) | `engines/pdf_plan.py` field added; `engines/docling_postprocess.py` runs the y-sort + drop-cap stages |
| `suppress_layout_label_text` | `True` | Custom `MmragChunkingSerializerProvider` strips `Other`/`Icon`/`Table` classification labels via `meta.classification` (`blocked_meta_names`) AND the legacy `PictureClassificationData` annotation path |
| `bitmap_area_threshold` | `0.92` | Plan field threaded into `EasyOcrOptions.bitmap_area_threshold` so cover artwork doesn't trigger OCR garbage |
| Min image size | 40x40 | Catches drop caps and small dingbats (smaller than the magazine 100x100) |
| `extract_backgrounds` | False | Novels have no editorial-photo backgrounds |
| Routing trigger | `domain == "literature" AND not is_scan AND page_count >= 50 AND median_dim small` | `_score_digital_literature` in `orchestration/profile_classifier.py` |

**Plan reference:** `docs/archive/PLAN_DOCLING_POSTPROCESSOR.md` documents the four
post-Docling stages (reading-order y-sort, drop-cap promotion, label-leak
filter, OCR gating) and the routing/wiring across `profile_classifier.py`,
`strategy_profiles.py`, `strategy_orchestrator.py`. Acceptance fixture:
`tests/test_docling_postprocessor_acceptance.py` (HARRY pages 13-30; passes
against live full-HARRY conversion).

---

## Implementation Status (v2.7.0)

| Feature | Status | Notes |
|---|---|---|
| Image extraction routing (I10) | Done | All types use Docling layout + picture classification. PyMuPDF tested and reverted (unreliable for magazines/papers). Classification disabled for scanned docs. |
| OCR routing | Done | Structural integrity pre-flight tests drive pathway. Digital guard skips OCR cascade. |
| Heading detection | Done | TOC bookmarks (PyMuPDF `get_toc()`), content-based magazine TOC fallback, `is_valid_heading()` filters, `_sanitize_chunk_for_export()` final gate. |
| HybridChunker | Done | Docling HybridChunker active for all profiles (350 max_tokens). Replaced custom 30-pass chunking. |
| VLM | Validated (cloud) | Cloud Qwen3-VL-Plus validated (2026-04-29): PCWorld text-reading 36.5%→22.2%, hard fallback 37.3%→21.4%. Local NuMarkdown comparison pending until local inference server is reachable. |
| Post-processing | Done | 4 multimodal validation layers: CorruptionInterceptor, POS Boundary Logic, Vision-Gated Hierarchy, Content-Type Classification. |
| Encoding corruption | Done | Heal-over strategy: keep HybridChunker, force refiner at threshold=0.0. |

## Open Items

1. **Magazine image quality** — composite page layouts (text+photo baked together) need rendered-region-crop architecture. Docling layout model is best-available but not ideal for separating editorial photos from page backgrounds.
2. **Class-specific chunking** — HybridChunker uses same `max_tokens=350` for all profiles. May benefit from per-profile tuning.
3. **`docling-hierarchical-pdf`** — integrated. Font-based heading reclassification runs after Docling conversion.
