# R3 code-router gap: precise diagnosis (2026-06-11)

Triggered by the Phase 5 full-corpus run: code-heavy books fail the R3
code-indentation gate (`QA_FAIL`, "degraded code indentation") on cleanly-extracted
(`degraded=0`) output. The obvious read - "the monospace router threshold (0.10) is
wrong, lower it" - is WRONG. The router is sound; the failure decomposes into three
distinct causes, only one of which is an unaddressed gap.

All analysis below is OFFLINE (PyMuPDF page signals + the in-tree router functions);
no inference server is involved, so it stands while the GX10 is down.

## The production routing (MineruQwenHybridEngine, router.py:383+)

Per page: `mono_ratio >= 0.10` -> Qwen (code); ELIF `page_has_code_block` AND NOT
`page_has_table` -> Qwen (diluted code block); ELSE -> MinerU. Two font-based code
signals, the second table-guarded (Qwen empties dense tables, so a code-block page
that also looks like a table stays on MinerU).

## The three root causes (measured)

| doc | pages | code pages (content) | residual code->MinerU | reason |
|---|--:|--:|--:|---|
| FluentPython | 766 | 450 | 11 (1.4%) | font-blind text |
| PythonDistilled | 1411 | 341 | 2 (0.1%) | font-blind text |
| HarryPotter (prose) | 327 | 0 | 0 | (no over-trigger) |
| **C++ Manual (R3-FAIL)** | 148 | **0 by text** | **148 (100%)** | **image-only scan** |

1. **Dilution - ALREADY FIXED.** A real monospace code block whose page-average ratio
   is pulled below 0.10 by surrounding prose. `page_has_code_block` (contiguous run of
   >=4 lines each >=0.6 mono) recovers these; FluentPython has 456 such pages caught.
   This was the first thing I went to "fix" - it was already there and table-guarded.

2. **Font-blind text code - TINY residual.** Code typeset in a font not in
   `MONO_FONT_TOKENS`, so both font signals read 0. Measured: 11 FluentPython +
   2 PythonDistilled pages (~1% of code pages). A font-independent content detector
   (indentation + code punctuation + keywords) catches them with **0% false-positive
   on prose** (HarryPotter 0, Grundlagen 0). Small, low-risk, optional.

3. **Image-only scanned code - THE REAL GAP.** The C++ manual is 100% image-only:
   every one of 148 pages has 0 text chars, 1 image, 100% image-area coverage. There
   is NO text layer, so EVERY font-based signal is blind by construction. All pages ->
   MinerU; MinerU OCRs the code images but its 1.2B recognizer mangles indentation
   (R3 0.44) -> the gate fails on real degraded code. A threshold tweak cannot touch
   this - there are no glyphs to weigh.

## The fix (ties into the already-designed Phase 3 quality-risk arbitration)

The image-only case is exactly what `PLAN_EXTRACTION_FIDELITY_V1` Section 5.4 /
charter Section 4.3 already specify but have not built: **action-on-flag specialist
re-extraction.** The R3 metric ALREADY detects degraded code post-extraction (it is
what fails these docs). Wire that flag to a bounded re-extraction of the flagged
page(s) through the Qwen code lane:

```
MinerU page -> R3/quality-risk flag (collapsed code suite) -> ONE re-extraction
of that page via the Qwen VLM (clean indentation) -> accept the better of the two.
```

This needs no new routing signal and no per-page image classifier - it reuses the
existing R3 detector as the trigger, on the engine-agnostic chunker side, and the
existing Qwen lane as the specialist. It fixes both the image-only scan (cause 3) and
the font-blind text residual (cause 2) for free, because both surface as the same
post-extraction R3 signal. It is the Section 5.4 consumer #1 ("conversion-time
specialist re-extraction") made concrete.

Optional, separable, lower-value: add the font-independent content detector as a 3rd
routing branch for cause 2 (measured 0% prose over-trigger). Skip if the Section 5.4
re-extraction lands - it subsumes this.

## What NOT to do

- Do NOT lower the 0.10 monospace threshold. It is calibrated (AIOS non-code <=0.02,
  code 0.19-0.98); lowering it floods Qwen with prose pages (cost/latency) and does
  nothing for the image-only case that actually fails.
- Do NOT route all image-only pages to Qwen. Most scanned docs are prose/forms where
  MinerU is the right engine; that would blow up the Qwen lane on every scan.

The lever is the post-extraction quality flag, not the pre-flight router.
