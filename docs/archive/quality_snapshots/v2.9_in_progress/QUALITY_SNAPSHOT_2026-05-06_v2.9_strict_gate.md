# Quality Snapshot 2026-05-06 — v2.9 strict-gate working state

> **Working snapshot, NOT a release.** v2.9 has not shipped. The
> `v2.9.0` tag was created on 2026-05-05 against a 32/34 reading
> from `qa_conversion_audit.py` alone, then deleted on 2026-05-06
> after a user-driven QA review surfaced defects the loose gate had
> missed. This file tracks the post-enrichment strict-gate state of
> `main`. The active baseline remains
> [`docs/QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md`](QUALITY_SNAPSHOT_2026-05-04_v2.8_after.md).

**Strict gate:** [`scripts/qa_full_conversion.py`](../scripts/qa_full_conversion.py)
— bundles `qa_conversion_audit.py` + `qa_universal_invariants.py` +
`qa_ingestion_hygiene.py` + `qa_semantic_fidelity.py`, plus
deterministic checks for missing pages (blank-page-aware when a
`--source-pdf` is provided), within-page text duplication, irreparable
corruption, image-description usability, asset health, and table
integrity. See `docs/TESTING.md` "Single-Conversion Full QA".

## Aggregate

**5 PASS / 3 WARN / 26 FAIL out of 34** (post-enrichment).

| Status | Count | Notes |
|---|---|---|
| `QA_PASS` | 5 | clean across all four scripts and deterministic checks |
| `QA_WARN` | 3 | one or more advisory issues, no hard-fail conditions |
| `QA_FAIL` | 26 | one or more hard-fail conditions |

## v2.9 work that did land on `main`

These are real fixes, even though the corpus does not yet meet the
ship gate.

- **chunk_id collision fix** — per-document monotonic ``position``
  hashed into the chunk-id seed; v2.8's 427 within-file dupes
  collapse on a fresh broad reconversion.
- **Refiner smart-routing** (`cli._decide_enable_refiner`) — gates
  the config-default refiner on `has_encoding_corruption=True`. HARRY
  no longer hammers qwen-plus per chunk; Combat still refines on
  encoding corruption.
- **Cross-page DocChunk page-coverage split** — Docling's
  HybridChunker emits multi-page chunks; the chunker now emits one
  IngestionChunk per source page so chapter-intro pages aren't lost.
- **Hybrid chunker `next_text_snippet` crash fix** — chunks with
  None semantic_context no longer trigger silent fallback to
  element-by-element extraction.
- **Mid-sentence merger and near-duplicate filter are page-scoped** —
  cross-page mid-sentence joins no longer reattribute one page's
  content to another. Same-content cross-page split copies survive
  the near-dup filter.
- **CorruptionInterceptor extended** — handles `Modality.TABLE` (was
  TEXT-only), and detects long em-dash / `CS` runs in addition to
  CIDFont / uniXXXX / replacement-character patterns. Irreparable
  chunks (>10% replacement chars) are quarantined instead of left in
  the canonical output.
- **FULL-PAGE-GUARD defers when no conversion-time VLM** — three
  sites changed from discard → defer with `vision_status="pending"`.
  Combat p4 (and similar pages whose only content is a full-page
  image) now produces a chunk that the post-conversion enrichment
  script picks up.
- **Phase 5b enrichment writes canonical `chunk.content`** — not
  just `metadata.visual_description`. Earlier v2.9 attempts reported
  `image_placeholder_ratio=1.0` because the canonical field was
  never updated.
- **Page-coverage / dup-excess / irreparable-corruption gates** in
  `qa_universal_invariants.py`.
- **Strict-gate wrapper** in `scripts/qa_full_conversion.py` plus
  the documentation entry in `docs/TESTING.md`.

## Per-document strict-gate result

| Status | Document | Failure codes |
|---|---|---|
| QA_WARN | HarryPotter_and_the_Sorcerers_Stone | (advisory only) |
| QA_PASS | Form_0013_invoice | — |
| QA_WARN | Form_betwistingsformulier | (advisory only) |
| QA_PASS | CarOK_voorraadtelling | — |
| QA_FAIL | AIOS_LLM_Agent_Operating_System | IMAGE_DESCRIPTION_UNUSABLE |
| QA_FAIL | A_comprehensive_review_on_hybrid_electri | IMAGE_DESCRIPTION_UNUSABLE |
| QA_FAIL | Hybrid_electric_vehicles | IMAGE_DESCRIPTION_UNUSABLE |
| QA_PASS | IRJET_Modeling_of_Solar_PV | — |
| QA_PASS | Recent_Trends_in_Transportation | — |
| QA_FAIL | Combat_Aircraft_August_2025 | LOCALIZED_CORRUPTION (p66 table); IMAGE_DESCRIPTION_UNUSABLE; SCRIPT_GATE_FAIL (qa_conversion_audit) |
| QA_FAIL | PCWorld_July_2025 | IMAGE_DESCRIPTION_UNUSABLE |
| QA_PASS | ATZ_Elektronik_German | — |
| QA_FAIL | Kimothi_RAG_Guide | MISSING_PAGES; IMAGE_DESCRIPTION_UNUSABLE |
| QA_FAIL | Integra_manual | MISSING_PAGES |
| QA_FAIL | Jungjun_AI_Agent | IMAGE_DESCRIPTION_UNUSABLE |
| QA_FAIL | Bourne_RAG_2024 | MISSING_PAGES; PAGE_CHUNK_OUTLIER; IMAGE_DESCRIPTION_UNUSABLE |
| QA_FAIL | Devlin_LLM_Agents | MISSING_PAGES; SCRIPT_GATE_FAIL (audit); IMAGE_DESCRIPTION_UNUSABLE |
| QA_FAIL | Raieli_AI_Agents | MISSING_PAGES; IMAGE_DESCRIPTION_UNUSABLE |
| QA_FAIL | Adedeji_GenAI_Google_Cloud | MISSING_PAGES; TABLE_CORRUPTION (p301) |
| QA_FAIL | Cronin_GenAI_Models | MISSING_PAGES |
| QA_FAIL | Hao_ML_Platform | MISSING_PAGES; IMAGE_DESCRIPTION_UNUSABLE |
| QA_FAIL | Nagasubramanian_Agentic_AI | MISSING_PAGES; IMAGE_DESCRIPTION_UNUSABLE |
| QA_FAIL | Sekar_MCP_Standard | MISSING_PAGES; IMAGE_DESCRIPTION_UNUSABLE |
| QA_FAIL | Python_Cookbook | MISSING_PAGES |
| QA_FAIL | ArcGIS_Python_Cookbook | MISSING_PAGES; IMAGE_DESCRIPTION_UNUSABLE |
| QA_FAIL | Fluent_Python | MISSING_PAGES; IMAGE_DESCRIPTION_UNUSABLE |
| QA_FAIL | Python_Distilled | MISSING_PAGES |
| QA_FAIL | Ayeva_Python_Patterns | MISSING_PAGES |
| QA_FAIL | Chaubal_PyTorch_Projects | MISSING_PAGES |
| QA_FAIL | Earthship_Vol1 | MISSING_PAGES; IMAGE_DESCRIPTION_UNUSABLE |
| QA_FAIL | Firearms | SCRIPT_GATE_FAIL (audit); IMAGE_DESCRIPTION_UNUSABLE |
| QA_FAIL | Greenhouse_Design | MISSING_PAGES; IMAGE_DESCRIPTION_UNUSABLE |
| QA_WARN | ChatGPT_Praktijk_handboek | (advisory only) |
| QA_FAIL | KI_En_ChatGPT_Praktische_Gids | SCRIPT_GATE_FAIL (audit + universal) |

## Failure-mode summary

**A. `MISSING_PAGES` on TOC / index pages (~18 docs).** Frontmatter
pages 5–12 (TOC, contents, brief contents, preface) and back-matter
index pages produce zero chunks. Removing the
`DocItemLabel.DOCUMENT_INDEX` filter at chunker time recovered some
pages (e.g. Hao p8–p10) but a deeper Docling- or pipeline-side
filter still drops the rest (Hao p5–p7 confirmed dropped after the
filter removal). Investigation paused with a reproducer in
`/tmp/hao_p5/`.

**B. `IMAGE_DESCRIPTION_UNUSABLE` from short cloud-VLM responses.**
The strict gate rejects descriptions under 20 chars. Cloud
`qwen3-vl-plus` occasionally emits valid but terse outputs (e.g.
"Venn diagram.", "Line chart.", "System schematic."). Several docs
hit single-digit miss counts (Hao 4/252; AIOS 10/10 etc.). Either
lower the threshold or accept short responses when
`vision_status="complete"`.

**C. Combat Aircraft p66 corrupted table.** The squadron-roster
table chunk still contains corrupted typography ("Ist
FTS(Note2)——————|PuebloMemorialAirport,Colorado—…"). The OCR-failure
regex catches em-dash runs of 6+; the surviving chunk's run is just
under that. Down from 5 corrupt chunks → 1.

**D. Specific-doc audit failures.** Adedeji p301 table corruption,
Devlin / Earthship / Firearms each have distinct
`qa_conversion_audit.py` failures, KI_En_ChatGPT EPUB has a
pre-existing LABEL orphan ratio above the 25% gate.

**E. Qdrant `mmrag_v2_8` not refreshed.** The collection currently
contains v2.8.0 ingest data only; v2.9 corpus has not been ingested
because v2.9 hasn't shipped.

## What it would take to actually ship v2.9

1. **Close the TOC/index page-drop.** This is the highest-leverage
   item — closing it likely brings the bulk of the 26 QA_FAIL docs
   to QA_PASS.
2. **Decide the short-VLM-description policy.** Either lower the
   gate threshold (e.g. 12 chars) or accept short responses when
   the VLM marked them complete. Update the gate accordingly.
3. **Combat p66 table corruption.** Lower the em-dash threshold
   from 6 → 4 in `OCR_FAILURE_PATTERNS`, or add a heuristic for
   "OCR table that lost its typography" beyond simple regex.
4. **Resolve the localized audit failures** doc by doc.
5. **Re-run the broad reconversion** on the affected docs, then
   re-run Phase 5b enrichment (only on placeholder image chunks
   so it skips already-enriched ones), then re-run the strict
   gate. 34/34 PASS is the bar.
6. **Drop and recreate `mmrag_v2_8`** from the v2.9 corpus.
7. **Tag `v2.9.0`** with a non-misleading AFTER snapshot.

## Reproducing this snapshot

```bash
# Convert (current main; ~10 hours sequential)
bash scripts/convert_books.sh

# Enrich (~3 hours, ~$10 cloud spend at qwen3-vl-plus)
python scripts/enrich_image_chunks_v29.py output/*/ingestion.jsonl

# Strict gate per document
for doc in output/<canonical>/ingestion.jsonl; do
  python scripts/qa_full_conversion.py "$doc" --source-pdf "<source>"
done
```

## Tag

The `v2.9.0` git tag is **not** present. Engine version on `main`
reads `2.9.0` (bumped during the in-flight work) but no annotated
tag has been created. Do not treat this commit chain as a release.
