# Quality Snapshot 2026-05-04 — v2.8 AFTER

**Purpose:** AFTER state for the v2.8 broad reconversion (Phase 5c).
Compare against `docs/QUALITY_SNAPSHOT_2026-05-03.md` for the BEFORE column.

**HEAD:** `c2e795e fix(audit): form TEXT-verdict contradiction + overnight pipeline scaffolding`

**Pre-flight evidence (committed in `5b0e13d`):**
- pytest tests/ -q: 590 passed, 2 skipped, 0 failed.
- bash scripts/smoke_multiprofile.sh: 11/11 GATE_PASS + 11/11 UNIVERSAL_PASS.
- HARRY pages-1-30 live acceptance: PASS.

## Aggregate (v2.8 canonical corpus only)
- **30/34 PASS** (includes form-pass class)
- **4 FAIL**
- **1 forms** (form acceptance class — invoices/short scanned docs)
- 22 legacy probes / exploration outputs (table 2)

## Per-document AFTER (canonical v2.8 corpus)

| Output dir | Chunks | Class | AFTER | BEFORE | Delta |
|---|---|---|---|---|---|
| A_comprehensive_review_on_hybrid_electri | 191 | digital | PASS | **FAIL (TEXT)** | ✓ FIXED |
| Adedeji_GenAI_Google_Cloud | 834 | digital | PASS | PASS | ✓ stable |
| AIOS_LLM_Agent_Operating_System | 146 | digital | PASS | PASS | ✓ stable |
| ArcGIS_Python_Cookbook | 928 | digital | PASS | — | NEW |
| ATZ_Elektronik_German | 60 | digital | PASS | PASS | ✓ stable |
| Ayeva_Python_Patterns | 753 | digital | FAIL | **FAIL (TEXT, CODE)** | still FAIL |
| Bourne_RAG_2024 | 1164 | digital | PASS | PASS | ✓ stable |
| CarOK_voorraadtelling | 76 | digital | PASS | — | NEW |
| ChatGPT_Praktijk_handboek | 298 | digital | PASS | — | NEW |
| Chaubal_PyTorch_Projects | 695 | digital | PASS | **FAIL (CODE)** | ✓ FIXED |
| Combat_Aircraft_August_2025 | 584 | digital | PASS | **FAIL (TEXT)** | ✓ FIXED |
| Cronin_GenAI_Models | 1288 | digital | PASS | PASS | ✓ stable |
| Devlin_LLM_Agents | 764 | digital | PASS | PASS | ✓ stable |
| Earthship_Vol1 | 675 | scanned | PASS | PASS | ✓ stable |
| Firearms | 1690 | scanned | FAIL | PASS | ⚠ REGRESSION |
| Fluent_Python | 1602 | digital | PASS | — | NEW |
| Form_0013_invoice | 20 | scanned | FORM_PASS | — | NEW |
| Form_betwistingsformulier | 8 | digital | PASS | — | NEW |
| Greenhouse_Design | 1148 | digital | PASS | — | NEW |
| Hao_ML_Platform | 1255 | digital | PASS | — | NEW |
| HarryPotter_and_the_Sorcerers_Stone | 447 | digital | PASS | PASS | ✓ stable |
| Hybrid_electric_vehicles | 125 | digital | PASS | — | NEW |
| Integra_manual | 244 | digital | PASS | — | NEW |
| IRJET_Modeling_of_Solar_PV | 40 | digital | PASS | PASS | ✓ stable |
| Jungjun_AI_Agent | 658 | digital | PASS | — | NEW |
| KI_En_ChatGPT_Praktische_Gids | 1210 | digital | FAIL | — | NEW |
| Kimothi_RAG_Guide | 680 | digital | PASS | PASS | ✓ stable |
| Nagasubramanian_Agentic_AI | 1017 | digital | PASS | — | NEW |
| PCWorld_July_2025 | 371 | digital | PASS | PASS | ✓ stable |
| Python_Cookbook | 389 | digital | PASS | — | NEW |
| Python_Distilled | 1042 | digital | PASS | — | NEW |
| Raieli_AI_Agents | 1509 | digital | PASS | — | NEW |
| Recent_Trends_in_Transportation | 20 | digital | PASS | PASS | ✓ stable |
| Sekar_MCP_Standard | 656 | digital | FAIL | — | NEW |

## Legacy / probe outputs (informational)

These predate or sit outside the v2.8 canonical corpus
(probes, partial-page reconverts, pre-fix _codex/_promptfix/_vlm runs).
Not counted in the aggregate above.

| Output dir | Chunks | Class | AFTER | BEFORE | Delta |
|---|---|---|---|---|---|
| ayeva_qa_20260501 | 627 | digital | PASS | PASS | ✓ stable |
| carok_fix | 76 | digital | PASS | — | NEW |
| carok_verify | 67 | digital | PASS | — | NEW |
| Combat_Aircraft_full_promptfix_v2 | 537 | digital | FAIL | **FAIL (TEXT)** | still FAIL |
| Combat_Aircraft_full_vlm_codex | 592 | digital | FAIL | — | NEW |
| Combat_Aircraft_promptfix_pages91_100 | 61 | digital | PASS | — | NEW |
| Fluent_Python_full_codex | 1633 | digital | PASS | PASS | ✓ stable |
| Fluent_Python_full_vlm_codex | 1634 | digital | PASS | PASS | ✓ stable |
| frontmatter_acceptance_harry_probe | 12 | digital | PASS | — | NEW |
| PCWorld_promptfix_pages1_20 | 65 | digital | PASS | — | NEW |
| probe_ayeva_p1_20 | 33 | digital | PASS | — | NEW |
| probe_boundary_closeout_rag_guide | 680 | digital | PASS | — | NEW |
| probe_carok | 0 | digital | FAIL | — | NEW |
| probe_carok_codex_verify | 76 | digital | PASS | — | NEW |
| probe_combat | 584 | digital | PASS | — | NEW |
| probe_contextual_retrieval_rag_guide | 680 | digital | PASS | — | NEW |
| probe_harry_potter | 458 | digital | PASS | — | NEW |
| probe_milestone2_rag_guide | 680 | digital | PASS | — | NEW |
| probe_rag_guide | 663 | digital | PASS | — | NEW |
| probe_rag_guide_guard | 680 | digital | PASS | — | NEW |
| v28_phase1_hybrid_review | 191 | digital | PASS | — | NEW |
| v28_phase3_combat_p60_70 | 133 | digital | PASS | — | NEW |

## Phase 5c gating decisions

**Conversion flags used** (per scripts/convert_books.sh):
```
python -m mmrag_v2.cli process <pdf> -o <out> -b <batch> \
  --vision-provider none --no-refiner --no-cache
```
Matches the Phase 0 BEFORE baseline so the delta column above is
apples-to-apples and isolates v2.8 code changes.

**Refiner smart-routing** (deferred to v2.9):
`cli.py:686` enables refiner whenever `~/.mmrag-v2.yml`
`refiner.enabled=true`, regardless of `has_encoding_corruption`.
This caused HARRY (clean prose, zero corruption) to hammer
qwen-plus during the first broad-reconversion attempt. Fix is
v2.9 scope: gate the config-default enable on the diagnostic
just like the explicit auto-override at `cli.py:1101` does.
After the fix, the broad reconversion can re-run without flags
and produce the same output for clean docs while still
auto-enabling refiner on encoding-corrupt ones.

## Qdrant re-ingestion — DEFERRED

The existing `multimodal-doc-converter-qdrant` Docling container
fails to start because of a stale collection lock:
```
Panic: Can't read collection version: Resource deadlock avoided
path: ./storage/collections/sekar_s__the_mcp_standard_.../version.info
```
The lock is held by a previous interrupted ingest run. Recovery
requires user input — clear the stale lock file or recreate the
container with a clean storage volume. Side-by-side ingest into
a new `mmrag_v2_8` collection is safe to do once the container
starts.

**Runbook (run in the morning):**
```bash
# 1. Inspect the broken container
docker logs multimodal-doc-converter-qdrant 2>&1 | tail -20

# 2. Choose recovery — pick ONE:
# (a) Surgical: clear only the stale lock
docker run --rm -v multimodal-doc-converter_qdrant_storage:/q alpine \
  rm -rf /q/collections/sekar_s__the_mcp_standard__a_developer_s_guide__building_universal_ai_tools_2026_pdf
# (b) Nuclear: drop the storage volume entirely (loses all collections)
docker rm -f multimodal-doc-converter-qdrant
docker volume rm multimodal-doc-converter_qdrant_storage

# 3. Start Qdrant (whichever recovery path)
docker start multimodal-doc-converter-qdrant   # if (a)
# OR re-create per project setup if (b)

# 4. Start Ollama (embedding backend)
open -a Ollama   # or: ollama serve &

# 5. Side-by-side ingest into mmrag_v2_8
for d in output/*/; do
  jsonl="$d/ingestion.jsonl"
  [ -f "$jsonl" ] || continue
  python scripts/ingest_to_qdrant.py "$jsonl" --collection mmrag_v2_8
done

# 6. Verify chunk counts match JSONL line counts (no silent drops)
# 7. Tag v2.8.0 once verification passes
git tag v2.8.0
git push origin v2.8.0   # only if pushing is desired
```
