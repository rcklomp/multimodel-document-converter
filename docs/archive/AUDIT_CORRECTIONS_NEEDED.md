# Audit Corrections Needed (Final Pass)

Goal: list every remaining issue in `docs/QUALITY_ARCHITECTURE_AUDIT.md` so Claude can fix all at once.

## Remaining Corrections

1. **Impact Assessment wording is inconsistent with verification status.**  
   In `docs/QUALITY_ARCHITECTURE_AUDIT.md` Impact Assessment still says “REQ‑MM‑03 Violated: Missing semantic context for assets/tables,” but the body now states image next_text is **verification required** (not a confirmed violation).  
   Fix: change Impact Assessment to say **tables are violated; images require QA verification**.

2. **Top‑level Impact Assessment is incomplete.**  
   The summary omits REQ‑PDF‑04 (page image generation) and REQ‑SENS drift, both called out later as violations.  
   Fix: add bullets for **REQ‑PDF‑04** and **REQ‑SENS** (or explicitly state the Impact Assessment is not exhaustive).

3. **Final summary list is missing in the report.**  
   The audit no longer includes a clear “Final Audit Summary” list of the **7 violations + 1 verification required**.  
   Fix: add a short summary section near the end, listing the seven violations and the one verification item explicitly.

## Optional Clarity Improvements (Non‑blocking)

4. **Clarify BatchProcessor atomic writes scope.**  
   The fix plan correctly flags non‑atomic writes in `batch_processor.py:1190-1272`. Add one sentence explicitly stating that this is a **core JSONL export path** (not just logging), to avoid minimization.

5. **REQ‑PDF‑04 description:**  
   The REQ‑PDF‑04 clarification notes `generate_picture_images` naming changes, but the actual quality issue is `generate_page_images=False`.  
   Fix: explicitly state that `generate_page_images=True` is required for padding consistency (REQ‑MM‑01), even if SRS wording is updated.
