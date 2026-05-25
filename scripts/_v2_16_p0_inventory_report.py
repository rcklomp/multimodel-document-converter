"""v2.16 Phase 0 step 5 — auto-draft `docs/CORPUS_EXPANSION_*.md` from
the classifier JSON output.

The plan (§3 Phase 0 step 5) requires a per-doc inventory report listing
basename, auto-routed profile, chunk count (text/table/image/code),
computed class flags, class-determining metrics, probe flags, extraction
warnings. This script consumes the JSON dump from
`classify_corpus_v2_16_p0.py --json` and emits the report.

User reviews + accepts probe-flagged docs by editing the report
directly.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path


def render(rows: list[dict]) -> str:
    today = datetime.date.today().isoformat()
    lines = [
        f"# v2.16 Phase 0 — Corpus Expansion Inventory",
        "",
        f"> Generated: {today}",
        f"> Source PDFs: `data/raw/` (7 files); outputs under `output/<basename>/ingestion.jsonl`",
        f"> Classifier: `scripts/classify_corpus_v2_16_p0.py` (recalibrated thresholds; see §2 below).",
        "",
        "## 1. Per-doc classification",
        "",
        "| basename | profile | chunks | text | code | table | image | code_ratio | table_ratio | uniq_tbl | non_ascii | intent_ML | flags | probes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        probe = r.get("probe_results") or {}
        probe_str = ", ".join(f"{k}={v}" for k, v in probe.items()) if probe else "—"
        lines.append(
            f"| `{r.get('basename', '?')}` "
            f"| {r.get('profile') or '—'} "
            f"| {r.get('n_total_chunks', 0)} "
            f"| {r.get('n_text_chunks', 0)} "
            f"| {r.get('n_code_chunks', 0)} "
            f"| {r.get('n_table_chunks', 0)} "
            f"| {r.get('n_image_chunks', 0)} "
            f"| {r.get('code_ratio', 0.0):.3f} "
            f"| {r.get('table_ratio', 0.0):.3f} "
            f"| {r.get('unique_table_template_patterns', 0)} "
            f"| {r.get('mean_non_ascii_ratio', 0.0):.4f} "
            f"| {r.get('intent_minority_lang_hit_rate', 0.0):.2f} "
            f"| {', '.join(r.get('flags') or [])} "
            f"| {probe_str} |"
        )
    lines.extend([
        "",
        "## 2. Recalibrated thresholds (vs plan defaults)",
        "",
        "Per PLAN_V2.16.md §3 Phase 0 step 2 (threshold pre-validation):",
        "the plan's defaults (code 0.30, table 0.40) fail the canonical-34",
        "sanity targets — Fluent_Python at 0.276 code-ratio misses 0.30 by",
        "construction; CarOK at 0.136 table-ratio misses 0.40 by",
        "construction. Recalibrated thresholds satisfy all 6 sanity",
        "targets via step-3 rules alone (Probe A not needed for CarOK at",
        "the new threshold).",
        "",
        "| Class | Plan default | v2.16 recalibrated | Sanity verdict |",
        "|---|---|---|---|",
        "| code-dense | `code/(text+code) ≥ 0.30` | **`≥ 0.25`** | Fluent_Python (0.276) PASS; HarryPotter/CarOK still NOT |",
        "| form-class | `table/total ≥ 0.40 AND uniq ≥ 3` | **`≥ 0.10 AND uniq ≥ 3`** | CarOK (0.136, uniq=10) PASS; Python_Distilled (0.039) still NOT |",
        "| minority-language | unchanged | unchanged | ATZ_Elektronik PASS via intent (no umlauts); HarryPotter false-positive noted as intent-classifier limitation |",
        "",
        "## 3. Probe coverage (per PLAN_V2.16.md §3 Phase 0 step 4)",
        "",
        "- **Probe A** — form-class re-extract via `--force-table-vlm`",
        "  fires only on docs with `profile ∈ {scanned, scanned_degraded}`",
        "  AND `image_chunks > 0` AND `table_chunks == 0`. See per-doc",
        "  flags above.",
        "- **Probe B** — borderline minority-language (signal-only). Fires",
        "  when intent classifier matches ≥1 chunk but total hit-rate <",
        "  0.30 AND non-ASCII < 0.03. See per-doc flags above.",
        "- **Probe C** — near-boundary classification (signal-only). Flags",
        "  docs within 5pp of the code-dense / form-class / minority-",
        "  language thresholds for user review.",
        "",
        "## 4. Class composition feeding Phase 2/3/4 scoping",
        "",
        "Per PLAN_V2.16.md §3 Phase 0 step 7:",
        "",
        "- **Minority-language docs** in this expansion: classified per",
        "  Phase 0 (German content count visible in §1 above).",
        "  Phase 2 (omlx-deficit class-level test) verdict already",
        "  recorded as multi-factor in",
        "  `docs/archive/diagnostics/DIAGNOSTIC_2026-05-25_v2.16_p2_omlx_deficit_root_cause.md`",
        "  — replication would require a hot dashscope collection which",
        "  was dropped in v2.14 P3. Phase 6 KILL is final.",
        "- **Form-class docs** in this expansion: see §1. Phase 4",
        "  generality measured via the programmatic dual gate",
        "  (`suppression_count > 0` AND no same-page Jaccard ≥ 0.5) once",
        "  the CarOK re-extract result lands.",
        "- **Code-dense docs** in this expansion: see §1. Phase 3",
        "  acceptance bar (≥85% Phase 1 validation on code-dense docs)",
        "  is structurally blocked by the inert-on-current-corpus state",
        "  of partial_code (see DECISIONS.md \"v2.16 Phase 3 …\").",
        "",
        "## 5. User-acceptance notes",
        "",
        "**Probe-flagged docs require explicit user acceptance** before",
        "Phase 0 step 6 (Qdrant snapshot + dense append + BM25 rebuild)",
        "runs. Acceptance is recorded by editing this report in place",
        "with one of:",
        "",
        "- `ACCEPTED: classification correct`",
        "- `RECLASSIFY: override profile to <X>` (re-run ingest with",
        "  `--profile-override <X>` before step 6.3)",
        "- `RECALIBRATE: adjust threshold <X> from <Y> → <Z>` (re-run",
        "  step 2 + step 3 with the new threshold).",
        "",
        "Edit each row of §1 to add an `ACCEPTED` / `RECLASSIFY` /",
        "`RECALIBRATE` column when reviewing.",
        "",
        "## 6. Source PDFs",
        "",
    ])
    for r in rows:
        lines.append(f"- `data/raw/<...>.pdf` → `output/{r.get('basename', '?')}/`")
    lines.extend([
        "",
        "## 7. Reproduce",
        "",
        "```bash",
        "# Re-ingest all 7:",
        "bash scripts/_v2_16_p0_ingest_all.sh",
        "",
        "# Re-classify:",
        "python scripts/classify_corpus_v2_16_p0.py output/{Bevestigingsmiddelen,ATZ_Aerodynamik_Nutzfahrzeugen,ATZ_ESF_Mercedes_2009,Schwungradspeicher,Eliasz_Zephyr_RTOS,Grundlagen_Fahrzeug_Motorentechnik,Digitale_Fotografie_Feb_2026}/ingestion.jsonl",
        "",
        "# Regenerate this report:",
        "python scripts/_v2_16_p0_inventory_report.py \\",
        "    --classifier-json output/_v2_16_p0_logs/classify_new7.json \\",
        "    --output docs/archive/misc/CORPUS_EXPANSION_2026-05-25_v2.16_p0.md",
        "```",
    ])
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classifier-json", type=Path, required=True,
                    help="Output of `classify_corpus_v2_16_p0.py --json`")
    ap.add_argument("--output", type=Path, required=True,
                    help="Inventory report path")
    args = ap.parse_args()

    text = args.classifier_json.read_text(encoding="utf-8")
    # The classifier prints the JSON to stdout; strip any leading
    # banner / shell-prompt lines.
    start = text.find("[")
    if start < 0:
        print(f"ERROR: no JSON array found in {args.classifier_json}", file=sys.stderr)
        return 2
    rows = json.loads(text[start:])
    md = render(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"Wrote {args.output}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
