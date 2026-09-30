#!/usr/bin/env python3
"""Build the v2.8 AFTER quality snapshot from a fresh audit log.

Reads the SUMMARY block emitted by qa_conversion_audit.py at the end of
a multi-file audit, parses each row, and writes a markdown snapshot doc
that compares against the BEFORE snapshot (QUALITY_SNAPSHOT_2026-05-03.md).

Usage:
    python scripts/v28_build_after_snapshot.py <audit_log> <output_md>
"""
from __future__ import annotations

import re
import subprocess
import sys
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BEFORE_SNAPSHOT = REPO_ROOT / "docs" / "QUALITY_SNAPSHOT_2026-05-03.md"
CONVERT_SCRIPT = REPO_ROOT / "scripts" / "convert_books.sh"


def canonical_output_names() -> set[str]:
    """Read the v2.8 corpus output names from convert_books.sh.

    Canonical name is the second `convert "..."` argument. Anything in
    output/ that isn't on this list is a legacy probe / exploration
    artifact and is excluded from the v2.8 aggregate.
    """
    if not CONVERT_SCRIPT.exists():
        return set()
    text = CONVERT_SCRIPT.read_text(encoding="utf-8")
    out = set()
    # Match: convert "src" "name" optional_batch
    for m in re.finditer(r'^convert\s+"[^"]+"\s+"([^"]+)"', text, re.MULTILINE):
        out.add(m.group(1))
    return out


# Format produced by qa_conversion_audit.py SUMMARY block:
#   ✓ Doc_Name                                       N chunks  digital   PASS
#   ✗ Doc_Name                                       N chunks  digital   FAIL
#   ✓ Form_Name                                      N chunks  scanned   FORM_PASS
SUMMARY_ROW = re.compile(
    r"^\s+(?P<icon>[✓✗◻])\s+(?P<name>\S(?:.*?\S)?)\s{2,}(?P<chunks>\d+)\s+chunks\s+(?P<doc_class>\S+)\s+(?P<status>\S+)\s*$"
)


def parse_audit_summary(audit_log: Path) -> list[dict]:
    """Extract rows from the SUMMARY block at the end of an audit log."""
    rows = []
    in_summary = False
    for line in audit_log.read_text(encoding="utf-8").splitlines():
        if line.strip() == "SUMMARY":
            in_summary = True
            continue
        if not in_summary:
            continue
        m = SUMMARY_ROW.match(line)
        if m:
            rows.append(m.groupdict())
    return rows


def parse_before_summary(before_md: Path) -> dict[str, str]:
    """Parse BEFORE table rows: output_dir -> audit verdict label."""
    text = before_md.read_text(encoding="utf-8")
    # Crude markdown-table row parser. The BEFORE table uses pipe-separated
    # columns where col 0 is the output dir and col 4 is the audit verdict.
    out = {}
    in_table = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("| Output dir") or s.startswith("| Output dir |"):
            in_table = True
            continue
        if in_table:
            if not s.startswith("|"):
                in_table = False
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            if len(cells) >= 4 and cells[0] and cells[0] not in {"---", "Output dir"}:
                # cells[3] is "Audit"
                # The header had "| Output dir | Source PDF | Date | Audit | Failing checks (BEFORE) | v2.8 phase |"
                key = cells[0].strip("*")
                verdict = cells[3]
                if "PASS" in verdict.upper() or "FAIL" in verdict.upper():
                    out[key] = verdict
    return out


def short_status(s: str) -> str:
    if s == "PASS":
        return "PASS"
    if s == "FORM_PASS":
        return "FORM_PASS"
    if s == "FORM_FAIL":
        return "FORM_FAIL"
    if s == "FAIL":
        return "FAIL"
    if s == "UNSUPPORTED":
        return "UNSUPPORTED"
    return s


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: v28_build_after_snapshot.py <audit_log> <output_md>", file=sys.stderr)
        return 2
    audit_log = Path(argv[1])
    out_md = Path(argv[2])

    if not audit_log.exists():
        print(f"audit log not found: {audit_log}", file=sys.stderr)
        return 2

    rows = parse_audit_summary(audit_log)
    if not rows:
        print(f"no SUMMARY rows parsed from {audit_log}", file=sys.stderr)
        return 2

    before = parse_before_summary(BEFORE_SNAPSHOT) if BEFORE_SNAPSHOT.exists() else {}
    canonical = canonical_output_names()

    # Split: canonical v2.8 corpus rows vs. legacy probes / exploration runs.
    # Aggregate stats apply only to canonical; probes are reported separately.
    canonical_rows = [r for r in rows if r["name"] in canonical]
    legacy_rows = [r for r in rows if r["name"] not in canonical]

    today = date.today().isoformat()

    pass_count = sum(1 for r in canonical_rows if r["status"] in {"PASS", "FORM_PASS"})
    fail_count = sum(1 for r in canonical_rows if r["status"] in {"FAIL", "FORM_FAIL"})
    form_count = sum(1 for r in canonical_rows if r["status"].startswith("FORM"))

    # Recent commit context
    try:
        head = subprocess.run(
            ["git", "log", "-1", "--format=%h %s"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        ).stdout.strip()
    except Exception:
        head = "(unknown)"

    md = []
    md.append(f"# Quality Snapshot {today} — v2.8 AFTER\n")
    md.append("**Purpose:** AFTER state for the v2.8 broad reconversion (Phase 5c).")
    md.append("Compare against `docs/QUALITY_SNAPSHOT_2026-05-03.md` for the BEFORE column.\n")
    md.append(f"**HEAD:** `{head}`\n")
    md.append("**Pre-flight evidence (committed in `5b0e13d`):**")
    md.append("- pytest tests/ -q: 590 passed, 2 skipped, 0 failed.")
    md.append("- bash scripts/smoke_multiprofile.sh: 11/11 GATE_PASS + 11/11 UNIVERSAL_PASS.")
    md.append("- HARRY pages-1-30 live acceptance: PASS.\n")

    md.append("## Aggregate (v2.8 canonical corpus only)")
    md.append(f"- **{pass_count}/{len(canonical_rows)} PASS** (includes form-pass class)")
    md.append(f"- **{fail_count} FAIL**")
    md.append(f"- **{form_count} forms** (form acceptance class — invoices/short scanned docs)")
    md.append(f"- {len(legacy_rows)} legacy probes / exploration outputs (table 2)\n")

    def _row(r: dict) -> str:
        name = r["name"]
        before_v = before.get(name, "—")
        after_v = short_status(r["status"])
        if before_v == "—":
            delta = "NEW"
        elif "PASS" in before_v.upper() and "PASS" in after_v.upper():
            delta = "✓ stable"
        elif "FAIL" in before_v.upper() and "PASS" in after_v.upper():
            delta = "✓ FIXED"
        elif "PASS" in before_v.upper() and "FAIL" in after_v.upper():
            delta = "⚠ REGRESSION"
        elif "FAIL" in before_v.upper() and "FAIL" in after_v.upper():
            delta = "still FAIL"
        else:
            delta = "—"
        return f"| {name} | {r['chunks']} | {r['doc_class']} | {after_v} | {before_v} | {delta} |"

    md.append("## Per-document AFTER (canonical v2.8 corpus)\n")
    md.append("| Output dir | Chunks | Class | AFTER | BEFORE | Delta |")
    md.append("|---|---|---|---|---|---|")
    for r in sorted(canonical_rows, key=lambda x: x["name"].lower()):
        md.append(_row(r))
    md.append("")

    if legacy_rows:
        md.append("## Legacy / probe outputs (informational)\n")
        md.append("These predate or sit outside the v2.8 canonical corpus")
        md.append("(probes, partial-page reconverts, pre-fix _codex/_promptfix/_vlm runs).")
        md.append("Not counted in the aggregate above.\n")
        md.append("| Output dir | Chunks | Class | AFTER | BEFORE | Delta |")
        md.append("|---|---|---|---|---|---|")
        for r in sorted(legacy_rows, key=lambda x: x["name"].lower()):
            md.append(_row(r))
        md.append("")

    md.append("## Phase 5c gating decisions\n")
    md.append("**Conversion flags used** (per scripts/convert_books.sh):")
    md.append("```")
    md.append("python -m mmrag_v2.cli process <pdf> -o <out> -b <batch> \\")
    md.append("  --vision-provider none --no-refiner --no-cache")
    md.append("```")
    md.append("Matches the Phase 0 BEFORE baseline so the delta column above is")
    md.append("apples-to-apples and isolates v2.8 code changes.\n")

    md.append("**Refiner smart-routing** (deferred to v2.9):")
    md.append("`cli.py:686` enables refiner whenever `~/.mmrag-v2.yml`")
    md.append("`refiner.enabled=true`, regardless of `has_encoding_corruption`.")
    md.append("This caused HARRY (clean prose, zero corruption) to hammer")
    md.append("qwen-plus during the first broad-reconversion attempt. Fix is")
    md.append("v2.9 scope: gate the config-default enable on the diagnostic")
    md.append("just like the explicit auto-override at `cli.py:1101` does.")
    md.append("After the fix, the broad reconversion can re-run without flags")
    md.append("and produce the same output for clean docs while still")
    md.append("auto-enabling refiner on encoding-corrupt ones.\n")

    md.append("## Qdrant re-ingestion — DEFERRED\n")
    md.append("The existing `multimodal-doc-converter-qdrant` Docling container")
    md.append("fails to start because of a stale collection lock:")
    md.append("```")
    md.append("Panic: Can't read collection version: Resource deadlock avoided")
    md.append("path: ./storage/collections/sekar_s__the_mcp_standard_.../version.info")
    md.append("```")
    md.append("The lock is held by a previous interrupted ingest run. Recovery")
    md.append("requires user input — clear the stale lock file or recreate the")
    md.append("container with a clean storage volume. Side-by-side ingest into")
    md.append("a new `mmrag_v2_8` collection is safe to do once the container")
    md.append("starts.\n")
    md.append("**Runbook (run in the morning):**")
    md.append("```bash")
    md.append("# 1. Inspect the broken container")
    md.append("docker logs multimodal-doc-converter-qdrant 2>&1 | tail -20")
    md.append("")
    md.append("# 2. Choose recovery — pick ONE:")
    md.append("# (a) Surgical: clear only the stale lock")
    md.append("docker run --rm -v multimodal-doc-converter_qdrant_storage:/q alpine \\")
    md.append("  rm -rf /q/collections/sekar_s__the_mcp_standard__a_developer_s_guide__building_universal_ai_tools_2026_pdf")
    md.append("# (b) Nuclear: drop the storage volume entirely (loses all collections)")
    md.append("docker rm -f multimodal-doc-converter-qdrant")
    md.append("docker volume rm multimodal-doc-converter_qdrant_storage")
    md.append("")
    md.append("# 3. Start Qdrant (whichever recovery path)")
    md.append("docker start multimodal-doc-converter-qdrant   # if (a)")
    md.append("# OR re-create per project setup if (b)")
    md.append("")
    md.append("# 4. Start Ollama (embedding backend)")
    md.append("open -a Ollama   # or: ollama serve &")
    md.append("")
    md.append("# 5. Side-by-side ingest into mmrag_v2_8")
    md.append("for d in output/*/; do")
    md.append("  jsonl=\"$d/ingestion.jsonl\"")
    md.append("  [ -f \"$jsonl\" ] || continue")
    md.append("  python scripts/ingest_to_qdrant.py \"$jsonl\" --collection mmrag_v2_8")
    md.append("done")
    md.append("")
    md.append("# 6. Verify chunk counts match JSONL line counts (no silent drops)")
    md.append("# 7. Tag v2.8.0 once verification passes")
    md.append("git tag v2.8.0")
    md.append("git push origin v2.8.0   # only if pushing is desired")
    md.append("```")

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out_md} ({pass_count}/{len(rows)} pass, {fail_count} fail)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
