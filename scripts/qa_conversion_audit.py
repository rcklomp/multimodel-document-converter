#!/usr/bin/env python3
"""
Automated conversion quality audit — replaces manual Gemini review.

Runs structural, image, and text quality checks on an ingestion.jsonl
and produces a pass/fail report with actionable diagnostics.

Usage:
    python scripts/qa_conversion_audit.py output/Combat_Aircraft_August_2025/ingestion.jsonl
    python scripts/qa_conversion_audit.py output/*/ingestion.jsonl   # audit all

Exit codes:
    0 — AUDIT_PASS (all checks green)
    1 — AUDIT_FAIL (one or more checks failed)
    2 — usage / file error
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Current pipeline version for provenance comparison
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from mmrag_v2.version import __schema_version__ as CURRENT_VERSION
except ImportError:
    CURRENT_VERSION = "unknown"
from typing import Optional


@dataclass
class AuditResult:
    source_file: str = ""
    profile_type: str = ""
    schema_version: str = ""
    total_chunks: int = 0
    text_chunks: int = 0
    image_chunks: int = 0
    table_chunks: int = 0
    total_assets: int = 0

    # Structural checks
    null_chunk_type: int = 0
    invalid_bbox: int = 0
    empty_text: int = 0
    missing_modality: int = 0

    # Image quality
    blank_images: int = 0
    low_detail_images: int = 0
    thin_strip_images: int = 0
    tiny_images: int = 0
    missing_asset_files: int = 0

    # Text quality
    encoding_artifacts: int = 0       # /C211, /uniFB01, \xc3
    high_corruption_chunks: int = 0   # corruption_score > 0.5
    oversize_chunks: int = 0          # > 1500 chars
    micro_chunks: int = 0             # < 30 chars (non-label)
    empty_refined: int = 0            # refined_content is null/empty

    # Heading / hierarchy quality
    text_with_heading: int = 0        # text chunks that have a parent_heading
    text_without_heading: int = 0     # text chunks with null parent_heading
    long_headings: int = 0            # parent_heading > 80 chars (misclassified)
    shallow_breadcrumbs: int = 0      # breadcrumb has only [doc, page] — no TOC hierarchy
    multi_sentence_headings: int = 0  # headings with > 1 sentence (". " count)
    unique_headings: int = 0          # count of distinct parent_heading values
    heading_fragmentation: float = 0.0  # unique_headings / text_chunks — high = noisy
    suspicious_headings: int = 0      # headings matching non-heading patterns

    # Schema consistency
    version_mismatches: int = 0
    wrong_version: str = ""

    # Diagnostics
    issues: list = field(default_factory=list)

    def add_issue(self, severity: str, msg: str):
        if len(self.issues) < 20:
            self.issues.append(f"[{severity}] {msg}")


def audit(jsonl_path: Path) -> AuditResult:
    r = AuditResult()
    assets_dir = jsonl_path.parent / "assets"

    # Count actual asset files
    if assets_dir.exists():
        r.total_assets = len([f for f in assets_dir.iterdir() if f.suffix == ".png"])

    versions = set()
    asset_refs = []
    all_headings_seen: set = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)

            # Metadata record
            if obj.get("object_type") == "ingestion_metadata":
                r.source_file = obj.get("source_file", "")
                r.profile_type = obj.get("profile_type", "unknown")
                r.schema_version = obj.get("schema_version", "")
                # Provenance checks
                pv = obj.get("pipeline_version", "")
                if pv and pv != CURRENT_VERSION:
                    r.add_issue("PROVENANCE", f"pipeline_version={pv} != current {CURRENT_VERSION}")
                sv = obj.get("schema_version", "")
                if sv and sv != CURRENT_VERSION:
                    r.add_issue("PROVENANCE", f"schema_version={sv} != current {CURRENT_VERSION}")
                if not obj.get("source_file_hash"):
                    r.add_issue("PROVENANCE", "missing source_file_hash")
                continue

            r.total_chunks += 1
            modality = obj.get("modality")
            meta = obj.get("metadata") or {}

            if not modality:
                r.missing_modality += 1
                continue

            if modality == "text":
                r.text_chunks += 1
            elif modality == "image":
                r.image_chunks += 1
            elif modality == "table":
                r.table_chunks += 1

            # Track schema versions
            sv = obj.get("schema_version", "")
            if sv:
                versions.add(sv)

            # Structural: null chunk_type on text
            if modality == "text" and not meta.get("chunk_type"):
                r.null_chunk_type += 1
                r.add_issue("STRUCT", f"pg{meta.get('page_number')}: text chunk missing chunk_type")

            # Structural: empty text content
            if modality == "text":
                content = (obj.get("content") or "").strip()
                if not content:
                    r.empty_text += 1

                # Text quality: encoding artifacts
                import re
                artifact_pat = re.compile(
                    r"/C\d{1,3}|/uni[A-F0-9]{4,}|\\x[0-9a-f]{2}|\ufffd"
                )
                if artifact_pat.search(content):
                    r.encoding_artifacts += 1

                # Text quality: oversize/micro
                if len(content) > 1500:
                    r.oversize_chunks += 1
                if len(content) < 30:
                    r.micro_chunks += 1

                # Corruption score
                cs = meta.get("corruption_score")
                if cs is not None and cs > 0.5:
                    r.high_corruption_chunks += 1

                # Refined content
                if meta.get("refined_content") is None:
                    r.empty_refined += 1

            # Heading / hierarchy quality
            if modality == "text":
                hierarchy = meta.get("hierarchy") or {}
                ph = hierarchy.get("parent_heading")
                bp = hierarchy.get("breadcrumb_path") or []

                if ph:
                    r.text_with_heading += 1
                    all_headings_seen.add(ph)
                    if len(ph) > 80:
                        r.long_headings += 1
                        r.add_issue("HEADING", f"pg{meta.get('page_number')}: heading > 80 chars: \"{ph[:60]}...\"")
                    if ph.count(". ") > 1 or ph.count(".\n") > 1:
                        r.multi_sentence_headings += 1
                        r.add_issue("HEADING", f"pg{meta.get('page_number')}: multi-sentence heading: \"{ph[:60]}...\"")
                    # Suspicious: structural markers misclassified as headings
                    import re as _re_sh
                    _ph_lower = ph.lower().strip()
                    if _re_sh.match(
                        r"^(this chapter covers|\(continued\)|listing \d[\d.]*\b"
                        r"|figure \d[\d.]*\b|table \d[\d.]*\b"
                        r"|example \d[\d.-]*\.)",  # "Example 4-20." not "Example 2: title"
                        _ph_lower
                    ):
                        r.suspicious_headings += 1
                else:
                    r.text_without_heading += 1

                # Shallow breadcrumb: only [doc_name, Page X] — no structural hierarchy
                if len(bp) <= 2:
                    r.shallow_breadcrumbs += 1

            # Structural: bbox validation
            spatial = meta.get("spatial") or {}
            bbox = spatial.get("bbox")
            if bbox and isinstance(bbox, list):
                for val in bbox:
                    if not isinstance(val, int) or val < 0 or val > 1000:
                        r.invalid_bbox += 1
                        break

            # Track asset refs for file check
            asset_ref = obj.get("asset_ref") or {}
            fp = asset_ref.get("file_path", "")
            if fp:
                asset_refs.append(fp)

    # Heading fragmentation: unique headings / text chunks
    r.unique_headings = len(all_headings_seen)
    if r.text_chunks > 0:
        r.heading_fragmentation = r.unique_headings / r.text_chunks

    # Check asset files exist
    for ref in asset_refs:
        asset_path = jsonl_path.parent / ref
        if not asset_path.exists():
            r.missing_asset_files += 1
            r.add_issue("ASSET", f"Missing: {ref}")

    # Image quality analysis (pixel-level)
    if assets_dir.exists() and r.total_assets > 0:
        try:
            from PIL import Image
            import numpy as np

            for f in sorted(assets_dir.iterdir()):
                if f.suffix != ".png":
                    continue
                try:
                    img = Image.open(f)
                    w, h = img.size
                    arr = np.array(img, dtype=np.float32)
                    mean_val = arr.mean()
                    std_val = arr.std()
                    aspect = max(w, h) / max(min(w, h), 1)
                    area = w * h

                    # Truly blank: near-uniform white/black. std < 8 avoids
                    # flagging light-background diagrams (std 12-20).
                    if (mean_val > 250 and std_val < 8) or (mean_val < 5 and std_val < 8):
                        r.blank_images += 1
                        r.add_issue("IMAGE", f"Blank: {f.name} (mean={mean_val:.0f}, std={std_val:.1f})")

                    # Aspect > 25 catches true structural lines (1px rules,
                    # header bars) without flagging wide flow diagrams (aspect 10-20).
                    if aspect > 25:
                        r.thin_strip_images += 1
                        r.add_issue("IMAGE", f"Thin strip: {f.name} ({w}x{h}, aspect={aspect:.0f})")

                    if area < 5000:
                        r.tiny_images += 1

                except Exception:
                    pass
        except ImportError:
            r.add_issue("WARN", "PIL/numpy not available — skipping image quality checks")

    # Version consistency
    if len(versions) > 1:
        r.version_mismatches = len(versions)
        r.add_issue("SCHEMA", f"Multiple schema versions: {versions}")
    if r.schema_version and versions and r.schema_version not in versions:
        r.wrong_version = f"metadata={r.schema_version}, chunks={versions}"
        r.add_issue("SCHEMA", f"Version mismatch: {r.wrong_version}")

    return r


def print_report(r: AuditResult, path: Path) -> bool:
    """Print audit report. Returns True if passed."""
    print(f"{'='*60}")
    print(f"AUDIT: {r.source_file or path.parent.name}")
    print(f"{'='*60}")
    print(f"  Profile: {r.profile_type}  Schema: {r.schema_version}")
    print(f"  Chunks: {r.total_chunks} (text={r.text_chunks}, image={r.image_chunks}, table={r.table_chunks})")
    print(f"  Assets: {r.total_assets}")
    print()

    # Structural
    struct_ok = (r.null_chunk_type == 0 and r.invalid_bbox == 0
                 and r.empty_text == 0 and r.missing_modality == 0)
    print(f"  STRUCTURAL:  {'PASS' if struct_ok else 'FAIL'}")
    if not struct_ok:
        if r.null_chunk_type: print(f"    null_chunk_type: {r.null_chunk_type}")
        if r.invalid_bbox: print(f"    invalid_bbox: {r.invalid_bbox}")
        if r.empty_text: print(f"    empty_text: {r.empty_text}")
        if r.missing_modality: print(f"    missing_modality: {r.missing_modality}")

    # Image quality
    img_ok = r.blank_images == 0 and r.missing_asset_files == 0 and r.thin_strip_images == 0
    print(f"  IMAGE:       {'PASS' if img_ok else 'FAIL'}")
    if not img_ok:
        if r.blank_images: print(f"    blank_images: {r.blank_images}")
        if r.thin_strip_images: print(f"    thin_strips: {r.thin_strip_images}")
        if r.missing_asset_files: print(f"    missing_files: {r.missing_asset_files}")
    if r.tiny_images:
        print(f"    tiny_images: {r.tiny_images} (advisory)")

    # Text quality
    text_ok = r.encoding_artifacts == 0 and r.oversize_chunks == 0
    print(f"  TEXT:         {'PASS' if text_ok else 'WARN' if r.encoding_artifacts < 5 else 'FAIL'}")
    if r.encoding_artifacts: print(f"    encoding_artifacts: {r.encoding_artifacts}")
    if r.high_corruption_chunks: print(f"    high_corruption: {r.high_corruption_chunks}")
    if r.oversize_chunks: print(f"    oversize_>1500: {r.oversize_chunks}")
    if r.micro_chunks: print(f"    micro_<30: {r.micro_chunks} (advisory)")

    # Heading quality
    heading_coverage = r.text_with_heading / max(r.text_chunks, 1)
    heading_ok = (
        r.long_headings == 0
        and r.multi_sentence_headings == 0
        and heading_coverage >= 0.80
        and r.suspicious_headings == 0
    )
    if heading_coverage >= 0.90:
        cov_label = "PASS"
    elif heading_coverage >= 0.70:
        cov_label = "WARN"
    else:
        cov_label = "FAIL"
    frag_label = "OK" if r.heading_fragmentation <= 0.40 else "HIGH"
    heading_label = "PASS" if heading_ok else "FAIL"
    print(f"  HEADING:     {heading_label}")
    print(f"    coverage: {r.text_with_heading}/{r.text_chunks} ({heading_coverage:.0%}) [{cov_label}]")
    print(f"    unique headings: {r.unique_headings} (fragmentation: {r.heading_fragmentation:.0%}) [{frag_label}]")
    if r.text_without_heading:
        print(f"    null_headings: {r.text_without_heading}")
    if r.long_headings:
        print(f"    long_headings (>80): {r.long_headings}")
    if r.multi_sentence_headings:
        print(f"    multi_sentence: {r.multi_sentence_headings}")
    if r.suspicious_headings:
        print(f"    suspicious_headings: {r.suspicious_headings}")
    if r.shallow_breadcrumbs:
        print(f"    shallow_breadcrumbs: {r.shallow_breadcrumbs} (advisory)")

    # Schema
    schema_ok = r.version_mismatches == 0
    print(f"  SCHEMA:      {'PASS' if schema_ok else 'FAIL'}")
    if not schema_ok:
        print(f"    version_mismatches: {r.version_mismatches}")

    # Issues
    if r.issues:
        print(f"\n  Issues ({len(r.issues)}):")
        for issue in r.issues[:10]:
            print(f"    {issue}")
        if len(r.issues) > 10:
            print(f"    ... and {len(r.issues) - 10} more")

    # Overall
    hard_fail = (
        not struct_ok
        or r.blank_images > 0
        or r.missing_asset_files > 0
        or not schema_ok
        or r.long_headings > 0
        or r.multi_sentence_headings > 0
        or r.suspicious_headings > 0
    )
    passed = not hard_fail
    print(f"\n  {'AUDIT_PASS' if passed else 'AUDIT_FAIL'}")
    print()
    return passed


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: qa_conversion_audit.py path/to/ingestion.jsonl [...]", file=sys.stderr)
        return 2

    paths = []
    for arg in argv[1:]:
        p = Path(arg)
        if p.exists():
            paths.append(p)
        else:
            print(f"warning: {arg} not found, skipping", file=sys.stderr)

    if not paths:
        return 2

    all_passed = True
    results = []

    for path in paths:
        r = audit(path)
        passed = print_report(r, path)
        results.append((path, r, passed))
        if not passed:
            all_passed = False

    # Summary table for multi-file runs
    if len(paths) > 1:
        print(f"{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for path, r, passed in results:
            name = path.parent.name
            status = "PASS" if passed else "FAIL"
            printf_str = f"  {'✓' if passed else '✗'} {name:45s} {r.total_chunks:5d} chunks  {status}"
            print(printf_str)
        total = len(results)
        passed_count = sum(1 for _, _, p in results if p)
        print(f"\n  {passed_count}/{total} passed")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
