#!/usr/bin/env python3
"""
Automated conversion quality audit — replaces manual Gemini review.

Runs structural, image, text, code, heading, and semantic quality checks
on an ingestion.jsonl and produces a pass/fail report with actionable
diagnostics.

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
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Current pipeline version for provenance comparison
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from mmrag_v2.version import __schema_version__ as CURRENT_VERSION
except ImportError:
    CURRENT_VERSION = "unknown"


# ---------------------------------------------------------------------------
# Patterns (ported from qa_ingestion_hygiene, evaluate_technical_manual_gates,
# qa_semantic_fidelity)
# ---------------------------------------------------------------------------

_PAGE_NUM_LINE = re.compile(r"(?m)^\s*\d{1,4}\s*$")

_LABEL_LIKE = re.compile(r"^(?:\d[\d.]*\s+)?[A-Z][A-Za-z0-9/&()' .,-]{1,55}:?$")

_CODE_SIG = re.compile(
    r"(?m)^\s*(def|class|import|from|return|yield|if\s+__name__|async\s+def)\b"
)
_CODE_INLINE = re.compile(
    r"(```|::|\bdef\s+\w+\(|\bclass\s+\w+|"
    r"\bimport\s+\w+|\bfrom\s+\w+\s+import\b|"
    r"[{}<>]=?|==|!=|:=|->|=>|\breturn\b)"
)

_INFIX_RE = re.compile(
    r"(?<![\n\r])(?<!^)"
    r"\b(?P<prev>[a-z][a-z'\-]{0,24})\s+"
    r"(?P<num>(?:[1-9]|[12]\d|3\d|40))\.\s+"
    r"(?P<next>[a-z][A-Za-z'\-]*)"
)

_ARTIFACT_PAT = re.compile(
    r"/C\d{1,3}|/uni[A-F0-9]{4,}|\\x[0-9a-f]{2}|\ufffd"
)

_SUSPICIOUS_HEADING = re.compile(
    r"^(this chapter covers|\(continued\)|listing \d[\d.]*\b"
    r"|figure \d[\d.]*\b|table \d[\d.]*\b"
    r"|example \d[\d.-]*\.)"
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _has_ctrl(s: str) -> bool:
    return any((ord(c) < 32 and c not in ("\n", "\t")) or ord(c) == 127 for c in s)


def _ctrl_count(s: str) -> int:
    return sum(1 for c in s if ((ord(c) < 32 and c not in ("\n", "\t")) or ord(c) == 127))


def _is_label_like(text: str) -> bool:
    s = (text or "").strip()
    if not s or len(s) > 60 or "\n" in s:
        return False
    if _PAGE_NUM_LINE.fullmatch(s):
        return False
    if not _LABEL_LIKE.match(s):
        return False
    # TOC entries end with a page number — they're complete as-is, not orphan labels
    if re.search(r"\s\d{1,4}$", s):
        return False
    if ":" in s and not s.endswith(":"):
        return False
    if ":" not in s:
        words = [w for w in re.split(r"\s+", s) if w]
        if len(words) > 6:
            return False
        if any(w.endswith((".", "?", "!")) for w in words):
            return False
    return True


def _is_code_chunk(meta: dict) -> bool:
    ct = str((meta or {}).get("chunk_type") or "").lower()
    cc = str((meta or {}).get("content_classification") or "").lower()
    return ct == "code" or cc == "code"


def _looks_code_like(text: str) -> bool:
    s = text or ""
    if not s.strip():
        return False
    if _CODE_SIG.search(s):
        return True
    return bool(_CODE_INLINE.search(s))


def _count_infix_artifacts(text: str) -> int:
    n = 0
    for m in _INFIX_RE.finditer(text or ""):
        prev = m.group("prev")
        nxt = m.group("next")
        start = m.start()
        left = (text or "")[max(0, start - 2):start]
        if left.endswith(("\n", "\r", ". ", ": ", "; ", "! ", "? ")):
            continue
        between = (text or "")[m.start("prev"):m.start("num")]
        if "\n" in between:
            continue
        if len(prev) <= 1 or len(nxt) <= 1:
            continue
        if prev in ("bis", "to", "from", "through", "vom", "von", "and", "or"):
            continue
        if nxt in ("bis", "to", "through"):
            continue
        n += 1
    return n


def _is_placeholder(content: str) -> bool:
    t = (content or "").strip().lower()
    if not t:
        return True
    if "extraction unavailable" in t:
        return True
    if re.match(r"^\[(figure|image|table)\b", t):
        return True
    if t.startswith("[vlm_failed"):
        return True
    if len(t) < 80 and ("figure on page" in t or "table on page" in t):
        return True
    return False


def _is_markdown_table(content: str) -> bool:
    t = (content or "").strip()
    if not t:
        return False
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    if "|" not in lines[0]:
        return False
    return any(re.search(r"\|\s*-{2,}", ln) for ln in lines[1:3])


def _infer_doc_class(profile_type: str, chunks: list) -> str:
    """Infer digital vs scanned from metadata."""
    digital_modalities = {"native_digital", "image_heavy"}
    modality_counts: Counter = Counter()
    for obj in chunks:
        md = obj.get("metadata") or {}
        modality = str(md.get("document_modality") or "").strip().lower()
        if modality:
            modality_counts[modality] += 1
    if modality_counts:
        top = modality_counts.most_common(1)[0][0]
        if top.startswith("scanned"):
            return "scanned"
        if top in digital_modalities:
            return "digital"
        scanned_votes = sum(c for m, c in modality_counts.items() if m.startswith("scanned"))
        digital_votes = sum(c for m, c in modality_counts.items() if m in digital_modalities)
        if scanned_votes > digital_votes:
            return "scanned"
    return "digital"


# ---------------------------------------------------------------------------
# AuditResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class AuditResult:
    source_file: str = ""
    profile_type: str = ""
    schema_version: str = ""
    doc_class: str = ""  # "digital" or "scanned"
    total_chunks: int = 0
    text_chunks: int = 0
    image_chunks: int = 0
    table_chunks: int = 0
    total_assets: int = 0

    # Structural checks
    null_chunk_type: int = 0
    invalid_bbox: int = 0
    bbox_missing_dims: int = 0
    empty_text: int = 0
    missing_modality: int = 0

    # Image quality
    blank_images: int = 0
    low_detail_images: int = 0
    thin_strip_images: int = 0
    tiny_images: int = 0
    missing_asset_files: int = 0
    image_placeholders: int = 0
    image_with_description: int = 0

    # Table quality
    table_placeholders: int = 0
    table_markdown: int = 0

    # Text quality
    encoding_artifacts: int = 0
    high_corruption_chunks: int = 0
    oversize_chunks: int = 0
    micro_chunks: int = 0
    micro_non_label: int = 0
    empty_refined: int = 0
    ctrl_chunks: int = 0
    ctrl_total: int = 0
    page_num_artifacts: int = 0
    infix_artifacts: int = 0

    # Code quality
    code_chunks: int = 0
    code_flat: int = 0
    code_with_indent: int = 0
    code_like_total: int = 0
    code_fragment_micro: int = 0

    # Label quality
    label_chunks: int = 0
    orphan_labels: int = 0

    # Heading / hierarchy quality
    text_with_heading: int = 0
    text_without_heading: int = 0
    long_headings: int = 0
    shallow_breadcrumbs: int = 0
    multi_sentence_headings: int = 0
    unique_headings: int = 0
    heading_fragmentation: float = 0.0
    suspicious_headings: int = 0

    # Schema consistency
    version_mismatches: int = 0
    wrong_version: str = ""

    # Diagnostics
    issues: list = field(default_factory=list)

    def add_issue(self, severity: str, msg: str):
        if len(self.issues) < 30:
            self.issues.append(f"[{severity}] {msg}")


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------

def audit(jsonl_path: Path) -> AuditResult:
    r = AuditResult()
    assets_dir = jsonl_path.parent / "assets"

    if assets_dir.exists():
        r.total_assets = len([f for f in assets_dir.iterdir() if f.suffix == ".png"])

    versions: set = set()
    asset_refs: list = []
    all_headings_seen: set = set()
    all_chunks: list = []

    # Collect text rows for multi-pass analysis (orphan labels, cross-page)
    text_rows: List[Tuple[str, int, str, dict]] = []

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
                pv = obj.get("pipeline_version", "")
                if pv and pv != CURRENT_VERSION:
                    r.add_issue("PROVENANCE", f"pipeline_version={pv} != current {CURRENT_VERSION}")
                sv = obj.get("schema_version", "")
                if sv and sv != CURRENT_VERSION:
                    r.add_issue("PROVENANCE", f"schema_version={sv} != current {CURRENT_VERSION}")
                if not obj.get("source_file_hash"):
                    r.add_issue("PROVENANCE", "missing source_file_hash")
                continue

            all_chunks.append(obj)
            r.total_chunks += 1
            modality = obj.get("modality")
            meta = obj.get("metadata") or {}

            if not modality:
                r.missing_modality += 1
                continue

            content = (obj.get("content") or "").strip()

            # --- Modality-specific counting ---
            if modality == "text":
                r.text_chunks += 1
            elif modality == "image":
                r.image_chunks += 1
                # Image description coverage
                if _is_placeholder(content):
                    r.image_placeholders += 1
                vd = obj.get("visual_description") or (meta.get("visual_description"))
                if vd:
                    r.image_with_description += 1
            elif modality == "table":
                r.table_chunks += 1
                if _is_placeholder(content):
                    r.table_placeholders += 1
                if _is_markdown_table(content):
                    r.table_markdown += 1

            # Track schema versions
            sv = obj.get("schema_version", "")
            if sv:
                versions.add(sv)

            # --- Structural: bbox validation ---
            spatial = meta.get("spatial") or {}
            bbox = spatial.get("bbox")
            if bbox and isinstance(bbox, list):
                for val in bbox:
                    if not isinstance(val, int) or val < 0 or val > 1000:
                        r.invalid_bbox += 1
                        break
                # Missing page dimensions with bbox present
                if not spatial.get("page_width") or not spatial.get("page_height"):
                    r.bbox_missing_dims += 1

            # Track asset refs
            asset_ref = obj.get("asset_ref") or {}
            fp = asset_ref.get("file_path", "")
            if fp:
                asset_refs.append(fp)

            # --- Text-specific checks ---
            if modality != "text":
                continue

            chunk_id = str(obj.get("chunk_id") or "")
            page_number = int(meta.get("page_number") or 0)
            text_rows.append((chunk_id, page_number, content, meta))

            is_code = _is_code_chunk(meta)
            is_label = _is_label_like(content)
            code_like = _looks_code_like(content)

            # Structural: null chunk_type
            if not meta.get("chunk_type"):
                r.null_chunk_type += 1
                r.add_issue("STRUCT", f"pg{page_number}: text chunk missing chunk_type")

            # Empty text
            if not content:
                r.empty_text += 1

            # Encoding artifacts
            if _ARTIFACT_PAT.search(content):
                r.encoding_artifacts += 1

            # Oversize / micro
            if len(content) > 1500:
                r.oversize_chunks += 1
            if len(content) < 30:
                r.micro_chunks += 1
                if not is_label and not is_code:
                    r.micro_non_label += 1

            # Corruption score
            cs = meta.get("corruption_score")
            if cs is not None and cs > 0.5:
                r.high_corruption_chunks += 1

            # Refined content
            if meta.get("refined_content") is None:
                r.empty_refined += 1

            # Control characters
            if _has_ctrl(content):
                r.ctrl_chunks += 1
                r.ctrl_total += _ctrl_count(content)
                if r.ctrl_chunks <= 3:
                    r.add_issue("CTRL", f"pg{page_number}: control chars in {chunk_id}")

            # Page number artifacts
            if _PAGE_NUM_LINE.search(content):
                r.page_num_artifacts += 1

            # Infix artifacts (skip code chunks — they legitimately contain "word N. word")
            if not is_code:
                r.infix_artifacts += _count_infix_artifacts(content)

            # Label tracking
            if is_label:
                r.label_chunks += 1

            # Code quality
            if is_code:
                r.code_chunks += 1
                if "\n" not in content:
                    r.code_flat += 1
                lines = [ln for ln in content.splitlines() if ln.strip()]
                has_indent = any(ln.startswith(("    ", "\t")) for ln in lines)
                has_repl = ">>>" in content or re.search(r"(?m)^\s*\.\.\.\s", content) is not None
                if has_indent or has_repl:
                    r.code_with_indent += 1

            if code_like:
                r.code_like_total += 1
                if not is_code and len(content.strip()) < 80:
                    r.code_fragment_micro += 1

            # Heading / hierarchy quality
            hierarchy = meta.get("hierarchy") or {}
            ph = hierarchy.get("parent_heading")
            bp = hierarchy.get("breadcrumb_path") or []

            if ph:
                r.text_with_heading += 1
                all_headings_seen.add(ph)
                if len(ph) > 80:
                    r.long_headings += 1
                    r.add_issue("HEADING", f"pg{page_number}: heading > 80 chars: \"{ph[:60]}...\"")
                if ph.count(". ") > 1 or ph.count(".\n") > 1:
                    r.multi_sentence_headings += 1
                    r.add_issue("HEADING", f"pg{page_number}: multi-sentence heading: \"{ph[:60]}...\"")
                if _SUSPICIOUS_HEADING.match(ph.lower().strip()):
                    r.suspicious_headings += 1
            else:
                r.text_without_heading += 1

            if len(bp) <= 2:
                r.shallow_breadcrumbs += 1

    # --- Multi-pass analysis ---

    # Heading fragmentation
    r.unique_headings = len(all_headings_seen)
    if r.text_chunks > 0:
        r.heading_fragmentation = r.unique_headings / r.text_chunks

    # Orphan label detection: label must be followed by body text within 5 rows
    for i, (_cid, page_no, txt, _meta) in enumerate(text_rows):
        if not _is_label_like(txt):
            continue
        attached = False
        for j in range(i + 1, min(len(text_rows), i + 5)):
            _nid, npg, ntxt, _nmeta = text_rows[j]
            if npg not in (page_no, page_no + 1):
                continue
            if len(ntxt) >= 20 and not _is_label_like(ntxt):
                attached = True
                break
        if not attached:
            r.orphan_labels += 1

    # Doc class inference
    r.doc_class = _infer_doc_class(r.profile_type, all_chunks)

    # Asset file existence check
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

                    if (mean_val > 250 and std_val < 8) or (mean_val < 5 and std_val < 8):
                        r.blank_images += 1
                        r.add_issue("IMAGE", f"Blank: {f.name} (mean={mean_val:.0f}, std={std_val:.1f})")

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


# ---------------------------------------------------------------------------
# Profile-aware thresholds
# ---------------------------------------------------------------------------

def _classify_content_type(r: AuditResult) -> str:
    """Classify document content type for gate selection.

    Returns one of:
        'code_heavy'   — programming book, code is structurally load-bearing
        'mixed_prose'  — technical manual with incidental code
        'non_code'     — magazine, academic paper, scanned form, etc.
    """
    if r.text_chunks == 0:
        return "non_code"
    code_ratio = r.code_chunks / r.text_chunks
    if code_ratio >= 0.15:  # 15%+ of text chunks are code
        return "code_heavy"
    if r.code_chunks >= 3:
        return "mixed_prose"
    return "non_code"


def _get_thresholds(doc_class: str, profile_type: str, content_type: str) -> dict:
    """Return pass/fail thresholds based on document class, profile, and content type.

    Gate levels:
        value    — hard gate, AUDIT_FAIL if exceeded
        None     — metric not gated (still reported)
        'warn'   — reported as warning, does not cause AUDIT_FAIL
    """
    # --- Base thresholds by document class ---
    if doc_class == "scanned":
        th = {
            "micro_non_label_ratio": 0.22,
            "oversize_ratio": 0.02,
            "orphan_label_ratio": 0.30,
            "orphan_label_min_count": 5,
            "code_frag_ratio": None,
            "code_flat_ratio": None,
            "code_indent_fidelity": None,
            "code_indent_gate": None,  # hard / warn / None
            "image_placeholder_ratio": 0.20,
            "image_description_coverage": 0.80,
            "table_placeholder_ratio": 0.20,
            "table_markdown_ratio": 0.80,
        }
    else:
        # digital
        orphan_limit = 0.65 if profile_type == "academic_whitepaper" else 0.25
        micro_limit = 0.22 if profile_type in ("digital_magazine", "academic_whitepaper") else 0.12
        th = {
            "micro_non_label_ratio": micro_limit,
            "oversize_ratio": 0.01,
            "orphan_label_ratio": orphan_limit,
            "orphan_label_min_count": 5,
            "code_frag_ratio": 0.05,
            "code_flat_ratio": 0.35,
            "code_indent_fidelity": 0.90,
            "code_indent_gate": "hard",
            "image_placeholder_ratio": 0.20,
            "image_description_coverage": 0.80,
            "table_placeholder_ratio": 0.20,
            "table_markdown_ratio": 0.80,
        }

    # --- Content-type overrides for code gates ---
    if content_type == "code_heavy":
        # Code is structurally load-bearing — strict gates
        th["code_indent_fidelity"] = 0.90
        th["code_indent_gate"] = "hard"
        th["code_flat_ratio"] = 0.35
        th["code_frag_ratio"] = 0.05
    elif content_type == "mixed_prose":
        # Incidental code — warn but don't block
        th["code_indent_fidelity"] = 0.90
        th["code_indent_gate"] = "warn"
        th["code_flat_ratio"] = 0.35
        th["code_frag_ratio"] = None  # too few code chunks to gate
    else:
        # Non-code — ignore code metrics entirely
        th["code_indent_fidelity"] = None
        th["code_indent_gate"] = None
        th["code_flat_ratio"] = None
        th["code_frag_ratio"] = None

    return th


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(r: AuditResult, path: Path) -> bool:
    """Print audit report. Returns True if passed."""
    content_type = _classify_content_type(r)
    th = _get_thresholds(r.doc_class, r.profile_type, content_type)

    print(f"{'='*65}")
    print(f"AUDIT: {r.source_file or path.parent.name}")
    print(f"{'='*65}")
    print(f"  Profile: {r.profile_type}  Class: {r.doc_class}  Content: {content_type}  Schema: {r.schema_version}")
    print(f"  Chunks: {r.total_chunks} (text={r.text_chunks}, image={r.image_chunks}, table={r.table_chunks})")
    print(f"  Assets: {r.total_assets}")
    print()

    fails: list = []

    # --- STRUCTURAL ---
    struct_ok = (r.null_chunk_type == 0 and r.invalid_bbox == 0
                 and r.empty_text == 0 and r.missing_modality == 0
                 and r.bbox_missing_dims == 0)
    print(f"  STRUCTURAL:  {'PASS' if struct_ok else 'FAIL'}")
    if not struct_ok:
        if r.null_chunk_type:
            print(f"    null_chunk_type: {r.null_chunk_type}")
        if r.invalid_bbox:
            print(f"    invalid_bbox: {r.invalid_bbox}")
        if r.bbox_missing_dims:
            print(f"    bbox_missing_dims: {r.bbox_missing_dims}")
        if r.empty_text:
            print(f"    empty_text: {r.empty_text}")
        if r.missing_modality:
            print(f"    missing_modality: {r.missing_modality}")
        fails.append("STRUCTURAL")

    # --- IMAGE ---
    img_placeholder_ratio = (r.image_placeholders / r.image_chunks) if r.image_chunks else 0.0
    img_desc_coverage = (r.image_with_description / r.image_chunks) if r.image_chunks else 1.0
    # When ALL images are placeholders, VLM was intentionally disabled — advisory only
    vlm_was_used = r.image_chunks > 0 and img_placeholder_ratio < 1.0
    img_ok = (r.blank_images == 0 and r.missing_asset_files == 0
              and r.thin_strip_images == 0)
    if r.image_chunks > 0 and vlm_was_used:
        if img_placeholder_ratio > th["image_placeholder_ratio"]:
            img_ok = False
        if img_desc_coverage < th["image_description_coverage"]:
            img_ok = False
    vlm_note = "" if vlm_was_used else " (no VLM — advisory)"
    print(f"  IMAGE:       {'PASS' if img_ok else 'FAIL'}{vlm_note}")
    if r.image_chunks > 0:
        print(f"    placeholder_ratio: {img_placeholder_ratio:.2%} (limit {th['image_placeholder_ratio']:.0%}){vlm_note}")
        print(f"    description_coverage: {img_desc_coverage:.2%} (limit {th['image_description_coverage']:.0%})")
    if r.blank_images:
        print(f"    blank_images: {r.blank_images}")
    if r.thin_strip_images:
        print(f"    thin_strips: {r.thin_strip_images}")
    if r.missing_asset_files:
        print(f"    missing_files: {r.missing_asset_files}")
    if r.tiny_images:
        print(f"    tiny_images: {r.tiny_images} (advisory)")
    if not img_ok:
        fails.append("IMAGE")

    # --- TABLE ---
    tbl_placeholder_ratio = (r.table_placeholders / r.table_chunks) if r.table_chunks else 0.0
    tbl_markdown_ratio = (r.table_markdown / r.table_chunks) if r.table_chunks else 1.0
    tbl_ok = True
    if r.table_chunks > 0:
        if tbl_placeholder_ratio > th["table_placeholder_ratio"]:
            tbl_ok = False
        if tbl_markdown_ratio < th["table_markdown_ratio"]:
            tbl_ok = False
    print(f"  TABLE:       {'PASS' if tbl_ok else 'FAIL'}")
    if r.table_chunks > 0:
        print(f"    placeholder_ratio: {tbl_placeholder_ratio:.2%} (limit {th['table_placeholder_ratio']:.0%})")
        print(f"    markdown_ratio: {tbl_markdown_ratio:.2%} (limit {th['table_markdown_ratio']:.0%})")
    if not tbl_ok:
        fails.append("TABLE")

    # --- TEXT ---
    micro_ratio = (r.micro_non_label / r.text_chunks) if r.text_chunks else 0.0
    oversize_ratio = (r.oversize_chunks / r.text_chunks) if r.text_chunks else 0.0
    text_ok = (
        r.encoding_artifacts == 0
        and r.oversize_chunks == 0
        and r.ctrl_chunks == 0
        and r.infix_artifacts == 0
        and micro_ratio <= th["micro_non_label_ratio"]
        and oversize_ratio <= th["oversize_ratio"]
    )
    # Encoding artifacts < 5 is WARN, not FAIL
    if r.encoding_artifacts > 0 and r.encoding_artifacts < 5:
        text_label = "WARN"
    else:
        text_label = "PASS" if text_ok else "FAIL"
    print(f"  TEXT:        {text_label}")
    print(f"    micro_non_label_ratio: {micro_ratio:.3f} (limit {th['micro_non_label_ratio']})")
    print(f"    oversize_ratio: {oversize_ratio:.3f} (limit {th['oversize_ratio']})")
    if r.encoding_artifacts:
        print(f"    encoding_artifacts: {r.encoding_artifacts}")
    if r.high_corruption_chunks:
        print(f"    high_corruption: {r.high_corruption_chunks}")
    if r.ctrl_chunks:
        print(f"    ctrl_char_chunks: {r.ctrl_chunks} (total: {r.ctrl_total})")
    if r.page_num_artifacts:
        print(f"    page_num_artifacts: {r.page_num_artifacts} (advisory)")
    if r.infix_artifacts:
        print(f"    infix_artifacts: {r.infix_artifacts}")
    if r.micro_chunks:
        print(f"    micro_<30: {r.micro_chunks} (advisory)")
    if not text_ok and r.encoding_artifacts >= 5:
        fails.append("TEXT")
    elif r.infix_artifacts > 0 or r.ctrl_chunks > 0:
        fails.append("TEXT")
    elif micro_ratio > th["micro_non_label_ratio"] or oversize_ratio > th["oversize_ratio"]:
        fails.append("TEXT")

    # --- CODE ---
    code_flat_ratio = (r.code_flat / r.code_chunks) if r.code_chunks else 0.0
    code_indent_fidelity = (r.code_with_indent / r.code_chunks) if r.code_chunks else 1.0
    code_frag_ratio = (r.code_fragment_micro / r.code_like_total) if r.code_like_total else 0.0
    code_ok = True
    code_warnings: list = []
    code_gate = th.get("code_indent_gate")  # "hard", "warn", or None
    # Only gate code metrics when there are enough code chunks to be meaningful
    if r.code_chunks >= 3:
        if th["code_flat_ratio"] is not None and code_flat_ratio > th["code_flat_ratio"]:
            if code_gate == "hard":
                code_ok = False
            else:
                code_warnings.append(f"flat_ratio={code_flat_ratio:.2f} (>{th['code_flat_ratio']})")
        if th["code_indent_fidelity"] is not None and code_indent_fidelity < th["code_indent_fidelity"]:
            if code_gate == "hard":
                code_ok = False
            else:
                code_warnings.append(f"indent_fidelity={code_indent_fidelity:.2f} (<{th['code_indent_fidelity']})")
    # Require at least 3 code-like occurrences before gating on fragmentation
    if r.code_like_total >= 3 and th["code_frag_ratio"] is not None and code_frag_ratio > th["code_frag_ratio"]:
        if code_gate == "hard":
            code_ok = False
        else:
            code_warnings.append(f"frag_ratio={code_frag_ratio:.3f} (>{th['code_frag_ratio']})")
    if code_warnings:
        code_label = "WARN"
    else:
        code_label = "PASS" if code_ok else "FAIL"
    print(f"  CODE:        {code_label} [{content_type}]")
    if r.code_chunks > 0:
        print(f"    code_chunks: {r.code_chunks}  flat: {r.code_flat}  flat_ratio: {code_flat_ratio:.2f}")
        print(f"    indentation_fidelity: {code_indent_fidelity:.2f}")
    if r.code_like_total > 0:
        print(f"    code_like_total: {r.code_like_total}  fragment_micro: {r.code_fragment_micro}  frag_ratio: {code_frag_ratio:.3f}")
    if code_warnings:
        for w in code_warnings:
            print(f"    ⚠ {w}")
    if not code_ok:
        fails.append("CODE")

    # --- LABEL ---
    orphan_ratio = (r.orphan_labels / r.label_chunks) if r.label_chunks else 0.0
    # Skip orphan check when label count is too low for ratio to be meaningful
    if r.label_chunks < th["orphan_label_min_count"]:
        orphan_ratio_gated = 0.0
    else:
        orphan_ratio_gated = orphan_ratio
    label_ok = orphan_ratio_gated <= th["orphan_label_ratio"]
    print(f"  LABEL:       {'PASS' if label_ok else 'FAIL'}")
    if r.label_chunks > 0:
        print(f"    labels: {r.label_chunks}  orphans: {r.orphan_labels}  orphan_ratio: {orphan_ratio:.3f} (limit {th['orphan_label_ratio']})")
        if r.label_chunks < th["orphan_label_min_count"]:
            print(f"    (skipped — fewer than {th['orphan_label_min_count']} labels)")
    if not label_ok:
        fails.append("LABEL")

    # --- HEADING ---
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
    if not heading_ok:
        fails.append("HEADING")

    # --- SCHEMA ---
    schema_ok = r.version_mismatches == 0
    print(f"  SCHEMA:      {'PASS' if schema_ok else 'FAIL'}")
    if not schema_ok:
        print(f"    version_mismatches: {r.version_mismatches}")
        fails.append("SCHEMA")

    # --- Issues ---
    if r.issues:
        print(f"\n  Issues ({len(r.issues)}):")
        for issue in r.issues[:15]:
            print(f"    {issue}")
        if len(r.issues) > 15:
            print(f"    ... and {len(r.issues) - 15} more")

    # --- Overall ---
    passed = len(fails) == 0
    print(f"\n  {'AUDIT_PASS' if passed else 'AUDIT_FAIL'}", end="")
    if fails:
        print(f" ({', '.join(fails)})", end="")
    print()
    print()
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        print(f"{'='*65}")
        print("SUMMARY")
        print(f"{'='*65}")
        for path, r, passed in results:
            name = path.parent.name
            status = "PASS" if passed else "FAIL"
            print(f"  {'✓' if passed else '✗'} {name:45s} {r.total_chunks:5d} chunks  {r.doc_class:8s}  {status}")
        total = len(results)
        passed_count = sum(1 for _, _, p in results if p)
        print(f"\n  {passed_count}/{total} passed")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
