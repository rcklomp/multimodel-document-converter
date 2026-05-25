"""v2.16 Phase 0 — corpus classification + probes for documented limitations.

Implements the classification rules from `docs/PLAN_V2.16.md` §3 Phase 0
steps 3 (programmatic rules) and 4 (probes), with thresholds calibrated
against the canonical 34-doc corpus (step 2 threshold pre-validation).

Recalibrated thresholds (verdict + rationale in
`docs/CORPUS_EXPANSION_2026-05-24_v2.16_p0.md`):

  * code-dense: `code_chunks / (text + code chunks) >= 0.25`
    (was 0.30; Fluent_Python at 0.276 fails 0.30 by construction)
  * form-class: `table_chunks / total_chunks >= 0.10`
                AND `unique_table_template_patterns >= 3`
    (was 0.40; CarOK at 0.136 fails 0.40 by construction)
  * minority-language: unchanged from plan
    `mean_non_ascii_ratio > 0.03` over >=10 sampled text chunks
    OR >=30% of sampled chunks `classify_intent() == "minority_language"`

Reads JSONL only; never touches source PDFs (bias discipline).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from mmrag_v2.retrieval.intent import classify_intent  # noqa: E402

# Recalibrated thresholds — see docstring.
CODE_RATIO_THRESHOLD = 0.25
TABLE_RATIO_THRESHOLD = 0.10
TABLE_UNIQ_PATTERNS_MIN = 3
NON_ASCII_RATIO_THRESHOLD = 0.03
INTENT_HIT_RATE_THRESHOLD = 0.30
SAMPLE_CHUNKS_MIN = 10

# Near-boundary deltas (Probe C).
NEAR_BOUNDARY_CODE_LOW = 0.20
NEAR_BOUNDARY_CODE_HIGH = CODE_RATIO_THRESHOLD  # 0.25
NEAR_BOUNDARY_TABLE_LOW = 0.08
NEAR_BOUNDARY_TABLE_HIGH = TABLE_RATIO_THRESHOLD  # 0.10
NEAR_BOUNDARY_LANG_LOW = 0.025
NEAR_BOUNDARY_LANG_HIGH = 0.035


@dataclass
class DocMetrics:
    basename: str
    profile: Optional[str]
    total_pages: Optional[int]
    n_total_chunks: int = 0
    n_text_chunks: int = 0
    n_code_chunks: int = 0
    n_table_chunks: int = 0
    n_image_chunks: int = 0
    unique_table_template_patterns: int = 0
    code_ratio: float = 0.0
    table_ratio: float = 0.0
    sampled_chunks: int = 0
    mean_non_ascii_ratio: float = 0.0
    intent_minority_lang_hit_rate: float = 0.0
    has_flat_text_corruption: bool = False
    has_encoding_corruption: bool = False
    flags: list[str] = field(default_factory=list)
    probe_results: dict[str, str] = field(default_factory=dict)


def _table_shape(content: str) -> tuple[int, int]:
    """(column_count, row_count) for a markdown table; (0, 0) if not table-shaped."""
    if not content:
        return (0, 0)
    rows = [r for r in content.strip().split("\n") if r.strip().startswith("|")]
    if not rows:
        return (0, 0)
    cols = rows[0].count("|") - 1
    body = [r for r in rows if not set(r) <= set("-: |")]
    return (cols, len(body))


def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) > 127) / len(text)


def compute_metrics(jsonl_path: Path, sample_n: int = 30) -> DocMetrics:
    m = DocMetrics(basename=jsonl_path.parent.name, profile=None, total_pages=None)
    tables: list[tuple[int, int]] = []
    text_samples: list[str] = []

    with jsonl_path.open() as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("object_type") == "ingestion_metadata":
                m.profile = d.get("profile_type")
                m.total_pages = d.get("total_pages")
                m.has_flat_text_corruption = bool(d.get("has_flat_text_corruption"))
                m.has_encoding_corruption = bool(d.get("has_encoding_corruption"))
                continue
            m.n_total_chunks += 1
            mod = d.get("modality")
            content = d.get("content", "") or ""
            md = d.get("metadata") or {}
            is_code = bool(md.get("is_code"))
            if mod == "table":
                m.n_table_chunks += 1
                tables.append(_table_shape(content))
            elif mod == "image":
                m.n_image_chunks += 1
            elif mod == "text":
                if is_code:
                    m.n_code_chunks += 1
                else:
                    m.n_text_chunks += 1
                    if len(text_samples) < sample_n and len(content) >= 50:
                        text_samples.append(content)

    text_total = m.n_text_chunks + m.n_code_chunks
    m.code_ratio = (m.n_code_chunks / text_total) if text_total else 0.0
    m.table_ratio = (m.n_table_chunks / m.n_total_chunks) if m.n_total_chunks else 0.0
    m.unique_table_template_patterns = len(set(tables))

    m.sampled_chunks = len(text_samples)
    if text_samples:
        m.mean_non_ascii_ratio = sum(_non_ascii_ratio(s) for s in text_samples) / len(text_samples)
        intent_hits = sum(1 for s in text_samples if classify_intent(s) == "minority_language")
        m.intent_minority_lang_hit_rate = intent_hits / len(text_samples)
    return m


def classify(m: DocMetrics) -> list[str]:
    flags: list[str] = []
    # Code-dense: primary rule + technical_manual fallback (kept inert until
    # has_code_evidence exists in diagnostic). See docstring.
    if m.code_ratio >= CODE_RATIO_THRESHOLD:
        flags.append("code_dense")
    # Form-class:
    if (
        m.table_ratio >= TABLE_RATIO_THRESHOLD
        and m.unique_table_template_patterns >= TABLE_UNIQ_PATTERNS_MIN
    ):
        flags.append("form_class")
    # Minority-language: either signal suffices (per plan).
    if m.sampled_chunks >= SAMPLE_CHUNKS_MIN:
        if (
            m.mean_non_ascii_ratio > NON_ASCII_RATIO_THRESHOLD
            or m.intent_minority_lang_hit_rate >= INTENT_HIT_RATE_THRESHOLD
        ):
            flags.append("minority_language")
    if not flags:
        flags.append("general")
    return flags


def near_boundary_flags(m: DocMetrics) -> list[str]:
    """Probe C — near-boundary signal-only flags (no auto-reclassification)."""
    flags: list[str] = []
    if NEAR_BOUNDARY_CODE_LOW <= m.code_ratio < NEAR_BOUNDARY_CODE_HIGH:
        flags.append("NEAR_BOUNDARY_CODE_DENSE")
    if NEAR_BOUNDARY_TABLE_LOW <= m.table_ratio < NEAR_BOUNDARY_TABLE_HIGH:
        flags.append("NEAR_BOUNDARY_FORM_CLASS")
    if NEAR_BOUNDARY_LANG_LOW <= m.mean_non_ascii_ratio <= NEAR_BOUNDARY_LANG_HIGH:
        flags.append("NEAR_BOUNDARY_MINORITY_LANGUAGE")
    return flags


def probe_b_borderline_minority(m: DocMetrics) -> Optional[str]:
    """Probe B — borderline minority-language with OCR-stripped diacritics.

    Signal-only; no auto-reclassification. Per plan:
      if intent_classifier fires on >=1 chunk
         AND total hit-rate < 0.30
         AND mean_non_ascii_ratio < 0.03:
          flag BORDERLINE_MINORITY_LANGUAGE
    """
    if m.sampled_chunks < SAMPLE_CHUNKS_MIN:
        return None
    intent_any = m.intent_minority_lang_hit_rate > 0.0
    if (
        intent_any
        and m.intent_minority_lang_hit_rate < INTENT_HIT_RATE_THRESHOLD
        and m.mean_non_ascii_ratio < NON_ASCII_RATIO_THRESHOLD
    ):
        return "BORDERLINE_MINORITY_LANGUAGE"
    return None


def probe_a_form_misclass_eligible(m: DocMetrics) -> bool:
    """Probe A trigger condition (whether to re-extract with --force-table-vlm).

    if profile in {"scanned", "scanned_degraded"}
       AND image_chunks > 0
       AND table_chunks == 0:
    """
    return (
        (m.profile in ("scanned", "scanned_degraded"))
        and m.n_image_chunks > 0
        and m.n_table_chunks == 0
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "jsonl_paths",
        nargs="+",
        help="Paths to ingestion.jsonl files (typically output/<basename>/ingestion.jsonl).",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of table.")
    args = ap.parse_args()

    results = []
    for p in args.jsonl_paths:
        path = Path(p)
        if not path.exists():
            print(f"MISSING: {p}", file=sys.stderr)
            continue
        m = compute_metrics(path)
        m.flags = classify(m)
        m.probe_results = {}
        if probe_a_form_misclass_eligible(m):
            m.probe_results["A"] = "ELIGIBLE_RE_EXTRACT"
        if (b := probe_b_borderline_minority(m)) is not None:
            m.probe_results["B"] = b
        for c in near_boundary_flags(m):
            m.probe_results.setdefault("C", "")
            m.probe_results["C"] = (m.probe_results["C"] + " " + c).strip()
        results.append(m)

    if args.json:
        print(json.dumps([asdict(m) for m in results], indent=2))
        return 0

    print(
        f"{'basename':40s} {'profile':22s} chunks code_r tab_r uniq nonascii intent flags"
    )
    for m in results:
        print(
            f"{m.basename:40s} {str(m.profile):22s} {m.n_total_chunks:5d} "
            f"{m.code_ratio:5.3f} {m.table_ratio:5.3f} {m.unique_table_template_patterns:4d} "
            f"{m.mean_non_ascii_ratio:7.4f} {m.intent_minority_lang_hit_rate:5.2f} "
            f"{','.join(m.flags)}  {m.probe_results or ''}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
