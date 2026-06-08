#!/usr/bin/env python3
"""
Semantic-fidelity QA for ingestion JSONL outputs.

Structural gates (size/orphan/crash) are necessary but not sufficient.
This script checks whether extracted content is semantically useful for RAG:
- Image chunks are descriptive (not placeholders)
- Table chunks are structured (Markdown) and not placeholders
- Code chunks preserve multiline structure
- Detect simple cross-page heading->body anchor risks
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Shared R3 code-indentation metric (single source of truth shared with
# qa_conversion_audit.py; see docs/PLAN_R3_CODE_GATE_REDESIGN.md). This script
# remains ADVISORY — the authoritative hard gate is qa_conversion_audit.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _code_quality as code_quality_mod  # noqa: E402


LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9/&()' .,-]{1,50}:?$")


def is_label_like(s: str) -> bool:
    s = (s or "").strip()
    if not s or len(s) > 60 or "\n" in s:
        return False
    if not LABEL_RE.match(s):
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


def is_placeholder_image_or_table(
    content: str,
    chunk: Optional[Dict[str, Any]] = None,
) -> bool:
    """Phase 3 Step 3 (companion to qa_full_conversion's gate):
    detect placeholder image/table content with the F4 hard-fallback
    exemption when chunk metadata is supplied.

    Backward-compatible: when ``chunk`` is None, falls through to the
    pre-Phase-3 string-only logic (table calls do this).
    """
    # F4 hard-fallback exemption: an image chunk with vision_status=
    # "hard_fallback" AND both vision_error and vision_provider_used
    # recorded is a documented no-VLM-signal state, NOT a placeholder
    # row. The placeholder-shaped canonical content is the contract;
    # the gate must not double-count it.
    if chunk is not None:
        meta = chunk.get("metadata") or {}
        if (
            meta.get("vision_status") == "hard_fallback"
            and (meta.get("vision_error") or "").strip()
            and (meta.get("vision_provider_used") or "").strip()
        ):
            return False

    t = (content or "").strip().lower()
    if not t:
        return True
    if "extraction unavailable" in t:
        return True
    if re.match(r"^\[(figure|image|table)\b", t):
        return True
    # VLM failure sentinels — distinguishable from intentional no-VLM placeholders.
    if t.startswith("[vlm_failed"):
        return True
    # Extremely short "context only" strings should count as low-fidelity placeholders.
    if len(t) < 80 and ("figure on page" in t or "table on page" in t):
        return True
    return False


def is_markdown_table(content: str) -> bool:
    t = (content or "").strip()
    if not t:
        return False
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    if "|" not in lines[0]:
        return False
    # Separator line (---) is required for robust table parsing.
    return any(re.search(r"\|\s*-{2,}", ln) for ln in lines[1:3])


# Mirrors of the production heuristics (kept in sync): F1 furniture masthead
# (batch_processor._FURNITURE_MASTHEAD_RE) and F3 heading sanity
# (context_state.is_valid_heading). The F3 bare-domain requires a path so tech
# headings ("ASP.NET Core") are not flagged (review #2).
_FURNITURE_MASTHEAD_RE = re.compile(
    r"https?://|www\.|\.(?:com|org|net|aero|edu|gov)\b", re.I
)
_INSANE_HEADING_RE = re.compile(
    r"https?://|www\.|@[\w.-]+\.[A-Za-z]{2,}"
    r"|\b[\w.-]+\.(?:com|org|net|aero|edu|gov)/",
    re.I,
)
_CJK_LATIN_MIX_RE = re.compile("[\u4e00-\u9fff][A-Za-z]|[A-Za-z][\u4e00-\u9fff]")


def count_insane_headings(texts: List[Dict[str, Any]]) -> int:
    """Count chunks whose parent_heading is content-garbage (F3 regression net).

    Mirrors the F3 additions to `is_valid_heading` on the resulting strings:
    URL/email/bare-domain mastheads, CJK-Latin mixed garble, or folio-shaped
    (page-number + separator) headings. After F3 this should be ~0; a rising
    ratio means garbage headings are leaking in (e.g. via carry-forward or TOC).
    """
    n = 0
    for r in texts:
        h = ((r.get("metadata") or {}).get("hierarchy") or {}).get("parent_heading")
        if not h:
            continue
        if (
            _INSANE_HEADING_RE.search(h)
            or _CJK_LATIN_MIX_RE.search(h)
            or re.match(r"^\d+\s*[|/]\s", h.strip())
        ):
            n += 1
    return n


_NON_VISUAL_SENTINEL = "no distinct non-text visuals"


def count_non_visual_images(images: List[Dict[str, Any]]) -> int:
    """F2 regression net: image chunks the VLM declared non-visual (text-as-image)."""
    return sum(
        1
        for r in images
        if _NON_VISUAL_SENTINEL
        in (
            (r.get("metadata") or {}).get("visual_description")
            or r.get("visual_description")
            or ""
        ).lower()
    )


def count_blank_images(images: List[Dict[str, Any]], output_dir: Path) -> int:
    """F7 regression net: image chunks whose asset is deterministically blank/
    low-information (independent of the VLM description). Best-effort I/O."""
    try:
        from PIL import Image, ImageStat
        from mmrag_v2.universal.asset_materializer import _is_low_information
    except Exception:
        return 0
    n = 0
    for r in images:
        fp = (r.get("asset_ref") or {}).get("file_path")
        if not fp:
            continue
        p = output_dir / fp
        if not p.exists():
            continue
        try:
            st = ImageStat.Stat(Image.open(p).convert("L"))
            if _is_low_information(st.mean[0], st.stddev[0]):
                n += 1
        except Exception:
            pass
    return n


def count_cross_page_dupes(texts: List[Dict[str, Any]]) -> int:
    """F6 regression net: EXCESS exact-duplicate TEXT chunks repeated across page
    boundaries (captions/headers/VLM loops). Counts occurrences beyond the first
    of any content >= 20 chars appearing on >= 3 distinct pages. TEXT only.

    NB (review #4): this counts ALL excess occurrences, whereas the production
    fix (`_dedup_cross_page_repeats`) KEEPS a duplicate when dropping it would
    orphan its page (the page-coverage guard). So on such a doc the metric can
    read slightly > 0 while the fix behaved correctly - this is expected and is
    why the default threshold is 0.03, not 0. Do not "fix" a small residual.
    """
    page_occ: Dict[str, set] = {}
    total: Dict[str, int] = {}
    for r in texts:
        content = (r.get("content") or "").strip()
        if len(content) < 20:
            continue
        n = re.sub(r"\s+", " ", content).lower()
        page_occ.setdefault(n, set()).add(
            (r.get("metadata") or {}).get("page_number")
        )
        total[n] = total.get(n, 0) + 1
    return sum(
        total[n] - 1 for n, pgs in page_occ.items() if len(pgs) >= 3
    )


def count_running_furniture(texts: List[Dict[str, Any]]) -> int:
    """Count running-header/footer/folio furniture in the FINAL output.

    Mirrors `batch_processor._filter_running_furniture` (PLAN_GATE_QUALITY_V1 F1)
    as the regression net: a short TEXT chunk in the top/bottom page margin whose
    digit-normalized text repeats across >= 3 pages, or that matches a masthead/
    URL folio pattern. After F1 this should be ~0; a rising ratio means furniture
    is leaking back into the index.
    """
    norm_pages: Dict[str, set] = {}
    band: Dict[int, str] = {}
    for i, r in enumerate(texts):
        content = (r.get("content") or "").strip()
        if not content or len(content) > 70:
            continue
        md = r.get("metadata") or {}
        bb = (md.get("spatial") or {}).get("bbox")
        if not bb or len(bb) != 4:
            continue
        y0, y1 = bb[1], bb[3]
        if not (y0 > 920 or y1 < 80):
            continue
        nz = re.sub(r"\d+", "#", re.sub(r"\s+", " ", content)).lower()
        norm_pages.setdefault(nz, set()).add(md.get("page_number"))
        band[i] = nz
    furniture = {i for i, nz in band.items() if len(norm_pages[nz]) >= 3}
    for i in band:
        if _FURNITURE_MASTHEAD_RE.search((texts[i].get("content") or "")):
            furniture.add(i)
    return len(furniture)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ingestion_jsonl", type=Path)
    parser.add_argument("--max-image-placeholder-ratio", type=float, default=0.20)
    parser.add_argument("--max-table-placeholder-ratio", type=float, default=0.20)
    parser.add_argument("--min-table-markdown-ratio", type=float, default=0.80)
    parser.add_argument("--max-code-flat-ratio", type=float, default=0.35)
    parser.add_argument("--min-code-indentation-fidelity", type=float, default=0.90)
    parser.add_argument("--max-cross-page-label-anchor-risk", type=float, default=0.10)
    parser.add_argument("--max-furniture-chunk-ratio", type=float, default=0.05)
    parser.add_argument("--max-heading-sanity-ratio", type=float, default=0.02)
    parser.add_argument("--max-non-visual-image-ratio", type=float, default=0.05)
    parser.add_argument("--max-blank-image-ratio", type=float, default=0.02)
    parser.add_argument("--max-cross-page-dupe-ratio", type=float, default=0.03)
    parser.add_argument("--min-code-fence-consistency", type=float, default=1.0)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    with args.ingestion_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # Skip the document-level metadata record (first line in v2.6+ JSONLs).
    rows = [r for r in rows if r.get("object_type") != "ingestion_metadata"]

    images = [r for r in rows if r.get("modality") == "image"]
    tables = [r for r in rows if r.get("modality") == "table"]
    texts = [r for r in rows if r.get("modality") == "text"]

    # Pass the chunk to the image check so the F4 hard-fallback exemption
    # applies. Tables don't have a vision pipeline; the chunk-less call is
    # backward-compatible and behaves as before.
    image_placeholders = sum(
        1 for r in images
        if is_placeholder_image_or_table(r.get("content") or "", chunk=r)
    )
    table_placeholders = sum(
        1 for r in tables if is_placeholder_image_or_table(r.get("content") or "")
    )
    table_markdown = sum(1 for r in tables if is_markdown_table(r.get("content") or ""))

    # R3 code-indentation metric via the shared module over the FULL chunk
    # population (modality=code + legacy text-code), with positive code-ID
    # (equations excluded) and judge-only-judgeable scoring. Replaces the dead
    # modality=text-only computation that the V3 modality=code promotion bypassed.
    cq = code_quality_mod.code_quality(rows)
    # Flat ratio is advisory and computed over the positively-identified code
    # population (single-line struct code) — kept for parity with the prior gate.
    struct_code = [
        r
        for r in rows
        if code_quality_mod.is_code_population(r)
        and code_quality_mod.has_code_structure(r.get("content") or "")
    ]
    code_flat = sum(1 for r in struct_code if "\n" not in (r.get("content") or ""))

    text_rows = []
    for r in texts:
        md = r.get("metadata") or {}
        text_rows.append((int(md.get("page_number") or 0), (r.get("content") or "").strip()))

    cross_page_anchor_risk = 0
    for i in range(len(text_rows) - 1):
        pg, s = text_rows[i]
        npg, ns = text_rows[i + 1]
        if npg == pg + 1 and is_label_like(s) and len(ns) >= 40 and not is_label_like(ns):
            cross_page_anchor_risk += 1

    image_placeholder_ratio = (image_placeholders / len(images)) if images else 0.0
    image_with_description = sum(
        1 for r in images
        if r.get("visual_description") or (r.get("metadata") or {}).get("visual_description")
    )
    image_description_coverage = (image_with_description / len(images)) if images else 1.0
    table_placeholder_ratio = (table_placeholders / len(tables)) if tables else 0.0
    table_markdown_ratio = (table_markdown / len(tables)) if tables else 1.0
    code_flat_ratio = (code_flat / len(struct_code)) if struct_code else 0.0
    code_indentation_fidelity = cq.indentation_fidelity
    # Normalize cross-page risk by number of text chunks to keep threshold stable.
    cross_page_anchor_risk_ratio = (
        cross_page_anchor_risk / len(text_rows) if text_rows else 0.0
    )

    # F1: running-header/footer/folio furniture surviving into the final output.
    furniture_chunks = count_running_furniture(texts)
    furniture_chunk_ratio = (furniture_chunks / len(texts)) if texts else 0.0

    # F3: garbage parent_heading strings (URL/email/masthead, CJK-Latin garble,
    # folio-shaped) surviving into the final output.
    texts_with_heading = [
        r for r in texts
        if ((r.get("metadata") or {}).get("hierarchy") or {}).get("parent_heading")
    ]
    insane_headings = count_insane_headings(texts)
    heading_sanity_ratio = (
        insane_headings / len(texts_with_heading) if texts_with_heading else 0.0
    )

    # F2/F7: text-as-image survivors + deterministically-blank image assets.
    non_visual_images = count_non_visual_images(images)
    non_visual_image_ratio = (non_visual_images / len(images)) if images else 0.0
    blank_images = count_blank_images(images, args.ingestion_jsonl.parent)
    blank_image_ratio = (blank_images / len(images)) if images else 0.0

    # F6: excess cross-page duplicate TEXT chunks (captions/headers/VLM loops).
    cross_page_dupes = count_cross_page_dupes(texts)
    cross_page_dupe_ratio = (cross_page_dupes / len(texts)) if texts else 0.0

    # F4: modality=code chunks must be Markdown-fenced (engine-agnostic parity).
    code_chunks = [r for r in rows if r.get("modality") == "code"]
    code_fenced = sum(
        1 for r in code_chunks if (r.get("content") or "").lstrip().startswith("```")
    )
    code_fence_consistency = (code_fenced / len(code_chunks)) if code_chunks else 1.0

    print(
        f"images={len(images)} image_placeholder_ratio={image_placeholder_ratio:.4f} "
        f"image_description_coverage={image_description_coverage:.4f}"
    )
    print(
        f"tables={len(tables)} table_placeholder_ratio={table_placeholder_ratio:.4f} "
        f"table_markdown_ratio={table_markdown_ratio:.4f}"
    )
    print(
        f"code_population={cq.n_population} struct={cq.n_struct} "
        f"math_excluded={cq.n_math_excluded} judgeable={cq.n_judgeable} "
        f"code_flat_ratio={code_flat_ratio:.4f}"
    )
    print(f"code_indentation_fidelity={code_indentation_fidelity:.4f}")
    print(
        "cross_page_label_anchor_risk="
        f"{cross_page_anchor_risk} ratio={cross_page_anchor_risk_ratio:.4f}"
    )
    print(
        f"furniture_chunks={furniture_chunks} "
        f"furniture_chunk_ratio={furniture_chunk_ratio:.4f}"
    )
    print(
        f"insane_headings={insane_headings} "
        f"heading_sanity_ratio={heading_sanity_ratio:.4f}"
    )
    print(
        f"non_visual_images={non_visual_images} "
        f"non_visual_image_ratio={non_visual_image_ratio:.4f} "
        f"blank_images={blank_images} blank_image_ratio={blank_image_ratio:.4f}"
    )
    print(
        f"cross_page_dupes={cross_page_dupes} "
        f"cross_page_dupe_ratio={cross_page_dupe_ratio:.4f}"
    )
    print(
        f"code_chunks={len(code_chunks)} code_fenced={code_fenced} "
        f"code_fence_consistency={code_fence_consistency:.4f}"
    )

    fails: List[str] = []
    if image_placeholder_ratio > args.max_image_placeholder_ratio:
        fails.append(
            f"image_placeholder_ratio={image_placeholder_ratio:.3f} "
            f"(>{args.max_image_placeholder_ratio:.2f})"
        )
    if images and image_description_coverage < 0.80:
        fails.append(
            f"image_description_coverage={image_description_coverage:.3f} (<0.80)"
        )
    if len(tables) > 0:
        if table_placeholder_ratio > args.max_table_placeholder_ratio:
            fails.append(
                f"table_placeholder_ratio={table_placeholder_ratio:.3f} "
                f"(>{args.max_table_placeholder_ratio:.2f})"
            )
        if table_markdown_ratio < args.min_table_markdown_ratio:
            fails.append(
                f"table_markdown_ratio={table_markdown_ratio:.3f} "
                f"(<{args.min_table_markdown_ratio:.2f})"
            )
    # Require at least 3 code chunks before penalising flat ratio — a single
    # flat snippet in a 20-page sample is a statistical artefact, not a bug.
    if len(struct_code) >= 3 and code_flat_ratio > args.max_code_flat_ratio:
        fails.append(
            f"code_flat_ratio={code_flat_ratio:.3f} "
            f"(>{args.max_code_flat_ratio:.2f})"
        )
    # Indentation fidelity is scored only over JUDGEABLE code (multi-line nested
    # blocks); flat/REPL/equation chunks are exempt. Advisory here — the
    # authoritative hard gate is qa_conversion_audit.py (Policy B).
    if (
        cq.n_judgeable >= code_quality_mod.DEFAULT_MIN_JUDGEABLE
        and code_indentation_fidelity < args.min_code_indentation_fidelity
    ):
        fails.append(
            f"code_indentation_fidelity={code_indentation_fidelity:.3f} "
            f"(<{args.min_code_indentation_fidelity:.2f})"
        )
    if cross_page_anchor_risk_ratio > args.max_cross_page_label_anchor_risk:
        fails.append(
            f"cross_page_label_anchor_risk_ratio={cross_page_anchor_risk_ratio:.3f} "
            f"(>{args.max_cross_page_label_anchor_risk:.2f})"
        )
    if furniture_chunk_ratio > args.max_furniture_chunk_ratio:
        fails.append(
            f"furniture_chunk_ratio={furniture_chunk_ratio:.3f} "
            f"(>{args.max_furniture_chunk_ratio:.2f})"
        )
    if heading_sanity_ratio > args.max_heading_sanity_ratio:
        fails.append(
            f"heading_sanity_ratio={heading_sanity_ratio:.3f} "
            f"(>{args.max_heading_sanity_ratio:.2f})"
        )
    if non_visual_image_ratio > args.max_non_visual_image_ratio:
        fails.append(
            f"non_visual_image_ratio={non_visual_image_ratio:.3f} "
            f"(>{args.max_non_visual_image_ratio:.2f})"
        )
    if blank_image_ratio > args.max_blank_image_ratio:
        fails.append(
            f"blank_image_ratio={blank_image_ratio:.3f} "
            f"(>{args.max_blank_image_ratio:.2f})"
        )
    if cross_page_dupe_ratio > args.max_cross_page_dupe_ratio:
        fails.append(
            f"cross_page_dupe_ratio={cross_page_dupe_ratio:.3f} "
            f"(>{args.max_cross_page_dupe_ratio:.2f})"
        )
    if code_chunks and code_fence_consistency < args.min_code_fence_consistency:
        fails.append(
            f"code_fence_consistency={code_fence_consistency:.3f} "
            f"(<{args.min_code_fence_consistency:.2f})"
        )

    if fails:
        print("SEMANTIC_FAIL: " + "; ".join(fails))
    else:
        print("SEMANTIC_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
