"""HARRY pages 1-30 acceptance fixture for the Docling post-processor pipeline.

Phase 0 of `docs/PLAN_DOCLING_POSTPROCESSOR.md`. The fixture binds the
pipeline's chunk emission order to the PDF's y-coordinate reading order.

Modes:
    HARRY_ACCEPTANCE_JSONL=<path>   read pre-converted ingestion.jsonl
    RUN_HARRY_ACCEPTANCE=1          run the pipeline live on pages 1-30

Without either env var the test is skipped (the live conversion is
multi-minute and unfit for the default unit run). The cached path is
the easiest way to demonstrate the failure on current main; the live
form proves the regression survives a fresh conversion.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "harry_potter_pages_1_to_30"
EXPECTED_FILE = FIXTURE_DIR / "expected_reading_order.txt"
HARRY_PDF = REPO_ROOT / "data" / "scanned" / "HarryPotter_and_the_Sorcerers_Stone.pdf"


def _ascii_normalize(text: str) -> str:
    return (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", "-")
        .replace("–", "-")
        .replace(" ", " ")
        .replace("…", "...")
    )


def _load_expected() -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    for raw in EXPECTED_FILE.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        page_str, anchor = raw.split("\t", 1)
        rows.append((int(page_str), anchor))
    return rows


def _slice_pdf(src_pdf: Path, dst_pdf: Path, last_page: int) -> None:
    import pymupdf

    src = pymupdf.open(src_pdf)
    dst = pymupdf.open()
    try:
        dst.insert_pdf(src, from_page=0, to_page=last_page - 1)
        dst.save(str(dst_pdf))
    finally:
        dst.close()
        src.close()


def _run_pipeline_live(workdir: Path) -> Path:
    if not HARRY_PDF.exists():
        pytest.skip(f"PDF missing: {HARRY_PDF}")
    pdf30 = workdir / "harry_pages_1_30.pdf"
    _slice_pdf(HARRY_PDF, pdf30, last_page=30)
    out_dir = workdir / "out"
    cmd = [
        sys.executable,
        "-m",
        "mmrag_v2.cli",
        "process",
        str(pdf30),
        "--output-dir",
        str(out_dir),
        "--vision-provider",
        "none",
        "--no-refiner",
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    candidates = sorted(out_dir.rglob("ingestion.jsonl"))
    if not candidates:
        candidates = sorted(out_dir.rglob("*.jsonl"))
    if not candidates:
        pytest.fail(f"Pipeline produced no JSONL under {out_dir}")
    return candidates[0]


@pytest.fixture(scope="module")
def harry_chunks() -> List[dict]:
    cached = os.environ.get("HARRY_ACCEPTANCE_JSONL", "").strip()
    run_live = bool(os.environ.get("RUN_HARRY_ACCEPTANCE", "").strip())
    if cached:
        path = Path(cached).expanduser().resolve()
        if not path.exists():
            pytest.skip(f"HARRY_ACCEPTANCE_JSONL points at missing file: {path}")
    elif run_live:
        tmp = Path(tempfile.mkdtemp(prefix="harry_acc_"))
        try:
            path = _run_pipeline_live(tmp)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    else:
        pytest.skip(
            "Set HARRY_ACCEPTANCE_JSONL=<path> or RUN_HARRY_ACCEPTANCE=1 to run "
            "the HARRY pages 1-30 acceptance fixture."
        )

    chunks: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("object_type") == "ingestion_metadata":
            continue
        chunks.append(obj)
    if not chunks:
        pytest.fail(f"No content chunks parsed from {path}")
    return chunks


def _chunk_page(chunk: dict) -> Optional[int]:
    md = chunk.get("metadata") or {}
    page = md.get("page_number")
    if page is None:
        page = chunk.get("page_number")
    try:
        return int(page) if page is not None else None
    except (TypeError, ValueError):
        return None


def _chunk_text(chunk: dict) -> str:
    return _ascii_normalize(chunk.get("content", "") or chunk.get("text", "") or "")


def _locate(
    anchor: str,
    page: int,
    chunks: List[dict],
) -> Optional[Tuple[int, int]]:
    """Return (chunk_index, char_offset) of the first match, page-restricted."""
    norm_anchor = _ascii_normalize(anchor).strip()
    if not norm_anchor:
        return None
    for idx, chunk in enumerate(chunks):
        if _chunk_page(chunk) != page:
            continue
        body = _chunk_text(chunk)
        offset = body.find(norm_anchor)
        if offset >= 0:
            return (idx, offset)
    return None


def test_harry_paragraph_order_matches_pdf(harry_chunks):
    """Chunk emission order must match PDF y-coordinate reading order."""
    expected = _load_expected()
    located: List[Tuple[int, str, Tuple[int, int]]] = []
    missing: List[Tuple[int, str]] = []
    for page, anchor in expected:
        hit = _locate(anchor, page, harry_chunks)
        if hit is None:
            missing.append((page, anchor))
        else:
            located.append((page, anchor, hit))

    swaps: List[str] = []
    last_pos: Tuple[int, int] = (-1, -1)
    last_anchor: Optional[str] = None
    for page, anchor, pos in located:
        if pos < last_pos:
            swaps.append(
                f"page {page}: anchor {anchor!r} found at {pos} but previous "
                f"anchor {last_anchor!r} was at {last_pos}"
            )
        if pos > last_pos:
            last_pos = pos
            last_anchor = anchor

    if swaps:
        # Missing anchors are reported as diagnostic noise (the chunker may
        # legitimately drop or consolidate items); only swaps fail the test.
        details = ["Reading-order swaps (chunk_idx, char_offset descending):"]
        details.extend(f"  - {s}" for s in swaps)
        if missing:
            details.append("")
            details.append(
                f"(diagnostic) {len(missing)}/{len(expected)} anchors not "
                "located in any chunk for their declared page"
            )
        pytest.fail("\n".join(details))
