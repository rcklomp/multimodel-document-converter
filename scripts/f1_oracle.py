#!/usr/bin/env python3
"""PLAN_F1 Section 6 independent fidelity oracle (WP-3).

THE exit criterion for the Phase 1 Mechanism-B spike. The audit gate's per-chunk
`indentation_ok` is a nesting-PRESENCE check that the repair satisfies by
construction (review finding 1), so it can NEVER be the sole pass signal. This
oracle is independent: it scores `ast.parse` success on repair-touched,
judgeable, Python-shaped code chunks.

Verdict per book (both required):
  (1) post-repair parse rate on repair-touched judgeable-Python chunks >= 0.85, and
  (2) strictly greater than the book's pre-repair parse rate.

Pre-repair rate = parse rate over judgeable-Python code chunks in the PRE jsonl
(the engine output before the lane existed; book-level, run-to-run honest - the
two extractions are different VLM runs, so this is a distributional comparison,
not per-chunk matched, and is documented as such).

Also writes a FIXED N-page side-by-side artifact set (raw text-layer lines vs
recovered chunk content) so the numbers say whether it passed and the artifacts
say HOW.

NOT a gate edit. Standalone. Reads JSONL produced by the shipping CLI.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Optional

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
import _code_quality as cq  # noqa: E402

_PY_HINT = ("def ", "class ", "import ", "from ", "return ", "self.", "for ", "if ", "while ")


def _load_chunks(jsonl: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        # skip the ingestion_metadata header line (no modality)
        if d.get("modality") is None and d.get("schema_version"):
            continue
        out.append(d)
    return out


def _is_python_shaped(content: str) -> bool:
    s = content or ""
    if any(h in s for h in _PY_HINT):
        return True
    try:
        ast.parse(s)
        return True
    except (SyntaxError, ValueError):
        return False


def _parses(content: str) -> bool:
    try:
        ast.parse(content or "")
        return True
    except (SyntaxError, ValueError):
        return False


def _judgeable_python(chunks: list[dict[str, Any]], touched_only: bool) -> list[dict[str, Any]]:
    out = []
    for ch in chunks:
        if not cq.is_code_population(ch):
            continue
        content = ch.get("content") or ""
        if not cq.is_judgeable(content):
            continue
        if not _is_python_shaped(content):
            continue
        if touched_only:
            md = ch.get("metadata") or {}
            if not md.get("code_repair_applied"):
                continue
        out.append(ch)
    return out


def _rate(chunks: list[dict[str, Any]]) -> tuple[float, int, int]:
    if not chunks:
        return 1.0, 0, 0
    ok = sum(1 for c in chunks if _parses(c.get("content") or ""))
    return ok / len(chunks), ok, len(chunks)


def _write_artifacts(
    post: list[dict[str, Any]],
    book: str,
    out_dir: Path,
    n_pages: int,
    source_pdf: Optional[Path] = None,
) -> int:
    """Fixed N-page side-by-side: recovered chunk content per page (sorted pages).

    BINDING acceptance requirement (user directive 2026-06-12): for each artifact
    page, render the SOURCE PAGE to PNG alongside the recovered chunk text so a
    human can compare source-to-output directly. The phase does not close on the
    oracle numbers alone - it closes after that human review of these bundles.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    touched = _judgeable_python(post, touched_only=True)
    by_page: dict[int, list[dict[str, Any]]] = {}
    for ch in touched:
        pno = (ch.get("metadata") or {}).get("page_number")
        if pno:
            by_page.setdefault(int(pno), []).append(ch)
    pages = sorted(by_page)[:n_pages]

    doc = None
    if source_pdf and Path(source_pdf).is_file():
        try:
            import fitz  # noqa: PLC0415
            doc = fitz.open(str(source_pdf))
        except Exception:
            doc = None

    art = out_dir / f"{book}_oracle_artifacts.txt"
    with art.open("w", encoding="utf-8") as fh:
        fh.write(f"# {book} - Section 6 oracle artifacts: {len(pages)} pages, recovered code\n")
        if doc is not None:
            fh.write("# source-page PNGs rendered alongside (see *_pNN_source.png) for human review\n")
        else:
            fh.write("# WARNING: no source PDF -> source-page PNGs NOT rendered (pass --source-pdf)\n")
        fh.write("\n")
        for pno in pages:
            png_note = ""
            if doc is not None and 1 <= pno <= len(doc):
                png = out_dir / f"{book}_p{pno:04d}_source.png"
                try:
                    doc[pno - 1].get_pixmap(dpi=130).save(str(png))
                    png_note = f"  source-page PNG: {png.name}"
                except Exception:
                    png_note = "  (source PNG render failed)"
            for ch in by_page[pno]:
                parses = "PARSE_OK" if _parses(ch.get("content") or "") else "PARSE_FAIL"
                fh.write(f"===== page {pno}  chunk {ch.get('chunk_id')}  [{parses}]{png_note} =====\n")
                fh.write((ch.get("content") or "") + "\n\n")
    if doc is not None:
        doc.close()
    return len(pages)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", required=True)
    ap.add_argument("--pre", required=True, help="pre-lane jsonl (engine output before the lane)")
    ap.add_argument("--post", required=True, help="post-lane jsonl (new run with Mechanism B)")
    ap.add_argument("--artifacts-dir", default=str(_REPO / "output" / "f1_oracle"))
    ap.add_argument("--source-pdf", default=None,
                    help="source PDF to render per-page PNGs for human review (BINDING)")
    ap.add_argument("--floor", type=float, default=0.85)
    ap.add_argument("--artifact-pages", type=int, default=10)
    args = ap.parse_args(argv)

    pre = _load_chunks(Path(args.pre))
    post = _load_chunks(Path(args.post))

    pre_rate, pre_ok, pre_n = _rate(_judgeable_python(pre, touched_only=False))
    post_touched = _judgeable_python(post, touched_only=True)
    post_rate, post_ok, post_n = _rate(post_touched)
    # diagnostic: book-level post rate over ALL judgeable-python (not only touched)
    post_all_rate, _, post_all_n = _rate(_judgeable_python(post, touched_only=False))

    pages = _write_artifacts(
        post, args.book, Path(args.artifacts_dir), args.artifact_pages,
        source_pdf=Path(args.source_pdf) if args.source_pdf else None,
    )

    floor_ok = post_n > 0 and post_rate >= args.floor
    improved = post_rate > pre_rate
    verdict = "ORACLE_PASS" if (floor_ok and improved) else "ORACLE_FAIL"

    print(f"=== F1 ORACLE: {args.book} ===")
    print(f"  pre-lane  judgeable-python parse rate: {pre_rate:.3f} ({pre_ok}/{pre_n})")
    print(f"  post-lane repair-touched parse rate:   {post_rate:.3f} ({post_ok}/{post_n})")
    print(f"  post-lane all-judgeable parse rate:    {post_all_rate:.3f} (n={post_all_n}) [diagnostic]")
    print(f"  floor>={args.floor}: {'PASS' if floor_ok else 'FAIL'}   improved(post>pre): {'PASS' if improved else 'FAIL'}")
    print(f"  artifacts: {pages} pages -> {args.artifacts_dir}/{args.book}_oracle_artifacts.txt")
    print(f"  {verdict}")
    return 0 if verdict == "ORACLE_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
