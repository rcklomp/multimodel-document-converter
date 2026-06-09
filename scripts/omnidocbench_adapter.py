#!/usr/bin/env python3
"""OmniDocBench fidelity-benchmark adapter (PLAN_OMNIDOCBENCH_EVAL Phase 0).

Bridges OmniDocBench's per-page-image / per-page-Markdown world to our
PDF-in / JSONL-out pipeline. STANDALONE on purpose: it shells out to the
`mmrag-v2` CLI and never imports `batch_processor` / `uir_chunker` / any
extraction module (PLAN_OMNIDOCBENCH_EVAL R4). The render half is pure stdlib.

Three subcommands, run in order:

  select      OmniDocBench.json  -> manifest.json  (English subset, by the
              page_attribute.language field -- NOT the `_eng_` filename, which
              tags only 5 of 755 English pages; see 12.1#3).
  build-pdfs  manifest.json      -> one lossless 1-page PDF per page image
              (img2pdf, Pillow-RGB fallback for alpha/CMYK images).
  run         manifest.json      -> per-page ingestion.jsonl by invoking
              `mmrag-v2 process <pdf> --batch-size 10 --vision-provider none`.
  render      ingestion.jsonl    -> one `<gt_image_basename>.md` per page,
              chunks joined in JSONL line order (reading_order is not a schema
              field, 12.1#1).

Render decisions resolved against the scorer (src/core/preprocess/extract.py)
on 2026-06-09:
  - R5 fence: the scorer strips ```markdown/```html/```latex wrappers and bare
    closing ``` (remove_markdown_fences). Do NOT wrap the page; our bare code
    fences are harmless.
  - R6 heading: text scoring normalizes to alphanumerics (clean_string strips
    '#', '*', '-', spaces). Heading markers are a no-op for the text metric, so
    headings render as PLAIN text. No '#'x level; the null-`level` fallback is
    moot. Title extraction is disabled in the scorer anyway (extract.py L814).
  - R7 image: GT `figure` blocks carry no scored text (0/10 in the demo) and the
    scorer deletes markdown image syntax. Emitting our long VLM descriptions as
    prose would inflate edit distance. So image-modality chunks are OMITTED.
    Captions our pipeline emits as TEXT chunks are kept (scored vs figure_caption).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
DEFAULT_GT_JSON = HOME / "omnidocbench-eval/data/OmniDocBench.json"
DEFAULT_IMAGES = HOME / "omnidocbench-eval/data/images"
DEFAULT_WORKSPACE = HOME / "omnidocbench-eval/run"

# OmniDocBench image-modality / figure handling: omit (R7).
_OMIT_MODALITIES = {"image"}


# --------------------------------------------------------------------------- #
# select
# --------------------------------------------------------------------------- #
def cmd_select(args: argparse.Namespace) -> int:
    gt = json.loads(Path(args.gt_json).read_text(encoding="utf-8"))
    language = args.language
    pages = []
    for page in gt:
        info = page["page_info"]
        attr = info.get("page_attribute", {})
        if language != "all" and attr.get("language") != language:
            continue
        image_path = info["image_path"]  # e.g. yanbaopptmerge_SE05.pdf_7.jpg
        pages.append(
            {
                "image_path": image_path,
                "name": Path(image_path).stem,  # gt key, no extension
                "language": attr.get("language"),
                "data_source": attr.get("data_source"),
            }
        )
    if args.limit:
        pages = pages[: args.limit]

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    manifest = {
        "gt_json": str(Path(args.gt_json).resolve()),
        "images_dir": str(Path(args.images_dir).resolve()),
        "language": language,
        "count": len(pages),
        "pages": pages,
    }
    out = workspace / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SELECT: {len(pages)} pages (language={language}) -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# build-pdfs
# --------------------------------------------------------------------------- #
def _image_to_pdf(image_path: Path, pdf_path: Path) -> str:
    """Wrap one page image as a single-page PDF. Returns the method used."""
    import img2pdf  # local import: only needed in mmrag-v2 env

    try:
        with open(pdf_path, "wb") as fh:
            fh.write(img2pdf.convert(str(image_path)))
        return "img2pdf"
    except Exception:
        # img2pdf rejects alpha/CMYK; re-encode via Pillow as a fallback.
        from PIL import Image

        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im.save(pdf_path, "PDF", resolution=float(im.info.get("dpi", (150,))[0]))
        return "pillow"


def cmd_build_pdfs(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace)
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    images_dir = Path(manifest["images_dir"])
    pdf_dir = workspace / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    pages = manifest["pages"]
    if args.limit:
        pages = pages[: args.limit]

    methods = {"img2pdf": 0, "pillow": 0}
    skipped = built = missing = 0
    for page in pages:
        src = images_dir / page["image_path"]
        pdf = pdf_dir / f"{page['name']}.pdf"
        if not src.exists():
            print(f"  MISSING image: {src}", file=sys.stderr)
            missing += 1
            continue
        if pdf.exists() and not args.force:
            skipped += 1
            continue
        method = _image_to_pdf(src, pdf)
        methods[method] += 1
        built += 1
    print(
        f"BUILD-PDFS: built={built} skipped={skipped} missing={missing} "
        f"(img2pdf={methods['img2pdf']} pillow={methods['pillow']}) -> {pdf_dir}"
    )
    return 1 if missing else 0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace)
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    pdf_dir = workspace / "pdfs"
    out_dir = workspace / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = manifest["pages"]
    if args.limit:
        pages = pages[: args.limit]

    done = failed = skipped = 0
    for idx, page in enumerate(pages, 1):
        name = page["name"]
        pdf = pdf_dir / f"{name}.pdf"
        page_out = out_dir / name
        ingestion = page_out / "ingestion.jsonl"
        if ingestion.exists() and not args.force:
            skipped += 1
            continue
        if not pdf.exists():
            print(f"  MISSING pdf (run build-pdfs first): {pdf}", file=sys.stderr)
            failed += 1
            continue
        cmd = [
            args.cli,
            "process",
            str(pdf),
            "--batch-size",
            "10",
            "--vision-provider",
            "none",
            "--output-dir",
            str(page_out),
        ]
        print(f"[{idx}/{len(pages)}] {name} ...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not ingestion.exists():
            print(f"  FAILED ({proc.returncode}): {name}", file=sys.stderr)
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
            for line in tail:
                print(f"    | {line}", file=sys.stderr)
            failed += 1
            continue
        done += 1
    print(f"RUN: done={done} skipped={skipped} failed={failed}")
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def _render_chunk(chunk: dict) -> str | None:
    """One chunk -> one Markdown block, or None to drop it.

    table content is already a Markdown grid; code is already fenced; text and
    list items render as plain content. Image regions are omitted (R7).
    """
    modality = chunk.get("modality")
    if modality in _OMIT_MODALITIES:
        return None
    content = (chunk.get("content") or "").strip()
    if not content:
        return None
    return content


def render_ingestion(ingestion_path: Path) -> dict[int, str]:
    """Render one ingestion.jsonl to {page_number: markdown}.

    The first line is a doc-metadata header (object_type == ingestion_metadata)
    and is skipped (12.1#2). Real chunks are keyed off metadata.page_number and
    emitted in JSONL line order (12.1#1).
    """
    blocks_by_page: dict[int, list[str]] = {}
    for raw in ingestion_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        if obj.get("object_type") == "ingestion_metadata":
            continue
        page_no = obj.get("metadata", {}).get("page_number")
        if page_no is None:
            continue
        block = _render_chunk(obj)
        if block is None:
            continue
        blocks_by_page.setdefault(int(page_no), []).append(block)
    return {pg: "\n\n".join(blocks) for pg, blocks in blocks_by_page.items()}


def cmd_render(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace)
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    out_dir = workspace / "out"
    preds_dir = workspace / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    pages = manifest["pages"]
    if args.limit:
        pages = pages[: args.limit]

    written = missing = empty = 0
    for page in pages:
        name = page["name"]
        ingestion = out_dir / name / "ingestion.jsonl"
        if not ingestion.exists():
            missing += 1
            continue
        rendered = render_ingestion(ingestion)
        # single-page PDF -> all chunks are page 1; join any pages defensively.
        markdown = "\n\n".join(rendered[pg] for pg in sorted(rendered)).strip()
        if not markdown:
            empty += 1
        (preds_dir / f"{name}.md").write_text(markdown + "\n", encoding="utf-8")
        written += 1
    print(
        f"RENDER: written={written} missing_jsonl={missing} empty={empty} -> {preds_dir}"
    )
    return 1 if missing else 0


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="run workspace dir")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sel = sub.add_parser("select", help="filter GT json to a language subset -> manifest.json")
    p_sel.add_argument("--gt-json", default=str(DEFAULT_GT_JSON))
    p_sel.add_argument("--images-dir", default=str(DEFAULT_IMAGES))
    p_sel.add_argument("--language", default="english", help="page_attribute.language to keep, or 'all'")
    p_sel.add_argument("--limit", type=int, default=0, help="cap page count (smoke)")
    p_sel.set_defaults(func=cmd_select)

    p_pdf = sub.add_parser("build-pdfs", help="page images -> 1-page PDFs")
    p_pdf.add_argument("--limit", type=int, default=0)
    p_pdf.add_argument("--force", action="store_true")
    p_pdf.set_defaults(func=cmd_build_pdfs)

    p_run = sub.add_parser("run", help="invoke mmrag-v2 process per PDF")
    p_run.add_argument("--cli", default="mmrag-v2", help="pipeline CLI entrypoint")
    p_run.add_argument("--limit", type=int, default=0)
    p_run.add_argument("--force", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_rnd = sub.add_parser("render", help="ingestion.jsonl -> per-page <name>.md")
    p_rnd.add_argument("--limit", type=int, default=0)
    p_rnd.set_defaults(func=cmd_render)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
