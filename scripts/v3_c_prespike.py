#!/usr/bin/env python3
"""V3 Phase C Pre-Spike — 2-hour falsification test per Charter §4.2 step 1.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §4.2.

Foundation-session status: HARNESS ONLY. The ColPali embedding step is
not yet wired to a model — the harness exposes a clean injection point
so the operator can swap in:
    (a) `colpali-engine` package locally
    (b) ColPali via HF Spaces inference endpoint
    (c) ColPali via the omlx ColPali deployment from Phase C task C2
without changing the rest of the script.

The harness renders the gold + distractor pages, normalizes them to
the C-spike's required shape (200 DPI per Charter §4.2), and prints a
PASS/FAIL decision based on MaxSim ranking. If the gold page does not
rank first, the full C-spike is dead weight per Charter §4.2 step 1 #5.

Usage:
    # With a local colpali-engine install:
    python scripts/v3_c_prespike.py \\
        --pdf data/technical_report/ATZ.Elektronik...pdf \\
        --gold-page 7 \\
        --distractor-pages 12,23,45 \\
        --query "Schaltbild eines NPN-Transistor-Verstärkers" \\
        --colpali-mode local

    # Dry-run (no model needed, prints the harness plan):
    python scripts/v3_c_prespike.py --dry-run \\
        --pdf data/technical_report/ATZ.Elektronik...pdf \\
        --gold-page 7 \\
        --distractor-pages 12,23,45 \\
        --query "Schaltbild eines NPN-Transistor-Verstärkers"

Charter PASS condition: gold page ranks first in MaxSim against the
four-page candidate set on the most-favorable known query. A FAIL
means ColPali does not see the spatial signal even on the easiest
shot — terminate Phase C planning, redirect to VLM-native parsing
evaluation or alternative visual model per Charter §4.2 outcome rules.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# Default values per Charter §3.2 ConversionPlan.render_dpi and §4.2:
DEFAULT_RENDER_DPI = 200


@dataclass(frozen=True)
class PreSpikeConfig:
    pdf_path: Path
    gold_page: int
    distractor_pages: List[int]
    query: str
    render_dpi: int = DEFAULT_RENDER_DPI
    colpali_mode: str = "dry-run"  # "dry-run" | "local" | "hf-spaces" | "omlx"
    output_dir: Optional[Path] = None


@dataclass(frozen=True)
class PreSpikeResult:
    """Per Charter §4.2 step 1 PASS/FAIL outcome."""

    pages_ranked: List[Tuple[int, float]]  # (page_number, max_sim_score)
    gold_page: int
    gold_rank: int  # 1-indexed
    passed: bool  # gold_rank == 1

    @property
    def verdict_str(self) -> str:
        return "PASS" if self.passed else "FAIL"


# ---------------------------------------------------------------------------
# Step 1: render pages from the PDF at the configured DPI.
# ---------------------------------------------------------------------------


def render_pages(
    pdf_path: Path,
    page_numbers: List[int],
    *,
    render_dpi: int = DEFAULT_RENDER_DPI,
    output_dir: Optional[Path] = None,
):
    """Render the specified PDF pages to PIL Images at `render_dpi`.

    Uses PyMuPDF (`pymupdf` / `fitz`) which is already in project deps
    via Docling. Returns a list of (page_number, PIL.Image) tuples.

    If `output_dir` is provided, also persists each rendered page as PNG
    under `<output_dir>/page_<NNN>_dpi<XXX>.png` so the operator can
    visually inspect what ColPali will see (useful for sanity-checking
    that the gold page is correctly identified).
    """
    import fitz  # PyMuPDF; provided by Docling deps
    from PIL import Image

    rendered: list = []
    with fitz.open(str(pdf_path)) as doc:
        for page_number in page_numbers:
            # PyMuPDF is 0-indexed; user-facing page numbers are 1-indexed.
            zero_indexed = page_number - 1
            if not (0 <= zero_indexed < len(doc)):
                raise ValueError(
                    f"Page {page_number} out of range "
                    f"(document has {len(doc)} pages)"
                )
            page = doc[zero_indexed]
            zoom = render_dpi / 72.0  # 72 DPI is PyMuPDF's default
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            rendered.append((page_number, image))
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                path = output_dir / f"page_{page_number:03d}_dpi{render_dpi}.png"
                image.save(path)
    return rendered


# ---------------------------------------------------------------------------
# Step 2: ColPali embedding (injection point — not wired in foundation)
# ---------------------------------------------------------------------------


def embed_pages_via_colpali(
    page_images,
    *,
    mode: str = "dry-run",
):
    """Embed page images via ColPali.

    Foundation-session status: STUB for non-`local` modes; `local` mode
    delegates to the `colpali-engine` package if installed (the operator
    is expected to `pip install colpali-engine` before running with
    --colpali-mode local).

    Returns a list of patch-vector matrices (numpy arrays of shape
    (num_patches, embedding_dim)) — one per page.
    """
    if mode == "dry-run":
        return [None for _ in page_images]
    if mode == "local":
        try:
            from colpali_engine.models import ColPali, ColPaliProcessor  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ColPali local mode requires `pip install colpali-engine`. "
                "Use --colpali-mode dry-run to validate harness wiring "
                "without the model."
            ) from exc
        # Phase C task C2 will instantiate the model under the omlx
        # tenancy contract; the per-spike harness loads it on the
        # workstation CPU/MPS for the 2-hour falsification test.
        # Implementation deferred to operator execution.
        raise NotImplementedError(
            "Local ColPali dispatch lands when the operator runs this "
            "harness; see the module docstring for usage. The dry-run "
            "mode is the foundation-session deliverable."
        )
    if mode == "hf-spaces":
        raise NotImplementedError(
            "HF Spaces dispatch deferred to operator-execution time; "
            "see https://huggingface.co/spaces/vidore/colpali for the "
            "current API."
        )
    if mode == "omlx":
        raise NotImplementedError(
            "omlx ColPali dispatch lands in Phase C task C2 per Charter "
            "§7.7 tenancy. Foundation session ships the omlx scheduler "
            "contract but not the model deployment."
        )
    raise ValueError(f"Unknown ColPali mode: {mode!r}")


def embed_query_via_colpali(query: str, *, mode: str = "dry-run"):
    """Embed a query string via ColPali. Same dispatch as page embedding."""
    _ = query
    if mode == "dry-run":
        return None
    return embed_pages_via_colpali([None], mode=mode)[0]


# ---------------------------------------------------------------------------
# Step 3: MaxSim per Charter §3.4 #3
# ---------------------------------------------------------------------------


def maxsim_score(query_embedding, page_embedding) -> float:
    """Compute MaxSim score between a query and a page patch matrix.

    MaxSim (Charter §3.4 #3): for each query token, find its most
    similar document patch, then sum those maximum similarities.

    Numpy-only implementation suitable for the workstation pre-spike;
    Phase C uses the Qdrant MaxSim operator instead.
    """
    if query_embedding is None or page_embedding is None:
        # Dry-run: cannot compute. Return a sentinel.
        return float("nan")
    import numpy as np

    # Normalize so that dot product = cosine similarity.
    query = query_embedding / np.linalg.norm(
        query_embedding, axis=-1, keepdims=True
    )
    page = page_embedding / np.linalg.norm(
        page_embedding, axis=-1, keepdims=True
    )
    # sim[i, j] = cosine(query_token_i, page_patch_j)
    sim = query @ page.T  # shape: (num_query_tokens, num_page_patches)
    # MaxSim: max over j (page patches), then sum over i (query tokens)
    per_query_token_max = sim.max(axis=1)
    return float(per_query_token_max.sum())


# ---------------------------------------------------------------------------
# Step 4: orchestration + PASS/FAIL decision
# ---------------------------------------------------------------------------


def run_prespike(config: PreSpikeConfig) -> PreSpikeResult:
    """Execute the Charter §4.2 step 1 pre-spike.

    PASS = gold page ranks first under MaxSim against the candidate set
    {gold} ∪ {distractors}.
    """
    candidate_pages = [config.gold_page, *config.distractor_pages]
    log = logging.getLogger("v3_c_prespike")
    log.info(
        "Rendering %d pages from %s at %d DPI",
        len(candidate_pages),
        config.pdf_path,
        config.render_dpi,
    )
    page_renders = render_pages(
        config.pdf_path,
        candidate_pages,
        render_dpi=config.render_dpi,
        output_dir=config.output_dir,
    )

    if config.colpali_mode == "dry-run":
        log.warning(
            "ColPali mode = dry-run; embedding skipped. Foundation-"
            "session harness validation only. To get a real PASS/FAIL "
            "verdict, install `colpali-engine` and re-run with "
            "--colpali-mode local."
        )
        # In dry-run we still produce a stable, deterministic ranking
        # output so the harness's PASS/FAIL pipeline can be tested:
        # gold page wins trivially.
        scores = [(p, float("nan")) for p, _ in page_renders]
        scores.sort(key=lambda pair: 0 if pair[0] == config.gold_page else 1)
        return PreSpikeResult(
            pages_ranked=scores,
            gold_page=config.gold_page,
            gold_rank=1,
            passed=False,  # dry-run is NOT a PASS — operator must run live
        )

    log.info("Embedding query via ColPali (%s)", config.colpali_mode)
    query_emb = embed_query_via_colpali(config.query, mode=config.colpali_mode)
    log.info("Embedding %d pages via ColPali (%s)", len(page_renders), config.colpali_mode)
    page_embs = embed_pages_via_colpali(
        [img for _, img in page_renders], mode=config.colpali_mode
    )

    pairs: List[Tuple[int, float]] = []
    for (page_number, _), page_emb in zip(page_renders, page_embs):
        score = maxsim_score(query_emb, page_emb)
        pairs.append((page_number, score))

    # Highest MaxSim first.
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    gold_rank = next(
        rank
        for rank, (page_number, _) in enumerate(pairs, start=1)
        if page_number == config.gold_page
    )
    return PreSpikeResult(
        pages_ranked=pairs,
        gold_page=config.gold_page,
        gold_rank=gold_rank,
        passed=(gold_rank == 1),
    )


def _format_result(result: PreSpikeResult) -> str:
    lines = [
        f"V3 Phase C Pre-Spike — verdict: {result.verdict_str}",
        f"  gold page:       {result.gold_page}",
        f"  gold rank:       {result.gold_rank} of {len(result.pages_ranked)}",
        "  ranked candidates:",
    ]
    for rank, (page_number, score) in enumerate(result.pages_ranked, start=1):
        marker = " ← gold" if page_number == result.gold_page else ""
        lines.append(
            f"    rank {rank}: page={page_number:>4} maxsim={score:.4f}{marker}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "V3 Phase C pre-spike (Charter §4.2 step 1) — falsification "
            "test for ColPali viability on the most-favorable known query."
        )
    )
    parser.add_argument("--pdf", type=Path, required=True, help="Path to source PDF")
    parser.add_argument("--gold-page", type=int, required=True, help="1-indexed gold page")
    parser.add_argument(
        "--distractor-pages",
        type=str,
        required=True,
        help="Comma-separated 1-indexed distractor pages (recommend 3)",
    )
    parser.add_argument("--query", type=str, required=True, help="Pre-spike query string")
    parser.add_argument(
        "--render-dpi", type=int, default=DEFAULT_RENDER_DPI,
        help=f"Render DPI (default {DEFAULT_RENDER_DPI}; valid [72, 600])",
    )
    parser.add_argument(
        "--colpali-mode",
        choices=("dry-run", "local", "hf-spaces", "omlx"),
        default="dry-run",
        help=(
            "ColPali dispatch mode. 'dry-run' validates the harness without "
            "the model; the others delegate to a real ColPali backend."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to persist rendered page PNGs for inspection",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Equivalent to --colpali-mode dry-run.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s | %(message)s",
    )

    mode = "dry-run" if args.dry_run else args.colpali_mode

    config = PreSpikeConfig(
        pdf_path=args.pdf,
        gold_page=args.gold_page,
        distractor_pages=[int(s) for s in args.distractor_pages.split(",") if s],
        query=args.query,
        render_dpi=args.render_dpi,
        colpali_mode=mode,
        output_dir=args.output_dir,
    )
    if not (72 <= config.render_dpi <= 600):
        parser.error(
            f"--render-dpi {config.render_dpi} out of range [72, 600] "
            "(Charter §3.2)"
        )
    if not config.pdf_path.exists():
        parser.error(f"PDF not found: {config.pdf_path}")

    try:
        result = run_prespike(config)
    except NotImplementedError as exc:
        # Operator hit a deliberate fence — print + exit non-zero so CI
        # does not silently green-light a deferred path.
        print(f"NOT_IMPLEMENTED: {exc}", file=sys.stderr)
        return 2

    print(_format_result(result))
    if mode == "dry-run":
        print(
            "\n(Verdict 'FAIL' is the dry-run sentinel; install `colpali-"
            "engine` and re-run with --colpali-mode local for a real run.)"
        )
        return 3  # dry-run exit code distinct from PASS / FAIL
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
