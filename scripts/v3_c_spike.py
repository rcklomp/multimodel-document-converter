#!/usr/bin/env python3
"""V3 Phase C C-Spike — Charter §4.2 step 2 (full 2–3 day quantitative test).

Single-doc test on the highest-deficit document (`ATZ_Elektronik_German`):

    1. Render all pages at 200 DPI.
    2. Embed all pages with ColPali (one-time; reused across queries).
    3. For each of N=20 hand-crafted queries (Charter "or hand-craft if
       fixture coverage is thin"):
        a. Visual retrieval: ColPali query embed → MaxSim against page
           matrices → top-K page ranking.
        b. Text retrieval: production `retrieve_hybrid_reranked` →
           top-5 chunks → page mapping.
    4. Compute aggregate metrics:
        - Visual top-1 accuracy
        - Text top-1 accuracy
        - PASS A: visual recovery rate on text-failed queries ≥60%
                  AND visual does not harm text-passed queries
        - Per-query trace for PASS B planning (the actual reranker
          discrimination measurement against the bounded-join candidate
          set is run as a separate step using the same trace).

Charter §4.2 step 2 outputs feed Phase C planning. Verdict is binary
PASS / FAIL — if either condition fails, scope expands to region-level
granularity (PASS B FAIL) or Phase C redirects entirely (PASS A FAIL).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse the pre-spike's PDF rendering + ColPali embed + MaxSim helpers
# so the two scripts stay in sync; if the pre-spike's local-mode
# dispatch ever changes, the C-spike inherits it.
import importlib.util

_SCRIPT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "v3_c_prespike", _SCRIPT_DIR / "v3_c_prespike.py"
)
v3_c_prespike = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("v3_c_prespike", v3_c_prespike)
_spec.loader.exec_module(v3_c_prespike)


# ATZ_Elektronik_German hand-crafted query set.
#
# 20 queries distributed across pages:
#   page 1 (cover + Lifecycle Management diagram):  5 queries (visual-favored)
#   page 2 (Einleitung, body text):                 3 queries
#   page 3 (LCM realization + author bio):          3 queries
#   page 4 (horizontal Bruch / process):            3 queries
#   page 5 (MESSINA + CTM/CTE):                     3 queries
#   page 6 (impressum + ads):                       3 queries (text-favored)
#
# Page-1 queries include diagram-label tokens (BMW, SWC, ECU, Komponenten-
# Modell) and the phase sequence (Anforderungsanalyse →
# Architekturentwicklung → Implementierungsphase → Test → System-
# Integration) that text retrieval cannot easily reconstruct because
# they live in the flowchart's spatial layout, not in body prose.
ATZ_QUERIES: List[Dict[str, Any]] = [
    # Page 1 — visual-favored
    {"id": "Q01", "gold_page": 1,
     "text": "Lifecycle Management Anforderungsanalyse bis System-Integration"},
    {"id": "Q02", "gold_page": 1,
     "text": "Effizientere Software Entwicklung durchgängige Prozesskette Spezifikation"},
    {"id": "Q03", "gold_page": 1,
     "text": "Bruch zwischen Spezifikation und Test in der Prozesskette"},
    {"id": "Q04", "gold_page": 1,
     "text": "BMW SWC ECU Komponenten-Modell ECU-Subsystem Lifecycle-Diagramm"},
    {"id": "Q05", "gold_page": 1,
     "text": "Anforderungsanalyse Architekturentwicklung Komponenten-Entwicklung Implementierung Test System-Integration Reihenfolge"},
    # Page 2 — Einleitung body
    {"id": "Q06", "gold_page": 2,
     "text": "OEM Modellpalette Herausforderungen Komplexität Softwareumfang"},
    {"id": "Q07", "gold_page": 2,
     "text": "ECU-SW-Entwicklungsprozess inhomogen Prozessbrüche heterogen"},
    {"id": "Q08", "gold_page": 2,
     "text": "vertikaler Bruch Entwicklungsphase Spezifikation Implementierung"},
    # Page 3 — LCM realization + author
    {"id": "Q09", "gold_page": 3,
     "text": "Klaus Eder Berner Mattner Systemtechnik Geschäftsführer"},
    {"id": "Q10", "gold_page": 3,
     "text": "Realisierung LCM für ECU-Software Säulen Werkzeuge"},
    {"id": "Q11", "gold_page": 3,
     "text": "FormalSpec Spezifikationssprache Anforderungen formale Notation"},
    # Page 4 — horizontal Bruch + process
    {"id": "Q12", "gold_page": 4,
     "text": "horizontaler Bruch Konzept Test-Toolkette Modularität"},
    {"id": "Q13", "gold_page": 4,
     "text": "Prozessdurchgängigkeit ohne kritische Brüche Entwicklungsphase"},
    {"id": "Q14", "gold_page": 4,
     "text": "modularer HiL Hardware-in-the-Loop Testumgebung modulare Architektur"},
    # Page 5 — MESSINA + CTM/CTE
    {"id": "Q15", "gold_page": 5,
     "text": "MESSINA Erweiterungen Klassifikationsbaum-Methode CTM CTE"},
    {"id": "Q16", "gold_page": 5,
     "text": "automatische Testfallgenerierung reaktive Tests"},
    {"id": "Q17", "gold_page": 5,
     "text": "modularHiL MESSINA Werkzeugkette Migrationsstrategie"},
    # Page 6 — impressum + ads (text-favored)
    {"id": "Q18", "gold_page": 6,
     "text": "Veranstaltungshinweise ATZ MTZ Jahreskalender recherchefreundlich"},
    {"id": "Q19", "gold_page": 6,
     "text": "Impressum Springer Vieweg Anzeigenleitung Verlag"},
    {"id": "Q20", "gold_page": 6,
     "text": "Abonnementbedingungen Bezugspreis ATZ Elektronik Jahrgang"},
]


ATZ_PDF_PATH = Path(
    "data/technical_report/"
    "ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters.pdf"
)
ATZ_DOC_ID = "6fccda8bd625"
ATZ_PAGE_COUNT = 6


@dataclass
class QueryResult:
    """Per-query record."""
    query_id: str
    query_text: str
    gold_page: int
    visual_top_pages: List[Tuple[int, float]] = field(default_factory=list)
    text_top_pages: List[Tuple[int, float]] = field(default_factory=list)
    text_top_chunks: List[Dict[str, Any]] = field(default_factory=list)
    timing_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def visual_top1_page(self) -> Optional[int]:
        return self.visual_top_pages[0][0] if self.visual_top_pages else None

    @property
    def text_top1_page(self) -> Optional[int]:
        return self.text_top_pages[0][0] if self.text_top_pages else None

    @property
    def visual_passes(self) -> bool:
        return self.visual_top1_page == self.gold_page

    @property
    def text_passes(self) -> bool:
        return self.text_top1_page == self.gold_page

    @property
    def gold_in_visual_top5(self) -> bool:
        return any(p == self.gold_page for p, _ in self.visual_top_pages[:5])

    @property
    def gold_in_text_top5(self) -> bool:
        return any(p == self.gold_page for p, _ in self.text_top_pages[:5])


# ---------------------------------------------------------------------------
# Visual retrieval (cached page embeddings + per-query MaxSim)
# ---------------------------------------------------------------------------


def embed_all_pages_once(
    pdf_path: Path,
    *,
    page_count: int,
    render_dpi: int,
    model_id: str,
    output_dir: Optional[Path] = None,
) -> Dict[int, Any]:
    """Render + embed all pages once. Returns {page_number: patch_matrix}."""
    pages = list(range(1, page_count + 1))
    log = logging.getLogger("v3_c_spike")
    log.info("Rendering %d pages at %d DPI", len(pages), render_dpi)
    renders = v3_c_prespike.render_pages(
        pdf_path, pages, render_dpi=render_dpi, output_dir=output_dir
    )
    log.info("Embedding %d pages via ColPali (local, %s)", len(pages), model_id)
    embs = v3_c_prespike.embed_pages_via_colpali(
        [img for _, img in renders], mode="local", model_id=model_id
    )
    return {pn: emb for (pn, _), emb in zip(renders, embs)}


def visual_rank_pages(
    query_text: str,
    page_embeddings: Dict[int, Any],
    *,
    model_id: str,
) -> List[Tuple[int, float]]:
    """Rank pages by MaxSim against the query embedding."""
    query_emb = v3_c_prespike.embed_query_via_colpali(
        query_text, mode="local", model_id=model_id
    )
    scored = []
    for page_number, page_emb in page_embeddings.items():
        scored.append(
            (page_number, v3_c_prespike.maxsim_score(query_emb, page_emb))
        )
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Text retrieval (production stack)
# ---------------------------------------------------------------------------


def text_rank_chunks_for_doc(
    query_text: str,
    *,
    target_doc_id: str,
    top_n_return: int = 25,
) -> Tuple[List[Tuple[int, float]], List[Dict[str, Any]]]:
    """Run production text retrieval, then map top chunks → pages for the target doc.

    Returns (per-page-rank, raw-chunk-list). Per-page-rank is the
    deduplicated sequence of pages encountered as we walk down the
    chunk list, with each page's score = max chunk score on that page.
    Pages NOT belonging to `target_doc_id` are dropped from the page
    ranking so we measure "did the retriever land on the right page
    within the doc" cleanly, but the raw chunks include everything
    returned (so global doc-id distribution is auditable separately).
    """
    from mmrag_v2.retrieval.pipeline import retrieve_hybrid_reranked

    raw = retrieve_hybrid_reranked(
        query_text,
        top_n_return=top_n_return,
        top_k_retrieve=25,
        top_n_fuse=25,
    )
    chunks = []
    page_best_score: Dict[int, float] = {}
    page_first_seen: Dict[int, int] = {}
    for idx, r in enumerate(raw):
        payload = r.get("payload") or {}
        chunk = {
            "chunk_id": payload.get("chunk_id"),
            "doc_id": payload.get("doc_id"),
            "page_number": payload.get("metadata", {}).get("page_number")
                if isinstance(payload.get("metadata"), dict)
                else payload.get("page_number"),
            "rerank_score": float(r.get("rerank_score", 0.0)),
            "score": float(r.get("score", 0.0)),
            "rank": idx,
        }
        chunks.append(chunk)
        if chunk["doc_id"] != target_doc_id:
            continue
        pg = chunk["page_number"]
        if pg is None:
            continue
        if pg not in page_best_score:
            page_best_score[pg] = chunk["rerank_score"]
            page_first_seen[pg] = idx
        else:
            page_best_score[pg] = max(page_best_score[pg], chunk["rerank_score"])
    page_rank = sorted(
        page_best_score.items(),
        key=lambda pair: (-pair[1], page_first_seen[pair[0]]),
    )
    return page_rank, chunks


# ---------------------------------------------------------------------------
# C-spike driver
# ---------------------------------------------------------------------------


@dataclass
class CSpikeReport:
    """Aggregate verdict + per-query traces."""
    query_results: List[QueryResult]
    target_doc_id: str
    target_doc_pdf: str
    page_count: int
    model_id: str
    render_dpi: int

    @property
    def n(self) -> int:
        return len(self.query_results)

    @property
    def visual_top1_accuracy(self) -> float:
        if not self.query_results:
            return 0.0
        return sum(1 for q in self.query_results if q.visual_passes) / self.n

    @property
    def text_top1_accuracy(self) -> float:
        if not self.query_results:
            return 0.0
        return sum(1 for q in self.query_results if q.text_passes) / self.n

    @property
    def text_failed_queries(self) -> List[QueryResult]:
        return [q for q in self.query_results if not q.text_passes]

    @property
    def text_passed_queries(self) -> List[QueryResult]:
        return [q for q in self.query_results if q.text_passes]

    @property
    def visual_recovery_rate_on_text_failed(self) -> float:
        """PASS A numerator: visual top-1 = gold among text-failed queries."""
        failed = self.text_failed_queries
        if not failed:
            return float("nan")  # no text-failed queries → recovery undefined
        return sum(1 for q in failed if q.visual_passes) / len(failed)

    @property
    def visual_harm_rate_on_text_passed(self) -> float:
        """PASS A denominator-guard: visual breaks queries text gets right."""
        passed = self.text_passed_queries
        if not passed:
            return float("nan")
        return sum(1 for q in passed if not q.visual_passes) / len(passed)

    @property
    def pass_a(self) -> bool:
        """Charter §4.2 step 2 #7: visual recovers ≥60% of text-failed
        queries AND does not harm queries text gets right."""
        recovery = self.visual_recovery_rate_on_text_failed
        harm = self.visual_harm_rate_on_text_passed
        # NaN-aware: if no text-failed queries, PASS A is vacuously OK
        # (text retrieval is already perfect). If no text-passed queries,
        # harm is undefined and we cannot certify "without harming".
        if recovery != recovery:  # NaN
            return True  # vacuous PASS — text is already perfect
        if harm != harm:  # NaN
            return False  # nothing to compare against → no certification
        return recovery >= 0.60 and harm <= 0.10


def run_cspike(
    *,
    pdf_path: Path,
    target_doc_id: str,
    page_count: int,
    queries: List[Dict[str, Any]],
    model_id: str,
    render_dpi: int,
    output_dir: Optional[Path] = None,
) -> CSpikeReport:
    log = logging.getLogger("v3_c_spike")

    page_embeddings = embed_all_pages_once(
        pdf_path,
        page_count=page_count,
        render_dpi=render_dpi,
        model_id=model_id,
        output_dir=output_dir,
    )

    results: List[QueryResult] = []
    for spec in queries:
        log.info("Query %s (gold=%d): %s",
                 spec["id"], spec["gold_page"], spec["text"][:60])
        qr = QueryResult(
            query_id=spec["id"],
            query_text=spec["text"],
            gold_page=spec["gold_page"],
        )

        t0 = time.monotonic()
        visual_ranking = visual_rank_pages(
            spec["text"], page_embeddings, model_id=model_id
        )
        qr.visual_top_pages = visual_ranking[:5]
        qr.timing_ms["visual"] = (time.monotonic() - t0) * 1000.0

        t0 = time.monotonic()
        text_page_rank, text_chunks = text_rank_chunks_for_doc(
            spec["text"], target_doc_id=target_doc_id, top_n_return=25
        )
        qr.text_top_pages = text_page_rank[:5]
        qr.text_top_chunks = text_chunks[:5]
        qr.timing_ms["text"] = (time.monotonic() - t0) * 1000.0

        log.info(
            "  visual top1 = page %s (gold=%d) %s",
            qr.visual_top1_page, qr.gold_page,
            "PASS" if qr.visual_passes else "miss",
        )
        log.info(
            "  text   top1 = page %s (gold=%d) %s",
            qr.text_top1_page, qr.gold_page,
            "PASS" if qr.text_passes else "miss",
        )
        results.append(qr)

    return CSpikeReport(
        query_results=results,
        target_doc_id=target_doc_id,
        target_doc_pdf=str(pdf_path),
        page_count=page_count,
        model_id=model_id,
        render_dpi=render_dpi,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_report(report: CSpikeReport) -> str:
    lines = [
        f"V3 Phase C C-Spike — {report.target_doc_id} ({Path(report.target_doc_pdf).name})",
        f"  model:        {report.model_id}",
        f"  render DPI:   {report.render_dpi}",
        f"  pages:        {report.page_count}",
        f"  queries:      {report.n}",
        "",
        "Per-query results:",
        f"  {'ID':<5} {'gold':>4} {'vT1':>4} {'tT1':>4} {'V?':>3} {'T?':>3} {'visTop3':<24} {'txtTop3':<24}",
    ]
    for q in report.query_results:
        vis_top3 = ",".join(f"{p}({s:.1f})" for p, s in q.visual_top_pages[:3])
        txt_top3 = ",".join(f"{p}({s:.2f})" for p, s in q.text_top_pages[:3]) or "(none)"
        lines.append(
            f"  {q.query_id:<5} {q.gold_page:>4} "
            f"{q.visual_top1_page if q.visual_top1_page else '-':>4} "
            f"{q.text_top1_page if q.text_top1_page else '-':>4} "
            f"{'P' if q.visual_passes else '.':>3} "
            f"{'P' if q.text_passes else '.':>3} "
            f"{vis_top3:<24} {txt_top3:<24}"
        )
    lines += [
        "",
        "Aggregate:",
        f"  Visual top-1 accuracy:  {report.visual_top1_accuracy:.2%} "
        f"({sum(1 for q in report.query_results if q.visual_passes)}/{report.n})",
        f"  Text   top-1 accuracy:  {report.text_top1_accuracy:.2%} "
        f"({sum(1 for q in report.query_results if q.text_passes)}/{report.n})",
        f"  Text-failed queries:    {len(report.text_failed_queries)}",
        f"  Text-passed queries:    {len(report.text_passed_queries)}",
    ]
    if report.text_failed_queries:
        lines.append(
            f"  Visual recovery on text-failed: "
            f"{report.visual_recovery_rate_on_text_failed:.2%} "
            f"(threshold ≥60%)"
        )
    if report.text_passed_queries:
        lines.append(
            f"  Visual harm on text-passed:     "
            f"{report.visual_harm_rate_on_text_passed:.2%} "
            f"(threshold ≤10%)"
        )
    lines.append(f"  Charter §4.2 step 2 PASS A verdict: "
                 f"{'PASS' if report.pass_a else 'FAIL'}")
    return "\n".join(lines)


def _report_to_json(report: CSpikeReport) -> str:
    payload = {
        "target_doc_id": report.target_doc_id,
        "target_doc_pdf": report.target_doc_pdf,
        "page_count": report.page_count,
        "model_id": report.model_id,
        "render_dpi": report.render_dpi,
        "n_queries": report.n,
        "visual_top1_accuracy": report.visual_top1_accuracy,
        "text_top1_accuracy": report.text_top1_accuracy,
        "n_text_failed": len(report.text_failed_queries),
        "n_text_passed": len(report.text_passed_queries),
        "visual_recovery_on_text_failed": report.visual_recovery_rate_on_text_failed,
        "visual_harm_on_text_passed": report.visual_harm_rate_on_text_passed,
        "pass_a": report.pass_a,
        "queries": [asdict(q) for q in report.query_results],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "V3 Phase C C-spike (Charter §4.2 step 2). Single-doc full "
            "quantitative test on ATZ_Elektronik_German."
        )
    )
    parser.add_argument("--pdf", type=Path, default=ATZ_PDF_PATH,
                        help="Source PDF (default: ATZ_Elektronik_German)")
    parser.add_argument("--doc-id", type=str, default=ATZ_DOC_ID,
                        help="doc_id of the ingested document for text-leg mapping")
    parser.add_argument("--page-count", type=int, default=ATZ_PAGE_COUNT)
    parser.add_argument("--render-dpi", type=int, default=200)
    parser.add_argument("--model-id", type=str, default="vidore/colpali-v1.3")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optional dir to persist page renders")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Optional path to write the full JSON report")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s | %(message)s",
    )

    if not args.pdf.exists():
        parser.error(f"PDF not found: {args.pdf}")

    report = run_cspike(
        pdf_path=args.pdf,
        target_doc_id=args.doc_id,
        page_count=args.page_count,
        queries=ATZ_QUERIES,
        model_id=args.model_id,
        render_dpi=args.render_dpi,
        output_dir=args.output_dir,
    )

    print(_format_report(report))
    if args.json_out:
        args.json_out.write_text(_report_to_json(report), encoding="utf-8")
        print(f"\nJSON written: {args.json_out}")
    return 0 if report.pass_a else 1


if __name__ == "__main__":
    sys.exit(main())
