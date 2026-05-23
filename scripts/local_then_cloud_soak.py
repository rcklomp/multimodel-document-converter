#!/usr/bin/env python3
"""v2.14 Phase 4d — Tie-breaker soak harness.

Two-tier judging that combines local-vLLM (free) for everything with
cloud qwen-max (costly) only for "contested" queries:

  Stage 1: Local-vLLM judges every query that doesn't already have a
           `judgment_local` cached. Result stored at q["judgment_local"].
  Stage 2: Contested queries get re-judged by cloud qwen-max. Result
           stored at q["judgment"] (the standard scoring field).
  Stage 3: For non-contested queries, q["judgment"] is copied from
           q["judgment_local"] so downstream report consumers see a
           uniform score regardless of which judge produced it.

Definition of "contested" (governed by `--contested-axis-floor`,
default 2): a query is contested when EITHER the local judge failed
to parse OR ANY axis score is strictly below the floor. With the
default floor of 2, only queries where local rated 2/2/2 across all
three axes are considered "uncontested" — every other query gets a
cloud cross-check.

Why this works for the v2.14 Phase 0 verdict (all axes RESTRICTED,
27B is strict on format): the local judge tends to downgrade some
genuine 2s to 1s. Treating every non-2 as contested catches all
those potential false negatives without paying cloud cost for the
unanimous 2/2/2 queries (which empirically agree across judges per
the 2026-05-23 calibration confusion matrix).

Cost model (illustrative, 518-query fixture):
  - Full cloud:    518 calls × $0.001 = ~$0.50
  - Local-only:    518 calls × $0 = $0 (but RESTRICTED-axis coverage)
  - Tie-breaker:   518 local-free + N contested cloud = ~$0.001·N
    With ~40-50% local-agreement-with-cloud-on-perfect-scores
    observed in the 27B calibration, expect N ≈ 250-300 → ~$0.30,
    a ~40-50% cost reduction with cloud quality on the uncertain set.

Usage:
  python scripts/local_then_cloud_soak.py \\
    --work-path output/soak/<run>/work.jsonl \\
    --report-path docs/SOAK_$(date +%Y-%m-%d)_v2.14_p4d_tiebreaker.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Reuse helpers + judge prompts from the main soak harness so prompts
# stay byte-identical between scripts (calibration depends on this).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_sk", SCRIPTS / "synthetic_soak.py")
_sk = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sk)
JUDGE_SYSTEM = _sk.JUDGE_SYSTEM
JUDGE_USER_TEMPLATE = _sk.JUDGE_USER_TEMPLATE
JUDGE_MODEL = _sk.JUDGE_MODEL  # cloud default: qwen-max
_extract_json = _sk._extract_json
_read_work = _sk._read_work
_write_work = _sk._write_work
_call_dashscope = _sk._call_dashscope
_call_vllm = _sk._call_vllm
VLLM_GEN_DEFAULT_URL = _sk.VLLM_GEN_DEFAULT_URL
VLLM_GEN_DEFAULT_MODEL = _sk.VLLM_GEN_DEFAULT_MODEL


def _build_judge_messages(query_text: str, gold: str, top1: dict) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
            query=query_text,
            gold=gold[:1500],
            source_file=top1.get("source_file") or "",
            page=top1.get("page_number"),
            retrieved=(top1.get("content") or "")[:1500],
        )},
    ]


def _parse_judgment(raw: str | None) -> dict | None:
    if not raw:
        return None
    parsed = _extract_json(raw, "object")
    if not isinstance(parsed, dict) or "relevance" not in parsed:
        return None
    try:
        return {
            "relevance": int(parsed.get("relevance", 0)),
            "format": int(parsed.get("format", 0)),
            "faithfulness": int(parsed.get("faithfulness", 0)),
            "rationale": str(parsed.get("rationale", ""))[:300],
        }
    except (ValueError, TypeError):
        return None


def _is_contested(judgment: dict | None, axis_floor: int) -> bool:
    """A query is contested when the local judge failed to parse OR any
    axis score is strictly below the floor. Default floor is 2 → every
    non-2/2/2 query gets a cloud cross-check."""
    if judgment is None:
        return True
    for axis in ("relevance", "format", "faithfulness"):
        try:
            if int(judgment.get(axis, -1)) < axis_floor:
                return True
        except (ValueError, TypeError):
            return True
    return False


def stage_local_judge(rows: list[dict], *, local_url: str, local_model: str) -> int:
    """Add `judgment_local` to every in-scope query that's missing one.
    Returns the number of new local judgments produced."""
    new_count = 0
    targets = [
        (r_idx, q_idx) for r_idx, r in enumerate(rows)
        for q_idx, q in enumerate(r.get("queries") or [])
        if not q.get("judgment_local")
        and (q.get("retrieval") or {}).get("top_k")
        and q.get("_tiebreak_in_scope")
    ]
    print(f"  local-judge: {len(targets)} queries to score on local vLLM")
    for n, (r_idx, q_idx) in enumerate(targets, 1):
        r = rows[r_idx]
        q = r["queries"][q_idx]
        top1 = (q["retrieval"]["top_k"] or [{}])[0]
        raw = _call_vllm(
            local_url, local_model,
            messages=_build_judge_messages(q["query_text"], r["gold_content"], top1),
            temperature=0.0, max_tokens=250,
        )
        judgment = _parse_judgment(raw)
        q["judgment_local"] = judgment or {"status": "parse_or_call_failed"}
        new_count += 1
        if n % 20 == 0:
            print(f"    [{n}/{len(targets)}] local-judged")
    return new_count


def stage_cloud_tiebreak(
    rows: list[dict], api_key: str, *, axis_floor: int, cloud_model: str,
) -> tuple[int, int, int]:
    """For contested queries, run cloud judge and overwrite `judgment`.
    For uncontested queries, copy `judgment_local` → `judgment` so
    downstream report consumers see a uniform field.

    Returns (cloud_calls_made, uncontested_promoted, contested_total)."""
    cloud_calls = 0
    uncontested_promoted = 0
    contested = 0
    for r in rows:
        for q in (r.get("queries") or []):
            if not q.get("_tiebreak_in_scope"):
                continue
            local = q.get("judgment_local") or {}
            # Treat the sentinel as a missing judgment.
            if isinstance(local, dict) and local.get("status") == "parse_or_call_failed":
                local = None
            if not _is_contested(local, axis_floor):
                # Uncontested: copy local to canonical judgment field if
                # not already set by a prior cloud run.
                if not q.get("judgment") or q["judgment"].get("relevance") is None:
                    q["judgment"] = {**local, "judge_source": "local"}
                    uncontested_promoted += 1
                continue
            # Contested: run cloud unless already done.
            contested += 1
            if q.get("judgment") and q["judgment"].get("relevance") is not None \
               and q["judgment"].get("judge_source") in (None, "cloud"):
                continue
            top = (q.get("retrieval") or {}).get("top_k") or []
            if not top:
                continue
            raw = _call_dashscope(
                api_key, cloud_model,
                messages=_build_judge_messages(q["query_text"], r["gold_content"], top[0]),
                temperature=0.0, max_tokens=200,
            )
            parsed = _parse_judgment(raw)
            if parsed is None:
                print(f"    ! cloud parse failed for {q.get('query_id')}; keeping local",
                      file=sys.stderr)
                if local is not None:
                    q["judgment"] = {**local, "judge_source": "local_fallback"}
                continue
            q["judgment"] = {**parsed, "judge_source": "cloud"}
            cloud_calls += 1
    return cloud_calls, uncontested_promoted, contested


def _agreement_on_re_judged(rows: list[dict]) -> dict:
    """For queries where BOTH local and cloud judgments exist, compute
    per-axis exact-match agreement. Tells us how often the local was
    actually right on the contested set (i.e. how much cloud paid for
    cases that didn't change)."""
    counts = {"n": 0, "rel": 0, "fmt": 0, "faith": 0,
              "rel_diff_sum": 0, "fmt_diff_sum": 0, "faith_diff_sum": 0}
    for r in rows:
        for q in (r.get("queries") or []):
            local = q.get("judgment_local") or {}
            cloud = q.get("judgment") or {}
            if not isinstance(local, dict) or local.get("status") == "parse_or_call_failed":
                continue
            if cloud.get("judge_source") != "cloud":
                continue
            counts["n"] += 1
            for axis_short, axis_full in [("rel", "relevance"), ("fmt", "format"),
                                          ("faith", "faithfulness")]:
                try:
                    l_val = int(local.get(axis_full, -1))
                    c_val = int(cloud.get(axis_full, -1))
                except (ValueError, TypeError):
                    continue
                if l_val == c_val:
                    counts[axis_short] += 1
                counts[f"{axis_short}_diff_sum"] += abs(l_val - c_val)
    return counts


def _write_report(rows: list[dict], report_path: Path, *,
                  local_model: str, cloud_model: str,
                  cloud_calls: int, contested: int, axis_floor: int,
                  total_queries: int, local_new: int) -> None:
    agree = _agreement_on_re_judged(rows)
    n = agree["n"]
    rel_pct = 100 * agree["rel"] / n if n else 0
    fmt_pct = 100 * agree["fmt"] / n if n else 0
    faith_pct = 100 * agree["faith"] / n if n else 0
    cost_savings_pct = 100 * (1 - cloud_calls / max(total_queries, 1))

    lines = [
        "# v2.14 Phase 4d — Tie-Breaker Soak Report",
        "",
        f"> Date: {datetime.now().strftime('%Y-%m-%d')}",
        f"> Local judge: `{local_model}`",
        f"> Cloud judge: `{cloud_model}` (Dashscope)",
        f"> Contested-axis floor: {axis_floor} (queries with any axis < {axis_floor} get cloud re-judge)",
        "",
        "## Headline",
        "",
        f"- Total queries with retrieval: {total_queries}",
        f"- Local judgments produced this run: {local_new}",
        f"- Contested queries (re-judged by cloud): {contested}",
        f"- Cloud API calls actually made: {cloud_calls}",
        f"- Cost reduction vs. full-cloud: **{cost_savings_pct:.1f}%**",
        f"  (full-cloud baseline = {total_queries} calls; this run = {cloud_calls})",
        "",
        "## Local-vs-cloud agreement on the re-judged set",
        "",
        f"Sample size: {n} queries (where both local + cloud judgments exist).",
        "",
        "| Axis | Exact match | Avg |delta| |",
        "|---|---:|---:|",
    ]
    for short, full in [("rel", "relevance"), ("fmt", "format"),
                        ("faith", "faithfulness")]:
        pct = 100 * agree[short] / n if n else 0
        avg_diff = agree[f"{short}_diff_sum"] / n if n else 0
        lines.append(f"| {full} | {pct:.1f}% | {avg_diff:.2f} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if n == 0:
        lines.append("No re-judged queries — cloud was not invoked on this run.")
    else:
        if rel_pct + fmt_pct + faith_pct >= 270:  # avg ≥ 90%
            lines.append(
                "High agreement on the re-judged set — most cloud calls "
                "CONFIRMED the local judgment. Consider raising the contested "
                "floor (e.g. only re-judge when ANY axis is 0) to save more "
                "cloud calls on the next run."
            )
        elif rel_pct + fmt_pct + faith_pct < 180:  # avg < 60%
            lines.append(
                "LOW agreement on the re-judged set — cloud frequently "
                "OVERRODE local. The tie-breaker is doing real work; do not "
                "raise the floor."
            )
        else:
            lines.append(
                "Moderate agreement — the tie-breaker pattern is operating "
                "in its intended band. Floor setting looks right for this "
                "endpoint's calibration profile."
            )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("Per-query `judgment.judge_source` records which side made the "
                 "final call:")
    sources: dict[str, int] = {}
    for r in rows:
        for q in (r.get("queries") or []):
            src = (q.get("judgment") or {}).get("judge_source", "(missing)")
            sources[src] = sources.get(src, 0) + 1
    for k, v in sorted(sources.items()):
        lines.append(f"- `judge_source={k}`: {v}")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  report: wrote {report_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-path", required=True,
                   help="Path to the soak work.jsonl. Must already have retrieval results.")
    p.add_argument("--local-url", default=VLLM_GEN_DEFAULT_URL,
                   help=f"Local vLLM endpoint (default: {VLLM_GEN_DEFAULT_URL})")
    p.add_argument("--local-model", default=VLLM_GEN_DEFAULT_MODEL,
                   help=f"Local model id (default: {VLLM_GEN_DEFAULT_MODEL})")
    p.add_argument("--cloud-model", default=JUDGE_MODEL,
                   help=f"Cloud judge model (default: {JUDGE_MODEL})")
    p.add_argument("--contested-axis-floor", type=int, default=2,
                   help="Any axis score strictly below this floor flags the "
                        "query as contested (re-judged by cloud). Default 2 → "
                        "every non-2/2/2 is contested.")
    p.add_argument("--report-path",
                   default=str(REPO_ROOT / "docs" /
                               f"SOAK_{datetime.now().strftime('%Y-%m-%d')}_v2.14_p4d_tiebreaker.md"))
    p.add_argument("--max-queries", type=int, default=0,
                   help="If >0, only process the first N queries (smoke).")
    p.add_argument("--skip-local", action="store_true",
                   help="Skip the local-judge stage (use existing judgment_local "
                        "cache; only run the cloud tie-break and report).")
    p.add_argument("--skip-cloud", action="store_true",
                   help="Skip the cloud tie-break stage (local-only run; useful "
                        "to populate the judgment_local cache before deciding "
                        "to spend cloud budget).")
    args = p.parse_args()

    work_path = Path(args.work_path)
    if not work_path.exists():
        print(f"ERROR: work file missing: {work_path}", file=sys.stderr)
        return 2

    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not args.skip_cloud and not api_key:
        print("ERROR: DASHSCOPE_API_KEY env var is not set; required for cloud "
              "tie-break stage. Pass --skip-cloud to run local-only.",
              file=sys.stderr)
        return 2

    rows = _read_work(work_path)
    if not rows:
        print(f"ERROR: empty work file: {work_path}", file=sys.stderr)
        return 2

    # Tag the first --max-queries queries with `_tiebreak_in_scope=True`.
    # Stages honor this flag (non-destructive); the persisted JSONL retains
    # ALL queries, the flag just narrows what gets newly judged this run.
    in_scope_only = bool(args.max_queries)
    if in_scope_only:
        kept = 0
        for r in rows:
            for q in (r.get("queries") or []):
                q["_tiebreak_in_scope"] = (kept < args.max_queries)
                kept += 1
    else:
        for r in rows:
            for q in (r.get("queries") or []):
                q["_tiebreak_in_scope"] = True

    total_queries = sum(
        1 for r in rows for q in (r.get("queries") or [])
        if (q.get("retrieval") or {}).get("top_k") and q.get("_tiebreak_in_scope")
    )
    print(f"Tie-breaker run: {total_queries} queries in scope")

    def _flush() -> None:
        """Strip the ephemeral in-scope flag before writing the work file
        so the persisted JSONL stays clean."""
        for r in rows:
            for q in (r.get("queries") or []):
                q.pop("_tiebreak_in_scope", None)
        _write_work(work_path, rows)
        # Re-mark in-scope so subsequent stages still filter correctly.
        if in_scope_only:
            kept = 0
            for r in rows:
                for q in (r.get("queries") or []):
                    q["_tiebreak_in_scope"] = (kept < args.max_queries)
                    kept += 1
        else:
            for r in rows:
                for q in (r.get("queries") or []):
                    q["_tiebreak_in_scope"] = True

    local_new = 0
    if not args.skip_local:
        t0 = time.time()
        local_new = stage_local_judge(
            rows, local_url=args.local_url, local_model=args.local_model,
        )
        _flush()
        print(f"  local-judge: {local_new} new in {time.time()-t0:.0f}s; work flushed")

    cloud_calls = uncontested = contested = 0
    if not args.skip_cloud:
        t0 = time.time()
        cloud_calls, uncontested, contested = stage_cloud_tiebreak(
            rows, api_key,
            axis_floor=args.contested_axis_floor,
            cloud_model=args.cloud_model,
        )
        _flush()
        print(f"  cloud-tiebreak: contested={contested} cloud_calls={cloud_calls} "
              f"uncontested_promoted={uncontested} in {time.time()-t0:.0f}s")

    _write_report(
        rows, Path(args.report_path),
        local_model=args.local_model, cloud_model=args.cloud_model,
        cloud_calls=cloud_calls, contested=contested,
        axis_floor=args.contested_axis_floor,
        total_queries=total_queries, local_new=local_new,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
