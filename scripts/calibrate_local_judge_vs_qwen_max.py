#!/usr/bin/env python3
"""v2.14 Phase 0 — calibrate a local LLM judge against qwen-max.

Reads a v2.13 P1-style soak work.jsonl (where qwen-max has already
judged each query on relevance / format / faithfulness), reconstructs
the exact JUDGE_SYSTEM + JUDGE_USER_TEMPLATE used by synthetic_soak.py,
sends the same prompts to a local OpenAI-compatible LLM (default:
Qwen2.5-14B-Instruct served via vLLM on the GX10 at
http://10.0.10.239:8000/v1), and computes per-axis agreement vs the
qwen-max ground truth.

Outputs a markdown calibration report.

Why this matters: v2.14's local-LLM integration (judge / HyDE /
soak query generation) needs concrete agreement evidence before
the local model can be trusted for the exploration loop. The
agreement threshold:
  - >85% per axis: local LLM trustworthy for exploration soaks
                   (e.g. RRF weight sweeps, top_k sweeps, prompt iteration)
  - 70-85%      : restricted to HyDE-only (weaker semantics still
                   help retrieval)
  - <70%        : not usable; either pick a stronger local model
                   or stay on cloud judging

Usage:
  python scripts/calibrate_local_judge_vs_qwen_max.py \\
    --work-path output/soak/v2.13_p1_omlx/work.jsonl \\
    --local-url http://10.0.10.239:8000/v1 \\
    --local-model Qwen/Qwen2.5-14B-Instruct \\
    --report-path docs/CALIBRATION_2026-05-22_v2.14_p0_local_judge.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Reuse the exact judge prompt from synthetic_soak so the comparison
# is apples-to-apples (same instructions to both judges).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_sk", SCRIPTS / "synthetic_soak.py")
_sk = _ilu.module_from_spec(_spec)
# synthetic_soak imports search_qdrant + ingest_to_qdrant at import time;
# we only need the JUDGE_* + _extract_json + _read_work helpers.
_spec.loader.exec_module(_sk)
JUDGE_SYSTEM = _sk.JUDGE_SYSTEM
JUDGE_USER_TEMPLATE = _sk.JUDGE_USER_TEMPLATE
_extract_json = _sk._extract_json
_read_work = _sk._read_work

DEFAULT_LOCAL_URL = "http://10.0.10.239:8000/v1"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-14B-Instruct"


def _post_chat(url: str, model: str, messages: list[dict], *,
               temperature: float = 0.0, max_tokens: int = 250,
               timeout: int = 60, retries: int = 2) -> str | None:
    """Call an OpenAI-compatible /v1/chat/completions endpoint and
    return the assistant message content (or None on failure)."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                f"{url.rstrip('/')}/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            choices = data.get("choices") or []
            if not choices:
                return None
            return (choices[0].get("message") or {}).get("content")
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8")[:200]
            except Exception:
                detail = ""
            last_err = f"HTTP {e.code}: {detail}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = repr(e)
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    print(f"    ! local-judge call failed: {last_err}", file=sys.stderr)
    return None


def _judge_local(url: str, model: str, query: str, gold: str,
                 retrieved: str, source_file: str, page) -> dict | None:
    """Apply the exact JUDGE prompt to the local LLM and parse JSON."""
    user_prompt = JUDGE_USER_TEMPLATE.format(
        query=query,
        gold=gold[:1500],
        source_file=source_file or "",
        page=page,
        retrieved=retrieved[:1500],
    )
    raw = _post_chat(
        url, model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0, max_tokens=250,
    )
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


def _agreement_table(pairs: list[tuple[int, int]]) -> dict:
    """Compute exact-match agreement + ±1 agreement + class-level breakdown."""
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    exact = sum(1 for a, b in pairs if a == b)
    within_1 = sum(1 for a, b in pairs if abs(a - b) <= 1)
    # Treat as binary (0 vs >=1) and (<=1 vs 2)
    binary_zero = sum(1 for a, b in pairs if (a == 0) == (b == 0))
    binary_top = sum(1 for a, b in pairs if (a == 2) == (b == 2))
    # Confusion matrix
    confusion = Counter()
    for gt, local in pairs:
        confusion[(gt, local)] += 1
    return {
        "n": n,
        "exact_pct": round(100 * exact / n, 1),
        "within1_pct": round(100 * within_1 / n, 1),
        "binary_zero_pct": round(100 * binary_zero / n, 1),
        "binary_top_pct": round(100 * binary_top / n, 1),
        "confusion": dict(confusion),  # {(gt, local): count}
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-path", required=True,
                   help="Path to the soak work.jsonl whose qwen-max judgments are the ground truth.")
    p.add_argument("--local-url", default=DEFAULT_LOCAL_URL,
                   help=f"Local OpenAI-compatible /v1 base URL (default: {DEFAULT_LOCAL_URL})")
    p.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL,
                   help=f"Local model id (default: {DEFAULT_LOCAL_MODEL})")
    p.add_argument("--report-path", default=None,
                   help="Output markdown report (default: docs/CALIBRATION_<date>_v2.14_p0_local_judge.md)")
    p.add_argument("--max-queries", type=int, default=0,
                   help="If >0, only judge the first N already-qwen-max-judged queries (for quick smoke).")
    p.add_argument("--results-cache", default=None,
                   help="JSON cache of per-query local judgments; resumes from cache if present.")
    args = p.parse_args()

    work_path = Path(args.work_path)
    if not work_path.exists():
        print(f"ERROR: work file missing: {work_path}", file=sys.stderr)
        return 2

    if args.report_path is None:
        args.report_path = f"docs/CALIBRATION_{datetime.now().strftime('%Y-%m-%d')}_v2.14_p0_local_judge.md"
    report_path = Path(args.report_path)

    if args.results_cache is None:
        cache_path = work_path.parent / "calibration_local_judgments.json"
    else:
        cache_path = Path(args.results_cache)

    rows = _read_work(work_path)
    targets = []  # (query_id, ground_truth_judgment, retrieval_top1, gold_content, query_text)
    for r in rows:
        gold = r.get("gold_content") or ""
        for q in (r.get("queries") or []):
            gt = q.get("judgment") or {}
            if gt.get("relevance") is None:
                continue  # only consider qwen-max-judged queries
            retrieval = q.get("retrieval") or {}
            top = retrieval.get("top_k") or []
            if not top:
                continue
            targets.append((q["query_id"], gt, top[0], gold, q["query_text"]))

    if args.max_queries:
        targets = targets[:args.max_queries]

    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            cache = {}
    print(f"Calibration: {len(targets)} qwen-max-judged queries; cache has {len(cache)} prior results.")
    print(f"Local judge: {args.local_model} @ {args.local_url}")

    new_count = 0
    for i, (qid, gt, top1, gold, query_text) in enumerate(targets, 1):
        if qid in cache:
            continue
        local = _judge_local(
            args.local_url, args.local_model,
            query=query_text, gold=gold,
            retrieved=top1.get("content") or "",
            source_file=top1.get("source_file") or "",
            page=top1.get("page_number"),
        )
        if local is None:
            cache[qid] = {"status": "parse_or_call_failed"}
        else:
            cache[qid] = {
                "status": "ok",
                "local": local,
                "qwen_max": {k: gt.get(k) for k in ("relevance", "format", "faithfulness", "rationale")},
            }
        new_count += 1
        if new_count % 20 == 0 or i == len(targets):
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
            print(f"  [{i}/{len(targets)}] cached {new_count} new judgments")
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    print(f"Cache final: {len(cache)} entries at {cache_path}")

    # Compute per-axis agreement.
    rel_pairs, fmt_pairs, faith_pairs = [], [], []
    parse_fails = 0
    for qid, entry in cache.items():
        if entry.get("status") != "ok":
            parse_fails += 1
            continue
        gt = entry["qwen_max"]
        lo = entry["local"]
        rel_pairs.append((int(gt["relevance"]), int(lo["relevance"])))
        fmt_pairs.append((int(gt["format"]), int(lo["format"])))
        faith_pairs.append((int(gt["faithfulness"]), int(lo["faithfulness"])))

    rel = _agreement_table(rel_pairs)
    fmt = _agreement_table(fmt_pairs)
    faith = _agreement_table(faith_pairs)

    lines = []
    lines.append(f"# v2.14 Phase 0 Calibration — Local Judge vs qwen-max")
    lines.append(f"")
    lines.append(f"> Date: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"> Ground truth: `{work_path}` (qwen-max judgments from v2.13 P1 soak)")
    lines.append(f"> Local judge: `{args.local_model}` @ `{args.local_url}`")
    lines.append(f"> Sample size: {len(rel_pairs)} queries judged on both sides ({parse_fails} parse/call failures excluded)")
    lines.append(f"")
    lines.append("## Headline agreement (per axis)")
    lines.append("")
    lines.append("| Axis | n | Exact match | ±1 | Binary (0 vs ≥1) | Binary (≤1 vs 2) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, a in [("relevance", rel), ("format", fmt), ("faithfulness", faith)]:
        lines.append(
            f"| {name} | {a['n']} | **{a.get('exact_pct', 0)}%** | {a.get('within1_pct', 0)}% | "
            f"{a.get('binary_zero_pct', 0)}% | {a.get('binary_top_pct', 0)}% |"
        )
    lines.append("")
    lines.append("## Disposition by exact-match %")
    lines.append("")
    lines.append("| Threshold | Recommended use |")
    lines.append("|---|---|")
    lines.append("| ≥85% | Local judge trustworthy for exploration soaks (hyperparameter sweeps, prompt iteration) |")
    lines.append("| 70-85% | Restrict to HyDE-only (weaker semantics still help retrieval) |")
    lines.append("| <70% | Not usable; pick a stronger local model or stay on cloud judging |")
    lines.append("")

    # Per-axis disposition
    lines.append("**Per-axis verdicts:**")
    lines.append("")
    for name, a in [("relevance", rel), ("format", fmt), ("faithfulness", faith)]:
        pct = a.get("exact_pct", 0)
        if pct >= 85:
            verdict = "✓ TRUSTWORTHY — use for exploration soaks"
        elif pct >= 70:
            verdict = "⚠ RESTRICTED — HyDE-only"
        else:
            verdict = "✗ NOT USABLE — stay on cloud for this axis"
        lines.append(f"- **{name}**: {pct}% exact → {verdict}")
    lines.append("")
    lines.append("## Confusion matrices (qwen-max → local)")
    lines.append("")
    for name, a in [("relevance", rel), ("format", fmt), ("faithfulness", faith)]:
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| qwen-max ↓ \\ local → | 0 | 1 | 2 |")
        lines.append("|---|---:|---:|---:|")
        conf = a.get("confusion", {})
        for gt_score in (0, 1, 2):
            row = [str(conf.get((gt_score, local_score), 0)) for local_score in (0, 1, 2)]
            lines.append(f"| **{gt_score}** | {row[0]} | {row[1]} | {row[2]} |")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(f"- {parse_fails} queries had a parse or call failure on the local side and are excluded from the agreement numbers.")
    lines.append("- Identical JUDGE prompt structure used on both sides (`JUDGE_SYSTEM` + `JUDGE_USER_TEMPLATE` from `scripts/synthetic_soak.py`).")
    lines.append("- Same retrieved chunks, same gold, same query texts — only the judge model differs.")
    lines.append("- Cache: `output/soak/v2.13_p1_omlx/calibration_local_judgments.json` (rerun with the same `--results-cache` to resume).")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
