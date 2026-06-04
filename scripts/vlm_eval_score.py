#!/usr/bin/env python
"""VLM-eval deterministic scorer (runs anywhere; NO model deps).

Reads each candidate's captured outputs (output/vlm_eval/runs/<label>/) + the
golden manifest and scores the deterministic axes, then prints a per-candidate
scorecard and a cross-candidate comparison. Format-agnostic (auto-detects
JSON / DocTags / markdown). The semantic axes (content completeness/fidelity)
are a SEPARATE judged pass; this covers what can be measured without a judge.

Usage:
    python scripts/vlm_eval_score.py output/vlm_eval/runs \
        --golden-dir output/vlm_eval/golden_set
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# A markdown grid: a pipe row plus a |---|---| separator row.
_MD_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$\n\s*\|[\s:|-]*-[\s:|-]*\|\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"```[\w-]*\n(.*?)```", re.DOTALL)
_REPEAT_RE = re.compile(r"(.{1,80}?)\1{7,}", re.DOTALL)
_BBOX_HINTS = ("bbox", '"box"', "<loc_", "location", '"poly"', "x_min", 'x1"')
_DOCTAGS_HINTS = ("<doctag", "<otsl", "<text>", "<picture>", "<table>", "<code>", "<loc_")


def _detect_format(text: str) -> str:
    t = text.lstrip()
    if any(h in text for h in _DOCTAGS_HINTS):
        return "doctags"
    if t.startswith("{") or t.startswith("["):
        return "json"
    return "markdown"


def _content_text(text: str, fmt: str) -> str:
    """The CONTENT to run markdown/code/repetition checks on.

    For JSON, the table markdown / fenced code lives INSIDE string values
    (escaped \\n), so parse and concatenate the content fields (json.loads
    un-escapes). For markdown/doctags the raw text is the content.
    """
    if fmt != "json":
        return text
    try:
        obj = json.loads(text)
    except Exception:
        return text
    parts: list[str] = []

    def _walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("content", "text", "markdown", "md") and isinstance(v, str):
                    parts.append(v)
                else:
                    _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)

    _walk(obj)
    return "\n".join(parts) if parts else text


def _parseable(text: str, fmt: str) -> bool:
    if fmt == "json":
        try:
            json.loads(text)
            return True
        except Exception:
            return False
    return bool(text.strip())  # markdown/doctags: non-empty is "parseable"


def _has_bbox(text: str, fmt: str) -> bool:
    return any(h in text for h in _BBOX_HINTS)


def _has_markdown_table(text: str) -> bool:
    return bool(_MD_TABLE_RE.search(text))


def _table_cells_nonempty(text: str) -> bool:
    # at least one grid data row with a non-empty cell (not just separators)
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and "---" not in s:
            cells = [c.strip() for c in s.strip("|").split("|")]
            if any(cells):
                return True
    return False


def _code_indent_score(text: str) -> float:
    m = _FENCE_RE.search(text)
    if not m:
        return 0.0  # no fenced code at all
    body = m.group(1)
    indented = sum(1 for ln in body.splitlines() if ln[:1] in (" ", "\t"))
    return 1.0 if indented >= 1 else 0.5  # fenced + indentation vs fenced flat


def _has_repetition(text: str) -> bool:
    return bool(_REPEAT_RE.search(text))


def score_candidate(run_dir: Path, manifest: list) -> dict:
    by_id = {e["id"]: e for e in manifest}
    rows = []
    for f in sorted(run_dir.glob("*.json")):
        if f.name == "run_meta.json":
            continue
        d = json.loads(f.read_text())
        pid = d["id"]
        cap = by_id.get(pid, {}).get("capability", "?")
        text = d.get("output") or ""
        fmt = _detect_format(text)
        content = _content_text(text, fmt)  # unescaped element content for JSON
        row = {
            "id": pid,
            "cap": cap,
            "fmt": fmt,
            "latency_s": d.get("latency_s"),
            "status": d.get("status"),
            "parseable": _parseable(text, fmt),
            "has_bbox": _has_bbox(text, fmt),
            "repetition": _has_repetition(content),
        }
        if cap == "table":
            row["table_md"] = _has_markdown_table(content) and _table_cells_nonempty(content)
        if cap == "code":
            row["code_indent"] = _code_indent_score(content)
        rows.append(row)
    return {"label": run_dir.name, "rows": rows}


def _agg(rows, key, cap=None, kind="mean"):
    vals = [r[key] for r in rows if key in r and (cap is None or r["cap"] == cap)]
    if not vals:
        return None
    if kind == "rate":
        return round(sum(bool(v) for v in vals) / len(vals), 2)
    return round(sum(vals) / len(vals), 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_root")
    ap.add_argument("--golden-dir", default="output/vlm_eval/golden_set")
    args = ap.parse_args()
    manifest = json.loads((Path(args.golden_dir) / "manifest.json").read_text())

    cards = []
    for run_dir in sorted(Path(args.runs_root).iterdir()):
        if run_dir.is_dir() and any(run_dir.glob("*.json")):
            cards.append(score_candidate(run_dir, manifest))

    if not cards:
        print(f"no candidate runs under {args.runs_root}")
        return 1

    hdr = (
        f"{'candidate':>16} | {'fmt':>8} {'bbox%':>5} {'parse%':>6} "
        f"{'tbl_md%':>7} {'code_ind':>8} {'rep%':>5} {'med_lat':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for c in cards:
        rows = c["rows"]
        fmts = {r["fmt"] for r in rows}
        lat = sorted(r["latency_s"] for r in rows if r.get("latency_s") is not None)
        print(
            f"{c['label']:>16} | {('/'.join(sorted(fmts)))[:8]:>8} "
            f"{_agg(rows,'has_bbox',kind='rate') or 0:>5} "
            f"{_agg(rows,'parseable',kind='rate') or 0:>6} "
            f"{_agg(rows,'table_md',cap='table',kind='rate') if any(r['cap']=='table' for r in rows) else '-':>7} "
            f"{_agg(rows,'code_indent',cap='code') if any(r['cap']=='code' for r in rows) else '-':>8} "
            f"{_agg(rows,'repetition',kind='rate') or 0:>5} "
            f"{(lat[len(lat)//2] if lat else '-'):>7}"
        )
    print(
        "\nLEGEND: bbox%=pages emitting per-element bboxes (structural fit); "
        "tbl_md%=table pages with a non-empty markdown grid (R2); code_ind=code "
        "pages with fenced+indented code (R3, 0-1); rep%=pages with a >=8x loop "
        "(R6); med_lat=median s/page (R8). Semantic completeness/fidelity is the "
        "separate judged pass."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
