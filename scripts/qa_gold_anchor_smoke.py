#!/usr/bin/env python3
"""Source-anchored acceptance smoke for one converted document.

PLAN_QUALITY_REMEDIATION_V1 WP-0.3. Every anchor in the spec is derived from the SOURCE
PDF (its text layer and rendered pages), never from a conversion run, so this check cannot
be satisfied by a fix that merely restates its own predicate (the regression-guard metrics in
qa_semantic_fidelity.py are mirrors and are not acceptance evidence).

Anchor kinds (see tests/fixtures/gold_anchor_specs/irjet.json):
  section_anchors      a phrase that lives in a known source section: the chunk holding it must
                       carry that section's heading as parent_heading. Evaluated only when the
                       engine labelled that section's heading (some chunk carries it as
                       parent_heading); otherwise recorded N/A with the reason. N/A is never a
                       pass, and too few evaluable anchors make the run inconclusive.
  front_matter_anchors a phrase whose chunk must NOT carry a given parent (the abstract must
                       not sit under INTRODUCTION).
  furniture_absent     source running header/footer strings: no TEXT chunk may contain them.
  reference_anchors    ALL tokens of one source reference entry must be found inside ONE chunk.
  table_anchors        a TABLE chunk with the expected tokens and at least N data rows.

Exit code 0 only when nothing failed and enough section anchors were evaluable.
Read-only over the JSONL; no network, no PDF access.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

Row = Dict[str, Any]
DEFAULT_SPEC = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "gold_anchor_specs" / "irjet.json"
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _parent(row: Row) -> str:
    return (
        ((row.get("metadata") or {}).get("hierarchy") or {}).get("parent_heading") or ""
    ).strip()


def load_rows(path: Path) -> List[Row]:
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [r for r in rows if r.get("object_type") != "ingestion_metadata"]


def _table_rows(content: str) -> int:
    n = 0
    for ln in (content or "").splitlines():
        s = ln.strip()
        if s.startswith("|") and not re.fullmatch(r"\|[\s:\-|]+\|?", s):
            n += 1
    return n


def evaluate(rows: List[Row], spec: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Return (anchor id, PASS|FAIL|NA, detail) triples."""
    texts = [r for r in rows if r.get("modality") == "text"]
    tables = [r for r in rows if r.get("modality") == "table"]
    sections: Dict[str, str] = spec.get("sections", {})
    present = {
        sid for sid, rx in sections.items() if any(re.search(rx, _parent(t), re.I) for t in texts)
    }
    out: List[Tuple[str, str, str]] = []

    for a in spec.get("section_anchors", []):
        aid, sec = "section:" + a["id"], a["section"]
        hits = [t for t in texts if re.search(a["phrase"], _norm(t.get("content")), re.I)]
        if not hits:
            out.append((aid, "FAIL", "anchor text is absent from every TEXT chunk (content lost)"))
            continue
        if sec not in present:
            out.append(
                (
                    aid,
                    "NA",
                    f"section '{sec}' heading was not assigned as a parent_heading by the engine in this output",
                )
            )
            continue
        bad = [
            f"{h.get('chunk_id')}: parent={_parent(h)!r}"
            for h in hits
            if not re.search(sections[sec], _parent(h), re.I)
        ]
        if bad:
            out.append((aid, "FAIL", "; ".join(bad[:3])))
        else:
            out.append((aid, "PASS", f"{len(hits)} chunk(s) under '{sec}'"))

    for a in spec.get("front_matter_anchors", []):
        aid = "front:" + a["id"]
        hits = [t for t in texts if re.search(a["phrase"], _norm(t.get("content")), re.I)]
        if not hits:
            out.append((aid, "FAIL", "anchor text is absent from every TEXT chunk (content lost)"))
            continue
        bad = [
            f"{h.get('chunk_id')}: parent={_parent(h)!r}"
            for h in hits
            if re.search(a["must_not_parent"], _parent(h), re.I)
        ]
        out.append(
            (aid, "FAIL", "; ".join(bad[:3])) if bad else (aid, "PASS", f"{len(hits)} chunk(s)")
        )

    for pat in spec.get("furniture_absent", []):
        aid = "furniture:" + pat
        hits = [
            t.get("chunk_id")
            for t in texts
            if re.search(re.escape(pat), _norm(t.get("content")), re.I)
        ]
        out.append(
            (aid, "FAIL", f"{len(hits)} chunk(s) contain it") if hits else (aid, "PASS", "absent")
        )

    for a in spec.get("reference_anchors", []):
        aid = "reference:" + a["id"]
        pats = a["all_of"]
        one = [t for t in texts if all(re.search(p, _norm(t.get("content")), re.I) for p in pats)]
        if one:
            out.append((aid, "PASS", f"intact in {one[0].get('chunk_id')}"))
            continue
        found = [p for p in pats if any(re.search(p, _norm(t.get("content")), re.I) for t in texts)]
        if len(found) == len(pats):
            out.append(
                (
                    aid,
                    "FAIL",
                    "entry is SPLIT across chunks (every token exists, none in one chunk)",
                )
            )
        else:
            missing = [p for p in pats if p not in found]
            out.append(
                (
                    aid,
                    "FAIL",
                    f"tokens not found intact in any chunk (missing or cut mid-token): {missing}",
                )
            )

    for a in spec.get("table_anchors", []):
        aid = "table:" + a["id"]
        ok = [
            t
            for t in tables
            if all(re.search(p, t.get("content") or "", re.I) for p in a["all_of"])
            and _table_rows(t.get("content")) >= a.get("min_rows", 1)
        ]
        out.append(
            (aid, "PASS", f"{ok[0].get('chunk_id')}")
            if ok
            else (aid, "FAIL", "no TABLE chunk with the expected tokens and row count")
        )
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    args = ap.parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    results = evaluate(load_rows(args.jsonl), spec)
    for aid, verdict, detail in results:
        print(f"ANCHOR {verdict:4} {aid} - {detail}")
    fails = sum(1 for _, v, _ in results if v == "FAIL")
    nas = sum(1 for _, v, _ in results if v == "NA")
    sec_eval = sum(1 for a, v, _ in results if a.startswith("section:") and v in ("PASS", "FAIL"))
    need = spec.get("min_evaluable_section_anchors", 0)
    print(f"summary fail={fails} na={nas} section_anchors_evaluable={sec_eval} (need >= {need})")
    if fails:
        print(f"GOLD_ANCHOR_FAIL {fails}")
        return 1
    if sec_eval < need:
        print(
            "GOLD_ANCHOR_INCONCLUSIVE too few section anchors were evaluable (the engine did not label enough headings)"
        )
        return 1
    print("GOLD_ANCHOR_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
