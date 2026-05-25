"""v2.16 Phase 1 — personal validation runner.

Loads JSON fixtures from `tests/fixtures/personal_validation_queries/`,
runs each query through `retrieve_hybrid_reranked` with v2.16-production
defaults, and reports per-class pass rates vs `target_pass_rate`.

Per-query PASS rule (ALL three must hold; see PLAN_V2.16.md §3 Phase 1):

  (a) top_5_gold_doc        — gold doc_id appears in retrieved top-5.
  (b) format_constraint     — table_value: top-1 modality == "table"
                              runnable_code: top-1 content `ast.parse`s
  (c) expected_anchor_regex — top-1 content matches ≥1 regex pattern
                              (chunk-shape-independent answer-correctness).

`gold_chunk_ids` (when provided) is RECORDED in the report for cross-cycle
sanity comparison; it does NOT gate PASS.

Exit code: nonzero if any class falls below its `target_pass_rate`.

Usage:
  python scripts/run_personal_validation.py \\
      --fixtures-dir tests/fixtures/personal_validation_queries \\
      --output docs/VALIDATION_REPORT_<today>.md
"""
from __future__ import annotations

import argparse
import ast
import datetime
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from mmrag_v2.retrieval.pipeline import retrieve_hybrid_reranked  # noqa: E402


@dataclass
class QueryResult:
    query_id: str
    query_text: str
    pass_overall: bool
    pass_gold_doc: Optional[bool]
    pass_format: Optional[bool]
    pass_anchor_regex: Optional[bool]
    top_1_doc_id: Optional[str]
    top_1_modality: Optional[str]
    top_1_chunk_id: Optional[str]
    top_5_doc_ids: list[str]
    matched_regex: Optional[str]
    gold_chunk_ids_authored: list[str]
    top_1_chunk_id_matches_gold: Optional[bool]
    note: Optional[str] = None


@dataclass
class ClassResult:
    class_name: str
    personal_importance: str
    target_pass_rate: float
    pass_rate: float = 0.0
    n_pass: int = 0
    n_total: int = 0
    queries: list[QueryResult] = field(default_factory=list)

    @property
    def meets_target(self) -> bool:
        return self.pass_rate >= self.target_pass_rate


def _load_canonical_basenames() -> set[str]:
    """Load the canonical-34 doc list from the rebuild script — the
    authoritative source of "which output/ directory names are
    canonical" (vs development snapshots like
    `CarOK_v2_14_p1_ocr` that share doc_id with the canonical
    `CarOK_voorraadtelling`)."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_rebuild_mod",
            str(_REPO_ROOT / "scripts/rebuild_mmrag_v2_8_for_rc1.py"),
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(mod)  # type: ignore
        # Post-Phase-0 rename will replace CANONICAL_34 with CANONICAL_DOCS;
        # tolerate both names so this script works through the transition.
        return set(getattr(mod, "CANONICAL_DOCS", None)
                   or getattr(mod, "CANONICAL_34", []))
    except Exception:
        return set()


def _build_doc_id_to_basename_map(output_root: Path) -> dict[str, str]:
    """Scan `output/*/ingestion.jsonl` headers to build a map from
    Qdrant payload `doc_id` (12-char content hash) → output basename
    (the canonical directory name, matching `CANONICAL_DOCS` and the
    validation fixture `class` field).

    Production Qdrant payloads carry `doc_id` as a hash and `source_file`
    as the ORIGINAL pdf filename (with spaces). Neither matches the
    snake_case basename used in the fixture `class` field, so we resolve
    via the ingestion.jsonl header.

    Many output/ directories are development snapshots that share a
    content-hash with the canonical doc. Preference order:
      1. Name is in the canonical doc list (rebuild script).
      2. Dotless name (no `.phaseN_baseline` suffix etc.).
      3. Shorter name.
    """
    mapping: dict[str, str] = {}
    if not output_root.exists():
        return mapping
    canonical = _load_canonical_basenames()
    for child in sorted(output_root.iterdir()):
        jsonl = child / "ingestion.jsonl"
        if not jsonl.exists():
            continue
        try:
            with jsonl.open() as fh:
                first = fh.readline()
            d = json.loads(first)
            if d.get("object_type") != "ingestion_metadata":
                continue
            hid = d.get("doc_id")
            if not hid:
                continue
            existing = mapping.get(hid)
            if existing is None:
                mapping[hid] = child.name
                continue
            existing_canon = existing in canonical
            child_canon = child.name in canonical
            if child_canon and not existing_canon:
                mapping[hid] = child.name
            elif existing_canon and not child_canon:
                pass
            else:
                # Same canonical status — fall back to dotless / shorter.
                e_dot = "." in existing
                c_dot = "." in child.name
                if e_dot and not c_dot:
                    mapping[hid] = child.name
                elif e_dot == c_dot and len(child.name) < len(existing):
                    mapping[hid] = child.name
        except (json.JSONDecodeError, OSError):
            continue
    return mapping


_DOC_ID_MAP: dict[str, str] = {}


def _doc_id_from_payload(payload: dict) -> Optional[str]:
    """Resolve the doc-level canonical basename from a chunk payload.

    Qdrant production payloads carry `doc_id` as a 12-char content hash
    (e.g. `46d689134b24`). The validation fixture `class` field uses
    the canonical basename (e.g. `CarOK_voorraadtelling`). The hash →
    basename map is built from `output/<basename>/ingestion.jsonl` headers
    on first call.
    """
    if not payload:
        return None
    hid = payload.get("doc_id")
    if hid and hid in _DOC_ID_MAP:
        return _DOC_ID_MAP[hid]
    # Soak-fixture path: `doc_dir` is already the basename.
    if payload.get("doc_dir"):
        return payload["doc_dir"]
    # Last resort — strip source_file extension (may not match basename).
    sf = payload.get("source_file")
    if sf:
        return Path(sf).stem.replace(" ", "_")
    return hid


def _modality_of(payload: dict) -> Optional[str]:
    return (payload or {}).get("modality")


def _content_of(payload: dict) -> str:
    return (payload or {}).get("content", "") or ""


def _ast_parses(text: str) -> bool:
    """Is `text` runnable Python? Strips common chunk-shell prefixes
    (>>>, leading "Example N-N." headers) before parsing."""
    if not text:
        return False
    # Strip Python REPL prompts and the common "Example N-N." header that
    # Fluent_Python chunks frequently carry.
    cleaned_lines = []
    for line in text.split("\n"):
        s = line.rstrip()
        if s.startswith(">>> "):
            cleaned_lines.append(s[4:])
        elif s == ">>>":
            continue
        elif s.startswith("```") or s.startswith("Example "):
            continue
        else:
            cleaned_lines.append(s)
    cleaned = "\n".join(cleaned_lines).strip()
    if not cleaned:
        return False
    try:
        ast.parse(cleaned)
        return True
    except SyntaxError:
        return False


def _check_format(format_constraint: Optional[str], payload: dict) -> Optional[bool]:
    if format_constraint is None:
        return None
    if format_constraint == "table_value":
        return _modality_of(payload) == "table"
    if format_constraint == "runnable_code":
        return _ast_parses(_content_of(payload))
    raise ValueError(f"unknown format_constraint: {format_constraint!r}")


def _check_regex(patterns: list[str], text: str) -> Optional[str]:
    """Return the first matching pattern, or None if none match.
    DOTALL is set so '.' crosses newlines (chunks contain newlines)."""
    if not patterns:
        return None
    for pat in patterns:
        if re.search(pat, text or "", flags=re.DOTALL | re.IGNORECASE):
            return pat
    return None


def evaluate_query(
    query: dict,
    class_name: str,
    *,
    dry_run: bool = False,
    retrieve_kwargs: Optional[dict] = None,
) -> QueryResult:
    """Run one query through retrieve_hybrid_reranked + evaluate PASS rules."""
    qid = query.get("id", "?")
    qtext = query.get("query_text", "")
    expected = query.get("expected") or {}
    require_gold_doc = bool(expected.get("top_5_gold_doc"))
    format_constraint = expected.get("format_constraint")  # optional
    regex_patterns = expected.get("expected_anchor_regexes") or []
    gold_chunks = expected.get("gold_chunk_ids") or []

    if dry_run:
        return QueryResult(
            query_id=qid, query_text=qtext,
            pass_overall=False,
            pass_gold_doc=None, pass_format=None, pass_anchor_regex=None,
            top_1_doc_id=None, top_1_modality=None, top_1_chunk_id=None,
            top_5_doc_ids=[], matched_regex=None,
            gold_chunk_ids_authored=gold_chunks,
            top_1_chunk_id_matches_gold=None,
            note="dry_run",
        )

    kwargs = dict(retrieve_kwargs or {})
    chunks = retrieve_hybrid_reranked(qtext, **kwargs)
    if not chunks:
        return QueryResult(
            query_id=qid, query_text=qtext,
            pass_overall=False,
            pass_gold_doc=False, pass_format=None, pass_anchor_regex=False,
            top_1_doc_id=None, top_1_modality=None, top_1_chunk_id=None,
            top_5_doc_ids=[], matched_regex=None,
            gold_chunk_ids_authored=gold_chunks,
            top_1_chunk_id_matches_gold=None,
            note="empty retrieval",
        )

    top_5_doc_ids = [_doc_id_from_payload(c.get("payload") or {}) for c in chunks[:5]]
    top_1 = chunks[0]
    p1 = top_1.get("payload") or {}
    top_1_doc_id = _doc_id_from_payload(p1)
    top_1_modality = _modality_of(p1)
    top_1_chunk_id = p1.get("chunk_id")
    top_1_content = _content_of(p1)

    pass_gold_doc = (class_name in top_5_doc_ids) if require_gold_doc else None
    pass_format = _check_format(format_constraint, p1)
    matched = _check_regex(regex_patterns, top_1_content)
    pass_anchor = (matched is not None) if regex_patterns else None
    top_1_matches_gold = (
        (top_1_chunk_id in gold_chunks) if gold_chunks else None
    )

    # PASS rule: every applicable check must be True. None == "not applicable".
    checks = [c for c in [pass_gold_doc, pass_format, pass_anchor] if c is not None]
    pass_overall = bool(checks) and all(checks)

    return QueryResult(
        query_id=qid, query_text=qtext,
        pass_overall=pass_overall,
        pass_gold_doc=pass_gold_doc,
        pass_format=pass_format,
        pass_anchor_regex=pass_anchor,
        top_1_doc_id=top_1_doc_id,
        top_1_modality=top_1_modality,
        top_1_chunk_id=top_1_chunk_id,
        top_5_doc_ids=top_5_doc_ids,
        matched_regex=matched,
        gold_chunk_ids_authored=gold_chunks,
        top_1_chunk_id_matches_gold=top_1_matches_gold,
    )


def load_fixture(path: Path) -> dict:
    """Load + minimally validate a fixture file."""
    d = json.loads(path.read_text(encoding="utf-8"))
    for k in ("class", "personal_importance", "target_pass_rate", "queries"):
        if k not in d:
            raise ValueError(f"{path}: missing required key {k!r}")
    if not isinstance(d["queries"], list) or len(d["queries"]) == 0:
        raise ValueError(f"{path}: queries must be a non-empty list")
    for q in d["queries"]:
        if "id" not in q or "query_text" not in q:
            raise ValueError(f"{path}: each query needs id + query_text")
        exp = q.get("expected") or {}
        if not exp.get("expected_anchor_regexes"):
            raise ValueError(
                f"{path}: query {q.get('id')} missing expected_anchor_regexes "
                f"(mandatory per PLAN_V2.16.md §3 Phase 1 step 3)"
            )
    return d


def run_class(fixture_path: Path, *, dry_run: bool = False,
              retrieve_kwargs: Optional[dict] = None) -> ClassResult:
    d = load_fixture(fixture_path)
    cls = ClassResult(
        class_name=d["class"],
        personal_importance=d["personal_importance"],
        target_pass_rate=float(d["target_pass_rate"]),
    )
    for q in d["queries"]:
        r = evaluate_query(q, cls.class_name, dry_run=dry_run,
                           retrieve_kwargs=retrieve_kwargs)
        cls.queries.append(r)
        cls.n_total += 1
        if r.pass_overall:
            cls.n_pass += 1
    cls.pass_rate = cls.n_pass / cls.n_total if cls.n_total else 0.0
    return cls


def render_report(results: list[ClassResult], *, generated_at: str,
                  label: str = "") -> str:
    """Render the markdown report."""
    title_suffix = f" — {label}" if label else ""
    lines = [
        f"# v2.16 Personal Validation Report{title_suffix}",
        "",
        f"> Generated: {generated_at}",
        f"> Fixture mechanic: per-class `target_pass_rate` is the gate.",
        f"> Per-query PASS rule (ALL must hold): (a) gold doc in top-5,",
        f"> (b) format_constraint matches top-1 modality / ast.parse,",
        f"> (c) `expected_anchor_regexes` matches top-1 content.",
        "",
        "## Summary",
        "",
        "| Class | personal_importance | target | pass_rate | result |",
        "|---|---|---:|---:|---|",
    ]
    for cls in results:
        verdict = "PASS" if cls.meets_target else "FAIL"
        lines.append(
            f"| {cls.class_name} | {cls.personal_importance} | "
            f"{cls.target_pass_rate*100:.0f}% | "
            f"{cls.pass_rate*100:.1f}% ({cls.n_pass}/{cls.n_total}) | {verdict} |"
        )
    lines.append("")
    for cls in results:
        lines.append(f"## {cls.class_name}")
        lines.append("")
        lines.append(
            f"- personal_importance: **{cls.personal_importance}** ; "
            f"target: {cls.target_pass_rate*100:.0f}% ; "
            f"got: **{cls.pass_rate*100:.1f}%** ({cls.n_pass}/{cls.n_total}) — "
            f"{'PASS' if cls.meets_target else 'FAIL'}"
        )
        lines.append("")
        lines.append("| Query | doc top-5 | format | regex | gold-match | top-1 chunk | overall |")
        lines.append("|---|---|---|---|---|---|---|")
        for q in cls.queries:
            bar = lambda x: "—" if x is None else ("✓" if x else "✗")  # noqa: E731
            chunk_disp = q.top_1_chunk_id[-24:] if q.top_1_chunk_id else "—"
            lines.append(
                f"| {q.query_id} | {bar(q.pass_gold_doc)} | "
                f"{bar(q.pass_format)} | {bar(q.pass_anchor_regex)} | "
                f"{bar(q.top_1_chunk_id_matches_gold)} | "
                f"`{chunk_disp}` | {bar(q.pass_overall)} |"
            )
        lines.append("")
        # Diagnostic dump for failed queries
        failed = [q for q in cls.queries if not q.pass_overall]
        if failed:
            lines.append("### Failed query detail")
            lines.append("")
            for q in failed:
                lines.append(f"**{q.query_id}** — `{q.query_text}`")
                lines.append("")
                lines.append(f"- top-1 doc: `{q.top_1_doc_id}` modality=`{q.top_1_modality}`")
                lines.append(f"- top-5 docs: {q.top_5_doc_ids}")
                if q.gold_chunk_ids_authored:
                    lines.append(f"- gold chunks authored: {q.gold_chunk_ids_authored}")
                if q.matched_regex:
                    lines.append(f"- matched regex: `{q.matched_regex}`")
                if q.note:
                    lines.append(f"- note: {q.note}")
                lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fixtures-dir",
        type=Path,
        default=_REPO_ROOT / "tests/fixtures/personal_validation_queries",
    )
    ap.add_argument(
        "--output", type=Path, default=None,
        help="Output report path; defaults to docs/VALIDATION_REPORT_<today>.md",
    )
    ap.add_argument("--label", default="",
                    help="Label appended to report title (e.g. 'v2.15.0_baseline')")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip retrieval; emit empty results (fixture syntax check only)")
    ap.add_argument("--reranker-backend", default=None,
                    help="omlx | dashscope | null (defaults to env)")
    args = ap.parse_args()

    fixtures = sorted(args.fixtures_dir.glob("*.json"))
    if not fixtures:
        print(f"No fixture files in {args.fixtures_dir}", file=sys.stderr)
        return 2

    # Build doc_id (hash) → basename map from output/<basename>/ingestion.jsonl
    # headers — required because production qdrant payloads carry hashes.
    global _DOC_ID_MAP
    _DOC_ID_MAP = _build_doc_id_to_basename_map(_REPO_ROOT / "output")

    retrieve_kwargs: dict = {}
    if args.reranker_backend:
        retrieve_kwargs["reranker_backend"] = args.reranker_backend

    results: list[ClassResult] = []
    for fx in fixtures:
        try:
            cls = run_class(fx, dry_run=args.dry_run, retrieve_kwargs=retrieve_kwargs)
        except ValueError as e:
            print(f"FIXTURE ERROR: {e}", file=sys.stderr)
            return 3
        results.append(cls)

    if args.output is None:
        today = datetime.date.today().isoformat()
        args.output = _REPO_ROOT / f"docs/VALIDATION_REPORT_{today}.md"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = render_report(
        results,
        generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
        label=args.label,
    )
    args.output.write_text(report, encoding="utf-8")
    rel = args.output.relative_to(_REPO_ROOT) if _REPO_ROOT in args.output.parents else args.output
    print(f"Wrote {rel}")

    any_fail = False
    for cls in results:
        flag = "PASS" if cls.meets_target else "FAIL"
        print(
            f"  {cls.class_name:35s} {cls.personal_importance:5s} "
            f"target={cls.target_pass_rate*100:.0f}% got={cls.pass_rate*100:.1f}% "
            f"({cls.n_pass}/{cls.n_total}) {flag}"
        )
        if not cls.meets_target:
            any_fail = True

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
