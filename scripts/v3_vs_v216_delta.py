#!/usr/bin/env python3
"""Compute per-doc V3-vs-V2.16 extraction deltas.

Reads:
    - V2.16 baseline stats from output/v3_soak/v216_baseline_stats.json
      (produced earlier in this session)
    - V3 batch manifest from output/v3_baselines/manifest.json

For each canonical doc that has both a V2.16 baseline and a V3 batch
entry (mapped via canonical-name layout), emits:
    canonical_name, v216_chunks, v3_chunks, delta, delta_pct,
    v216_modalities, v3_modalities, v3_routing, v3_seconds

Writes Markdown to ``output/v3_soak/v3_vs_v216_delta.md`` and a JSON
side-car to ``output/v3_soak/v3_vs_v216_delta.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _v3_modality_counts(jsonl: Path) -> dict:
    if not jsonl.exists():
        return {}
    mods: Counter[str] = Counter()
    with jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            if c.get("object_type") == "ingestion_metadata":
                continue
            m = c.get("element_type") or c.get("modality") or "text"
            mods[m] += 1
    return dict(mods)


def _resolve_v3_jsonl(
    canonical_name: str, canonical_root: Path
) -> Path | None:
    """Find the V3 JSONL for a canonical name via the v3_canonical layout."""
    candidate = canonical_root / canonical_name / "ingestion.jsonl"
    return candidate if candidate.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v216-stats",
        type=Path,
        default=REPO_ROOT / "output" / "v3_soak" / "v216_baseline_stats.json",
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=REPO_ROOT / "output" / "v3_canonical",
        help="Where V3 ingestion.jsonl per canonical doc lives. Run "
             "scripts/build_v3_canonical_layout.py first to populate.",
    )
    parser.add_argument(
        "--v3-manifest",
        type=Path,
        default=REPO_ROOT / "output" / "v3_baselines" / "manifest.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=REPO_ROOT / "output" / "v3_soak" / "v3_vs_v216_delta.md",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "output" / "v3_soak" / "v3_vs_v216_delta.json",
    )
    args = parser.parse_args(argv)

    if not args.v216_stats.exists():
        print(f"missing v2.16 stats: {args.v216_stats}", file=sys.stderr)
        return 2
    if not args.canonical_root.exists():
        print(
            f"missing v3 canonical layout: {args.canonical_root} — run "
            "scripts/build_v3_canonical_layout.py first",
            file=sys.stderr,
        )
        return 2

    v216 = json.loads(args.v216_stats.read_text(encoding="utf-8"))

    # Walk meta.json files under output/v3_baselines/ directly. The
    # batch manifest only covers what the most recent run iterated, so
    # docs completed in earlier runs but filtered out by --max-pages
    # never get a manifest entry. The meta.json files are the canonical
    # per-doc record and survive across restarts.
    v3_by_basename: dict[str, dict] = {}
    baselines_root = REPO_ROOT / "output" / "v3_baselines"
    if baselines_root.exists():
        for meta_path in baselines_root.rglob("meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            src = meta.get("source_pdf") or ""
            if not src:
                continue
            base = Path(src).stem.lower()
            v3_by_basename[base] = meta

    rows = []
    for canonical_name, v216_row in v216.items():
        v3_jsonl = _resolve_v3_jsonl(canonical_name, args.canonical_root)
        v3_chunks = 0
        v3_mods: dict[str, int] = {}
        if v3_jsonl is not None:
            v3_mods = _v3_modality_counts(v3_jsonl)
            v3_chunks = sum(v3_mods.values())
        # Match manifest entry by canonical-name token overlap (best-effort).
        # Tokenize on any non-alphanumeric so dots / underscores / hyphens in
        # source filenames don't hide real overlaps (ATZ filenames use dots).
        import re
        manifest_entry: dict = {}
        canonical_tokens = set(re.findall(r"[a-z0-9]+", canonical_name.lower()))
        best_overlap = 0
        for base, entry in v3_by_basename.items():
            base_tokens = set(re.findall(r"[a-z0-9]+", base))
            overlap = len(canonical_tokens & base_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                manifest_entry = entry
        v216_chunks = int(v216_row.get("chunks", 0) or 0)
        delta = v3_chunks - v216_chunks
        delta_pct = (
            f"{(delta / v216_chunks * 100):+.1f}%" if v216_chunks else "n/a"
        )
        rows.append(
            {
                "canonical_name": canonical_name,
                "v216_chunks": v216_chunks,
                "v3_chunks": v3_chunks,
                "delta": delta,
                "delta_pct": delta_pct,
                "v216_modalities": v216_row.get("modalities", {}),
                "v3_modalities": v3_mods,
                "v3_routing": manifest_entry.get("routing", {}),
                "v3_elapsed_seconds": manifest_entry.get("elapsed_seconds"),
                "v3_status": manifest_entry.get("status", "unmapped"),
            }
        )

    # Aggregate
    v216_total = sum(r["v216_chunks"] for r in rows)
    v3_total = sum(r["v3_chunks"] for r in rows)
    matched = sum(1 for r in rows if r["v3_chunks"] > 0)
    missing_v3 = [r["canonical_name"] for r in rows if r["v3_chunks"] == 0]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "summary": {
                    "matched_docs": matched,
                    "total_canonical": len(rows),
                    "v216_total_chunks": v216_total,
                    "v3_total_chunks": v3_total,
                    "overall_delta_pct": (
                        f"{((v3_total - v216_total) / v216_total * 100):+.1f}%"
                        if v216_total else "n/a"
                    ),
                    "missing_v3": missing_v3,
                },
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    md = ["# V3 vs V2.16 — Per-Doc Extraction Delta", "", f"Matched {matched}/{len(rows)} canonical docs. " f"V2.16 total chunks: {v216_total:,}. V3 total chunks: {v3_total:,}.", ""]
    md.append("| Doc | V2.16 chunks | V3 chunks | Δ | Δ% | V3 routing (vlm/docling/fallback) | V3 elapsed |")
    md.append("|---|---:|---:|---:|---:|---|---:|")
    for r in sorted(rows, key=lambda x: -abs(x["delta"])):
        routing = r["v3_routing"] or {}
        routing_str = (
            f"{routing.get('vlm', 0)}/{routing.get('docling', 0)}"
            f"/{routing.get('docling_fallback', 0)}"
            if routing else "—"
        )
        elapsed = r["v3_elapsed_seconds"]
        elapsed_str = f"{elapsed:.0f}s" if isinstance(elapsed, (int, float)) else "—"
        md.append(
            f"| {r['canonical_name']} | {r['v216_chunks']} | {r['v3_chunks']} | "
            f"{r['delta']:+d} | {r['delta_pct']} | {routing_str} | {elapsed_str} |"
        )
    if missing_v3:
        md.append("")
        md.append(f"**V3 missing for {len(missing_v3)} canonical docs:** {missing_v3}")
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote {args.out_md} and {args.out_json}")
    print(
        f"Matched {matched}/{len(rows)} docs; V2.16 total {v216_total} chunks → "
        f"V3 total {v3_total} chunks "
        f"({((v3_total - v216_total) / v216_total * 100):+.1f}%)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
