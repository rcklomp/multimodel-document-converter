#!/usr/bin/env python3
"""v2.15 Phase 1 — narrow-fixture sampler for HyDE bridging mini-soak.

Produces a 5-doc stratified fixture with the v0.9 layout:

  - 100 from ATZ_Elektronik_German   (German subgroup, R7F1 n-bump
                                       to clear binomial false-positive
                                       rate at +10pp gate)
  - 20  from Python_Cookbook         \
  - 20  from IRJET_Modeling_of_Solar_PV  > code-dense subgroup
  - 20  from Hybrid_electric_vehicles    > (4/4 quorum gate)
  - 20  from Greenhouse_Design       /

Total n=180 queries (after stage_generate writes 1 query per
sampled chunk — Phase 1 uses single-query-per-chunk; reuses the
existing GENERATE_USER_TEMPLATE).

Output: `output/soak/v2.15_p1_narrow/work.jsonl` ready for
`synthetic_soak.py --stage generate` + `--stage retrieve` +
`--stage judge`.

Usage:
  python scripts/sample_phase1_narrow_fixture.py --seed 42

Then drive the rest via synthetic_soak.py for both arms:
  python scripts/synthetic_soak.py --stage generate \\
    --work output/soak/v2.15_p1_narrow/work.jsonl

  # HyDE-off baseline arm (default; no --use-hyde flag)
  python scripts/synthetic_soak.py --stage retrieve \\
    --work output/soak/v2.15_p1_narrow_hyde_off/work.jsonl \\
    --hybrid --provider omlx --collection mmrag_v2_8__qwen3_local

  # HyDE-on arm (auto-intent on the same 5 docs)
  python scripts/synthetic_soak.py --stage retrieve \\
    --work output/soak/v2.15_p1_narrow_hyde_on/work.jsonl \\
    --hybrid --provider omlx --collection mmrag_v2_8__qwen3_local \\
    --use-hyde

  python scripts/synthetic_soak.py --stage judge --work output/soak/v2.15_p1_narrow_<arm>/work.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from synthetic_soak import (  # noqa: E402
    DOCS_ROOT,
    _is_eligible_text_chunk,
    _load_chunks,
)

# v0.9 fixture spec — German subgroup gets 100; 4 code-dense docs
# get 20 each. Total n=180.
PHASE1_TARGET_DOCS: dict[str, int] = {
    "ATZ_Elektronik_German":             100,  # German subgroup
    "Python_Cookbook":                    20,  # code-dense subgroup
    "IRJET_Modeling_of_Solar_PV":         20,
    "Hybrid_electric_vehicles":           20,
    "Greenhouse_Design":                  20,
}
DEFAULT_OUTPUT = REPO_ROOT / "output/soak/v2.15_p1_narrow/work.jsonl"


def sample(seed: int, output: Path) -> None:
    if output.exists():
        print(f"  refusing to overwrite existing {output} — delete first to re-sample",
              file=sys.stderr)
        raise SystemExit(2)
    rng = random.Random(seed)
    rows: list[dict] = []
    missing: list[str] = []
    short: list[tuple[str, int, int]] = []
    for doc_name, target in PHASE1_TARGET_DOCS.items():
        chunks = [c for c in _load_chunks(doc_name) if _is_eligible_text_chunk(c)]
        if not chunks:
            missing.append(doc_name)
            continue
        if len(chunks) < target:
            short.append((doc_name, len(chunks), target))
            take = len(chunks)
        else:
            take = target
        picks = rng.sample(chunks, take)
        for p in picks:
            rows.append({
                "doc_dir": doc_name,
                "gold_chunk_id": p.get("chunk_id"),
                "gold_doc_id": p.get("doc_id"),
                "gold_source_file": (p.get("metadata") or {}).get("source_file"),
                "gold_page_number": (p.get("metadata") or {}).get("page_number"),
                "gold_content": (p.get("content") or "").strip(),
                "queries": [],
            })
        print(f"  {doc_name}: sampled {take}/{len(chunks)} (target {target})")
    if missing:
        print(f"\nERROR: docs not found in {DOCS_ROOT}: {missing}", file=sys.stderr)
        raise SystemExit(3)
    rng.shuffle(rows)
    for i, row in enumerate(rows, start=1):
        row["sample_id"] = f"P1-{i:04d}"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n  wrote {len(rows)} chunks to {output}")
    if short:
        print(f"\n  WARNING: short fixtures (took all available):")
        for doc_name, available, target in short:
            print(f"    {doc_name}: only {available} eligible chunks (wanted {target})")
        print(f"  Acceptance-gate stats below specified n will have wider CIs.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed (default 42)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output JSONL path (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})")
    args = p.parse_args()
    sample(args.seed, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
