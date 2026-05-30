#!/usr/bin/env python3
"""Lay V3 baseline outputs under canonical-doc names for synthetic_soak.

The soak harness's ``stage_sample`` reads
``output/<canonical_name>/ingestion.jsonl`` per ``CANONICAL_DOCS`` in
``scripts/synthetic_soak.py``. The V3 batch ingester writes to
``output/v3_baselines/<category>/<doc_stem>/ingestion.jsonl`` — different
naming. This script builds a parallel canonical layout
(``output/v3_canonical/<canonical_name>/ingestion.jsonl``) by mapping V3
output dirs to canonical names via best-prefix fuzzy match.

The V3 baselines are NOT touched. The canonical layout uses real copies
(not symlinks) of the V2-shaped JSONL so the soak's loader sees the
right schema.

Prereq:
    1. ``scripts/v3_batch_ingest.py`` has populated ``output/v3_baselines/``
    2. ``scripts/v3_to_v2_jsonl.py`` has populated ``output/v3_baselines_v2shape/``

Usage:
    python scripts/build_v3_canonical_layout.py \\
        --v3-shape-dir output/v3_baselines_v2shape \\
        --out-dir      output/v3_canonical
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Imported lazily from synthetic_soak to stay in sync.


def _load_canonical_docs() -> List[str]:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import synthetic_soak  # type: ignore
    return list(synthetic_soak.CANONICAL_DOCS)


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(s: str) -> List[str]:
    return _WORD_RE.findall(s.lower())


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _best_match(
    canonical: str, candidates: List[Tuple[str, Path]]
) -> Optional[Tuple[float, Path]]:
    """Return (score, path) of the best-matching candidate, or None."""
    canon_tokens = _tokenize(canonical)
    best: Optional[Tuple[float, Path]] = None
    for stem, path in candidates:
        cand_tokens = _tokenize(stem)
        score = _jaccard(canon_tokens, cand_tokens)
        if best is None or score > best[0]:
            best = (score, path)
    return best


def _collect_v3_candidates(v3_shape_dir: Path) -> List[Tuple[str, Path]]:
    """Return list of (doc_stem, ingestion_jsonl_path) found under v3_shape_dir."""
    out: List[Tuple[str, Path]] = []
    for jsonl in v3_shape_dir.rglob("ingestion.jsonl"):
        if jsonl.stat().st_size == 0:
            continue
        # Stem is the directory name containing ingestion.jsonl
        stem = jsonl.parent.name
        out.append((stem, jsonl))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v3-shape-dir",
        type=Path,
        default=REPO_ROOT / "output" / "v3_baselines_v2shape",
        help="Root of V2-shaped V3 baselines.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "v3_canonical",
        help="Canonical-layout output root.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.30,
        help="Reject matches below this Jaccard score.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the mapping but don't copy.",
    )
    args = parser.parse_args(argv)

    canonical = _load_canonical_docs()
    v3_shape_dir: Path = args.v3_shape_dir.resolve()
    out_dir: Path = args.out_dir.resolve()
    if not v3_shape_dir.exists():
        print(f"v3-shape-dir does not exist: {v3_shape_dir}", file=sys.stderr)
        return 2

    candidates = _collect_v3_candidates(v3_shape_dir)
    if not candidates:
        print(f"no ingestion.jsonl found under {v3_shape_dir}", file=sys.stderr)
        return 2

    print(
        f"matching {len(canonical)} canonical names against {len(candidates)} "
        f"V3 candidates (min Jaccard score = {args.min_score})"
    )
    mapped: Dict[str, Path] = {}
    unmatched: List[str] = []
    used_paths: set[Path] = set()
    for name in canonical:
        canon_tokens = _tokenize(name)
        canon_str = "_".join(canon_tokens)
        # Score = Jaccard + a bonus when one normalized name is a substring
        # of the other (handles Form_ prefixes, trailing _and_their_challenges
        # suffixes, etc.). Bonus pushes substring matches above the threshold.
        scored = []
        for stem, path in candidates:
            stem_tokens = _tokenize(stem)
            stem_str = "_".join(stem_tokens)
            jaccard_score = _jaccard(canon_tokens, stem_tokens)
            substr_bonus = 0.0
            if canon_str and stem_str:
                if canon_str in stem_str or stem_str in canon_str:
                    # Reward shared tokens that form a contiguous run.
                    substr_bonus = 0.40
                else:
                    # Cheap per-token substring check for distinctive tokens
                    # (e.g. "0013" canonical vs "0013_140302..." stem).
                    distinctive = [
                        t for t in canon_tokens
                        if len(t) >= 4 and t not in {"form", "and", "the"}
                    ]
                    if distinctive and any(t in stem_str for t in distinctive):
                        substr_bonus = 0.25
            scored.append((jaccard_score + substr_bonus, stem, path))
        scored.sort(key=lambda t: -t[0])
        picked: Optional[Tuple[float, str, Path]] = None
        for score, stem, path in scored:
            if score < args.min_score:
                break
            if path in used_paths:
                continue
            picked = (score, stem, path)
            break
        if picked is None:
            unmatched.append(name)
            continue
        mapped[name] = picked[2]
        used_paths.add(picked[2])
        print(f"  {name} ← {picked[2].parent.name} (score={picked[0]:.2f})")

    if unmatched:
        print(f"UNMATCHED ({len(unmatched)}): {unmatched}", file=sys.stderr)

    if args.dry_run:
        print(f"dry-run: would create {len(mapped)} canonical dirs at {out_dir}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for canonical_name, src_jsonl in mapped.items():
        dst_dir = out_dir / canonical_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_jsonl = dst_dir / "ingestion.jsonl"
        shutil.copy2(src_jsonl, dst_jsonl)
        # Also bring meta.json across so downstream debugging keeps the
        # routing decisions reachable.
        src_meta = src_jsonl.parent / "meta.json"
        if src_meta.exists():
            shutil.copy2(src_meta, dst_dir / "meta.json")
    print(
        f"laid out {len(mapped)} canonical docs at {out_dir} "
        f"({len(unmatched)} unmatched)"
    )
    return 0 if not unmatched else 1


if __name__ == "__main__":
    sys.exit(main())
