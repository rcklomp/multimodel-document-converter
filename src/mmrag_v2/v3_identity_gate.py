"""V3 Phase A identity-half gate.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §3.2 (semantic-identity gate)
+ §8.2 (normalization rules).

The semantic-identity gate is in two halves:

    1. **Identity half (≥95% of v2.16 chunks):** chunk-by-chunk match
       under the normalization rules below.
    2. **Explained-delta half (≤5%):** every remaining delta enumerated
       in docs/PHASE_A_INTENTIONAL_DELTAS.md with v2.X documented-defect
       cross-reference.

This module is the CI-callable identity-half engine. It compares a
v2.16 fixture (`tests/fixtures/retrieval_regression_v2_X.json` or an
ingestion.jsonl snapshot) to a v3.0.0 output, applies the §8.2
normalization rules, and emits the identity ratio + list of differing
chunks.

Foundation-session status: FUNCTIONAL on synthetic fixtures. The
real v2.16-vs-v3.0.0 comparison runs at the end of Phase A task A5
(corpus rebuild + strict gate). The module is shipped now so Phase A
task A2 can use it as the "did this refactor break anything" feedback
loop from day 1.

§8.2 normalization rules applied before hashing:
    1. JSON field-order normalization (sort_keys=True).
    2. Confidence-value rounding ±0.01 tolerance.
    3. Identity-relevant vs metadata-only field enumeration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Charter §8.2 identity-relevant vs metadata-only field table:
_IDENTITY_RELEVANT_FIELDS = frozenset(
    {
        "chunk_type",
        "content",
        "page_number",
        "bbox",
        "modality",
        "parent_heading",
        "parent_element_id",
        "structural_flags",
        # confidence_breakdown handled separately (rounded with ±0.01)
    }
)
_METADATA_ONLY_FIELDS = frozenset(
    {
        "chunk_id",
        "extraction_warnings",
        "pipeline_version",
        "schema_version",
        "source_file_hash",
        "sanitizer_model_id",
        "sanitizer_prompt_version",
        "sanitization_status",
        "content_original",
        "content_sanitized",
        "uir_version",
        "continuation_group_id",
        "created_at",
        "timestamp",
    }
)
_CONFIDENCE_ROUNDING_DECIMALS = 2  # ±0.01 tolerance per §8.2


@dataclass(frozen=True)
class IdentityHashableChunk:
    """A chunk reduced to its identity-relevant projection.

    Used as the comparison unit between v2.16 baseline and v3.0.0
    output. Two chunks have the same identity hash iff they are
    "byte-identical" under the §8.2 normalization rules.
    """

    payload: Dict[str, Any] = field(default_factory=dict)

    def identity_hash(self) -> str:
        normalized = json.dumps(self.payload, sort_keys=True)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_content_for_projection(content: str) -> str:
    """Apply platform-level content normalizations per Charter §3.2 matching policy.

    Charter §3.2 identity-half matching: "content must match (modulo
    trailing whitespace)". Platform-level normalizations (Unicode NFC,
    line-ending CRLF→LF) are applied so byte-for-byte differences
    from non-semantic sources (NFD vs NFC, Windows vs Unix line
    endings) do not produce spurious 'regression' alerts.

    Internal whitespace is NOT collapsed in the content projection —
    chunkers emitting different internal whitespace IS a real
    behavioral change and should land in the explained-delta half
    rather than be silently masked.
    """
    import unicodedata

    normalized = unicodedata.normalize("NFC", content)
    normalized = normalized.replace("\r\n", "\n")
    # Strip trailing whitespace from each line (Charter "modulo trailing
    # whitespace"). The body's trailing whitespace at the very end is
    # also stripped.
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).rstrip()


def normalize_chunk_for_identity(chunk: Dict[str, Any]) -> IdentityHashableChunk:
    """Apply Charter §8.2 normalization rules and return identity projection.

    Drops metadata-only fields entirely (they don't affect identity).
    Identity-relevant fields are kept verbatim, except `content` which
    is normalized per Charter §3.2 matching policy (NFC + CRLF→LF +
    trailing-whitespace strip — see _normalize_content_for_projection).
    Confidence fields are rounded to 2 decimal places (±0.01 tolerance
    per §8.2 #2).
    """
    projection: Dict[str, Any] = {}
    for key, value in chunk.items():
        if key in _METADATA_ONLY_FIELDS:
            continue
        if key == "confidence_breakdown" and isinstance(value, dict):
            projection[key] = {
                sub_key: round(sub_val, _CONFIDENCE_ROUNDING_DECIMALS)
                if isinstance(sub_val, (int, float))
                else sub_val
                for sub_key, sub_val in value.items()
            }
            continue
        if key == "structural_flags":
            # §8.2 row: structural_flags are additive, sorted for hash
            # determinism regardless of producer's order.
            if isinstance(value, (list, set, tuple)):
                projection[key] = sorted(value)
                continue
            if isinstance(value, dict):
                # v2.X format: Dict[str, bool] of flags-that-fired
                projection[key] = sorted(
                    k for k, v in value.items() if v
                )
                continue
        if key == "content" and isinstance(value, str):
            projection[key] = _normalize_content_for_projection(value)
            continue
        if key in _IDENTITY_RELEVANT_FIELDS:
            projection[key] = value
    return IdentityHashableChunk(payload=projection)


def _chunk_key(chunk: Dict[str, Any]) -> str:
    """Stable identity key per Charter §3.2 identity-half row 1.

    `(doc_id, page_number, content_hash_prefix)` where
    content_hash_prefix = first 16 hex chars of SHA-256 of content
    after Unicode NFC normalization + whitespace collapse + CRLF→LF.
    """
    import re
    import unicodedata

    doc_id = chunk.get("doc_id", "unknown")
    page_number = chunk.get("page_number", -1)
    content = chunk.get("content", "")
    # Per §3.2: Unicode NFC, internal whitespace collapse, CRLF → LF.
    content_normalized = unicodedata.normalize("NFC", content)
    content_normalized = content_normalized.replace("\r\n", "\n")
    content_normalized = re.sub(r"\s+", " ", content_normalized).strip()
    content_hash = hashlib.sha256(
        content_normalized.encode("utf-8")
    ).hexdigest()[:16]
    return f"{doc_id}|{page_number}|{content_hash}"


@dataclass
class IdentityGateReport:
    """Result of comparing v2.16 baseline to v3.0.0 candidate output."""

    baseline_chunk_count: int
    candidate_chunk_count: int
    matched: int  # baseline chunks that have an identity-hash match in candidate
    differing_baseline_ids: List[str] = field(default_factory=list)
    missing_baseline_ids: List[str] = field(default_factory=list)
    new_candidate_ids: List[str] = field(default_factory=list)

    @property
    def identity_ratio(self) -> float:
        if self.baseline_chunk_count == 0:
            return 1.0
        return self.matched / self.baseline_chunk_count

    def passes(self, threshold: float = 0.95) -> bool:
        """Charter §3.2 identity-half row 1: ≥95% chunk match by stable identity key."""
        return self.identity_ratio >= threshold


def compare_for_identity(
    *,
    baseline_chunks: List[Dict[str, Any]],
    candidate_chunks: List[Dict[str, Any]],
) -> IdentityGateReport:
    """Identity-half comparison per Charter §3.2.

    Algorithm:
        1. Group both sides by stable identity key (doc_id|page|content_hash).
        2. For each baseline key:
           a. If the candidate has the same key: identity-hash both sides
              after §8.2 normalization. Match iff hashes equal.
           b. If the candidate has no entry at that key: "missing" delta.
        3. Candidate keys not in baseline: "new_candidate" delta.
        4. Compute identity ratio = matched / baseline_count.
    """
    baseline_by_key: Dict[str, Dict[str, Any]] = {}
    for chunk in baseline_chunks:
        key = _chunk_key(chunk)
        # If the baseline has duplicates at the same key, the last one wins —
        # this is the same behavior the v2.X chunker emits (chunk_ids are
        # unique by construction), so duplicates would themselves be a bug.
        baseline_by_key[key] = chunk
    candidate_by_key: Dict[str, Dict[str, Any]] = {}
    for chunk in candidate_chunks:
        key = _chunk_key(chunk)
        candidate_by_key[key] = chunk

    matched = 0
    differing: List[str] = []
    missing: List[str] = []
    for key, baseline_chunk in baseline_by_key.items():
        cand = candidate_by_key.get(key)
        if cand is None:
            missing.append(key)
            continue
        baseline_proj = normalize_chunk_for_identity(baseline_chunk)
        cand_proj = normalize_chunk_for_identity(cand)
        if baseline_proj.identity_hash() == cand_proj.identity_hash():
            matched += 1
        else:
            differing.append(key)

    new_candidate = [
        key for key in candidate_by_key if key not in baseline_by_key
    ]

    return IdentityGateReport(
        baseline_chunk_count=len(baseline_by_key),
        candidate_chunk_count=len(candidate_by_key),
        matched=matched,
        differing_baseline_ids=differing,
        missing_baseline_ids=missing,
        new_candidate_ids=new_candidate,
    )


def load_jsonl_chunks(path: str) -> List[Dict[str, Any]]:
    """Load an ingestion.jsonl-style file into a list of chunk dicts.

    Each line is either a chunk record or a metadata record (v2.X
    convention). Foundation session: returns ALL JSON lines; the caller
    can filter by `chunk_type` or `_kind` if needed.
    """
    chunks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point — compare baseline.jsonl vs candidate.jsonl."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "V3 Phase A identity-half gate — Charter §3.2 + §8.2. "
            "Compares a v2.16 baseline ingestion.jsonl against a v3.0.0 "
            "candidate ingestion.jsonl and reports the identity ratio."
        )
    )
    parser.add_argument("--baseline", required=True, help="v2.16 baseline JSONL")
    parser.add_argument("--candidate", required=True, help="v3.0.0 candidate JSONL")
    parser.add_argument(
        "--threshold", type=float, default=0.95,
        help="Identity-ratio threshold (Charter default 0.95).",
    )
    parser.add_argument(
        "--limit-diff-output", type=int, default=10,
        help="Cap on number of differing chunk IDs printed (default 10).",
    )
    args = parser.parse_args(argv)

    baseline = load_jsonl_chunks(args.baseline)
    candidate = load_jsonl_chunks(args.candidate)
    report = compare_for_identity(
        baseline_chunks=baseline,
        candidate_chunks=candidate,
    )
    print(f"V3 Phase A identity-half gate")
    print(f"  baseline chunks:  {report.baseline_chunk_count}")
    print(f"  candidate chunks: {report.candidate_chunk_count}")
    print(f"  matched:          {report.matched}")
    print(f"  identity ratio:   {report.identity_ratio:.4f}")
    print(f"  threshold:        {args.threshold:.4f}")
    print(f"  verdict:          {'PASS' if report.passes(args.threshold) else 'FAIL'}")
    if report.differing_baseline_ids:
        print(
            f"  differing (showing {min(len(report.differing_baseline_ids), args.limit_diff_output)} "
            f"of {len(report.differing_baseline_ids)}):"
        )
        for key in report.differing_baseline_ids[: args.limit_diff_output]:
            print(f"    {key}")
    if report.missing_baseline_ids:
        print(
            f"  missing in candidate (showing {min(len(report.missing_baseline_ids), args.limit_diff_output)} "
            f"of {len(report.missing_baseline_ids)}):"
        )
        for key in report.missing_baseline_ids[: args.limit_diff_output]:
            print(f"    {key}")
    return 0 if report.passes(args.threshold) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
