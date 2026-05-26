# Phase A Intentional Deltas Registry

**Charter:** [`ARCHITECTURE_V3_DRAFT_0.5.md`](ARCHITECTURE_V3_DRAFT_0.5.md) §3.2 "Explained-delta half"
**Gate:** every v2.16 → v3.0.0 chunk delta MUST be enumerated here with v2.X
documented-defect cross-reference and reviewer sign-off. CI tooling compares
the v2.16 baseline fixture to the v3.0.0 output; an unenumerated delta fails
the build.

## Status

**EMPTY** at foundation-session start (2026-05-26). Phase A code work has
not begun. The table below establishes the schema only.

## Phase A delta table

| v2.16 chunk_id (or batch) | v3.0.0 chunk_id (or batch) | Delta type | v2.X cross-ref | Affected doc(s) | Reviewer sign-off |
|---|---|---|---|---|---|

## Allowed delta types

Per Charter §3.2:

- `cross_page_partial_code_repair` — A2 chunk attributed to wrong page
  in v2.16; v3.0.0 attributes to correct starting page AND sets
  `StructuralFlag.PARTIAL_CODE_CROSS_PAGE`. Cross-ref:
  `docs/DECISIONS.md` "v2.16 partial_code adjacency fetch shipped INERT
  for the cross-page case."
- `reading_order_correction` — page emits elements in a corrected reading
  order under the UIR-native cleanup site (replaces the per-profile
  y_sort_with_dropcap branches). Cross-ref required: which v2.X heuristic
  fired for this doc.
- `flag_addition` — chunk gains a new `StructuralFlag` enum member not
  present in v2.16's `Dict[str, bool]`. Strictly additive; no v2.16
  flag may go missing (that's a regression, not a delta).
- `chunk_id_re-derivation` — element ordering / page assignment / modality
  classification change in Phase A produces a new `chunk_id` for the
  same content. MUST be paired with an entry in
  `docs/CHUNK_ID_REWRITE_MAP_3.0.0.csv` (generated in task A6).
- Other types must be approved by the user before being added.

## Empty file = behavioral identity

An empty delta table after Phase A close means the explained-delta half
of the semantic-identity gate was unused — the inert
`partial_code_cross_page` repair did not trigger on any corpus doc. That
is an acceptable outcome (it just means the Charter's "controlled-delta
refactoring" produced no behavioral change in practice).
