# User Issues — MM-Converter-V2

**Purpose**: append-only registry of user-filed issues against
documented-limitation document classes. Read by
`scripts/analyze_doc_class_telemetry.py` to feed the
`open_user_issues` signal into the v2.15 Option F telemetry
promotion rule (per
`docs/DECISIONS.md` "v2.15 Documented-Limitation Telemetry Threshold").

## How this file is used

Per `docs/CYCLE_OPEN_CHECKLIST.md`, every cycle-open includes a
"Review USER_ISSUES.md for new entries since prior tag" step.
`analyze_doc_class_telemetry.py` parses this file's table rows and
counts entries per `doc_class` since the prior cycle's tag date.
The count flows into the v2.15 telemetry promotion rule:

- **Standard promotion arm** fires on `hit-rate >= 5%` AND
  (`severe_defect_tag = True` OR `open_user_issues >= 1`).
- **Defect-override arm** fires on `severe_defect_tag = True` AND
  `hit-rate >= 1%` (independent of issue count).
- **Closure arm** blocked when `open_user_issues >= 1` (file
  evidence beats telemetry; class cannot auto-close if real
  complaints exist).

## Format

Append rows to the table below. **DO NOT delete or modify
existing rows** — this is append-only by design (audit trail).
If an issue is resolved, append a new row with the same
`doc_class` and `observed_behavior = "RESOLVED: <commit hash>"`
rather than editing in place. The analyzer counts all rows for
the per-class total; resolution rows are tallied separately
via grep when needed.

Column meanings:
- **date**: ISO `YYYY-MM-DD` of issue filing
- **doc_class**: must match a `name` entry in
  `src/mmrag_v2/retrieval/documented_limitations.py`
  (e.g. `CarOK_voorraadtelling`, `Fluent_Python`)
- **query**: the user query that surfaced the issue (or "n/a" if
  not query-driven — e.g. a corpus-ingestion bug)
- **observed_behavior**: what the user saw (one line)
- **expected_behavior**: what they expected (one line)

The parser is a regex against `| YYYY-MM-DD | <doc_class> | …`
so keep the date in the first column and the doc_class in the
second column. Other column orderings are fine for human reading
but won't be picked up by the analyzer.

## Active issues

| date | doc_class | query | observed_behavior | expected_behavior |
|---|---|---|---|---|

<!-- No issues filed yet. v2.15 documented-limitation entries:
     CarOK_voorraadtelling, Fluent_Python (both severe_defect_tag=True
     on entry per their prior-cycle defect history; promotion-eligible
     via defect-override arm regardless of issue count). -->

## Resolution log (informational)

Append resolution markers here as one-line entries. Not parsed
by the analyzer; for human audit only.

<!-- Format suggestion:
     YYYY-MM-DD <commit-hash> <doc_class>: <one-line resolution>
-->
