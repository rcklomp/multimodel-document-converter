# Cycle-Open Checklist

**Purpose**: single source of truth for "things that should fire
each cycle-open" — removes the failure mode where deferred items
rot because no process watches them.

Read this at the **start of every v2.X cycle** (i.e., when authoring
`docs/PLAN_V2.X.md`). The line items below produce inputs to the
opening plan's Carry-Forwards table or schedule blocking pre-Phase
work.

## Cycle-open line items

### 1. Telemetry analysis (Phase 3 [F] outputs)

Run:
```
python scripts/analyze_doc_class_telemetry.py --current-cycle v2.X
```

Writes `docs/TELEMETRY_REPORT_<today>.md`. Read it; copy the
per-class disposition (PROMOTE / CLOSE / ESCALATE / defer) for
each registered documented-limitation class into the opening plan's
Carry-Forwards table.

Disposition handling:
- **PROMOTE**: class earns Option A treatment this cycle —
  extraction-lane investment (Phase 2 in plan v2.15 ordering).
- **CLOSE**: class converts to Option E (documented-limitation
  closure) — add explicit DECISIONS.md entry; remove from
  `src/mmrag_v2/retrieval/documented_limitations.py` registry.
- **ESCALATE**: class has persisted in middle band ≥3 cycles;
  cycle plan MUST adjudicate explicitly (cannot defer to a 4th
  middle-band cycle). Author one of:
  - Convert to Option A with explicit reasoning
  - Convert to Option E with explicit reasoning
- **defer**: continue telemetry; no action needed this cycle.

### 2. USER_ISSUES.md review

Open `docs/USER_ISSUES.md` and scan for new rows since the prior
cycle's tag date. Per-class issue counts already flow into the
telemetry analyzer above (item 1), but newly-filed issues warrant
human review even if no trigger fires — they may suggest a class
that should gain `severe_defect_tag = True` in
`src/mmrag_v2/retrieval/documented_limitations.py`.

### 3. Docling release-notes review (carry-forward 6.1 trigger)

Check Docling release notes since the last cycle. Trigger:

> If Docling minor version increments to ≥2.87 (current production
> pin: 2.86.0 per `pyproject.toml`), reopen Phase 4 Approach 2
> evaluation per `docs/PLAN_V2.15.md` carry-forward 6.1.

Also re-check every 90 days regardless of version (the time-bound
half of the trigger). The 90-day clock resets on each successful
review (whether or not it triggers a Phase 4 reopen).

5-minute changelog check; outcome is a yes/no on whether to add a
Phase 4 re-evaluation to the opening plan's phase set.

### 4. Phase 0 calibration freshness

Check:
```
if phase_0_expiration_date - 72h <= today:
    schedule scripts/calibrate_local_judge_vs_qwen_max.py
    as a BLOCKING pre-Phase-1 step
```

`phase_0_expiration_date` is the prior cycle's last successful
re-cal date + 30 days. If expiring within 72h of cycle-open,
schedule the re-cal now (don't wait for the T-72h pre-tag
checkpoint to fire it).

### 5. cycle_slip.log review (carryover from prior cycle close-out)

Open `docs/cycle_slip.log` and check whether any blocking-step
slip entries exist from the prior cycle. If so:
- They've already been resolved (the prior cycle did ship, after
  all), so no action needed
- But the entries may inform whether to plan more buffer in this
  cycle's tag date (e.g., if the prior cycle slipped 48h on
  calibration review, plan 48h earlier)

The log is per-cycle (rotated by tag); the prior cycle's log is
read-only carryover. The current cycle creates a new empty log.

## cycle_slip.log tracker

This section documents the format of `docs/cycle_slip.log`. The
file is append-only, written by automated pre-tag scripts and
read by the silent-default notification dispatch.

### Format

One JSON-like line per slip event:
```
{iso_timestamp} | {trigger_name} | elapsed_hours={N} | old_tag_date={D} | new_projected_tag_date={D'}
```

Where:
- `iso_timestamp`: when the slip was detected
- `trigger_name`: which automated check fired
  (`"T-72h re-cal"` for the v2.15 Phase 6 case;
  future cycles may add others)
- `elapsed_hours`: how long the blocking step ran past the
  threshold that triggered slip-log inclusion
- `old_tag_date`: prior `effective_tag_date` before this slip
- `new_projected_tag_date`: new `effective_tag_date` after
  applying this slip's buffer

### Multi-slip composition

Multiple slip entries compose: latest `new_projected_tag_date`
wins. The silent-default notification dispatch script reads ALL
entries and uses `max(initial_planned_tag, max(slip.new_projected_tag_date for slip in log))`
to compute the effective tag date.

### v2.15 Phase 6 T-72h slip-log trigger

Per `docs/PLAN_V2.15.md` §Phase 6 (T-72h pre-tag checkpoint),
the slip log fires when EITHER:

(a) re-cal wall-clock execution > 24h, OR
(b) at `re_cal_start_timestamp + 24h`, no
    `docs/CALIBRATION_*_v2.15_p0_*.md` file exists with
    `mtime > re_cal_start_timestamp` (file-existence check —
    verifies the verdict has actually landed; absent file =
    human review is the bottleneck)

When (a) or (b) fires, append a line to `docs/cycle_slip.log`
with `new_projected_tag_date = now() + 48h` (gives the
maintainer time to finish review without the auto-activation
breathing down their neck).

## Audit trail

This checklist was added in v2.15 per Round-4 audit Finding 1
(telemetry trigger without a defined reader/process invalidates
the entire Phase 3 design). Subsequent line items added per
Round-4 Finding 6 (Docling version watcher), Round-6 Finding 6
(cycle_slip.log integration), and Round-7 Finding 4 (T-72h
file-existence sub-trigger).

Two-round-minimum audit recommendation: any future structural
changes to this checklist should themselves get an audit pass
before going live, per `docs/PLAN_V2.15.md` §9 stopping rule.
