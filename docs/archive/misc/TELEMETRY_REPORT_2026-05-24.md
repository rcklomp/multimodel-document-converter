# v2.15+ Documented-Limitation Telemetry Report

> Generated: 2026-05-24
> Current cycle: v2.16
> Telemetry log: `output/telemetry/document_class_hits.jsonl`
> Total rows in log: 448
> Qualified queries (30d window, `rerank_top_5_non_empty=True`): 448
> Qualified queries (60d window): 448
> User issues counted since: (all-time)

Per-class disposition follows. Disposition priority is:
promotion > closure > escalation > defer-to-next-cycle.

---

## CarOK_voorraadtelling
- added_cycle: v2.15  (current: v2.15 + 1 cycles → grace_period_elapsed: False)
- severe_defect_tag: True
- 30-day hit-rate: 0.0% (0 / 448 qualified queries)
- 60-day hit-rate: 0.0% (0 / 448 qualified queries)
- open_user_issues: 0
- consecutive_middle_cycles: 0
- PROMOTION TRIGGER (standard arm: >=5% AND pain-signal): NOT FIRED
- PROMOTION TRIGGER (defect-override arm: defect-tag AND >=1%): NOT FIRED
- CLOSURE TRIGGER (<1% AND 0 issues AND no defect-tag AND grace elapsed): NOT FIRED
- MIDDLE-BAND ESCALATION (>=3 consecutive cycles): NOT FIRED
- v2.X disposition: Defer to next cycle (continue telemetry)
- defect_summary: v2.14 P1 mini-soak Format -26.9pp regression; VLM tables + flat-prose duplicates coexist post force_table_vlm; retrieval picks prose 29/30 times. See QUALITY_SNAPSHOT_2026-05-23_v2.14_after.md §1 Phase 1 PARTIAL row.

## Fluent_Python
- added_cycle: v2.15  (current: v2.15 + 1 cycles → grace_period_elapsed: False)
- severe_defect_tag: True
- 30-day hit-rate: 0.0% (0 / 448 qualified queries)
- 60-day hit-rate: 0.0% (0 / 448 qualified queries)
- open_user_issues: 0
- consecutive_middle_cycles: 0
- PROMOTION TRIGGER (standard arm: >=5% AND pain-signal): NOT FIRED
- PROMOTION TRIGGER (defect-override arm: defect-tag AND >=1%): NOT FIRED
- CLOSURE TRIGGER (<1% AND 0 issues AND no defect-tag AND grace elapsed): NOT FIRED
- MIDDLE-BAND ESCALATION (>=3 consecutive cycles): NOT FIRED
- v2.X disposition: Defer to next cycle (continue telemetry)
- defect_summary: Docling extraction-layer prose+code intermixing at page boundaries; truncated CODE chunks (e.g. p326 ends mid-statement at '    return'). HybridChunker post-merge tested and reverted (fires 0x in production). See PROJECT_STATUS.md v2.14 Phase 6 PARTIAL row.
