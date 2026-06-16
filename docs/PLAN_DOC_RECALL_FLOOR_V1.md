# PLAN - the ~6% doc-recall floor (the "bigger lever"), evidence-grounded

Status: DRAFT (2026-06-16). Branch `feat/omnidocbench-phase0`.
Companion to the shipped HyDE-blend work (commit c2cb870) and memory
`project_retrieval_findings`. This plan is deliberately honest: the
characterization below shows the obvious cheap fixes are MARGINAL and
NOT statistically significant through the production pipeline. Read
Section 1 before proposing any fix - several "obvious" levers are
already measured and rejected here.

---

## 0. The problem (recap)

~6% of soak queries (31/514) never retrieve the right DOCUMENT at dense
top-100. The RAG answers well when the right doc is retrieved, so this
floor is the main remaining retrieval gap. HyDE-blend (shipped) recovers
~11/31 at the chunk level; this plan is about the rest.

## 1. What we MEASURED this session (do not re-derive; these are paired,
   judge-free, full-514 unless noted)

**1a. Root cause of the floor = empty image-placeholder chunks flood
dense retrieval.** For a representative miss ("Why were some of the
proposed designs not seriously considered..."), the dense **top-200 is
100% empty-content `modality=image` chunks from a SINGLE document**
(`af66389a8345`), score band 0.349-0.363. The gold text chunk sits past
rank 600. These placeholders carry no text - their only payload is
`visual_description = "[image: <hash>.png]"` - so they embed to a tight
noise cluster that outscores low-scoring gold chunks. (`/tmp/debug_flood.py`,
`/tmp/debug_depth.py`.)

**1b. Filtering empties recovers the DOC at top-100 (raw dense): 29/31
misses (94%); gold CHUNK into top-100: 26/31 (84%).** (`/tmp/measure_filter.py`.)
This is why the floor LOOKS like a cheap win.

**1c. BUT it collapses through the production pipeline.** Production
retrieves top-50 dense candidates then reranks to 10. The reranker
ALREADY absorbs most of the flood (raw-dense gold-chunk@10 = 81.9% but
post-rerank b50 = 87.4%). Measured production-faithful (top-K -> omlx
rerank -> gold-chunk@10, `/tmp/measure_depth.py`):

| arm                         | gold-chunk@10 | vs current (McNemar) |
|-----------------------------|---------------|----------------------|
| b50  (CURRENT prod)         | 87.4%         | -                    |
| b100 (depth 100, no filter) | 87.7%         | won 5 / lost 3, p=0.73 ns |
| f100 (filter + depth 100)   | 88.3%         | won 8 / lost 3, p=0.23 ns |

Filter-on-top-of-depth alone: won 3 / lost 0 (p=0.25 ns). Pure dense@10
with filter (no rerank): +0.0pp (won 0 / lost 0) - the recovered chunks
land at rank 30-90, inside top-100 but outside top-10, so only the
rerank pool depth converts them.

**Conclusion of Section 1:** the empty-chunk flood is real and dramatic
at the raw-dense layer, but the omlx reranker already mitigates it, so
empty-filtering + deeper retrieval nets **+0.9pp, NOT significant, with 3
regressions**. This is the same marginal territory as the three levers
rejected earlier this session (empties +0pp, rerank-sort -10pp,
hybrid-as-default ~0). **The residual 6% is a genuine
chunk-representation / embedder-ceiling problem, not a cheap pool-hygiene
bug.** Do not ship "empty-filter + deep retrieval" as THE fix for the
floor; it does not clear a significance gate.

## 2. The two regression cases (must understand before any filter ships)

The 3 filter regressions are queries whose gold chunk may itself be a
low-text asset chunk (image/table the answer lives in). Filtering
empty-content chunks would remove a legitimately-empty gold. ACTION
(entry gate for any filter work): pull the 3 regressed qids from
`logs/measure_depth.log` re-run, inspect their gold chunk modality +
content. If any gold is a content-less asset chunk, a content-emptiness
filter is unsafe as-is and must exempt gold-eligible asset modalities.

## 3. Candidate levers (ranked; each has a pre-registered decision rule)

### Lever A (RECOMMENDED bet) - contextual-chunk re-embedding
**Mechanism.** The residual misses are inferential / context-dependent
("the passage", "this cruise", "ch05/template.py", "why were designs not
considered"). Their gold chunk discusses the answer but in vocabulary far
from the question, and lacks self-identifying context. Anthropic's
"contextual retrieval" prepends a short LLM-generated doc/section context
string to each chunk BEFORE embedding (published ~35-49% reduction in
retrieval-failure rate). This directly targets "the gold chunk doesn't
embed near the query because it is context-stripped."
**Cost.** One local-LLM call per chunk at ingest (GX10, offline, one-off)
+ a re-embed + re-ingest of the 29-doc soak corpus into a SHADOW
collection. No production-path change until measured.
**Pre-registered decision rule.** Build `mmrag_v3__qwen3_ctx` (contextual
re-embed). Re-run `/tmp/measure_depth.py` style production-faithful
gold-chunk@10 + doc-recall@100 over the 514. SHIP only if: doc-recall@100
on the 31 misses recovers >= 12 (significant on McNemar p<0.05 vs current)
AND full-514 gold-chunk@10 does not regress (losses <= wins, no
significant drop). Otherwise DISCARD and accept the floor (Lever D).
**Risk.** Context strings can add noise for already-findable chunks
(the regression direction). The decision rule's no-regress clause guards it.

### Lever B (cheap hygiene, ship-independently-of-the-floor) - empty
de-prioritization + modest depth bump
Even though it does not clear the floor's significance gate, f100 never
LOST net (won 8 / lost 3) and the empty flood is objectively wrong
(content-less chunks competing in semantic search). Option: at ingest,
set `search_priority` low for content-empty image chunks (machinery
already exists: `ingest_to_qdrant.resolve_search_priority`), and/or raise
`top_k_retrieve` 50 -> 100 in the retrieval default. **Gate:** ship only
after Section 2's regression inspection clears; treat as hygiene, NOT as a
floor fix; measure no full-514 regression. Expect ~+0.5-0.9pp, ns.

### Lever C (DEFERRED, high-cost) - stronger / multi-vector embedder
ColBERT-style late interaction or a larger embedder could lift the
genuinely-deep gold chunks. Project policy is wary of the embedder-swap
reflex (`feedback_no_gx10_model_swap_reflex`), and the cost is high. Only
revisit if Lever A fails AND the floor is judged worth more investment.

### Lever D (the honest default) - ACCEPT the 6% floor
The RAG already answers well when the doc is retrieved; doc-recall is ~94%
and post-rerank chunk@10 is 87.4%. If Lever A does not clear its gate, the
correct engineering call is to accept the floor and stop spending on it -
the marginal levers are measured and do not move the answer bar.

## 4. Recommended sequence

1. **Section 2 regression inspection** (30 min, no risk) - decide if any
   empty-filter is even safe.
2. **Lever A pilot** (the one real bet): contextual re-embed into a shadow
   collection, measure against the pre-registered rule. This is the only
   approach that targets the residual's actual mechanism.
3. Decide from the measurement: ship Lever A if it clears the gate; else
   ship Lever B as hygiene only and adopt Lever D (accept the floor).
4. Do NOT pursue Lever C unless A fails and the floor is re-prioritized.

## 5. Reusable artifacts (already on disk)

- `output/v3_soak_code/doc_recall.jsonl` - per-query gold-doc rank (baseline).
- `output/v3_soak_code/hyde_blend.jsonl` - cached hypotheticals + base/hyde/blend doc-ranks.
- `output/v3_soak_code/hyde_ab.jsonl` - judge-free gold-chunk@10 base vs blend.
- `/tmp/measure_depth.py` - production-faithful retrieve+rerank gold-chunk@10
  with McNemar (the harness to clone for Lever A/B measurement).
- `/tmp/debug_flood.py`, `/tmp/measure_filter.py` - flood diagnosis + raw-dense recovery.

## 6. What NOT to do (measured + rejected)

- Do not ship empty-filtering as a floor fix (+0.9pp, ns, through prod).
- Do not sort by rerank_score (-10pp, prior finding).
- Do not make hybrid/BM25 the default (no gain + German regression).
- Do not reflexively swap the embedder before Lever A is measured.
- Do not re-measure at raw-dense@10 and conclude "+0pp" or at top-100 and
  conclude "94% recovery" - both are the wrong layer. The production
  pipeline (top-K -> rerank -> @10) is the only honest measurement layer.
