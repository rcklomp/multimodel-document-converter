#!/usr/bin/env python3
"""v2.10 synthetic-soak harness — LLM-judged retrieval quality eval.

Substitutes a "two weeks of real RAG usage" soak with an automated
LLM-as-judge protocol that:

1. SAMPLE — stratified-sample ~300 text chunks across the 34-doc
   `mmrag_v2_8` corpus. Heuristics: ≥ 150 chars, not pure-code, not
   advertisement.
2. GENERATE — for each sampled chunk, ask Dashscope qwen-max to write
   2 natural user queries whose answer is in the chunk (~600 queries).
3. RETRIEVE — embed each query via Ollama llava, search `mmrag_v2_8`
   top-5.
4. JUDGE — ask Dashscope qwen-max to grade each top-1 result on
   relevance / format / faithfulness (0-2 each).
5. REPORT — aggregate per-doc and corpus-wide metrics + a list of the
   worst (lowest-scoring) (query, chunk) pairs as v2.10.x candidate
   defects. Writes
   `docs/QUALITY_SNAPSHOT_<DATE>_v2.10_soak.md`.

Pinned design choices (2026-05-16):
- Judge provider: Dashscope `qwen-max` (best judgment quality in this
  pipeline; matches Phase 5b enrichment provider).
- Query count: 300 chunks × 2 queries = 600.
- Threshold: report-only. No ship gate; humans read the numbers.
- Cadence: run on every tagged release.

Resumable. Re-running the script picks up where the work file left
off so an API hiccup mid-stage doesn't waste prior calls. Stages
can also be run individually:

  python scripts/synthetic_soak.py --stage sample
  python scripts/synthetic_soak.py --stage generate
  python scripts/synthetic_soak.py --stage retrieve
  python scripts/synthetic_soak.py --stage judge
  python scripts/synthetic_soak.py --stage report
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from search_qdrant import embed as _embed_ollama, search  # noqa: E402
from ingest_to_qdrant import embed_text_dashscope  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "output" / "soak" / "v2.10"
DEFAULT_WORK_PATH = OUTPUT_DIR / "work.jsonl"
DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / (
    f"QUALITY_SNAPSHOT_{datetime.now().strftime('%Y-%m-%d')}_v2.10_soak.md"
)
DOCS_ROOT = REPO_ROOT / "output"
COLLECTION_DEFAULT_DASHSCOPE = "mmrag_v2_8__qwen3_dashscope"
COLLECTION_DEFAULT_OLLAMA = "mmrag_v2_8"
EMBED_MODEL_OLLAMA = "llava"
EMBED_MODEL_DASHSCOPE = "text-embedding-v4"
TOP_K = 5

# Dashscope OpenAI-compatible endpoint (matches scripts/convert_all.sh /
# refiner.py pattern).
DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
JUDGE_MODEL = "qwen-max"
GENERATOR_MODEL = "qwen-max"

# Heuristic content filters for chunk sampling.
MIN_CHUNK_CHARS = 150
MAX_CODE_RATIO = 0.4
ADVERT_KEYWORDS = ("subscribe", "buy now", "click here", "limited time", "discount")

# 34 canonical doc directories. Mirrors scripts/rebuild_mmrag_v2_8_for_rc1.py.
CANONICAL_34 = [
    "HarryPotter_and_the_Sorcerers_Stone", "Form_0013_invoice", "Form_betwistingsformulier",
    "CarOK_voorraadtelling", "AIOS_LLM_Agent_Operating_System",
    "A_comprehensive_review_on_hybrid_electri", "Hybrid_electric_vehicles",
    "IRJET_Modeling_of_Solar_PV", "Recent_Trends_in_Transportation",
    "Combat_Aircraft_August_2025", "PCWorld_July_2025", "ATZ_Elektronik_German",
    "Kimothi_RAG_Guide", "Integra_manual", "Jungjun_AI_Agent", "Bourne_RAG_2024",
    "Devlin_LLM_Agents", "Raieli_AI_Agents", "Adedeji_GenAI_Google_Cloud",
    "Cronin_GenAI_Models", "Hao_ML_Platform", "Nagasubramanian_Agentic_AI",
    "Sekar_MCP_Standard", "Python_Cookbook", "ArcGIS_Python_Cookbook",
    "Fluent_Python", "Python_Distilled", "Ayeva_Python_Patterns",
    "Chaubal_PyTorch_Projects", "Earthship_Vol1", "Firearms", "Greenhouse_Design",
    "ChatGPT_Praktijk_handboek", "KI_En_ChatGPT_Praktische_Gids",
]


def _load_chunks(doc_name: str) -> list[dict]:
    jsonl = DOCS_ROOT / doc_name / "ingestion.jsonl"
    if not jsonl.exists():
        return []
    chunks: list[dict] = []
    for i, line in enumerate(jsonl.open("r", encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if i == 0 and obj.get("object_type") == "ingestion_metadata":
            continue
        chunks.append(obj)
    return chunks


def _is_eligible_text_chunk(chunk: dict) -> bool:
    if chunk.get("modality") != "text":
        return False
    content = (chunk.get("content") or "").strip()
    if len(content) < MIN_CHUNK_CHARS:
        return False
    # Reject mostly-code chunks (heuristic: many short indented lines)
    lines = content.splitlines()
    if lines:
        code_like = sum(1 for ln in lines if ln.startswith(("    ", "\t", "  ")) or ln.strip().startswith((">>>", "...")))
        if code_like / max(1, len(lines)) > MAX_CODE_RATIO:
            return False
    lowered = content.lower()
    if any(kw in lowered for kw in ADVERT_KEYWORDS):
        return False
    return True


def stage_sample(seed: int, n_chunks: int, work_path: Path) -> None:
    if work_path.exists():
        print(f"  sample: work file already exists at {work_path}; skip (delete to re-sample)")
        return
    rng = random.Random(seed)
    print(f"  sample: stratified across {len(CANONICAL_34)} docs, target n={n_chunks}, seed={seed}")
    per_doc_target = max(1, n_chunks // len(CANONICAL_34))
    sampled: list[dict] = []
    for doc_name in CANONICAL_34:
        chunks = [c for c in _load_chunks(doc_name) if _is_eligible_text_chunk(c)]
        if not chunks:
            print(f"    {doc_name}: 0 eligible (skip)")
            continue
        take = min(per_doc_target, len(chunks))
        picks = rng.sample(chunks, take)
        for p in picks:
            sampled.append({
                "doc_dir": doc_name,
                "gold_chunk_id": p.get("chunk_id"),
                "gold_doc_id": p.get("doc_id"),
                "gold_source_file": (p.get("metadata") or {}).get("source_file"),
                "gold_page_number": (p.get("metadata") or {}).get("page_number"),
                "gold_content": (p.get("content") or "").strip(),
                "queries": [],
            })
        print(f"    {doc_name}: sampled {take}/{len(chunks)}")
    # Cap to exactly n_chunks if oversampled (rare with the floor)
    rng.shuffle(sampled)
    if len(sampled) > n_chunks:
        sampled = sampled[:n_chunks]
    # Assign deterministic sample_ids in shuffled order
    for i, row in enumerate(sampled, start=1):
        row["sample_id"] = f"S{i:04d}"
    work_path.parent.mkdir(parents=True, exist_ok=True)
    with work_path.open("w", encoding="utf-8") as fh:
        for row in sampled:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  sample: wrote {len(sampled)} rows to {work_path}")


def _read_work(work_path: Path) -> list[dict]:
    if not work_path.exists():
        return []
    rows: list[dict] = []
    for line in work_path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_work(work_path: Path, rows: list[dict]) -> None:
    tmp = work_path.with_suffix(work_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(work_path)


def _call_dashscope(api_key: str, model: str, messages: list[dict],
                    temperature: float = 0.0, max_tokens: int = 600,
                    timeout: int = 60, retries: int = 3) -> str | None:
    """OpenAI-compatible chat-completions call against Dashscope. Returns
    response text or None on failure."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(DASHSCOPE_URL, data=body, method="POST")
            req.add_header("Authorization", f"Bearer {api_key}")
            req.add_header("Content-Type", "application/json")
            resp = urllib.request.urlopen(req, timeout=timeout)
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            try:
                detail = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                detail = ""
            print(f"    ! Dashscope HTTP {e.code}: {detail}", file=sys.stderr)
            return None
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue
        except Exception as e:
            print(f"    ! Dashscope error: {e}", file=sys.stderr)
            return None
    print(f"    ! Dashscope failed after {retries} retries: {last_err}", file=sys.stderr)
    return None


_JSON_ARRAY_RE = re.compile(r"\[\s*(?:\".*?\"\s*,?\s*)+\]", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_json(text: str, expect: str) -> Any:
    """Robust JSON extraction (LLMs sometimes wrap output in prose/markdown)."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if expect == "array":
        m = _JSON_ARRAY_RE.search(text)
    else:
        m = _JSON_OBJECT_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


GENERATE_SYSTEM = (
    "You write evaluation queries for a retrieval-augmented generation system. "
    "Return ONLY a JSON array of exactly 2 short natural questions (5-15 words each). "
    "Do not include any prose around the JSON."
)

GENERATE_USER_TEMPLATE = """Given this passage, write 2 natural user questions whose answer is in this passage.
Requirements:
- Questions should be specific enough that this passage (or a closely related passage from the same document) would be a strong answer.
- Do not quote the passage verbatim.
- Make the two questions DIFFERENT in shape — e.g. one fact-seeking ("what is..."), one explanatory or how-to.
- Use the same language as the passage (English, Dutch, or German).

Passage:
\"\"\"
{content}
\"\"\"

Return ONLY a JSON array of 2 strings."""


def stage_generate(work_path: Path, api_key: str) -> None:
    rows = _read_work(work_path)
    if not rows:
        print("  generate: no work file; run --stage sample first", file=sys.stderr)
        return
    needed = sum(1 for r in rows if not r.get("queries"))
    print(f"  generate: {needed}/{len(rows)} rows need queries")
    done = 0
    for i, row in enumerate(rows):
        if row.get("queries"):
            continue
        content = row["gold_content"][:2500]  # cap input size
        result = _call_dashscope(
            api_key, GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": GENERATE_SYSTEM},
                {"role": "user", "content": GENERATE_USER_TEMPLATE.format(content=content)},
            ],
            temperature=0.3, max_tokens=300,
        )
        queries: list[str] = []
        if result:
            parsed = _extract_json(result, "array")
            if isinstance(parsed, list):
                queries = [str(q).strip() for q in parsed if isinstance(q, (str, int))]
        if not queries or len(queries) < 2:
            print(f"    [{row['sample_id']}] generation failed; skipping", file=sys.stderr)
            row["queries"] = []  # leave empty so a re-run can retry
            continue
        row["queries"] = [
            {"query_id": f"{row['sample_id']}.Q{j+1}", "query_text": q.strip()}
            for j, q in enumerate(queries[:2])
        ]
        done += 1
        if done % 10 == 0 or i + 1 == len(rows):
            _write_work(work_path, rows)
            print(f"    [{i+1}/{len(rows)}] generated {done} so far (work flushed)")
    _write_work(work_path, rows)
    print(f"  generate: wrote queries for {done} rows")


def _embed_query(text: str, provider: str, model: str,
                 ollama_url: str, api_key: str) -> list[float]:
    if provider == "dashscope":
        return embed_text_dashscope(text, model, api_key)
    return _embed_ollama(text, model=model, ollama_url=ollama_url)


def stage_retrieve(work_path: Path, qdrant_url: str, ollama_url: str,
                   collection: str, provider: str, embed_model: str,
                   api_key: str) -> None:
    rows = _read_work(work_path)
    if not rows:
        print("  retrieve: no work file", file=sys.stderr)
        return
    queries_total = sum(len(r.get("queries") or []) for r in rows)
    queries_done = sum(1 for r in rows for q in (r.get("queries") or []) if q.get("retrieval"))
    print(f"  retrieve: {queries_done}/{queries_total} queries already retrieved "
          f"(collection={collection}, provider={provider}, model={embed_model})")
    flushed = 0
    for r in rows:
        for q in (r.get("queries") or []):
            if q.get("retrieval"):
                continue
            try:
                vec = _embed_query(q["query_text"], provider, embed_model,
                                    ollama_url, api_key)
                results = search(vec, collection, limit=TOP_K, qdrant_url=qdrant_url)
            except Exception as e:
                print(f"    ! retrieval failed for {q['query_id']}: {e}", file=sys.stderr)
                continue
            top = []
            for hit in results[:TOP_K]:
                payload = hit.get("payload") or {}
                top.append({
                    "chunk_id": payload.get("chunk_id") or str(hit.get("id")),
                    "doc_id": payload.get("doc_id"),
                    "source_file": payload.get("source_file"),
                    "page_number": payload.get("page_number"),
                    "modality": payload.get("modality"),
                    "score": round(float(hit.get("score") or 0.0), 6),
                    "content": (payload.get("content") or "").strip()[:1500],
                })
            q["retrieval"] = {"top_k": top}
            flushed += 1
            if flushed % 20 == 0:
                _write_work(work_path, rows)
    _write_work(work_path, rows)
    print(f"  retrieve: completed {flushed} new retrievals")


JUDGE_SYSTEM = (
    "You evaluate retrieval-augmented generation (RAG) quality. "
    "Return ONLY a JSON object with integer fields relevance, format, faithfulness "
    "(each 0, 1, or 2) and a short string field rationale. No prose outside the JSON."
)

JUDGE_USER_TEMPLATE = """Grade how well the RETRIEVED chunk answers the USER QUERY.

USER QUERY:
{query}

GOLD PASSAGE (the chunk that was used to generate the query — for context only, do not penalize the retrieved chunk for being a different chunk from the same document):
\"\"\"
{gold}
\"\"\"

RETRIEVED CHUNK (top-1 from the retrieval system):
source_file: {source_file}
page: {page}
\"\"\"
{retrieved}
\"\"\"

Score on three axes, each 0/1/2:

1. relevance: Does the retrieved chunk's content answer the user query?
   2 = answers it directly. 1 = same topic but doesn't really answer. 0 = wrong domain.

2. format: Is the retrieved chunk content well-formed prose / code / table?
   2 = clean and readable. 1 = minor issues (some truncation, odd whitespace).
   0 = broken (leaked markup, garbled OCR, marker artifacts, severe truncation).

3. faithfulness: Would a user reading ONLY this chunk get a correct answer?
   2 = self-contained correct answer. 1 = partial / needs more context.
   0 = misleading or wrong.

Return ONLY: {{"relevance": <0-2>, "format": <0-2>, "faithfulness": <0-2>, "rationale": "<one sentence>"}}"""


def stage_judge(work_path: Path, api_key: str) -> None:
    rows = _read_work(work_path)
    if not rows:
        print("  judge: no work file", file=sys.stderr)
        return
    queries_total = sum(len(r.get("queries") or []) for r in rows)
    queries_done = sum(
        1 for r in rows for q in (r.get("queries") or [])
        if q.get("judgment") and q["judgment"].get("relevance") is not None
    )
    print(f"  judge: {queries_done}/{queries_total} queries already judged")
    flushed = 0
    for r in rows:
        for q in (r.get("queries") or []):
            judgment = q.get("judgment") or {}
            if judgment.get("relevance") is not None:
                continue
            retrieval = q.get("retrieval") or {}
            top = (retrieval.get("top_k") or [])
            if not top:
                continue
            top1 = top[0]
            content = top1.get("content") or ""
            result = _call_dashscope(
                api_key, JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                        query=q["query_text"],
                        gold=r["gold_content"][:1500],
                        source_file=top1.get("source_file") or "",
                        page=top1.get("page_number"),
                        retrieved=content[:1500],
                    )},
                ],
                temperature=0.0, max_tokens=200,
            )
            parsed = _extract_json(result or "", "object") if result else None
            if not isinstance(parsed, dict) or "relevance" not in parsed:
                print(f"    ! judge parse failed for {q['query_id']}; skipping", file=sys.stderr)
                continue
            try:
                q["judgment"] = {
                    "relevance": int(parsed.get("relevance", 0)),
                    "format": int(parsed.get("format", 0)),
                    "faithfulness": int(parsed.get("faithfulness", 0)),
                    "rationale": str(parsed.get("rationale", ""))[:300],
                }
            except (ValueError, TypeError):
                print(f"    ! judge cast failed for {q['query_id']}; skipping", file=sys.stderr)
                continue
            flushed += 1
            if flushed % 20 == 0:
                _write_work(work_path, rows)
                print(f"    judged {flushed} so far (work flushed)")
    _write_work(work_path, rows)
    print(f"  judge: completed {flushed} new judgments")


def stage_report(work_path: Path, report_path: Path,
                 collection: str, provider: str, embed_model: str) -> None:
    rows = _read_work(work_path)
    if not rows:
        print("  report: no work file", file=sys.stderr)
        return

    n_chunks = len(rows)
    queries: list[dict] = []
    for r in rows:
        for q in (r.get("queries") or []):
            queries.append({"row": r, "query": q})

    # Recall metrics (deterministic, no LLM)
    recall_at_1_chunk = 0
    recall_at_5_chunk = 0
    recall_at_5_doc = 0
    judged = 0
    rel_sum = fmt_sum = fait_sum = 0
    rel_max = fmt_max = fait_max = 0
    per_doc: dict[str, dict[str, int]] = {}
    weakest: list[tuple[int, dict]] = []  # (total_score, query+row pair)

    for entry in queries:
        r = entry["row"]
        q = entry["query"]
        gold_chunk = r["gold_chunk_id"]
        gold_doc = r["gold_doc_id"]
        doc_dir = r["doc_dir"]
        slot = per_doc.setdefault(doc_dir, {"queries": 0, "r1_chunk": 0, "r5_chunk": 0, "r5_doc": 0,
                                            "rel_sum": 0, "fmt_sum": 0, "fait_sum": 0, "judged": 0})
        slot["queries"] += 1
        retrieval = (q.get("retrieval") or {}).get("top_k") or []
        if retrieval:
            top_ids = [hit["chunk_id"] for hit in retrieval]
            top_docs = [hit.get("doc_id") for hit in retrieval]
            if top_ids and top_ids[0] == gold_chunk:
                recall_at_1_chunk += 1
                slot["r1_chunk"] += 1
            if gold_chunk in top_ids[:5]:
                recall_at_5_chunk += 1
                slot["r5_chunk"] += 1
            if gold_doc in top_docs[:5]:
                recall_at_5_doc += 1
                slot["r5_doc"] += 1
        judgment = q.get("judgment") or {}
        if judgment.get("relevance") is not None:
            judged += 1
            rel = int(judgment["relevance"]); fmt = int(judgment["format"]); fait = int(judgment["faithfulness"])
            rel_sum += rel; fmt_sum += fmt; fait_sum += fait
            rel_max += 2; fmt_max += 2; fait_max += 2
            slot["rel_sum"] += rel; slot["fmt_sum"] += fmt; slot["fait_sum"] += fait
            slot["judged"] += 1
            total = rel + fmt + fait
            weakest.append((total, entry))

    weakest.sort(key=lambda x: x[0])
    weakest_top = weakest[:15]

    def pct(num, denom):
        return f"{(num / denom * 100):.1f}%" if denom else "n/a"

    now = datetime.now(timezone.utc)
    lines: list[str] = []
    lines.append(f"# Quality Snapshot {now.strftime('%Y-%m-%d')} — SOAK ({collection})")
    lines.append("")
    lines.append("> **Status:** synthetic-soak report.")
    try:
        rel_src = work_path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        rel_src = work_path
    lines.append(f"> Source: `{rel_src}`.")
    lines.append(f"> Judge: Dashscope `{JUDGE_MODEL}`. Generator: `{GENERATOR_MODEL}`. Embedder: `{embed_model}` (provider={provider}). Collection: `{collection}`.")
    lines.append("> No QA threshold; this snapshot is informational.")
    lines.append("")
    lines.append("## 1. Corpus summary")
    lines.append("")
    lines.append(f"- Sampled chunks: **{n_chunks}** across {len(per_doc)} docs.")
    lines.append(f"- Queries generated: **{len(queries)}**.")
    lines.append(f"- Queries judged: **{judged}/{len(queries)}** ({pct(judged, len(queries))}).")
    lines.append("")
    lines.append("## 2. Headline metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Recall@1 (gold chunk_id is top-1) | {pct(recall_at_1_chunk, len(queries))} ({recall_at_1_chunk}/{len(queries)}) |")
    lines.append(f"| Recall@5 (gold chunk_id in top-5) | {pct(recall_at_5_chunk, len(queries))} ({recall_at_5_chunk}/{len(queries)}) |")
    lines.append(f"| Recall@5 (gold doc_id in top-5)   | {pct(recall_at_5_doc, len(queries))} ({recall_at_5_doc}/{len(queries)}) |")
    lines.append(f"| Relevance score                   | {pct(rel_sum, rel_max)} ({rel_sum}/{rel_max}) |")
    lines.append(f"| Format score                      | {pct(fmt_sum, fmt_max)} ({fmt_sum}/{fmt_max}) |")
    lines.append(f"| Faithfulness score                | {pct(fait_sum, fait_max)} ({fait_sum}/{fait_max}) |")
    lines.append("")
    lines.append("## 3. Per-document metrics")
    lines.append("")
    lines.append("| Doc | Queries | R@1 | R@5 (chunk) | R@5 (doc) | Relevance | Format | Faith |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for doc_dir in sorted(per_doc.keys()):
        s = per_doc[doc_dir]
        lines.append(
            f"| {doc_dir} | {s['queries']} | "
            f"{pct(s['r1_chunk'], s['queries'])} | "
            f"{pct(s['r5_chunk'], s['queries'])} | "
            f"{pct(s['r5_doc'], s['queries'])} | "
            f"{pct(s['rel_sum'], s['judged']*2)} | "
            f"{pct(s['fmt_sum'], s['judged']*2)} | "
            f"{pct(s['fait_sum'], s['judged']*2)} |"
        )
    lines.append("")
    lines.append("## 4. Weakest 15 (query, top-1) pairs — v2.10.x defect candidates")
    lines.append("")
    for total, entry in weakest_top:
        q = entry["query"]; r = entry["row"]
        retrieval = (q.get("retrieval") or {}).get("top_k") or []
        top1 = retrieval[0] if retrieval else {}
        jud = q.get("judgment") or {}
        lines.append(
            f"- **{q['query_id']}** total={total}/6 (r={jud.get('relevance')}, "
            f"f={jud.get('format')}, faith={jud.get('faithfulness')})"
        )
        lines.append(f"  - Query: {q['query_text']!r}")
        lines.append(f"  - Gold doc: `{r['doc_dir']}` (chunk `{r['gold_chunk_id']}`)")
        lines.append(f"  - Top-1: `{top1.get('source_file')}` p={top1.get('page_number')} score={top1.get('score')}")
        lines.append(f"  - Judge rationale: {jud.get('rationale')}")
    lines.append("")
    lines.append("## 5. Methodology")
    lines.append("")
    lines.append(f"- Sampled {n_chunks} text chunks (≥ {MIN_CHUNK_CHARS} chars, ≤ {int(MAX_CODE_RATIO*100)}% code-like lines, no advertisement keywords). Stratified across the 34-doc canonical corpus.")
    lines.append(f"- Each chunk → 2 queries generated by `{GENERATOR_MODEL}` (temperature 0.3).")
    lines.append(f"- Each query → top-{TOP_K} retrieved from `{collection}` via `{provider}` provider, model `{embed_model}`.")
    lines.append(f"- Each top-1 chunk → graded by `{JUDGE_MODEL}` (temperature 0.0) on relevance / format / faithfulness, each 0-2.")
    lines.append("- Gold passage is shown to the judge for context; the judge is instructed NOT to penalize a different-chunk same-document retrieval.")
    lines.append("")
    lines.append("## 6. Revision log")
    lines.append("")
    lines.append("| Date | Change |")
    lines.append("|---|---|")
    lines.append(f"| {now.strftime('%Y-%m-%d')} | Initial v2.10.0-rc1 soak snapshot. |")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  report: wrote {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["sample", "generate", "retrieve", "judge", "report", "all"],
                        default="all")
    parser.add_argument("--n-chunks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-path", default=str(DEFAULT_WORK_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--collection", default=None,
                        help="Qdrant collection to retrieve from. Provider-aware defaults: "
                             f"'{COLLECTION_DEFAULT_DASHSCOPE}' (dashscope), "
                             f"'{COLLECTION_DEFAULT_OLLAMA}' (ollama).")
    parser.add_argument("--provider", default="dashscope", choices=["ollama", "dashscope"],
                        help="Embedding provider for query-side (default: dashscope as of v2.11.0). "
                             "Must match how the target collection was built.")
    parser.add_argument("--embed-model", default=None,
                        help="Query-side embedding model. Default 'text-embedding-v4' for dashscope; 'llava' for ollama.")
    args = parser.parse_args()

    work_path = Path(args.work_path)
    report_path = Path(args.report_path)
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    needs_key = args.stage in ("generate", "judge", "all") or (
        args.stage in ("retrieve", "all") and args.provider == "dashscope"
    )
    if needs_key and not api_key:
        print("ERROR: DASHSCOPE_API_KEY env var is not set; required for generate/judge "
              "and for retrieve when --provider dashscope.", file=sys.stderr)
        return 2

    if args.collection is None:
        args.collection = (COLLECTION_DEFAULT_DASHSCOPE if args.provider == "dashscope"
                          else COLLECTION_DEFAULT_OLLAMA)
    if args.embed_model is None:
        args.embed_model = (EMBED_MODEL_DASHSCOPE if args.provider == "dashscope"
                            else EMBED_MODEL_OLLAMA)

    if args.stage in ("sample", "all"):
        print("[stage] sample")
        stage_sample(args.seed, args.n_chunks, work_path)
    if args.stage in ("generate", "all"):
        print("[stage] generate")
        stage_generate(work_path, api_key)
    if args.stage in ("retrieve", "all"):
        print("[stage] retrieve")
        stage_retrieve(work_path, args.qdrant_url, args.ollama_url,
                       args.collection, args.provider, args.embed_model, api_key)
    if args.stage in ("judge", "all"):
        print("[stage] judge")
        stage_judge(work_path, api_key)
    if args.stage in ("report", "all"):
        print("[stage] report")
        stage_report(work_path, report_path,
                     args.collection, args.provider, args.embed_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
