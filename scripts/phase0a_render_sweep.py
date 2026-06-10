#!/usr/bin/env python3
"""Phase 0A - VLM render-setting sweep (operational profiling + quality artifacts).

PLAN_EXTRACTION_FIDELITY_V1 Section 7 Phase 0A + Section 7.1. Answers the M5-load
question BEFORE any routing redesign: when we change page render resolution, what
happens to per-page cost (vision-token count via prompt_tokens, TTFT/prefill, decode,
total s/page, payload bytes, pages/hour, cold vs warm) AND to output quality (an
OmniDocBench-scored fidelity delta for the OmniDocBench-subset pages; saved artifacts
for the internal-corpus pages, where no ground truth exists).

Read-only + standalone by design: stdlib + fitz + urllib only for the SWEEP, so it
can never alter routing. It calls the VLM endpoint but persists NO chunks - only a
metrics JSONL and per-(page,setting) transcription artifacts under --out. The `score`
subcommand shells out to the OmniDocBench scorer in the `omnidocbench` conda env
(never imports it into this process).

It does NOT change server config (project guardrail: never reconfigure the M5 server
autonomously). On-host GPU-busy% / peak-temp must be sampled ON the M5 in parallel
(e.g. `sudo powermetrics --samplers gpu_power -i 1000`, user-permitted); this harness
captures every client-side signal (TTFT, decode tok/s, prompt/vision tokens, payload,
total s/page, pages/hour, cold/warm split, concurrent-queue-depth proxy) which is what
isolates a render/serialization bottleneck from a model-bound one.

Subcommands
-----------
  sweep             render each manifest page at each setting, transcribe, save
                    metrics + artifacts. OmniDocBench-subset artifacts also land
                    in <out>/preds/<setting>/<gt_name>.md for later scoring.
  score             score each setting's OmniDocBench preds vs GT (omnidocbench env)
                    and print the per-setting fidelity delta vs the dpi200 default.
  batch-probe       fire K concurrent requests at one setting to measure the
                    queue-depth / saturation proxy (wall vs sum-of-sequential).
  deindent-precheck strip indentation from a code-page markdown and measure how far
                    per-page edit distance moves UNDER THE SCORER'S NORMALIZATION
                    (clean_string strips whitespace) - the Section 7.3 seed point.

Usage
-----
  python scripts/phase0a_render_sweep.py sweep \
    --endpoint http://10.0.10.235:8000/v1 \
    --model mlx-community/Qwen3-VL-8B-Instruct-8bit \
    --out output/phase0a [--smoke] [--limit N] [--manifest manifest.json]
  python scripts/phase0a_render_sweep.py score --out output/phase0a
  python scripts/phase0a_render_sweep.py batch-probe --endpoint ... --model ... -k 3
  python scripts/phase0a_render_sweep.py deindent-precheck --md <code_page>.md
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import fitz  # PyMuPDF

HOME = Path.home()
EVAL_ROOT = HOME / "omnidocbench-eval"
ODB_REPO = EVAL_ROOT / "OmniDocBench"
ODB_PY = HOME / "miniforge3/envs/omnidocbench/bin/python"
# Full English GT (superset of every built page) - we filter our subset out of it.
GT_ENGLISH = EVAL_ROOT / "run" / "gt_english.json"
GT_BAKEOFF = EVAL_ROOT / "bakeoff" / "gt_bakeoff.json"
ODB_PDFS = EVAL_ROOT / "run" / "pdfs"

# Render settings swept (Section 7.1 / plan Phase 0A). dpi200 = production default
# and the baseline every fidelity delta is measured against. Keep dpi200 FIRST.
SETTINGS: list[tuple[str, str, int]] = [
    ("dpi200", "dpi", 200),
    ("dpi150", "dpi", 150),
    ("cap1600", "cap", 1600),
    ("cap1400", "cap", 1400),
]
BASELINE_SETTING = "dpi200"

# Small, representative INTERNAL two-corpus subset (real files in data/). No ground
# truth exists here, so these are ARTIFACT-VERDICT only (Section 7.1, A7): saved
# outputs for side-by-side, never a numeric delta.
INTERNAL_MANIFEST: list[dict] = [
    {"label": "carok_dutch_automotive_table", "corpus": "internal", "pdf": "data/data_spreadsheet/CarOK voorraadtelling 2021-04.pdf", "page": 1},
    {"label": "combat_aircraft_dense_magazine", "corpus": "internal", "pdf": "data/digital_magazine/Combat Aircraft - August 2025 UK.pdf", "page": 8},
    # page 65, NOT 60: p60 is a blank section break (zero text at every render
    # setting); p65 is the first genuine dense-code page (31-line indented JSON).
    {"label": "python_design_patterns_code", "corpus": "internal", "pdf": "data/technical_manual/Ayeva K. Mastering Python Design Patterns...essential Python patterns...3ed 2024.pdf", "page": 65},
    {"label": "dutch_business_form", "corpus": "internal", "pdf": "data/business_form/betwistingsformulier_aankoop_niet_ontvangen.pdf", "page": 0},
    {"label": "scanned_form_0013", "corpus": "internal", "pdf": "data/business_form/0013_140302111325_001.pdf", "page": 0},
]

# Small, FIXED OmniDocBench subset (prose, tables, form, multi-column). These are the
# already-built 1-page PDFs under ~/omnidocbench-eval/run/pdfs; their GT lives in the
# English GT, so the `score` step yields a COMPUTED OmniDocBench fidelity delta (A7).
# Chosen from the bake-off's stratified 44-page set so GT + PDF are guaranteed present.
_ODB_PAGES: list[tuple[str, str]] = [
    ("odb_academic_table", "docstructbench_llm-raw-scihub-o.O-ceat.200407001.pdf_3"),
    ("odb_magazine_prose", "docstructbench_enbook-zlib-o.O-21353024.pdf_39"),
    ("odb_form_fillable", "docstructbench_llm-raw-the-eye-o.O-Character%20Sheet%20-%20Form%20Fillable.pdf_2"),
    ("odb_newspaper_multicol", "newspaper_2a6b4fa088699701a6fa9ccecfb5c25d_1"),
]
OMNIDOCBENCH_MANIFEST: list[dict] = [
    {"label": label, "corpus": "omnidocbench", "gt_name": name,
     "pdf": str(ODB_PDFS / f"{name}.pdf"), "page": 0}
    for label, name in _ODB_PAGES
]

DEFAULT_MANIFEST: list[dict] = INTERNAL_MANIFEST + OMNIDOCBENCH_MANIFEST

VERBATIM_PROMPT = (
    "Transcribe ALL text on this page verbatim. Preserve layout, reading order, "
    "table structure (as a Markdown grid), and code indentation exactly. Describe "
    "any logos/figures/diagrams briefly. Do not summarize or omit anything."
)


# --------------------------------------------------------------------------- #
# rendering + VLM call (shared by sweep / batch-probe)
# --------------------------------------------------------------------------- #
def render(pdf: str, page_idx: int, mode: str, value: int) -> tuple[bytes, int, int]:
    """Render one page to PNG at the given setting. Returns (png_bytes, px_w, px_h)."""
    doc = fitz.open(pdf)
    if page_idx >= doc.page_count:
        raise IndexError(f"page {page_idx} out of range ({pdf} has {doc.page_count})")
    page = doc[page_idx]
    if mode == "dpi":
        pix = page.get_pixmap(dpi=value, alpha=False)
    elif mode == "cap":
        longest_pts = max(page.rect.width, page.rect.height) or 1.0
        zoom = value / longest_pts  # longest side -> `value` px
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    else:
        raise ValueError(f"unknown render mode {mode!r}")
    return pix.tobytes("png"), pix.width, pix.height


def call_vlm(endpoint: str, model: str, api_key: str, png: bytes, max_tokens: int,
             timeout: int) -> dict:
    """Stream one transcription. Returns timing + usage + text (or an error dict)."""
    img = "data:image/png;base64," + base64.b64encode(png).decode()
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": VERBATIM_PROMPT},
            {"type": "image_url", "image_url": {"url": img}},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    t0 = time.time()
    ttft = None
    chunks: list[str] = []
    usage = None
    try:
        for raw in urllib.request.urlopen(req, timeout=timeout):
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                chunks.append(ch[0]["delta"]["content"])
    except Exception as e:  # noqa: BLE001 - a probe; surface any transport/HTTP error
        return {"error": f"{type(e).__name__}: {e}", "total_s": round(time.time() - t0, 2)}
    text = "".join(chunks)
    total = time.time() - t0
    ttft = ttft or total
    decode_s = max(total - ttft, 1e-6)
    ct = (usage or {}).get("completion_tokens") or len(chunks)
    return {
        "error": None,
        "ttft_s": round(ttft, 2),
        "total_s": round(total, 2),
        "decode_s": round(decode_s, 2),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": ct,
        "decode_tok_s": round(ct / decode_s, 1),
        "chars": len(text),
        "text": text,
    }


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #
def cmd_sweep(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else DEFAULT_MANIFEST
    settings = SETTINGS[:1] if args.smoke else SETTINGS
    pages = manifest[:1] if args.smoke else manifest

    out = Path(args.out)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    # Record the manifest actually swept so `score` can find the OmniDocBench pages.
    (out / "manifest_used.json").write_text(json.dumps(manifest, indent=2))
    rows: list[dict] = []
    n = 0
    # Setting-OUTER / page-INNER: the first page of each setting block is the COLD
    # call for that setting (A7 cold-vs-warm); later pages are warm.
    with metrics_path.open("w") as mf:
        for slabel, mode, value in settings:
            cold_seen = False
            for item in pages:
                pdf, page = item["pdf"], int(item["page"])
                if not Path(pdf).exists():
                    print(f"SKIP (missing): {pdf}")
                    continue
                if args.limit and n >= args.limit:
                    break
                n += 1
                cold = not cold_seen
                cold_seen = True
                try:
                    png, pw, ph = render(pdf, page, mode, value)
                except Exception as e:  # noqa: BLE001
                    print(f"  RENDER FAIL {item['label']} {slabel}: {e}")
                    continue
                payload_kb = round(len(png) / 1024, 1)
                tag = "COLD" if cold else "warm"
                print(f"[{n}] {slabel}/{tag} {item['label']} p{page} ({pw}x{ph}, {payload_kb}KB) ...", flush=True)
                r = call_vlm(args.endpoint, args.model, args.api_key, png, args.max_tokens, args.timeout)
                row = {
                    "label": item["label"], "corpus": item.get("corpus", "internal"),
                    "gt_name": item.get("gt_name"), "pdf": pdf, "page": page,
                    "setting": slabel, "cold": cold, "px_w": pw, "px_h": ph,
                    "payload_kb": payload_kb, **{k: v for k, v in r.items() if k != "text"},
                }
                if r.get("error"):
                    print(f"    ERROR: {r['error']}")
                else:
                    pph = round(3600 / r["total_s"], 1) if r["total_s"] else None
                    row["pages_per_hour_est"] = pph
                    art = out / "artifacts" / f"{item['label']}__p{page}__{slabel}.md"
                    art.write_text(r["text"])
                    row["artifact"] = str(art)
                    # OmniDocBench-subset output is ALSO written to a per-setting preds
                    # dir named by GT page so `score` can compute a fidelity delta.
                    if item.get("corpus") == "omnidocbench" and item.get("gt_name"):
                        preds = out / "preds" / slabel
                        preds.mkdir(parents=True, exist_ok=True)
                        (preds / f"{item['gt_name']}.md").write_text(r["text"])
                    print(f"    {r['total_s']}s/page (ttft {r['ttft_s']}s, decode {r['decode_tok_s']} tok/s), "
                          f"prompt_tok={r['prompt_tokens']}, out_tok={r['completion_tokens']}, "
                          f"{r['chars']} chars, ~{pph} pages/hr")
                mf.write(json.dumps(row) + "\n")
                mf.flush()
                rows.append(row)

    _print_sweep_summary(rows, settings, metrics_path, out)
    return 0


def _print_sweep_summary(rows, settings, metrics_path, out) -> None:
    ok = [r for r in rows if not r.get("error")]

    def mean(s, k):
        vals = [r[k] for r in s if r.get(k) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n===== PHASE 0A SWEEP SUMMARY (mean per render setting; cold split out) =====")
    print(f"{'setting':>9} | {'n':>2} | {'cold s':>7} | {'warm s':>7} | {'ttft':>6} | "
          f"{'prompt_tok':>10} | {'payloadKB':>9} | {'pages/hr':>8}")
    for slabel, _, _ in settings:
        s = [r for r in ok if r["setting"] == slabel]
        if not s:
            continue
        cold = [r for r in s if r.get("cold")]
        warm = [r for r in s if not r.get("cold")]
        print(f"{slabel:>9} | {len(s):>2} | {mean(cold, 'total_s'):>7.1f} | "
              f"{mean(warm, 'total_s'):>7.1f} | {mean(s, 'ttft_s'):>6.1f} | "
              f"{mean(s, 'prompt_tokens'):>10.0f} | {mean(s, 'payload_kb'):>9.1f} | "
              f"{mean(s, 'pages_per_hour_est'):>8.1f}")
    errs = [r for r in rows if r.get("error")]
    print(f"\n{len(ok)} ok / {len(errs)} errored. Metrics: {metrics_path}")
    print(f"Internal artifacts (ARTIFACT VERDICT, no number): {out / 'artifacts'}")
    print(f"OmniDocBench preds (score with `score` subcommand): {out / 'preds'}")
    print("NOTE: internal column is an ARTIFACT VERDICT - compare artifacts across "
          "settings to SEE quality loss; never present it as a measured delta (A7).")


# --------------------------------------------------------------------------- #
# score (OmniDocBench fidelity delta [computed], A7)
# --------------------------------------------------------------------------- #
def _build_gt_subset(gt_names: set[str], dest: Path) -> int:
    """Filter the full English GT down to the swept OmniDocBench pages."""
    src = GT_ENGLISH if GT_ENGLISH.exists() else GT_BAKEOFF
    gt = json.loads(src.read_text(encoding="utf-8"))
    sub = [p for p in gt if os.path.splitext(p["page_info"]["image_path"])[0] in gt_names]
    dest.write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
    return len(sub)


def _score_one(gt_json: Path, preds_dir: Path, score_dir: Path) -> dict | None:
    """Run the OmniDocBench scorer (omnidocbench env) for one preds dir."""
    score_dir.mkdir(parents=True, exist_ok=True)
    cfg = score_dir / "score_config.yaml"
    cfg.write_text(
        "end2end_eval:\n"
        "  dataset:\n"
        "    dataset_name: end2end_dataset\n"
        "    ground_truth:\n"
        f"      data_path: {gt_json}\n"
        "    match_method: quick_match\n"
        "    match_workers: 8\n"
        "    match_timeout_sec: 420\n"
        "    prediction:\n"
        f"      data_path: {preds_dir}\n"
        "    quick_match_truncated_timeout_sec: 300\n"
        "    timeout_fallback_max_chunk_span: 10\n"
        "    timeout_fallback_order_penalty: 0.1\n"
        "  metrics:\n"
        "    text_block:\n      metric: [Edit_dist]\n"
        "    table:\n      metric: [TEDS, Edit_dist]\n      teds_workers: 8\n"
        "    reading_order:\n      metric: [Edit_dist]\n",
        encoding="utf-8",
    )
    # The scorer names its output by the PREDS-DIR basename (preds/<setting> ->
    # "<setting>_quick_match_metric_result.json"), not a fixed "preds_" prefix.
    src = ODB_REPO / "result" / f"{preds_dir.name}_quick_match_metric_result.json"
    if src.exists():  # clear any stale result so a failed run can't copy an old score
        src.unlink()
    proc = subprocess.run([str(ODB_PY), "pdf_validation.py", "--config", str(cfg)], cwd=str(ODB_REPO))
    if not src.exists():
        print(f"  SCORE FAIL (rc={proc.returncode}): no metric_result for {preds_dir}")
        return None
    dest = score_dir / "metric_result.json"
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    r = json.loads(dest.read_text(encoding="utf-8"))
    try:
        return {
            "text_ED": r["text_block"]["page"]["Edit_dist"]["ALL"],
            "reading_ED": r["reading_order"]["page"]["Edit_dist"]["ALL"],
            "table_TEDS": r["table"]["all"]["TEDS"]["all"],
        }
    except KeyError:
        return {"text_ED": None, "reading_ED": None, "table_TEDS": None}


def cmd_score(args: argparse.Namespace) -> int:
    # Absolute: the scorer runs with cwd=ODB_REPO, so relative config/GT/preds
    # paths would resolve under the repo and vanish.
    out = Path(args.out).resolve()
    preds_root = out / "preds"
    if not preds_root.exists():
        print(f"No preds dir at {preds_root}; run `sweep` first.")
        return 1
    gt_names = {p.stem for s in preds_root.iterdir() if s.is_dir() for p in s.glob("*.md")}
    if not gt_names:
        print("No OmniDocBench preds found to score.")
        return 1
    gt_subset = out / "odb_gt_subset.json"
    n_gt = _build_gt_subset(gt_names, gt_subset)
    print(f"GT subset: {n_gt} pages -> {gt_subset}")

    results: dict[str, dict] = {}
    for slabel, _, _ in SETTINGS:
        preds_dir = preds_root / slabel
        if not preds_dir.exists():
            continue
        print(f"\n--- scoring setting {slabel} ({len(list(preds_dir.glob('*.md')))} preds) ---")
        m = _score_one(gt_subset, preds_dir, out / "score" / slabel)
        if m:
            results[slabel] = m

    base = results.get(BASELINE_SETTING, {})
    print("\n===== OmniDocBench FIDELITY DELTA (computed; vs dpi200 baseline) =====")
    print(f"{'setting':>9} | {'text ED':>8} | {'dED':>7} | {'read ED':>8} | {'TEDS':>7} | {'dTEDS':>7}")
    print("(text/read ED: lower=better, +dED=worse; TEDS: higher=better, -dTEDS=worse)")
    for slabel, _, _ in SETTINGS:
        m = results.get(slabel)
        if not m:
            continue
        ded = (m["text_ED"] - base["text_ED"]) if (m.get("text_ED") is not None and base.get("text_ED") is not None) else None
        dteds = (m["table_TEDS"] - base["table_TEDS"]) if (m.get("table_TEDS") is not None and base.get("table_TEDS") is not None) else None
        def f(x, w=8, p=4):
            return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'-':>{w}}"
        print(f"{slabel:>9} | {f(m.get('text_ED'))} | {f(ded, 7)} | {f(m.get('reading_ED'))} | "
              f"{f(m.get('table_TEDS'), 7)} | {f(dteds, 7)}")
    (out / "odb_scores.json").write_text(json.dumps(results, indent=2))
    print(f"\nScores -> {out / 'odb_scores.json'}")
    print("CAVEAT: synthetic image-PDF, scanned-lane routing; EN-only; small fixed "
          "subset (failure-mode exposure, not a benchmark). dpi200 = production default.")
    return 0


# --------------------------------------------------------------------------- #
# batch-probe (queue-depth / saturation proxy, A7 single-vs-batch request shape)
# --------------------------------------------------------------------------- #
def cmd_batch_probe(args: argparse.Namespace) -> int:
    item = next((p for p in DEFAULT_MANIFEST if p["label"] == args.label), DEFAULT_MANIFEST[0])
    pdf, page = item["pdf"], int(item["page"])
    if not Path(pdf).exists():
        print(f"probe page missing: {pdf}")
        return 1
    slabel, mode, value = SETTINGS[0]
    png, pw, ph = render(pdf, page, mode, value)
    k = args.concurrency

    # 1) sequential baseline (single-page request shape): one warmup + one timed.
    call_vlm(args.endpoint, args.model, args.api_key, png, args.max_tokens, args.timeout)
    seq = call_vlm(args.endpoint, args.model, args.api_key, png, args.max_tokens, args.timeout)
    seq_s = seq.get("total_s")

    # 2) K concurrent (small-batch request shape): wall time of K simultaneous calls.
    results: list[dict] = [None] * k  # type: ignore[list-item]

    def worker(i: int) -> None:
        results[i] = call_vlm(args.endpoint, args.model, args.api_key, png, args.max_tokens, args.timeout)

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(k)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    per = [r.get("total_s") for r in results if r and not r.get("error")]
    errs = [r for r in results if r and r.get("error")]
    # If the server serialized the K calls, wall ~= k * seq_s (queue depth ~k).
    # If it parallelized, wall ~= seq_s. The ratio is the saturation proxy.
    ratio = round(wall / seq_s, 2) if seq_s else None
    print("\n===== BATCH-SHAPE / QUEUE-DEPTH PROBE (dpi200) =====")
    print(f"page={item['label']} ({pw}x{ph})  concurrency k={k}")
    print(f"sequential single-call: {seq_s}s/page")
    print(f"k concurrent: wall={round(wall,2)}s  per-call mean={round(sum(per)/len(per),2) if per else 'n/a'}s  errors={len(errs)}")
    print(f"wall / single ratio = {ratio}  (~k => fully serialized/queue-bound; ~1 => parallel)")
    if errs:
        print(f"  probe errors: {[e['error'] for e in errs][:3]}")
    return 0


# --------------------------------------------------------------------------- #
# deindent-precheck (Section 7.3 seed: how blind is the text metric to indentation?)
# --------------------------------------------------------------------------- #
def _clean_string(s: str) -> str:
    """Replicates OmniDocBench text_postprocess.clean_string (the text-metric
    normalization). KEY: it deletes ALL whitespace - so indentation loss is
    invisible to the scored text edit distance. Kept in sync with the scorer."""
    import re
    s = str(s or "")
    for a, b in [("\\t", ""), ("\\n", ""), ("\t", ""), ("\n", ""), ("/t", ""),
                 ("/n", ""), (" ", ""), ("✓", "✔"), ("√", "✔"),
                 ("-", "—"), ("∼", "～"), ("Ø", "∅")]:
        s = s.replace(a, b)
    s = re.sub(r"_{4,}", "____", s)
    s = re.sub(r" {4,}", "    ", s)
    return s


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cmd_deindent_precheck(args: argparse.Namespace) -> int:
    if args.md:
        original = Path(args.md).read_text()
    else:
        # Render + transcribe the default code page once to get ground-truthish md.
        item = next(p for p in INTERNAL_MANIFEST if p["label"] == "python_design_patterns_code")
        png, _, _ = render(item["pdf"], int(item["page"]), "dpi", 200)
        r = call_vlm(args.endpoint, args.model, args.api_key, png, args.max_tokens, args.timeout)
        if r.get("error"):
            print(f"precheck render/transcribe failed: {r['error']}")
            return 1
        original = r["text"]

    deindented = "\n".join(line.lstrip() for line in original.splitlines())

    # RAW (unnormalized) edit distance - what a layout-faithful metric would see.
    raw_ed = _levenshtein(original, deindented)
    raw_norm = raw_ed / max(len(original), len(deindented), 1)
    # SCORER-NORMALIZED edit distance - what OmniDocBench actually scores.
    co, cd = _clean_string(original), _clean_string(deindented)
    scorer_ed = _levenshtein(co, cd)
    scorer_norm = scorer_ed / max(len(co), len(cd), 1)

    result = {
        "chars_original": len(original),
        "chars_deindented": len(deindented),
        "raw_edit_dist": raw_ed,
        "raw_norm_ed": round(raw_norm, 6),
        "scorer_norm_chars_original": len(co),
        "scorer_norm_chars_deindented": len(cd),
        "scorer_edit_dist": scorer_ed,
        "scorer_norm_ed": round(scorer_norm, 6),
        "finding": ("scorer text-ED is BLIND to indentation (clean_string strips "
                    "whitespace): scorer_norm_ed ~0 while the raw layout-ED is "
                    f"{round(raw_norm, 4)}"),
    }
    print("\n===== DE-INDENT PRE-CHECK (Section 7.3 seed) =====")
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")
    return 0


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def add_vlm_args(p):
        p.add_argument("--endpoint", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--api-key", default="local")
        p.add_argument("--max-tokens", type=int, default=4096, help="production cap")
        p.add_argument("--timeout", type=int, default=600)

    p = sub.add_parser("sweep")
    add_vlm_args(p)
    p.add_argument("--out", default="output/phase0a")
    p.add_argument("--manifest", help="JSON list of {label,corpus,pdf,page[,gt_name]}")
    p.add_argument("--smoke", action="store_true", help="first page, first setting only")
    p.add_argument("--limit", type=int, default=0, help="cap number of (page,setting) calls")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("score")
    p.add_argument("--out", default="output/phase0a")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("batch-probe")
    add_vlm_args(p)
    p.add_argument("--label", default=OMNIDOCBENCH_MANIFEST[0]["label"])
    p.add_argument("-k", "--concurrency", type=int, default=3)
    p.set_defaults(func=cmd_batch_probe)

    p = sub.add_parser("deindent-precheck")
    add_vlm_args(p)
    p.add_argument("--md", help="code-page markdown; if omitted, render+transcribe the default code page")
    p.add_argument("--out", help="write the result JSON here")
    p.set_defaults(func=cmd_deindent_precheck)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
