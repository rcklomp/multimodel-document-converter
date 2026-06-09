#!/usr/bin/env python3
"""OmniDocBench extractor bake-off (PLAN_OMNIDOCBENCH_EVAL Phase 1, Section 6 item 1).

Objectively confirm/revisit the MinerU2.5 + Qwen-for-code choice on LABELED ground
truth. Runs the SAME OmniDocBench page subset through our shipping pipeline under
several engine routes, renders each to per-page Markdown (reusing the Phase 0
adapter's render half), scores each route with OmniDocBench, and reports a
side-by-side table.

STANDALONE orchestration: it sets engine-selection env vars and shells out to the
`mmrag-v2` CLI (via scripts/omnidocbench_adapter.py) -- it does NOT import the
extraction path (R4). Scoring shells out to the isolated `omnidocbench` env.

Engines (route env per src/mmrag_v3/processor.py:extract precedence):
  docling_fast  USE_DOCLING_FAST=1                          (Phase 0 offline baseline)
  mineru        USE_MINERU_ENGINE=1 + MINERU_ENDPOINT/MODEL (pure MinerU2.5)
  qwen3vl       USE_VLM_ENGINE=1    + VLM_NATIVE_*           (pure Qwen3-VL)
  hybrid        MINERU_ENDPOINT + VLM_NATIVE_* (default route) (MinerU+Qwen-for-code)
  paddleocr     USE_VLM_ENGINE=1 + VLM_NATIVE_MODEL=PaddleOCR (stretch; smoke-gated)
  granite       USE_VLM_ENGINE=1 + VLM_NATIVE_MODEL=granite   (stretch; emits DocTags)

The M5 box (10.0.10.235:8000) serves all four models (verified /v1/models 2026-06-09).
paddleocr/granite go through the generic VLM engine with a different model id; they
are smoke-gated -- kept only if the smoke produces usable Markdown, else logged as
deferred (need a dedicated adapter), never silently dropped.

Subcommands:
  prep                build the bake-off subset (N/source from the stratified
                      manifest) -> per-engine workspaces with symlinked PDFs.
  smoke   --engine X  run ONE code/table-dense page through engine X (verify wiring).
  run     --engine X  run + render the whole subset through engine X.
  score   --engine X  score engine X's preds vs the bake-off GT (omnidocbench env).
  report              collect every engine's metric_result.json into one table.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
ADAPTER = Path(__file__).resolve().parent / "omnidocbench_adapter.py"
EVAL_ROOT = HOME / "omnidocbench-eval"
STRAT_MANIFEST = EVAL_ROOT / "run" / "manifest_full.json"  # full english, to sub-sample
GT_ENGLISH = EVAL_ROOT / "run" / "gt_english.json"
SRC_PDFS = EVAL_ROOT / "run" / "pdfs"                       # already-built 1-page PDFs
BAKEOFF_ROOT = EVAL_ROOT / "bakeoff"
ODB_REPO = EVAL_ROOT / "OmniDocBench"

MMRAG_PY = HOME / "miniforge3/envs/mmrag-v2/bin/python"
MMRAG_CLI = HOME / "miniforge3/envs/mmrag-v2/bin/mmrag-v2"
ODB_PY = HOME / "miniforge3/envs/omnidocbench/bin/python"

M5 = "http://10.0.10.235:8000"
VLM_V1 = f"{M5}/v1"

# Engine-selection env. Empty-string values are explicitly UNSET before applying
# (so e.g. the qwen3vl route does not inherit a stray MINERU_ENDPOINT).
ENGINES: dict[str, dict[str, str]] = {
    "docling_fast": {
        "USE_DOCLING_FAST": "1",
    },
    "mineru": {
        "USE_MINERU_ENGINE": "1",
        "MINERU_ENDPOINT": M5,  # base URL, NO /v1 (mineru_vl_utils http-client)
        "MINERU_MODEL": "mlx-community/MinerU2.5-2509-1.2B-bf16",  # FULL served id
        "MINERU_API_KEY": "local",
    },
    "qwen3vl": {
        "USE_VLM_ENGINE": "1",
        "VLM_NATIVE_ENDPOINT": VLM_V1,
        "VLM_NATIVE_MODEL": "mlx-community/Qwen3-VL-8B-Instruct-8bit",
        "VLM_NATIVE_API_KEY": "local",
    },
    "hybrid": {  # default route: MINERU_ENDPOINT set + VLM_NATIVE_* set, no force flag
        "MINERU_ENDPOINT": M5,
        "MINERU_MODEL": "mlx-community/MinerU2.5-2509-1.2B-bf16",
        "MINERU_API_KEY": "local",
        "VLM_NATIVE_ENDPOINT": VLM_V1,
        "VLM_NATIVE_MODEL": "mlx-community/Qwen3-VL-8B-Instruct-8bit",
        "VLM_NATIVE_API_KEY": "local",
    },
    "paddleocr": {  # stretch: generic VLM engine, different model id. Probe 2026-06-09
                    # confirmed it returns Markdown + tables via the chat API.
        "USE_VLM_ENGINE": "1",
        "VLM_NATIVE_ENDPOINT": VLM_V1,
        "VLM_NATIVE_MODEL": "mlx-community/PaddleOCR-VL-1.5-8bit",
        "VLM_NATIVE_API_KEY": "local",
    },
    # DEFERRED: granite-docling-258M-mlx. The M5 mlx server cannot LOAD it
    # (HTTP 500 "Unrecognized image processor", probe 2026-06-09) -- a server-side
    # model-load failure, not our pipeline. Also emits DocTags, not Markdown, so it
    # would need a dedicated adapter even once the server can serve it. Re-add here
    # if/when both are resolved.
}

# Every engine-selection key any route sets -- cleared before applying a route so
# routes never leak into each other.
_ALL_ENGINE_KEYS = sorted({k for env in ENGINES.values() for k in env} | {
    "USE_MINERU_ENGINE", "USE_VLM_ENGINE", "USE_DOCLING_FAST",
    "USE_HYBRID_ENGINE", "USE_MINERU_QWEN_HYBRID",
})


def _engine_env(engine: str) -> dict[str, str]:
    base = dict(os.environ)
    for key in _ALL_ENGINE_KEYS:
        base.pop(key, None)
    base.update(ENGINES[engine])
    return base


def _workspace(engine: str) -> Path:
    return BAKEOFF_ROOT / engine


# --------------------------------------------------------------------------- #
def cmd_prep(args: argparse.Namespace) -> int:
    from collections import defaultdict

    man = json.loads(STRAT_MANIFEST.read_text(encoding="utf-8"))
    gt = json.loads(GT_ENGLISH.read_text(encoding="utf-8"))
    by_src: dict[str, list] = defaultdict(list)
    for p in man["pages"]:
        by_src[p["data_source"]].append(p)
    picked = []
    for src in sorted(by_src):
        picked.extend(sorted(by_src[src], key=lambda x: x["name"])[: args.per_source])
    picked_names = {p["name"] for p in picked}

    BAKEOFF_ROOT.mkdir(parents=True, exist_ok=True)
    # GT subset matching the bake-off pages
    gt_sub = [p for p in gt if os.path.splitext(p["page_info"]["image_path"])[0] in picked_names]
    (BAKEOFF_ROOT / "gt_bakeoff.json").write_text(json.dumps(gt_sub, ensure_ascii=False), encoding="utf-8")

    for engine in ENGINES:
        ws = _workspace(engine)
        ws.mkdir(parents=True, exist_ok=True)
        # per-engine manifest (same pages, same images dir)
        man_e = dict(man)
        man_e["pages"] = picked
        man_e["count"] = len(picked)
        (ws / "manifest.json").write_text(json.dumps(man_e, indent=2, ensure_ascii=False), encoding="utf-8")
        # share the already-built PDFs via a symlink
        link = ws / "pdfs"
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
        if not link.exists():
            link.symlink_to(SRC_PDFS)
    print(
        f"PREP: {len(picked)} pages ({args.per_source}/source) | gt_bakeoff={len(gt_sub)} | "
        f"engines={list(ENGINES)} -> {BAKEOFF_ROOT}"
    )
    return 0


def _adapter(engine: str, sub: str, env: dict[str, str], extra: list[str] | None = None) -> int:
    cmd = [str(MMRAG_PY), str(ADAPTER), "--workspace", str(_workspace(engine)), sub]
    if extra:
        cmd += extra
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def cmd_smoke(args: argparse.Namespace) -> int:
    """One page (the table/title-bearing book page) through engine X."""
    engine = args.engine
    env = _engine_env(engine)
    # limit-1 run + render; the first manifest page is deterministic
    rc = _adapter(engine, "run", {**env, "OMNI_SMOKE": "1"}, ["--cli", str(MMRAG_CLI), "--limit", "1", "--force"])
    if rc != 0:
        print(f"SMOKE {engine}: run FAILED rc={rc}")
        return rc
    rc = _adapter(engine, "render", env, ["--limit", "1"])
    # show the produced markdown for eyeballing
    ws = _workspace(engine)
    man = json.loads((ws / "manifest.json").read_text())
    name = man["pages"][0]["name"]
    md = ws / "preds" / f"{name}.md"
    print(f"\n===== SMOKE {engine}: {name} =====")
    print(md.read_text(encoding="utf-8")[:1200] if md.exists() else "(no markdown produced)")
    return rc


def cmd_run(args: argparse.Namespace) -> int:
    engine = args.engine
    env = _engine_env(engine)
    rc = _adapter(engine, "run", env, ["--cli", str(MMRAG_CLI)])
    if rc != 0:
        print(f"RUN {engine}: rc={rc} (some pages failed; render proceeds on what exists)")
    _adapter(engine, "render", env)
    return rc


def cmd_score(args: argparse.Namespace) -> int:
    engine = args.engine
    ws = _workspace(engine)
    cfg = ws / "score_config.yaml"
    cfg.write_text(
        "end2end_eval:\n"
        "  dataset:\n"
        "    dataset_name: end2end_dataset\n"
        "    ground_truth:\n"
        f"      data_path: {BAKEOFF_ROOT / 'gt_bakeoff.json'}\n"
        "    match_method: quick_match\n"
        "    match_workers: 8\n"
        "    match_timeout_sec: 420\n"
        "    prediction:\n"
        f"      data_path: {ws / 'preds'}\n"
        "    quick_match_truncated_timeout_sec: 300\n"
        "    timeout_fallback_max_chunk_span: 10\n"
        "    timeout_fallback_order_penalty: 0.1\n"
        "  metrics:\n"
        "    text_block:\n      metric: [Edit_dist]\n"
        "    table:\n      metric: [TEDS, Edit_dist]\n      teds_workers: 8\n"
        "    reading_order:\n      metric: [Edit_dist]\n",
        encoding="utf-8",
    )
    out = ws / "score"
    out.mkdir(exist_ok=True)
    # pdf_validation writes to ./result relative to cwd; run inside the repo and copy.
    proc = subprocess.run([str(ODB_PY), "pdf_validation.py", "--config", str(cfg)], cwd=str(ODB_REPO))
    src = ODB_REPO / "result" / "preds_quick_match_metric_result.json"
    if src.exists():
        (out / "metric_result.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"SCORE {engine}: -> {out / 'metric_result.json'}")
    else:
        print(f"SCORE {engine}: metric_result.json NOT produced (rc={proc.returncode})")
    return proc.returncode


def _metrics(engine: str) -> dict | None:
    f = _workspace(engine) / "score" / "metric_result.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text(encoding="utf-8"))
    try:
        return {
            "text_ED": r["text_block"]["page"]["Edit_dist"]["ALL"],
            "reading_ED": r["reading_order"]["page"]["Edit_dist"]["ALL"],
            "table_TEDS": r["table"]["all"]["TEDS"]["all"],
            "table_TEDS_struct": r["table"]["all"]["TEDS_structure_only"]["all"],
        }
    except KeyError:
        return None


def cmd_report(args: argparse.Namespace) -> int:
    print(f"\n{'engine':<14} {'text ED':>9} {'reading ED':>11} {'table TEDS':>11} {'TEDS struct':>12}")
    print("-" * 60)
    for engine in ENGINES:
        m = _metrics(engine)
        if not m:
            print(f"{engine:<14} {'(no score)':>9}")
            continue
        print(f"{engine:<14} {m['text_ED']:>9.4f} {m['reading_ED']:>11.4f} "
              f"{m['table_TEDS']:>11.4f} {m['table_TEDS_struct']:>12.4f}")
    print("\n(text ED / reading ED: lower=better; TEDS: higher=better)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prep"); p.add_argument("--per-source", type=int, default=6); p.set_defaults(func=cmd_prep)
    p = sub.add_parser("smoke"); p.add_argument("--engine", required=True, choices=list(ENGINES)); p.set_defaults(func=cmd_smoke)
    p = sub.add_parser("run"); p.add_argument("--engine", required=True, choices=list(ENGINES)); p.set_defaults(func=cmd_run)
    p = sub.add_parser("score"); p.add_argument("--engine", required=True, choices=list(ENGINES)); p.set_defaults(func=cmd_score)
    p = sub.add_parser("report"); p.set_defaults(func=cmd_report)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
