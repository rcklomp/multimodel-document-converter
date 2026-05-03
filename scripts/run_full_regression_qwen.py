#!/usr/bin/env python3
"""Run full PDF corpus conversion with online Qwen VLM/refiner."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from mmrag_v2.config import load_config


def _slug(path: Path) -> str:
    text = f"{path.parent.name}__{path.stem}"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return text[:140]


def _jsonl_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run_full_regression_qwen.py OUTPUT_ROOT", file=sys.stderr)
        return 2

    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "_run.log"
    manifest_path = root / "_manifest.jsonl"
    summary_path = root / "_conversion_summary.jsonl"
    private_config_path = root / "_qwen_runtime_config.yml"

    cfg = load_config()
    api_key = cfg.refiner.api_key or cfg.vlm.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("No API key available from config/env", file=sys.stderr)
        return 2

    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    vision_model = "qwen3-vl-plus"
    refiner_model = cfg.refiner.model or "qwen-plus"
    per_doc_timeout = int(os.environ.get("MMRAG_FULL_REGRESSION_TIMEOUT", "7200"))

    private_config_path.write_text(
        "\n".join(
            [
                "vlm:",
                "  provider: openai",
                f"  model: {vision_model}",
                f"  base_url: {base_url}",
                f"  api_key: {api_key}",
                "  timeout: 180",
                "refiner:",
                "  enabled: true",
                "  provider: openai",
                f"  model: {refiner_model}",
                f"  base_url: {base_url}",
                f"  api_key: {api_key}",
                "code_enrichment:",
                "  enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    private_config_path.chmod(0o600)

    pdfs = sorted(Path("data").rglob("*.pdf")) + sorted(Path("data").rglob("*.PDF"))
    unique: list[Path] = []
    seen: set[Path] = set()
    for pdf in pdfs:
        resolved = pdf.resolve()
        if resolved not in seen:
            unique.append(pdf)
            seen.add(resolved)

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for idx, pdf in enumerate(unique, 1):
            outdir = root / _slug(pdf)
            manifest.write(
                json.dumps(
                    {"index": idx, "pdf": str(pdf), "output_dir": str(outdir)},
                    ensure_ascii=False,
                )
                + "\n"
            )

    with log_path.open("a", encoding="utf-8") as log, summary_path.open(
        "a", encoding="utf-8"
    ) as summary:
        header = {
            "event": "start",
            "time": datetime.now(timezone.utc).isoformat(),
            "pdf_count": len(unique),
            "output_root": str(root),
            "vision_provider": "openai",
            "vision_model": vision_model,
            "vision_base_url": base_url,
            "refiner_provider": "openai",
            "refiner_model": refiner_model,
            "batch_size": 10,
            "per_doc_timeout": per_doc_timeout,
        }
        log.write(json.dumps(header, ensure_ascii=False) + "\n")
        log.flush()
        print(
            f"FULL_REGRESSION_START root={root} "
            f"pdf_count={len(unique)} model={vision_model}",
            flush=True,
        )

        for idx, pdf in enumerate(unique, 1):
            outdir = root / _slug(pdf)
            outdir.mkdir(parents=True, exist_ok=True)
            doc_log = outdir / "_convert.log"
            ingestion = outdir / "ingestion.jsonl"
            if ingestion.exists():
                record = {
                    "index": idx,
                    "pdf": str(pdf),
                    "output_dir": str(outdir),
                    "exit_code": 0,
                    "elapsed_seconds": 0.0,
                    "ingestion_jsonl": str(ingestion),
                    "jsonl_lines": _jsonl_count(ingestion),
                    "log": str(doc_log),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "skipped_existing": True,
                }
                summary.write(json.dumps(record, ensure_ascii=False) + "\n")
                summary.flush()
                print(
                    f"[{idx:02d}/{len(unique):02d}] SKIP existing "
                    f"lines={record['jsonl_lines']} out={outdir}",
                    flush=True,
                )
                continue

            cmd = [
                sys.executable,
                "-m",
                "mmrag_v2.cli",
                "process",
                str(pdf),
                "--output-dir",
                str(outdir),
                "--vision-provider",
                "openai",
                "--vision-model",
                vision_model,
                "--vision-base-url",
                base_url,
                "--vlm-timeout",
                "180",
                "--enable-refiner",
                "--refiner-provider",
                "openai",
                "--refiner-model",
                refiner_model,
                "--refiner-base-url",
                base_url,
                "--batch-size",
                "10",
            ]
            start = time.time()
            print(f"[{idx:02d}/{len(unique):02d}] START {pdf}", flush=True)
            env = os.environ.copy()
            env["MMRAG_CONFIG"] = str(private_config_path)
            with doc_log.open("w", encoding="utf-8") as handle:
                try:
                    proc = subprocess.run(
                        cmd,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                        env=env,
                        timeout=per_doc_timeout,
                    )
                    exit_code = proc.returncode
                    timed_out = False
                except subprocess.TimeoutExpired:
                    handle.write(f"\n[TIMEOUT] exceeded {per_doc_timeout}s\n")
                    exit_code = 124
                    timed_out = True
            elapsed = time.time() - start
            line_count = _jsonl_count(ingestion)
            record = {
                "index": idx,
                "pdf": str(pdf),
                "output_dir": str(outdir),
                "exit_code": exit_code,
                "elapsed_seconds": round(elapsed, 2),
                "ingestion_jsonl": str(ingestion) if ingestion.exists() else None,
                "jsonl_lines": line_count,
                "log": str(doc_log),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "timed_out": timed_out,
            }
            summary.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary.flush()
            log.write(json.dumps({"event": "doc_done", **record}, ensure_ascii=False) + "\n")
            log.flush()
            status = "DONE" if exit_code == 0 and ingestion.exists() else "FAIL"
            print(
                f"[{idx:02d}/{len(unique):02d}] {status} "
                f"exit={exit_code} seconds={elapsed:.1f} "
                f"lines={line_count} out={outdir}",
                flush=True,
            )

    print(f"FULL_REGRESSION_DONE root={root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
