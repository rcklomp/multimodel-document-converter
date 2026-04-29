#!/usr/bin/env python3
"""Evaluate the configured VLM prompt on image chunks from an ingestion JSONL.

This is a prompt-optimization harness, not a production converter. It re-sends
each image asset to the configured OpenAI-compatible VLM with the current
visual-only prompt and stores one JSONL result per image for review.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any

import requests
import yaml

from mmrag_v2.vision.vision_prompts import (
    build_text_reading_retry_prompt,
    build_visual_prompt,
    sanitize_text_reading_response,
    validate_vlm_response,
)


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _image_chunks(ingestion: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with ingestion.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "metadata":
                continue
            if row.get("modality") == "image" and (row.get("asset_ref") or {}).get("file_path"):
                chunks.append(row)
    return chunks


def _call_vlm(
    *,
    base_url: str,
    api_key: str,
    model: str,
    image_path: Path,
    prompt: str,
    timeout: int,
) -> tuple[str, float, str | None]:
    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    data_url = f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 120,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    start = time.time()
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=(10, timeout),
        )
        elapsed = time.time() - start
        if r.status_code != 200:
            return "", elapsed, f"HTTP {r.status_code}: {r.text[:500]}"
        data = r.json()
        return data["choices"][0]["message"]["content"].strip(), elapsed, None
    except Exception as exc:  # pragma: no cover - diagnostic script
        return "", time.time() - start, f"{type(exc).__name__}: {exc}"


def _call_validated_vlm(
    *,
    base_url: str,
    api_key: str,
    model: str,
    image_path: Path,
    prompt: str,
    timeout: int,
) -> tuple[str, str, list[str], float, str | None]:
    raw, elapsed, error = _call_vlm(
        base_url=base_url,
        api_key=api_key,
        model=model,
        image_path=image_path,
        prompt=prompt,
        timeout=timeout,
    )
    if error:
        return raw, "", [error], elapsed, error

    validation = validate_vlm_response(raw)
    total_elapsed = elapsed
    if not validation.is_valid:
        retry_prompt = (
            build_text_reading_retry_prompt(prompt)
            if validation.text_reading_detected
            else prompt
        )
        retry_raw, retry_elapsed, retry_error = _call_vlm(
            base_url=base_url,
            api_key=api_key,
            model=model,
            image_path=image_path,
            prompt=retry_prompt,
            timeout=timeout,
        )
        total_elapsed += retry_elapsed
        if retry_error:
            return retry_raw, "", [retry_error], total_elapsed, retry_error
        raw = retry_raw
        validation = validate_vlm_response(raw)

    if validation.is_valid:
        return raw, validation.cleaned_response, [], total_elapsed, None
    if validation.text_reading_detected:
        sanitized = sanitize_text_reading_response(raw)
        sanitized_validation = validate_vlm_response(sanitized)
        if sanitized_validation.is_valid:
            return raw, sanitized_validation.cleaned_response, validation.issues, total_elapsed, None
        return raw, "Dense typographic layout; no distinct non-text visuals.", validation.issues, total_elapsed, None
    return raw, "[VLM_FAILED: response invalid]", validation.issues, total_elapsed, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ingestion_jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path.home() / ".mmrag-v2.yml")
    parser.add_argument("--config-section", choices=("vlm", "refiner"), default="vlm")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=220)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    vlm_cfg = cfg.get(args.config_section, {})
    base_url = str(args.base_url or vlm_cfg["base_url"]).rstrip("/")
    api_key = str(vlm_cfg["api_key"])
    model = str(args.model or vlm_cfg["model"])

    ingestion = args.ingestion_jsonl.resolve()
    out_root = ingestion.parent
    chunks = _image_chunks(ingestion)
    if args.limit:
        chunks = chunks[: args.limit]

    done: set[str] = set()
    if args.resume and args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["chunk_id"])

    # Use the generic prompt for this eval. It is the safest cross-domain
    # baseline and avoids technical-diagram bias from cheap pixel heuristics.
    prompt = build_visual_prompt(context_section=None, is_diagram=False, is_photograph=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as out:
        for idx, chunk in enumerate(chunks, 1):
            chunk_id = chunk["chunk_id"]
            if chunk_id in done:
                continue
            asset_ref = chunk["asset_ref"]["file_path"]
            image_path = out_root / asset_ref
            raw_description, description, validation_issues, elapsed, error = _call_validated_vlm(
                base_url=base_url,
                api_key=api_key,
                model=model,
                image_path=image_path,
                prompt=prompt,
                timeout=args.timeout,
            )
            record = {
                "index": idx,
                "chunk_id": chunk_id,
                "page_number": chunk.get("metadata", {}).get("page_number"),
                "extraction_method": chunk.get("metadata", {}).get("extraction_method"),
                "asset_path": asset_ref,
                "asset_width": chunk.get("asset_ref", {}).get("width_px"),
                "asset_height": chunk.get("asset_ref", {}).get("height_px"),
                "old_description": chunk.get("content"),
                "raw_description": raw_description,
                "new_description": description,
                "validation_issues": validation_issues,
                "elapsed_s": round(elapsed, 3),
                "error": error,
                "model": model,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            status = "ERR" if error else "OK"
            print(f"[{idx}/{len(chunks)}] {status} p{record['page_number']} {chunk_id} {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
