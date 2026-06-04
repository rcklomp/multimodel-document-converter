#!/usr/bin/env python
"""VLM-eval inference via an OpenAI-compatible HTTP endpoint.

Runs the golden set against a SERVED model (e.g. the Qwen3-VL server already up
on the M5) - avoids loading a second copy of a large model in-process (which
GPU-address-faults when the server already holds one). Output format matches
vlm_eval_infer_mlx.py so the same scorer applies. Runs from any machine (hits
the endpoint over HTTP); no model deps.

    python scripts/vlm_eval_infer_http.py \
        --endpoint http://macbook-pro-m5.lan:8000/v1 \
        --model mlx-community/Qwen3-VL-8B-Instruct-8bit \
        --label qwen_baseline --golden-dir output/vlm_eval/golden_set \
        --out-dir output/vlm_eval/runs
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import requests

DEFAULT_PROMPT = (
    "Extract every element on this page as strict JSON: an array of objects "
    "{type: text|image|table|code|form, content, bbox:[x0,y0,x1,y1]}. Tables MUST "
    "be markdown grids in content; code fenced with exact indentation. Output JSON only."
)


def _png_data_uri(path: Path, max_side: int) -> str:
    from PIL import Image

    with Image.open(path) as im:
        if max_side > 0 and max(im.size) > max_side:
            scale = max_side / max(im.size)
            im = im.resize((int(im.size[0] * scale), int(im.size[1] * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="OpenAI-compatible base (.../v1)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--golden-dir", default="output/vlm_eval/golden_set")
    ap.add_argument("--out-dir", default="output/vlm_eval/runs")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-side", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    manifest = json.loads((Path(args.golden_dir) / "manifest.json").read_text())
    out_dir = Path(args.out_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    url = args.endpoint.rstrip("/") + "/chat/completions"
    print(f"# {args.label} -> {url} ({args.model}); prompt={args.prompt[:60]!r}", flush=True)

    for entry in manifest:
        pid = entry["id"]
        data_uri = _png_data_uri(Path(args.golden_dir) / f"{pid}.png", args.max_side)
        payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
        }
        t0 = time.time()
        try:
            r = requests.post(url, json=payload, timeout=args.timeout)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            text, status = f"[HTTP_ERROR] {type(exc).__name__}: {exc}", "error"
        dt = round(time.time() - t0, 1)
        (out_dir / f"{pid}.json").write_text(
            json.dumps(
                {
                    "id": pid,
                    "capability": entry["capability"],
                    "model": args.model,
                    "prompt": args.prompt,
                    "latency_s": dt,
                    "status": status,
                    "out_chars": len(text),
                    "output": text,
                    "render_px": entry.get("render_px"),
                },
                ensure_ascii=False,
            )
        )
        print(f"  {pid:28s} [{entry['capability']:6s}] {dt:>6}s {status} {len(text)}ch", flush=True)
    print(f"# done -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
