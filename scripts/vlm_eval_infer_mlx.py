#!/usr/bin/env python
"""VLM-eval inference runner (mlx_vlm, runs on the M5 in an ISOLATED env).

Runs ONE candidate model over the golden test set and captures raw output +
latency per page. Model-agnostic: any mlx_vlm-loadable model (PaddleOCR-VL,
Granite-Docling, Qwen3-VL, dots.ocr, ...). Scoring is a SEPARATE step
(scripts/vlm_eval_score.py) so this script has only mlx_vlm deps and the scorer
stays clean - honoring the "VLM deps never in the mmrag-v2 env" rule.

ENV: create an isolated env on the M5 (which already has mlx for the Qwen
server), e.g.:
    python -m venv ~/vlm-eval && source ~/vlm-eval/bin/activate
    pip install -U mlx-vlm pillow
Then, per candidate:
    python scripts/vlm_eval_infer_mlx.py \
        --model mlx-community/PaddleOCR-VL-1.5-8bit \
        --label paddleocr_vl \
        --golden-dir output/vlm_eval/golden_set \
        --out-dir output/vlm_eval/runs \
        --prompt "OCR this page to structured markdown with tables."

Outputs output/vlm_eval/runs/<label>/<page_id>.json
({id, model, prompt, latency_s, output, render_px}) + a run_meta.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Per-model default prompts. The doc-specialist models each want their own
# instruction; override with --prompt. These are starting points to refine.
DEFAULT_PROMPTS = {
    "paddleocr": "Convert this document page to structured output: markdown for "
    "text and tables (use | grid syntax), with layout regions.",
    "granite-docling": "Convert this page to docling.",
    "dots": "Parse the layout of this page and return the elements as JSON with "
    "category, bbox, and text/markdown content.",
    "qwen": "Extract every element on this page as strict JSON: per-element type "
    "(text/image/table/code/form), content (markdown grid for tables, fenced code "
    "with exact indentation), and bbox.",
    "default": "Convert this document page to structured markdown, preserving "
    "tables as markdown grids and code with exact indentation.",
}


def _default_prompt(model: str) -> str:
    m = model.lower()
    for key, prompt in DEFAULT_PROMPTS.items():
        if key in m:
            return prompt
    return DEFAULT_PROMPTS["default"]


def _maybe_resize(path: Path, max_side: int) -> str:
    """Optionally downscale a huge render so models don't OOM; return a path."""
    if max_side <= 0:
        return str(path)
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
        if max(w, h) <= max_side:
            return str(path)
        scale = max_side / max(w, h)
        im2 = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        out = path.with_name(path.stem + f"_max{max_side}.png")
        im2.save(out)
        return str(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id (mlx-community/...)")
    ap.add_argument("--label", required=True, help="short name for the output dir")
    ap.add_argument("--golden-dir", default="output/vlm_eval/golden_set")
    ap.add_argument("--out-dir", default="output/vlm_eval/runs")
    ap.add_argument("--prompt", default=None, help="override the per-model default")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-side", type=int, default=2048, help="downscale cap (0=off)")
    args = ap.parse_args()

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    prompt = args.prompt or _default_prompt(args.model)
    manifest = json.loads((Path(args.golden_dir) / "manifest.json").read_text())
    out_dir = Path(args.out_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"# loading {args.model} ...", flush=True)
    t_load = time.time()
    model, processor = load(args.model)
    config = load_config(args.model)
    print(f"# loaded in {time.time()-t_load:.0f}s; prompt={prompt!r}", flush=True)

    rows = []
    for entry in manifest:
        pid = entry["id"]
        img = _maybe_resize(Path(args.golden_dir) / f"{pid}.png", args.max_side)
        formatted = apply_chat_template(processor, config, prompt, num_images=1)
        t0 = time.time()
        try:
            result = generate(
                model, processor, formatted, [img], max_tokens=args.max_tokens, verbose=False
            )
            text = result if isinstance(result, str) else getattr(result, "text", str(result))
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - capture per-page failures
            text, status = f"[INFER_ERROR] {type(exc).__name__}: {exc}", "error"
        dt = round(time.time() - t0, 1)
        (out_dir / f"{pid}.json").write_text(
            json.dumps(
                {
                    "id": pid,
                    "capability": entry["capability"],
                    "model": args.model,
                    "prompt": prompt,
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
        rows.append({"id": pid, "latency_s": dt, "status": status})

    lat = sorted(r["latency_s"] for r in rows if r["status"] == "ok")
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "label": args.label,
                "prompt": prompt,
                "pages": len(rows),
                "ok": sum(r["status"] == "ok" for r in rows),
                "median_latency_s": lat[len(lat) // 2] if lat else None,
            },
            indent=2,
        )
    )
    print(f"# done -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
