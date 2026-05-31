#!/usr/bin/env python3
"""VLM throughput + fidelity profiler for any OpenAI-compatible vision endpoint.

Generalises the ad-hoc probe used on the GX10 (2026-05-30) into a reusable tool.
Renders one PDF page to an image, sends it as a streaming chat/completions
request, and measures the two numbers that decide whether local vision-native
extraction is viable:

  * **prefill / TTFT** — time to the first generated token (vision-encode + prompt
    prefill). This is the per-page FLOOR that weight-quantisation does NOT reduce.
  * **decode tok/s** — generation rate, which IS bandwidth/precision-bound.

It also prints the transcription text and (optionally) checks that expected
content survived — the fidelity guard that caught vLLM-FP8 silently emitting
blank-page garbage (a real 1.73x "speedup" on nothing). Speed without fidelity
is worthless.

Usage
-----
    python scripts/vlm_profile_probe.py \
        --endpoint http://10.0.10.246:8000/v1 \
        --model Qwen3-VL-8B-Instruct-8bit \
        --pdf data/business_form/0013_140302111325_001.pdf --page 0 \
        --api-key "$MLX_API_KEY" \
        --expect "Level Automotive,Castrol,1.949,60"

Reference baselines (Qwen3-VL-8B, ~17k-ctx page), to contextualise the result:
    GX10 / GB10  BF16   : prefill ~57 tok/s   decode 11.3 tok/s   ~132 s/page
    M1 Max (24c) 8-bit  : prefill 162 tok/s   decode 15.1 tok/s   (omlx.ai bench)
    M5 Max  (extrapol.) : prefill ~250-450    decode ~21-26 (8b) / ~30-40 (4b)
The decision number is **seconds/page** at your typical output length: a
corpus of N VLM-routed pages costs ~N * (prefill + out_tokens/decode) seconds.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request


def render_page_png(pdf_path: str, page_index: int, dpi: int) -> bytes:
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    if page_index >= doc.page_count:
        sys.exit(f"page {page_index} out of range (doc has {doc.page_count} pages)")
    pix = doc[page_index].get_pixmap(dpi=dpi)
    return pix.tobytes("png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--endpoint", required=True, help="OpenAI-compatible base URL, e.g. http://host:8000/v1"
    )
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--pdf", required=True, help="source PDF")
    ap.add_argument("--page", type=int, default=0, help="0-indexed page to render")
    ap.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="render DPI (do NOT lower for quality — see FINDINGS_LOG)",
    )
    ap.add_argument(
        "--api-key", default="EMPTY", help="bearer token (vLLM/omlx accept any if unauth)"
    )
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument(
        "--prompt",
        default="Transcribe ALL text on this page verbatim, preserve layout, "
        "and describe any logos/figures in detail.",
    )
    ap.add_argument(
        "--expect",
        default="",
        help="comma-separated substrings that SHOULD appear (fidelity check)",
    )
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args(argv)

    png = render_page_png(args.pdf, args.page, args.dpi)
    img = "data:image/png;base64," + base64.b64encode(png).decode()
    body = json.dumps(
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.prompt},
                        {"type": "image_url", "image_url": {"url": img}},
                    ],
                }
            ],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        args.endpoint.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"},
    )

    t0 = time.time()
    ttft = None
    chunks: list[str] = []
    usage = None
    try:
        for raw in urllib.request.urlopen(req, timeout=args.timeout):
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
    except Exception as e:  # noqa: BLE001 — a probe; surface any transport/HTTP error plainly
        print(f"PROBE ERROR: {type(e).__name__}: {e}")
        return 2

    text = "".join(chunks)
    total = time.time() - t0
    ttft = ttft or total
    decode_t = max(total - ttft, 1e-6)
    pt = (usage or {}).get("prompt_tokens")
    ct = (usage or {}).get("completion_tokens") or len(chunks)

    print(f"\n=== {args.model} @ {args.endpoint} (page {args.page} of {args.pdf}) ===")
    print(f"prompt_tokens (incl. vision): {pt}")
    print(f"completion_tokens:            {ct}")
    print(
        f"prefill / TTFT:               {ttft:.1f}s" + (f"  (~{pt/ttft:.0f} tok/s)" if pt else "")
    )
    print(f"decode:                       {decode_t:.1f}s  ->  {ct/decode_t:.1f} tok/s")
    print(
        f"TOTAL wall:                   {total:.1f}s/page   (prefill {100*ttft/total:.0f}% / decode {100*decode_t/total:.0f}%)"
    )

    if args.expect:
        wants = [s.strip() for s in args.expect.split(",") if s.strip()]
        hit = [w for w in wants if w.lower() in text.lower()]
        miss = [w for w in wants if w.lower() not in text.lower()]
        verdict = "FIDELITY OK" if not miss else "FIDELITY FAIL (content dropped?)"
        print(
            f"fidelity:                     {len(hit)}/{len(wants)} expected substrings present -> {verdict}"
        )
        if miss:
            print(f"  missing: {miss}")
    print(f"\n--- transcription ({len(text)} chars) ---\n{text[:1200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
