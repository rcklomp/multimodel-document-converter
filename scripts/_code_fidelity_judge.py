"""LLM-judge code-fidelity sampler (cross-book, repo-free, language-agnostic).

For each ingested code book, deterministically samples N code chunks and asks a
strong LLM to judge EXTRACTION fidelity (is the text garbled/truncated/de-indented
by the OCR/VLM pipeline?) — NOT whether the code is a complete program (book
snippets are fragments). Works on every book including the C/C++ ones the repo-diff
oracle and ast.parse cannot handle.

Verdict per chunk: CLEAN | MINOR | CORRUPTED. Aggregated per book.
Judge = GX10 vLLM chat endpoint (registry). Deterministic sampling (evenly spaced
by sorted chunk_id), no RNG. Standalone instrument; reads shipping-CLI JSONL.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import _code_quality as cq  # noqa: E402
from synthetic_soak import _call_vllm  # noqa: E402
from mmrag_v2.endpoints import endpoint  # noqa: E402

_RUBRIC = (
    "You are auditing a CODE SNIPPET extracted from a programming book by an "
    "OCR/VLM pipeline. Judge ONLY extraction FIDELITY of the text, NOT whether the "
    "code is a complete or runnable program — book snippets are deliberately partial.\n"
    "Verdicts:\n"
    "- CLEAN: identifiers, operators, strings and punctuation are intact and "
    "plausible; indentation is structurally consistent.\n"
    "- MINOR: trivial artifacts only (a stray callout digit, a lost blank line, "
    "slight indent wobble) that don't change meaning.\n"
    "- CORRUPTED: garbled/merged/split identifiers (e.g. raise_for_status -> "
    "extract_for_status, httpx.Client -> httpx(Client_), llm_complete -> "
    "llmcomplete), missing/extra characters, OCR errors, or flattened indentation "
    "that destroys nesting.\n"
    'Respond with ONLY compact JSON: {"verdict":"CLEAN|MINOR|CORRUPTED","reason":"<=12 words"}'
)


def _sample(jsonl: Path, n: int) -> list[dict]:
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    code = [
        r for r in rows
        if cq.is_code_population({"modality": (r.get("metadata") or {}).get("modality") or r.get("modality"),
                                  "metadata": r.get("metadata") or {}})
        and len((r.get("content") or "").strip()) >= 40
        and not cq.is_repl(r.get("content") or "")
    ]
    code.sort(key=lambda r: str(r.get("chunk_id") or ""))
    if len(code) <= n:
        return code
    step = len(code) / n
    return [code[int(i * step)] for i in range(n)]


def _judge(content: str, url: str, model: str, key) -> tuple[str, str]:
    msg = [{"role": "user", "content": _RUBRIC + "\n\nSNIPPET:\n" + content[:2400]}]
    out = _call_vllm(url, model, msg, api_key=key, temperature=0.0, max_tokens=120)
    if not out:
        return "ERROR", "no response"
    m = re.search(r'"verdict"\s*:\s*"(CLEAN|MINOR|CORRUPTED)"', out, re.I)
    v = m.group(1).upper() if m else ("CORRUPTED" if "CORRUPT" in out.upper() else "ERROR")
    rm = re.search(r'"reason"\s*:\s*"([^"]*)"', out)
    return v, (rm.group(1) if rm else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="output/backlog/*/ingestion.jsonl")
    ap.add_argument("--extra", nargs="*", default=["output/devlin_full_fixb/ingestion.jsonl"])
    ap.add_argument("--sample", type=int, default=20)
    args = ap.parse_args()

    e = endpoint("chat")
    url = getattr(e, "chat_url", None) or getattr(e, "base_url")
    model, key = e.model, getattr(e, "api_key", None)

    files = sorted(glob.glob(args.glob)) + [f for f in args.extra if os.path.exists(f)]
    print(f"judge: {model} @ {url} | sample={args.sample}/book\n")
    print(f"{'book':40s} {'n':>3} {'CLEAN':>6} {'MINOR':>6} {'CORRUPT':>7} {'fidelity':>8}")
    overall = {"CLEAN": 0, "MINOR": 0, "CORRUPTED": 0, "ERROR": 0}
    corrupt_examples = []
    for f in files:
        sample = _sample(Path(f), args.sample)
        c = {"CLEAN": 0, "MINOR": 0, "CORRUPTED": 0, "ERROR": 0}
        for r in sample:
            v, reason = _judge(r.get("content") or "", url, model, key)
            c[v] = c.get(v, 0) + 1
            overall[v] = overall.get(v, 0) + 1
            if v == "CORRUPTED" and len(corrupt_examples) < 30:
                corrupt_examples.append((os.path.basename(os.path.dirname(f))[:24],
                                         reason, " ".join((r.get("content") or "").split())[:80]))
        n = sum(c.values())
        good = c["CLEAN"] + c["MINOR"]
        fid = good / n if n else 0.0
        name = os.path.basename(os.path.dirname(f))[:40]
        print(f"{name:40s} {n:3d} {c['CLEAN']:6d} {c['MINOR']:6d} {c['CORRUPTED']:7d} {fid:8.2f}")
    N = sum(overall.values())
    good = overall["CLEAN"] + overall["MINOR"]
    print(f"\nOVERALL: {N} chunks judged | CLEAN={overall['CLEAN']} MINOR={overall['MINOR']} "
          f"CORRUPTED={overall['CORRUPTED']} ERROR={overall['ERROR']} | fidelity={good/N if N else 0:.3f}")
    if corrupt_examples:
        print("\n--- CORRUPTED examples ---")
        for bk, reason, snip in corrupt_examples:
            print(f"  [{bk}] {reason} :: {snip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
