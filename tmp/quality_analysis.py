#!/usr/bin/env python3
"""
Quality analysis script for 5 converted JSONL files.
Runs CHECK 1-8 for each file and reports results.
"""

import json
import ast
import re
import sys
from pathlib import Path
from collections import Counter

FILES = {
    "python_cookbook": "/Users/ronald/Projects/MM-Converter-V2.4.1/output/python_cookbook_v1/ingestion.jsonl",
    "fluent_python":   "/Users/ronald/Projects/MM-Converter-V2.4.1/output/fluent_python_v1/ingestion.jsonl",
    "devlin_llm_agents": "/Users/ronald/Projects/MM-Converter-V2.4.1/output/devlin_llm_agents_v1/ingestion.jsonl",
    "bourne_rag":      "/Users/ronald/Projects/MM-Converter-V2.4.1/output/bourne_rag_v1/ingestion.jsonl",
    "arcgis_python":   "/Users/ronald/Projects/MM-Converter-V2.4.1/output/arcgis_python_v1/ingestion.jsonl",
}

PYTHON_OPERATORS = set("=([{")


def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append(obj)
    return chunks


def get_chunk_type(chunk):
    meta = chunk.get("metadata", {})
    return meta.get("chunk_type", None)


def get_modality(chunk):
    meta = chunk.get("metadata", {})
    return meta.get("modality", None)


def get_page(chunk):
    meta = chunk.get("metadata", {})
    return meta.get("page_number", None)


def get_content(chunk):
    return chunk.get("content", "") or ""


# ---------------------------------------------------------------------------
# CHECK 1 — Summary stats
# ---------------------------------------------------------------------------
def check1(chunks):
    total = len(chunks)
    ct_counts = Counter(get_chunk_type(c) for c in chunks)
    mod_counts = Counter(get_modality(c) for c in chunks)

    result = {"total": total, "chunk_types": {}, "modalities": {}}
    for label in ["paragraph", "code", "list_item", None]:
        n = ct_counts.get(label, 0)
        pct = 100.0 * n / total if total else 0
        key = str(label) if label is not None else "null"
        result["chunk_types"][key] = (n, pct)
    for mod in ["text", "image", "table"]:
        result["modalities"][mod] = mod_counts.get(mod, 0)
    return result


# ---------------------------------------------------------------------------
# CHECK 2 — def/import/from misclassified as paragraph
# ---------------------------------------------------------------------------
def check2(chunks):
    bad = []
    for c in chunks:
        if get_chunk_type(c) != "paragraph":
            continue
        content = get_content(c)
        stripped = content.lstrip()
        if stripped.startswith(("def ", "import ", "from ")):
            bad.append((get_page(c), content[:120]))
    return {"count": len(bad), "examples": bad[:5]}


# ---------------------------------------------------------------------------
# CHECK 3 — Prose misclassified as code via >>> markers
# ---------------------------------------------------------------------------
def check3(chunks):
    """
    Among code chunks, find >>> lines that contain NO Python operators (=, (, [, {)
    after the '>>> ' prefix — i.e., look like prose / index entries rather than code.
    A chunk qualifies if ANY of its >>> lines looks like prose.
    """
    bad = []
    for c in chunks:
        if get_chunk_type(c) != "code":
            continue
        content = get_content(c)
        lines = content.splitlines()
        prompt_lines = [l for l in lines if l.startswith(">>> ")]
        if not prompt_lines:
            continue
        # Check if any prompt line has no operators
        prose_prompt_lines = []
        for l in prompt_lines:
            after = l[4:]  # strip ">>> "
            if not any(op in after for op in PYTHON_OPERATORS):
                prose_prompt_lines.append(l)
        if prose_prompt_lines:
            bad.append((get_page(c), content[:150]))
    return {"count": len(bad), "examples": bad[:3]}


# ---------------------------------------------------------------------------
# CHECK 4 — "Click here to view code image" noise
# ---------------------------------------------------------------------------
def check4(chunks):
    count = 0
    for c in chunks:
        if get_chunk_type(c) != "code":
            continue
        if "click here to view code image" in get_content(c).lower():
            count += 1
    return {"count": count}


# ---------------------------------------------------------------------------
# CHECK 5 — Repetition garbage
# ---------------------------------------------------------------------------
KNOWN_ARTIFACT = "a lot"

def check5(chunks):
    count = 0
    examples = []
    for c in chunks:
        content = get_content(c)
        # Exclude known source-artifact chunk
        if KNOWN_ARTIFACT in content and content.count("a lot") > 5:
            continue
        tokens = content.split()
        found = False
        for i in range(len(tokens) - 4):
            window = tokens[i:i+5]
            if len(set(window)) == 1:
                found = True
                break
        if found:
            count += 1
            examples.append((get_page(c), content[:120]))
    return {"count": count, "examples": examples[:5]}


# ---------------------------------------------------------------------------
# CHECK 6 — Oversize chunks
# ---------------------------------------------------------------------------
def check6(chunks):
    oversize = []
    for c in chunks:
        content = get_content(c)
        if len(content) > 2000:
            oversize.append((get_page(c), len(content), content[:120]))
    return {"count": len(oversize), "examples": oversize[:20]}


# ---------------------------------------------------------------------------
# CHECK 7 — Code quality: short legitimate assignments
# ---------------------------------------------------------------------------
def check7(chunks):
    count = 0
    for c in chunks:
        if get_chunk_type(c) != "code":
            continue
        content = get_content(c)
        if "\n" in content:
            continue  # multi-line, skip
        try:
            ast.parse(content)
            count += 1
        except SyntaxError:
            pass
    return {"count": count}


# ---------------------------------------------------------------------------
# CHECK 8 — Index entries misclassified as code
# ---------------------------------------------------------------------------
# Pattern: after ">>> ", text contains comma + 1-4 digits, NO operators
INDEX_PATTERN = re.compile(r",\s*\d{1,4}$")

def check8(chunks):
    bad = []
    for c in chunks:
        if get_chunk_type(c) != "code":
            continue
        content = get_content(c)
        lines = content.splitlines()
        prompt_lines = [l for l in lines if l.startswith(">>> ")]
        if not prompt_lines:
            continue
        # ALL prompt lines must match index pattern
        all_index = True
        for l in prompt_lines:
            after = l[4:]
            if any(op in after for op in PYTHON_OPERATORS):
                all_index = False
                break
            if not INDEX_PATTERN.search(after):
                all_index = False
                break
        if all_index:
            bad.append((get_page(c), content[:150]))
    return {"count": len(bad), "examples": bad[:3]}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_all():
    results = {}
    for name, path in FILES.items():
        print(f"  Loading {name}...", file=sys.stderr)
        try:
            chunks = load_chunks(path)
        except Exception as e:
            print(f"  ERROR loading {name}: {e}", file=sys.stderr)
            results[name] = None
            continue
        results[name] = {
            "check1": check1(chunks),
            "check2": check2(chunks),
            "check3": check3(chunks),
            "check4": check4(chunks),
            "check5": check5(chunks),
            "check6": check6(chunks),
            "check7": check7(chunks),
            "check8": check8(chunks),
        }
        print(f"  Done {name}: {len(chunks)} chunks", file=sys.stderr)
    return results


def print_results(results):
    book_names = list(FILES.keys())
    abbrev = {
        "python_cookbook": "PyCook",
        "fluent_python":   "FluentPy",
        "devlin_llm_agents": "DevlinLLM",
        "bourne_rag":      "BourneRAG",
        "arcgis_python":   "ArcGIS",
    }

    print("\n" + "="*80)
    print("CHECK 1 — Summary Stats")
    print("="*80)
    header = f"{'Metric':<28}" + "".join(f"{abbrev[b]:>12}" for b in book_names)
    print(header)
    print("-"*80)
    # Total
    row = f"{'Total chunks':<28}" + "".join(
        f"{results[b]['check1']['total']:>12}" for b in book_names
    )
    print(row)
    # Chunk types
    for ct in ["paragraph", "code", "list_item", "null"]:
        row = f"  {ct:<26}" + "".join(
            f"{results[b]['check1']['chunk_types'][ct][0]:>6} ({results[b]['check1']['chunk_types'][ct][1]:4.1f}%)"
            for b in book_names
        )
        print(row)
    # Modalities
    for mod in ["text", "image", "table"]:
        row = f"  modality:{mod:<18}" + "".join(
            f"{results[b]['check1']['modalities'].get(mod, 0):>12}" for b in book_names
        )
        print(row)

    print("\n" + "="*80)
    print("CHECK 2 — def/import/from misclassified as paragraph")
    print("="*80)
    row = f"{'Count':<28}" + "".join(
        f"{results[b]['check2']['count']:>12}" for b in book_names
    )
    print(row)
    for b in book_names:
        exs = results[b]['check2']['examples']
        if exs:
            print(f"\n  [{abbrev[b]}] Examples:")
            for pg, snippet in exs:
                print(f"    page={pg}: {repr(snippet)}")

    print("\n" + "="*80)
    print("CHECK 3 — Prose misclassified as code via >>> markers (no operators)")
    print("="*80)
    row = f"{'Count':<28}" + "".join(
        f"{results[b]['check3']['count']:>12}" for b in book_names
    )
    print(row)
    for b in book_names:
        exs = results[b]['check3']['examples']
        if exs:
            print(f"\n  [{abbrev[b]}] Examples:")
            for pg, snippet in exs:
                print(f"    page={pg}: {repr(snippet)}")

    print("\n" + "="*80)
    print("CHECK 4 — 'Click here to view code image' noise in code chunks")
    print("="*80)
    row = f"{'Count':<28}" + "".join(
        f"{results[b]['check4']['count']:>12}" for b in book_names
    )
    print(row)

    print("\n" + "="*80)
    print("CHECK 5 — Repetition garbage (5+ consecutive identical tokens)")
    print("="*80)
    row = f"{'Count':<28}" + "".join(
        f"{results[b]['check5']['count']:>12}" for b in book_names
    )
    print(row)
    for b in book_names:
        exs = results[b]['check5']['examples']
        if exs:
            print(f"\n  [{abbrev[b]}] Examples:")
            for pg, snippet in exs:
                print(f"    page={pg}: {repr(snippet)}")

    print("\n" + "="*80)
    print("CHECK 6 — Oversize chunks (len > 2000 chars)")
    print("="*80)
    row = f"{'Count':<28}" + "".join(
        f"{results[b]['check6']['count']:>12}" for b in book_names
    )
    print(row)
    for b in book_names:
        exs = results[b]['check6']['examples']
        if exs:
            print(f"\n  [{abbrev[b]}] Oversize chunks:")
            for pg, length, snippet in exs:
                print(f"    page={pg} len={length}: {repr(snippet)}")

    print("\n" + "="*80)
    print("CHECK 7 — Single-line valid Python code chunks (ast.parse passes)")
    print("="*80)
    row = f"{'Count':<28}" + "".join(
        f"{results[b]['check7']['count']:>12}" for b in book_names
    )
    print(row)

    print("\n" + "="*80)
    print("CHECK 8 — Index entries misclassified as code (>>> + comma + digits)")
    print("="*80)
    row = f"{'Count':<28}" + "".join(
        f"{results[b]['check8']['count']:>12}" for b in book_names
    )
    print(row)
    for b in book_names:
        exs = results[b]['check8']['examples']
        if exs:
            print(f"\n  [{abbrev[b]}] Examples:")
            for pg, snippet in exs:
                print(f"    page={pg}: {repr(snippet)}")

    print()


if __name__ == "__main__":
    results = run_all()
    print_results(results)
