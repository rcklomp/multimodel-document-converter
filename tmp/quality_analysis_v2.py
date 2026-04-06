#!/usr/bin/env python3
"""
Quality analysis script for 5 converted JSONL files — v2.
Adapted for on-disk format where metadata.chunk_type is ALWAYS null
and chunk classification must be inferred from content heuristics.

KEY FINDING REPORTED: chunk_type is universally null in all 5 files.
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

# Heuristic thresholds
CODE_KEYWORDS = {"def ", "class ", "import ", "from ", "return ", "yield ", "async ", "await "}


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


def get_chunk_type_from_metadata(chunk):
    """Read metadata.chunk_type — expected to be None in these files."""
    meta = chunk.get("metadata", {})
    return meta.get("chunk_type", None)


def infer_chunk_type(chunk):
    """
    Content-based heuristic classification since metadata.chunk_type is null.

    Rules (applied in order):
    1. modality != 'text' -> None (not a text classification)
    2. '>>>' in content -> 'code'
    3. content has def/class + newline, or starts with import/from and has newline -> 'code'
    4. starts with common list markers -> 'list_item'
    5. else -> 'paragraph'
    """
    modality = chunk.get("modality", "")
    if modality != "text":
        return None
    content = chunk.get("content", "") or ""
    stripped = content.lstrip()

    # Code detection
    if ">>>" in content:
        return "code"
    if "\n" in content:
        first_line = stripped.split("\n")[0].strip()
        if any(first_line.startswith(kw) for kw in CODE_KEYWORDS):
            return "code"

    # List item detection
    if stripped.startswith(("■", "•", "–", "—", "·")):
        return "list_item"
    if stripped.startswith(("- ", "* ")):
        return "list_item"
    if len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".):":
        return "list_item"

    return "paragraph"


def get_modality(chunk):
    return chunk.get("modality", None)


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

    # Metadata chunk_type (what the file actually contains)
    meta_ct_counts = Counter(get_chunk_type_from_metadata(c) for c in chunks)

    # Inferred chunk_type (content heuristics)
    inferred_ct_counts = Counter(infer_chunk_type(c) for c in chunks)

    # Modality from root field
    mod_counts = Counter(get_modality(c) for c in chunks)

    result = {
        "total": total,
        "meta_chunk_types": {},
        "inferred_chunk_types": {},
        "modalities": {},
    }

    # Metadata-based (all null in these files)
    for label in ["paragraph", "code", "list_item", None]:
        n = meta_ct_counts.get(label, 0)
        pct = 100.0 * n / total if total else 0
        key = str(label) if label is not None else "null"
        result["meta_chunk_types"][key] = (n, pct)

    # Content-heuristic-based
    for label in ["paragraph", "code", "list_item", None]:
        n = inferred_ct_counts.get(label, 0)
        pct = 100.0 * n / total if total else 0
        key = str(label) if label is not None else "null(non-text)"
        result["inferred_chunk_types"][key] = (n, pct)

    for mod in ["text", "image", "table"]:
        result["modalities"][mod] = mod_counts.get(mod, 0)

    return result


# ---------------------------------------------------------------------------
# CHECK 2 — def/import/from misclassified as paragraph
# Uses INFERRED classification: chunks where content heuristic says 'paragraph'
# but content actually starts with def/import/from (heuristic boundary cases)
# ---------------------------------------------------------------------------
def check2(chunks):
    """
    Find text chunks where content.lstrip() starts with def /import /from
    BUT our heuristic classified them as 'paragraph' (e.g., single-line, no newline).
    These are code lines mislabeled as prose.
    """
    bad = []
    for c in chunks:
        if get_modality(c) != "text":
            continue
        content = get_content(c)
        stripped = content.lstrip()
        # Must start with code keyword
        if not stripped.startswith(("def ", "import ", "from ")):
            continue
        # And our heuristic says it's a paragraph (not code)
        if infer_chunk_type(c) == "paragraph":
            bad.append((get_page(c), content[:120]))
    return {"count": len(bad), "examples": bad[:5]}


# ---------------------------------------------------------------------------
# CHECK 3 — Prose misclassified as code via >>> markers
# ---------------------------------------------------------------------------
def check3(chunks):
    """
    Among code-like chunks (content has '>>>'), find those where the >>> lines
    have NO Python operators (=, (, [, {) — i.e., look like prose or index entries
    rather than Python code.
    A chunk qualifies if ANY of its >>> lines looks like prose.
    """
    bad = []
    for c in chunks:
        if ">>>" not in get_content(c):
            continue
        if get_modality(c) != "text":
            continue
        content = get_content(c)
        lines = content.splitlines()
        prompt_lines = [l for l in lines if l.startswith(">>> ") or l == ">>>"]
        if not prompt_lines:
            # '>>>' appears embedded but not as a REPL prompt
            continue
        # Check if any prompt line has no Python operators
        prose_prompt_lines = []
        for l in prompt_lines:
            after = l[4:] if len(l) > 4 else ""  # strip ">>> "
            if not any(op in after for op in PYTHON_OPERATORS):
                prose_prompt_lines.append(l)
        if prose_prompt_lines:
            bad.append((get_page(c), content[:150]))
    return {"count": len(bad), "examples": bad[:3]}


# ---------------------------------------------------------------------------
# CHECK 4 — "Click here to view code image" noise
# ---------------------------------------------------------------------------
def check4(chunks):
    """Count ALL chunks (any modality) containing the noise string."""
    count = 0
    examples = []
    for c in chunks:
        if "click here to view code image" in get_content(c).lower():
            count += 1
            examples.append((get_page(c), get_content(c)[:120]))
    return {"count": count, "examples": examples[:5]}


# ---------------------------------------------------------------------------
# CHECK 5 — Repetition garbage
# ---------------------------------------------------------------------------
KNOWN_ARTIFACT_TRIGGER = "a lot"

def check5(chunks):
    count = 0
    examples = []
    for c in chunks:
        content = get_content(c)
        # Exclude known source-artifact chunk about 'a lot' * 10
        if KNOWN_ARTIFACT_TRIGGER in content and content.count("a lot") >= 8:
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
            oversize.append((get_page(c), len(content), get_modality(c), content[:120]))
    return {"count": len(oversize), "examples": oversize}


# ---------------------------------------------------------------------------
# CHECK 7 — Code quality: single-line valid Python via ast.parse
# Applies to all text chunks where inferred type = 'code' AND no '\n'
# ---------------------------------------------------------------------------
def check7(chunks):
    count = 0
    for c in chunks:
        if get_modality(c) != "text":
            continue
        if infer_chunk_type(c) != "code":
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
INDEX_PATTERN = re.compile(r",\s*\d{1,4}$")

def check8(chunks):
    """
    Among code-like chunks (content has '>>>'), find those where ALL >>> lines
    match the index pattern: comma + 1-4 digits at end, NO operators.
    """
    bad = []
    for c in chunks:
        if ">>>" not in get_content(c):
            continue
        if get_modality(c) != "text":
            continue
        content = get_content(c)
        lines = content.splitlines()
        prompt_lines = [l for l in lines if l.startswith(">>> ")]
        if not prompt_lines:
            continue
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

    SEP = "=" * 90
    SEP2 = "-" * 90

    def header_row(label_width=32):
        return f"{'Metric':<{label_width}}" + "".join(f"{abbrev[b]:>12}" for b in book_names)

    print()
    print(SEP)
    print("*** CRITICAL PRE-FINDING: metadata.chunk_type field ***")
    print(SEP)
    print("metadata.chunk_type is NULL for 100% of chunks in ALL 5 files.")
    print("These files were produced before the chunk_type computed_field was populated.")
    print("Checks 2/3/7/8 below use CONTENT-BASED HEURISTICS to infer paragraph/code/list_item.")
    print()

    print(SEP)
    print("CHECK 1 — Summary Stats")
    print(SEP)
    print(header_row())
    print(SEP2)

    row = f"{'Total chunks':<32}" + "".join(
        f"{results[b]['check1']['total']:>12}" for b in book_names
    )
    print(row)

    print(f"\n  [metadata.chunk_type — from file]")
    for ct in ["paragraph", "code", "list_item", "null"]:
        row = f"  {'meta:'+ct:<30}" + "".join(
            f"{results[b]['check1']['meta_chunk_types'][ct][0]:>6} ({results[b]['check1']['meta_chunk_types'][ct][1]:4.1f}%)"
            for b in book_names
        )
        print(row)

    print(f"\n  [inferred chunk_type — content heuristics]")
    for ct in ["paragraph", "code", "list_item", "null(non-text)"]:
        row = f"  {'inferred:'+ct:<30}" + "".join(
            f"{results[b]['check1']['inferred_chunk_types'][ct][0]:>6} ({results[b]['check1']['inferred_chunk_types'][ct][1]:4.1f}%)"
            for b in book_names
        )
        print(row)

    print(f"\n  [modality — from root field]")
    for mod in ["text", "image", "table"]:
        row = f"  {'modality:'+mod:<30}" + "".join(
            f"{results[b]['check1']['modalities'].get(mod, 0):>12}" for b in book_names
        )
        print(row)

    print()
    print(SEP)
    print("CHECK 2 — def/import/from misclassified as paragraph")
    print("(Text chunks starting with def/import/from that heuristic classifies as paragraph)")
    print(SEP)
    row = f"{'Count':<32}" + "".join(
        f"{results[b]['check2']['count']:>12}" for b in book_names
    )
    print(row)
    for b in book_names:
        exs = results[b]['check2']['examples']
        if exs:
            print(f"\n  [{abbrev[b]}] Examples:")
            for pg, snippet in exs:
                print(f"    page={pg}: {repr(snippet)}")

    print()
    print(SEP)
    print("CHECK 3 — Prose misclassified as code via >>> markers (no Python operators after >>>)")
    print(SEP)
    row = f"{'Count':<32}" + "".join(
        f"{results[b]['check3']['count']:>12}" for b in book_names
    )
    print(row)
    for b in book_names:
        exs = results[b]['check3']['examples']
        if exs:
            print(f"\n  [{abbrev[b]}] Examples:")
            for pg, snippet in exs:
                print(f"    page={pg}: {repr(snippet)}")

    print()
    print(SEP)
    print("CHECK 4 — 'Click here to view code image' noise (any chunk type)")
    print(SEP)
    row = f"{'Count':<32}" + "".join(
        f"{results[b]['check4']['count']:>12}" for b in book_names
    )
    print(row)
    for b in book_names:
        exs = results[b]['check4']['examples']
        if exs:
            print(f"\n  [{abbrev[b]}] Examples:")
            for pg, snippet in exs:
                print(f"    page={pg}: {repr(snippet)}")

    print()
    print(SEP)
    print("CHECK 5 — Repetition garbage (5+ consecutive identical whitespace-tokens)")
    print(SEP)
    row = f"{'Count':<32}" + "".join(
        f"{results[b]['check5']['count']:>12}" for b in book_names
    )
    print(row)
    for b in book_names:
        exs = results[b]['check5']['examples']
        if exs:
            print(f"\n  [{abbrev[b]}] Examples:")
            for pg, snippet in exs:
                print(f"    page={pg}: {repr(snippet)}")

    print()
    print(SEP)
    print("CHECK 6 — Oversize chunks (len(content) > 2000 chars)")
    print(SEP)
    row = f"{'Count':<32}" + "".join(
        f"{results[b]['check6']['count']:>12}" for b in book_names
    )
    print(row)
    for b in book_names:
        exs = results[b]['check6']['examples']
        if exs:
            print(f"\n  [{abbrev[b]}] Oversize chunks:")
            for pg, length, mod, snippet in exs:
                print(f"    page={pg} len={length} modality={mod}: {repr(snippet)}")

    print()
    print(SEP)
    print("CHECK 7 — Single-line valid Python code chunks (inferred code + ast.parse passes, no newline)")
    print(SEP)
    row = f"{'Count':<32}" + "".join(
        f"{results[b]['check7']['count']:>12}" for b in book_names
    )
    print(row)

    print()
    print(SEP)
    print("CHECK 8 — Index entries misclassified as code (>>> + comma + digits, no operators)")
    print(SEP)
    row = f"{'Count':<32}" + "".join(
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
