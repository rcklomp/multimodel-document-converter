"""Re-process already-ingested code books with Fix (b), replace production points.

For each target book: compute CURRENT production R3 (from the dense collection's
chunks), re-convert with Fix (b), compute NEW R3. Replace production ONLY if R3 does
not regress (user gate: "verify R3 lifts before replacing"). On replace: overwrite the
book's phase5_reextract dir (so the sparse rebuild picks up the new text), delete the
doc's old dense points by doc_id, and ingest the new chunks. Sparse rebuild + retrieval
validation are run separately AFTER this completes. Throwaway driver (underscore name).

Run (background):
  MINERU_ENDPOINT=... VLM_NATIVE_ENDPOINT=... python scripts/_reprocess_code_books.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, "scripts")
from _code_quality import code_quality  # noqa: E402

PY = sys.executable
QDRANT = "http://localhost:6333"
DENSE = "mmrag_v3__qwen3_local"
REEXTRACT = Path("output/phase5_reextract")
WORK = Path("output/reprocess")

# (label, source_pdf, doc_id, reextract_dir)
BOOKS = [
    ("Ayeva Python Design Patterns",
     "data/technical_manual/Ayeva K. Mastering Python Design Patterns...essential Python patterns...3ed 2024.pdf",
     "289fd158f828", "Ayeva_K_Mastering_Python_Design_Patterns_essenti"),
    ("AIOS LLM Agent OS",
     "data/academic_journal/AIOS LLM Agent Operating System.pdf",
     "07a1232cccf4", "AIOS_academic"),
    ("Cronin Building/Training GenAI",
     "data/technical_manual/Cronin I. Building and Training Generative AI Models. A Practical Guide...2026.pdf",
     "0054f66093d6", "Cronin_I_Building_and_Training_Generative_AI_Mode"),
    ("Sekar The MCP Standard",
     "data/technical_manual/Sekar S. The MCP Standard. A Developer's Guide..Building Universal AI Tools 2026.pdf",
     "47bcf7e2f91b", "Sekar_S_The_MCP_Standard_A_Developer_s_Guide_Bui"),
    ("Kimothi Simple Guide to RAG",
     "data/technical_manual/A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf",
     "12c8aaab4fa1", "A_Simple_Guide_to_Retrieval_Augmented_Generation"),
    ("Drupal Commerce E-commerce",
     "data/technical_manual/Building E-commerce Sites with Drupal Commerce.pdf",
     "9b5d80a7d85f", "Building_E_commerce_Sites_with_Drupal_Commerce"),
]


def _http(method: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{QDRANT}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def _adapt(d):
    m = d.get("metadata", {}) or {}
    return {"content": d.get("content") or "", "modality": m.get("modality") or d.get("modality"),
            "metadata": m}


def production_r3(doc_id: str):
    """R3 over the doc's current dense chunks (content + modality from payload)."""
    chunks, offset = [], None
    while True:
        body = {"limit": 1000, "with_payload": ["content", "modality", "chunk_type"],
                "with_vector": False,
                "filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}}
        if offset:
            body["offset"] = offset
        r = _http("POST", f"/collections/{DENSE}/points/scroll", body)["result"]
        for p in r["points"]:
            pl = p.get("payload") or {}
            chunks.append({"content": pl.get("content") or "",
                           "modality": pl.get("modality"),
                           "metadata": {"chunk_type": pl.get("chunk_type")}})
        offset = r.get("next_page_offset")
        if not offset:
            break
    cq = code_quality(chunks)
    return cq.indentation_fidelity, cq.n_judgeable, cq.n_judgeable_fail, len(chunks)


def file_r3(jsonl: Path):
    rows = [json.loads(l) for l in open(jsonl)]
    cq = code_quality([_adapt(d) for d in rows])
    hdr = rows[0]
    return cq.indentation_fidelity, cq.n_judgeable, cq.n_judgeable_fail, hdr


def convert(pdf: str, outdir: Path) -> bool:
    if outdir.exists():
        shutil.rmtree(outdir)
    cmd = [PY.replace("python", "mmrag-v2") if False else "/Users/Shared/miniforge3/envs/mmrag-v2/bin/mmrag-v2",
           "process", pdf, "--batch-size", "10", "--output-dir", str(outdir), "--vision-provider", "none"]
    print(f"    converting -> {outdir}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = (outdir / "ingestion.jsonl").exists()
    if not ok:
        print(f"    CONVERT FAILED rc={r.returncode}; tail:\n{r.stderr[-800:]}", flush=True)
    return ok


def delete_doc_points(doc_id: str) -> None:
    _http("POST", f"/collections/{DENSE}/points/delete",
          {"filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}})


def ingest_dense(jsonl: Path) -> bool:
    cmd = ["/Users/Shared/miniforge3/envs/mmrag-v2/bin/python", "scripts/ingest_to_qdrant.py",
           str(jsonl), "--provider", "omlx", "--collection", DENSE, "--qdrant-url", QDRANT]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    INGEST FAILED rc={r.returncode}; tail:\n{r.stderr[-800:]}", flush=True)
    return r.returncode == 0


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    summary = []
    for label, pdf, doc_id, rdir in BOOKS:
        print(f"\n=== {label} (doc_id={doc_id}) ===", flush=True)
        if not Path(pdf).exists():
            print("    SOURCE MISSING; skip", flush=True)
            summary.append((label, "SKIP_NO_SOURCE", None, None))
            continue
        cur_r3, cur_j, cur_f, cur_n = production_r3(doc_id)
        print(f"    current prod R3={cur_r3:.3f} (judgeable={cur_j} fail={cur_f}, {cur_n} chunks)", flush=True)
        outdir = WORK / rdir
        if not convert(pdf, outdir):
            summary.append((label, "CONVERT_FAIL", cur_r3, None))
            continue
        new_r3, new_j, new_f, hdr = file_r3(outdir / "ingestion.jsonl")
        risk = hdr.get("extraction_quality_risk_pages")
        rep = hdr.get("extraction_code_repaired_pages")
        new_doc_id = hdr.get("doc_id")
        print(f"    new R3={new_r3:.3f} (judgeable={new_j} fail={new_f}) repaired={rep}/{risk} doc_id={new_doc_id}", flush=True)
        if new_doc_id != doc_id:
            print(f"    DOC_ID MISMATCH ({new_doc_id} != {doc_id}); skip to avoid orphaning", flush=True)
            summary.append((label, "DOCID_MISMATCH", cur_r3, new_r3))
            continue
        if new_r3 < cur_r3 - 1e-9:
            print(f"    R3 REGRESSION ({new_r3:.3f} < {cur_r3:.3f}); KEEP production, skip replace", flush=True)
            summary.append((label, "SKIP_REGRESSION", cur_r3, new_r3))
            continue
        # Replace: overwrite reextract dir (for sparse), delete old dense, ingest new.
        dest = REEXTRACT / rdir
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(outdir / "ingestion.jsonl", dest / "ingestion.jsonl")
        if (outdir / "assets").exists():
            if (dest / "assets").exists():
                shutil.rmtree(dest / "assets")
            shutil.copytree(outdir / "assets", dest / "assets")
        delete_doc_points(doc_id)
        if not ingest_dense(dest / "ingestion.jsonl"):
            summary.append((label, "INGEST_FAIL", cur_r3, new_r3))
            continue
        print(f"    REPLACED production (R3 {cur_r3:.3f} -> {new_r3:.3f})", flush=True)
        summary.append((label, f"REPLACED rep={rep}/{risk}", cur_r3, new_r3))

    print("\n===== SUMMARY =====", flush=True)
    for label, status, c, n in summary:
        cs = f"{c:.3f}" if isinstance(c, float) else "-"
        ns = f"{n:.3f}" if isinstance(n, float) else "-"
        print(f"  {label:32s} {status:24s} R3 {cs} -> {ns}", flush=True)
    replaced = sum(1 for _, s, _, _ in summary if s.startswith("REPLACED"))
    print(f"\nREPROCESS_DONE replaced={replaced}/{len(BOOKS)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
