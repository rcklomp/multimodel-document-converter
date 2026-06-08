"""Text-as-image drop + blank-image guards (PLAN_GATE_QUALITY_V1 F2/F7).

F2: a text region misclassified as an image is described "...no distinct non-text
visuals" - dropped post-enrichment (the description only exists then), behind a
page-coverage guard. F7: a deterministically blank asset is skipped pre-VLM and
flagged by an advisory metric.

Offline/deterministic: synthetic JSONL + synthetic chunk dicts, no VLM.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _img(page, desc, fp="assets/x.png"):
    return {
        "object_type": "chunk",
        "modality": "image",
        "asset_ref": {"file_path": fp},
        "metadata": {"page_number": page, "visual_description": desc},
    }


def _txt(page):
    return {
        "object_type": "chunk",
        "modality": "text",
        "content": "Body text.",
        "metadata": {"page_number": page},
    }


def _write(tmp_path, recs):
    p = tmp_path / "ingestion.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return p


def test_f2_drops_non_visual_on_covered_page(tmp_path):
    enrich = _load("enrich_image_chunks_v29", "scripts/enrich_image_chunks_v29.py")
    p = _write(tmp_path, [
        _txt(2),
        _img(2, "Dense typographic layout; no distinct non-text visuals."),
    ])
    dropped = enrich._drop_non_visual_images(p)
    assert dropped == 1
    survivors = [json.loads(l) for l in p.read_text().splitlines()]
    assert all(r.get("modality") != "image" for r in survivors)


def test_f2_keeps_non_visual_only_chunk_on_page(tmp_path):
    enrich = _load("enrich_image_chunks_v29", "scripts/enrich_image_chunks_v29.py")
    p = _write(tmp_path, [
        _txt(1),
        _img(7, "no distinct non-text visuals"),  # page 7: image only
    ])
    assert enrich._drop_non_visual_images(p) == 0  # page guard keeps it


def test_f2_keeps_real_descriptions(tmp_path):
    enrich = _load("enrich_image_chunks_v29", "scripts/enrich_image_chunks_v29.py")
    p = _write(tmp_path, [
        _txt(1),
        _img(1, "A photovoltaic cell schematic with labeled resistors and a diode."),
    ])
    assert enrich._drop_non_visual_images(p) == 0


def test_non_visual_metric_counts_sentinel():
    qa = _load("qa_semantic_fidelity", "scripts/qa_semantic_fidelity.py")
    images = [
        _img(1, "A real figure of an aircraft."),
        _img(2, "Dense typographic layout; no distinct non-text visuals."),
        _img(3, "no distinct non-text visuals"),
    ]
    assert qa.count_non_visual_images(images) == 2
