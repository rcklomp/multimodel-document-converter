from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from mmrag_v2.vision.vision_prompts import VISUAL_ONLY_PROMPT, validate_vlm_response

_ENRICH_PATH = Path(__file__).resolve().parent.parent / "scripts" / "enrich_image_chunks_v29.py"
_MOD_NAME = "enrich_image_chunks_v29"
_spec = importlib.util.spec_from_file_location(_MOD_NAME, _ENRICH_PATH)
enrich = importlib.util.module_from_spec(_spec)
sys.modules[_MOD_NAME] = enrich
_spec.loader.exec_module(enrich)


def _record(
    *,
    content: str = "Venn diagram.",
    bbox: list[int] | None = None,
    retry_attempted: bool = False,
    asset_path: str = "assets/img.png",
) -> dict:
    metadata = {
        "vision_status": "complete",
        "visual_description": content,
        "page_number": 1,
        "spatial": {"bbox": bbox or [0, 0, 500, 500]},
    }
    if retry_attempted:
        metadata["vision_detail_retry_attempted"] = True
    return {
        "chunk_id": "doc_001_image_deadbeef",
        "doc_id": "doc",
        "modality": "image",
        "content": content,
        "visual_description": content,
        "asset_ref": {"file_path": asset_path},
        "metadata": metadata,
    }


def _manager(response: str) -> SimpleNamespace:
    provider = SimpleNamespace(describe_image=Mock(return_value=response))
    return SimpleNamespace(_provider=provider)


def test_detail_retry_resolves_short_complex_response(tmp_path: Path) -> None:
    rec = _record(content="Venn diagram.")
    manager = _manager(
        "Layered circular visual with three overlapping colored regions and "
        "connector shapes arranged around a central comparison area."
    )

    result = enrich._maybe_retry_for_detail(rec, manager, object(), tmp_path)

    assert result.triggered is True
    assert result.resolved is True
    assert result.hard_fallback is False
    assert rec["metadata"]["vision_status"] == "complete"
    assert len(rec["content"].strip()) >= enrich.SHORT_DESC_THRESHOLD_CHARS
    prompt = manager._provider.describe_image.call_args.args[1]
    assert prompt.startswith(VISUAL_ONLY_PROMPT.rstrip())
    assert "Do not transcribe or paraphrase any text from the image." in prompt
    assert manager._provider.describe_image.call_count == 1


def test_detail_retry_does_not_trigger_for_simple_short_asset(tmp_path: Path) -> None:
    rec = _record(content="Small icon.", bbox=[0, 0, 50, 50])
    manager = _manager("Should not be called")

    result = enrich._maybe_retry_for_detail(rec, manager, object(), tmp_path)

    assert result.triggered is False
    assert manager._provider.describe_image.call_count == 0
    assert rec["metadata"]["vision_status"] == "complete"


def test_detail_retry_short_response_becomes_hard_fallback(tmp_path: Path) -> None:
    rec = _record(content="System schematic.")
    manager = _manager("Diagram.")

    result = enrich._maybe_retry_for_detail(rec, manager, object(), tmp_path)

    assert result.triggered is True
    assert result.resolved is False
    assert result.hard_fallback is True
    assert rec["metadata"]["vision_status"] == "hard_fallback"
    assert rec["metadata"]["vision_error"] == enrich.SHORT_RESPONSE_SENTINEL
    assert rec["metadata"]["vision_provider_used"] == "qwen3-vl-plus"
    assert rec["metadata"]["vision_detail_retry_attempted"] is True
    assert manager._provider.describe_image.call_count == 1


def test_detail_retry_rejects_text_reading_response(tmp_path: Path) -> None:
    leaking_response = 'The text says "INTRODUCTION TO RETRIEVAL".'
    assert validate_vlm_response(leaking_response).is_valid is False
    rec = _record(content="Interface panel.")
    manager = _manager(leaking_response)

    result = enrich._maybe_retry_for_detail(rec, manager, object(), tmp_path)

    assert result.triggered is True
    assert result.hard_fallback is True
    assert rec["metadata"]["vision_status"] == "hard_fallback"
    assert rec["metadata"]["vision_error"] == enrich.SHORT_RESPONSE_SENTINEL
    assert "Text transcription detected" in rec["metadata"]["vision_validation_issues"]
    assert manager._provider.describe_image.call_count == 1


def test_detail_retry_cap_is_one_attempt(tmp_path: Path) -> None:
    rec = _record(content="Line chart.", retry_attempted=True)
    manager = _manager("Should not be called")

    result = enrich._maybe_retry_for_detail(rec, manager, object(), tmp_path)

    assert result.triggered is False
    assert manager._provider.describe_image.call_count == 0


def test_existing_complete_retry_path_loads_asset_and_updates_record(tmp_path: Path) -> None:
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    Image.new("RGB", (20, 20), color=(255, 255, 255)).save(asset_dir / "img.png")
    rec = _record(content="Logo graphic.")
    manager = _manager(
        "Small geometric symbol with dark angular shapes on a pale background "
        "and a centered balanced composition."
    )

    rec, result = enrich._retry_existing_complete_short_description(
        rec,
        manager,
        tmp_path,
    )

    assert result.triggered is True
    assert result.resolved is True
    assert rec["metadata"]["vision_status"] == "complete"
    assert rec["metadata"]["vision_detail_retry_attempted"] is True
    assert len(rec["metadata"]["visual_description"]) >= enrich.SHORT_DESC_THRESHOLD_CHARS
