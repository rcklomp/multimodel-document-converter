"""PLAN_QUALITY_REMEDIATION_V1 WP-D7: an IMAGE chunk embeds its FULL description.

`metadata.visual_description` is a mirror capped at 400 characters (contractual and pinned); the full
description lives in `content`. The ingest used the mirror as the embedding text, so a 702-character
flowchart description was embedded as its 400-character prefix ending "Recalculate o...". No
retrieval-lift claim is made (unmeasured); this fixes a false docstring claim ("loses nothing
retrievable") at the one consumer that embeds the mirror. Other consumers of the mirror (the Qdrant
payload, search display) are unchanged and listed in DECISIONS.
"""

import importlib.util
import sys
from pathlib import Path

from mmrag_v2.schema.ingestion_schema import _fit_visual_description

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _ingest():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "ingest_to_qdrant_d7", SCRIPTS / "ingest_to_qdrant.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ingest_to_qdrant_d7"] = mod
    spec.loader.exec_module(mod)
    return mod


LONG = "Flowchart depicting the PSO algorithm steps: " + " -> ".join(
    f"step {i} of the procedure" for i in range(40)
)


def test_a_long_description_is_embedded_in_full_not_as_its_400_char_mirror():
    mod = _ingest()
    mirror = _fit_visual_description(LONG)
    assert len(LONG) > 400 and mirror.endswith("...") and len(mirror) <= 400
    chunk = {"content": LONG}
    assert mod.image_embedding_text(chunk, {"visual_description": mirror}, LONG) == LONG


def test_a_short_description_is_unchanged():
    mod = _ingest()
    short = "A circuit diagram with a diode."
    assert (
        mod.image_embedding_text({"content": short}, {"visual_description": short}, short) == short
    )


def test_a_mirror_that_is_not_a_cut_of_content_keeps_the_old_preference():
    # enrichment lane: visual_description is the authoritative description, content a placeholder
    mod = _ingest()
    assert (
        mod.image_embedding_text(
            {"content": "[Figure on page 3]"},
            {"visual_description": "Real enriched description..."},
            "[Figure on page 3]",
        )
        == "Real enriched description..."
    )


def test_missing_mirror_falls_back_to_content_and_chunk_level_mirror_is_honoured():
    mod = _ingest()
    assert (
        mod.image_embedding_text({"content": "only content"}, {}, "only content") == "only content"
    )
    assert (
        mod.image_embedding_text({"visual_description": "top level mirror"}, {}, "")
        == "top level mirror"
    )


def test_a_natural_ellipsis_within_the_cap_is_not_mistaken_for_a_cut():
    mod = _ingest()
    text = "The curve keeps rising and then... "
    text = text.strip()
    assert mod.image_embedding_text({"content": text}, {"visual_description": text}, text) == text
