"""Engine-agnostic code fencing (PLAN_GATE_QUALITY_V1 F4).

VLM-promoted code chunks were not Markdown-fenced while the MinerU lane fenced
via _fence_code, so mixed-route docs had inconsistent code formatting. F4 fences
at the chunker chokepoint (idempotent: MinerU's already-fenced code is
unaffected) and indentation inside the fence is preserved verbatim.
"""

from __future__ import annotations

from mmrag_v2.chunking.uir_chunker import _code_element_to_uirchunk, _fence_code
from mmrag_v2.universal.intermediate import (
    ElementType,
    Modality,
    PageClassification,
    UniversalPage,
    create_element,
)

CODE = "class SysCall(Thread):\n    def __init__(self):\n        self.x = 1"


def test_fence_code_wraps_unfenced():
    out = _fence_code(CODE)
    assert out.startswith("```\n") and out.rstrip().endswith("```")
    # indentation inside preserved verbatim
    assert "    def __init__(self):" in out
    assert "        self.x = 1" in out


def test_fence_code_idempotent_on_already_fenced():
    already = "```\n" + CODE + "\n```"
    assert _fence_code(already) == already


def test_fence_code_noop_on_empty():
    assert _fence_code("") == ""
    assert _fence_code("\n\n") == "\n\n"


def test_code_chunk_is_fenced_end_to_end():
    page = UniversalPage(
        page_number=1,
        elements=[create_element(ElementType.TEXT, CODE, bbox=[80, 100, 900, 400], element_index=0)],
        classification=PageClassification.DIGITAL,
        dimensions=(612, 792),
    )
    el = page.elements[0]
    chunk = _code_element_to_uirchunk(el, page, "qwen3-vl-8b", 0)
    assert chunk.modality is Modality.CODE
    assert chunk.content.lstrip().startswith("```")
    assert "        self.x = 1" in chunk.content  # indentation survives
