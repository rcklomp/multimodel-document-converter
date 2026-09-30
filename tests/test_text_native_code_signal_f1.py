"""PLAN_F1 WP-1: the text_native_code page signal + over-trigger contract.

Calibrates _score_text_native_code against the frozen Workstream B negatives, the
Phase 0(e) positive controls, AND a NEW prose-with-indents negative (poetry /
nested list - R2), which Phase 0(e) proved the indentation channel alone cannot
distinguish from code. The contract: every negative must NOT fire; the code
positives must fire. If a negative fires, fix the signal, never the fixture.

The existing Workstream B negative tests (test_code_enrichment_decision.py) remain
green and unmodified - this signal is additive and shares the keyword-START set.
"""

from __future__ import annotations

import pytest

from mmrag_v2.batch_processor import _score_text_native_code

# --- Frozen Workstream B negatives (verbatim shape from the WS-B contract) ----
_INCIDENTAL_SHELL = (
    "Greenhouse Design and Control\n\nChapter 4: Environmental Control Systems\n\n"
    "Temperature control is critical for plant growth. The PID controller\n"
    "adjusts the HVAC system output. Installation requires:\n\n"
    "    pip install greenhouse-controller\n\n"
    "After installation run the diagnostic tool.\n\n"
    "    python check_sensors.py\n\n"
    "The sensors communicate via Modbus RTU.\nWiring diagrams are in Chapter 5.\n"
    "The control loop executes every 500ms.\n"
)
_SPARSE_FENCED = ("Some prose line.\n" * 30) + ("```python\ncode here\n```\n" * 2)
_MAGAZINE_PROSE = (
    "Combat Aircraft Monthly\nThe F-22 Raptor is a fifth-generation fighter aircraft.\n"
    "It entered service in 2005 and remains unmatched in air superiority.\n"
    "Avionics include AN/APG-77 AESA radar and advanced electronic warfare.\n"
) * 5
_MAGAZINE_KEYVALUE = (
    "Combat Aircraft\nAugust 2025\n\nWing/Group: 19th Air Force\n"
    "Squadron: 22nd Tactical Fighter\nLocation: Iwo To, Japan\n"
    "Aircraft: F-35C Lightning II\nTailCode: NE-200\n\n"
    "The aircraft were stationed aboard the USS George Washington for\n"
    "Field training during the deployment.\n"
) * 6
# NEW R2 negatives: indented prose the c1 channel over-triggers on (Phase 0e).
_NESTED_LIST = (
    "Shopping plan:\n  - produce\n    - apples\n    - pears\n  - dairy\n"
    "    - milk\n    - cheese\nNotes follow below.\n"
) * 6
_POETRY = (
    "    Whose woods these are I think I know.\n"
    "    His house is in the village though;\n"
    "        He will not see me stopping here\n"
    "    To watch his woods fill up with snow.\n"
) * 6

_NEGATIVES = {
    "incidental_shell": _INCIDENTAL_SHELL,
    "sparse_fenced": _SPARSE_FENCED,
    "magazine_prose": _MAGAZINE_PROSE,
    "magazine_keyvalue": _MAGAZINE_KEYVALUE,
    "nested_list_R2": _NESTED_LIST,
    "poetry_R2": _POETRY,
}

# --- Positives: born-digital code pages ---------------------------------------
_CHAUBAL_CODE = "\n".join(
    [
        "import torch",
        "import torch.nn as nn",
        "class CNN(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        "        self.conv1 = nn.Conv2d(3, 16, 3)",
        "    def forward(self, x):",
        "        x = self.conv1(x)",
        "        return x",
    ]
    * 5
)
_FLUENT_FENCED = (
    "```python\ndef factorial(n):\n    return 1 if n < 2 else n * factorial(n - 1)\n```\n"
    * 4
)
# Recalibration positive: body-heavy code as REAL PDF get_text() yields it -
# indentation flattened to FLUSH (x-positioned in the PDF, lost by get_text), and
# few def/class headers. The old keyword+leading-whitespace signal scored this ~0
# (the WP-4 Jungjun false-negative); the code-line-ratio signal must fire on it.
_FLUSH_BODY_CODE = "\n".join(
    [
        "import torch",
        "model = build_model(config)",
        "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)",
        "for epoch in range(epochs):",
        "for batch in loader:",
        "loss = model(batch)",
        "loss.backward()",
        "optimizer.step()",
        "optimizer.zero_grad()",
        "scheduler.step(val_acc)",
        "lr = scheduler.get_last_lr()",
        "print(f'epoch {epoch}: loss={loss.item()}')",
    ]
    * 3
)


@pytest.mark.parametrize("name", list(_NEGATIVES))
def test_negatives_do_not_fire(name):
    fired, ch = _score_text_native_code(_NEGATIVES[name])
    assert fired is False, f"{name} over-triggered text_native_code: {ch}"


def test_keyword_dense_code_fires():
    fired, ch = _score_text_native_code(_CHAUBAL_CODE)
    assert fired is True, ch


def test_fenced_code_fires():
    fired, ch = _score_text_native_code(_FLUENT_FENCED)
    assert fired is True, ch


def test_flush_body_heavy_code_fires():
    # The WP-4 recalibration target: real get_text() yields FLUSH code (no leading
    # whitespace) that is body-heavy (few def/class headers). Must fire.
    fired, ch = _score_text_native_code(_FLUSH_BODY_CODE)
    assert fired is True, ch
    assert ch["code_ratio"] >= 0.40


def test_short_text_layer_is_not_text_native():
    # Raster pages / near-empty pages: below the real-text-layer precondition.
    fired, ch = _score_text_native_code("model = build_model(config)")
    assert fired is False
    assert ch["chars"] < 100


def test_code_syntax_required_not_indentation_alone():
    # Poetry/nested lists HAVE indentation but ~no code syntax -> must not fire.
    # The discriminator is code-line ratio, not indentation.
    fired, ch = _score_text_native_code(_POETRY)
    assert ch["code_ratio"] < 0.40
    assert fired is False
    fired2, ch2 = _score_text_native_code(_NESTED_LIST)
    assert ch2["code_ratio"] < 0.40
    assert fired2 is False
