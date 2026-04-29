"""Tests for _has_fenced_flat_code detection (Workstream B).

These tests verify that the module-level helper correctly identifies code
chunks whose body lines have been squished onto a single line inside
backtick fences — the pattern observed in Ayeva_Python_Patterns and
Chaubal_PyTorch_Projects.
"""

import pytest

from mmrag_v2.batch_processor import _has_fenced_flat_code


# ---------------------------------------------------------------------------
# True-positive cases: should return True
# ---------------------------------------------------------------------------

def test_fenced_flat_python_class():
    """Single-line class body inside a fence — Ayeva-style corruption."""
    # Real squished lines from code-heavy books are >>120 chars
    body = (
        "class CreditCard(PaymentBase): def process_payment(self): "
        "msg = 'Processing payment...' print(msg) self._log(msg) return True"
    )
    assert len(body) > 120, "test body must exceed the 120-char threshold"
    txt = f"```python\n{body}\n```"
    assert _has_fenced_flat_code(txt) is True


def test_fenced_flat_with_language_tag():
    """Python language tag on the fence line should not confuse the detector."""
    txt = (
        "Some prose before.\n"
        "```python\n"
        "def train_epoch(model, loader, optimizer): for batch in loader: loss = model(batch) loss.backward() optimizer.step() return loss\n"
        "```\n"
        "Some prose after."
    )
    assert _has_fenced_flat_code(txt) is True


def test_fenced_flat_no_language_tag():
    """Fence without language tag should still be detected."""
    body = (
        "def foo(self): return self.bar if self.bar else "
        "self.compute_default_value_from_registry_cache() if self.registry else None"
    )
    assert len(body) > 120, "test body must exceed the 120-char threshold"
    txt = f"```\n{body}\n```"
    assert _has_fenced_flat_code(txt) is True


def test_fenced_flat_tilde_fence():
    """Tilde-style fences are also handled."""
    txt = (
        "~~~\n"
        "class Model(nn.Module): def __init__(self): super().__init__() self.fc = nn.Linear(128, 10) def forward(self, x): return self.fc(x)\n"
        "~~~"
    )
    assert _has_fenced_flat_code(txt) is True


def test_multiple_fenced_blocks_one_flat():
    """Only one of two fenced blocks is flat; still returns True."""
    good_block = "```python\ndef foo():\n    return 1\n```\n"
    flat_body = (
        "def bar(self): return self.x if self.x else "
        "self.compute_default_value_from_registry_or_fallback() if self.registry else None"
    )
    assert len(flat_body) > 120, "test body must exceed the 120-char threshold"
    flat_block = f"```python\n{flat_body}\n```"
    assert _has_fenced_flat_code(good_block + flat_block) is True


# ---------------------------------------------------------------------------
# True-negative cases: should return False
# ---------------------------------------------------------------------------

def test_normal_fenced_code_not_flagged():
    """Properly indented fenced code should NOT be flagged."""
    txt = (
        "```python\n"
        "def process_payment(self):\n"
        "    msg = 'Processing...'\n"
        "    print(msg)\n"
        "    return True\n"
        "```"
    )
    assert _has_fenced_flat_code(txt) is False


def test_no_fence_plain_flat_code():
    """A flat code line NOT inside a fence is handled by the original check, not this helper."""
    txt = "def process_payment(self): msg = 'Processing' print(msg) return True"
    assert _has_fenced_flat_code(txt) is False


def test_empty_string():
    assert _has_fenced_flat_code("") is False


def test_plain_prose_not_flagged():
    """Long prose line inside a fence should not be flagged."""
    txt = (
        "```\n"
        "This is a very long descriptive sentence about how the algorithm works in theory and practice.\n"
        "```"
    )
    assert _has_fenced_flat_code(txt) is False


def test_short_flat_line_inside_fence_not_flagged():
    """A short flat line inside a fence is below the 120-char threshold."""
    txt = "```python\ndef foo(self): return 1\n```"
    assert _has_fenced_flat_code(txt) is False


def test_single_keyword_long_line_not_flagged():
    """Only one keyword hit — requires at least two to fire."""
    # 120+ chars but only "return" matches
    line = "return " + "x" * 120
    txt = f"```python\n{line}\n```"
    assert _has_fenced_flat_code(txt) is False
