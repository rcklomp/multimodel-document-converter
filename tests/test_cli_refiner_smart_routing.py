"""
v2.9 Phase 2 — refiner smart-routing regression tests.

Pins the contract that the CLI's refiner-enable decision is gated on
``has_encoding_corruption`` (when the config default is the only signal),
so clean-prose docs no longer hammer the refiner per-chunk.

Background: v2.8 broad reconversion's first attempt left HARRY (clean
prose, zero encoding corruption) calling qwen-plus per chunk because
the eager config-default at ``cli.py:686`` set ``enable_refiner=True``
before the diagnostic engine ran. Refinements were rejected ("Edit
ratio 53.16% exceeds budget") but each call still cost a round trip.
The v2.8 workaround was ``--no-refiner`` everywhere; v2.9 fixes the
underlying gate so the workaround can come out of
``scripts/convert_books.sh``.

The decision is centralized in ``cli._decide_enable_refiner``; these
tests exercise that helper directly so the contract is easy to
reproduce without spinning up the whole typer app.
"""

from __future__ import annotations

from mmrag_v2.cli import _decide_enable_refiner


def test_refiner_off_by_default_when_clean_prose() -> None:
    """v2.8 bug surface: clean prose + ``cfg.refiner.enabled=true`` +
    no explicit CLI flag. v2.8 left enable_refiner=True (eager bug).
    v2.9 must leave it False because there's no encoding corruption.
    """
    result = _decide_enable_refiner(
        cli_flag=False,
        config_default_enabled=True,
        explicit_disable=False,
        explicit_enable=False,
        has_encoding_corruption=False,
    )
    assert result is False


def test_refiner_on_when_encoding_corruption_detected() -> None:
    """Combat-class doc surface: encoding corruption + config default
    enabled. The refiner MUST auto-enable so the heal-over path runs.
    """
    result = _decide_enable_refiner(
        cli_flag=False,
        config_default_enabled=True,
        explicit_disable=False,
        explicit_enable=False,
        has_encoding_corruption=True,
    )
    assert result is True


def test_refiner_explicit_no_refiner_always_wins() -> None:
    """``--no-refiner`` is the user's escape hatch; nothing the
    diagnostic finds may override it (this is what
    ``scripts/convert_books.sh`` relied on in v2.8).
    """
    result = _decide_enable_refiner(
        cli_flag=False,
        config_default_enabled=True,
        explicit_disable=True,
        explicit_enable=False,
        has_encoding_corruption=True,  # Even with corruption.
    )
    assert result is False


def test_refiner_explicit_enable_refiner_bypasses_diagnostic() -> None:
    """``--enable-refiner`` forces refiner on regardless of what the
    diagnostic engine says. The user's explicit choice is never
    overridden by absence of corruption.
    """
    result = _decide_enable_refiner(
        cli_flag=True,
        config_default_enabled=False,
        explicit_disable=False,
        explicit_enable=True,
        has_encoding_corruption=False,
    )
    assert result is True


def test_refiner_off_when_neither_flag_nor_config_default() -> None:
    """Belt-and-suspenders: with no explicit flag and config default
    OFF, the refiner stays off even if corruption is detected. This
    pins the rule that the auto-override is gated on the config
    default — corruption alone does not opt-in a user who never asked.
    """
    result = _decide_enable_refiner(
        cli_flag=False,
        config_default_enabled=False,
        explicit_disable=False,
        explicit_enable=False,
        has_encoding_corruption=True,
    )
    assert result is False
