import pytest

from mmrag_v2.refiner import (
    ConsistencyValidator,
    ContextualRefiner,
    NoiseScanner,
    RefinerConfig,
    SemanticRefiner,
)


def test_noise_scanner_detects_kerning():
    scanner = NoiseScanner()
    text = "G u l f s t r e a m G650"
    score = scanner.calculate_corruption_score(text)
    assert score > 0.0


def test_noise_scanner_clean_text_low_score():
    scanner = NoiseScanner()
    text = "The Gulfstream G650 has a range of 7,000 nm."
    score = scanner.calculate_corruption_score(text)
    assert score < 0.1


def test_consistency_validator_rejects_over_editing():
    validator = ConsistencyValidator(max_edit_ratio=0.15)
    original = "The Gulfstream G650 has a range of 7,000 nm."
    over_edited = "The luxurious Gulfstream G650 executive jet can travel very far."
    result = validator.validate(original, over_edited)
    assert result.is_valid is False
    assert "Edit ratio" in (result.rejection_reason or "")


def test_consistency_validator_protected_token_lock():
    validator = ConsistencyValidator()
    original = "Install module ECU-001-A before test."
    refined = "Install module ECU-001-B before test."
    result = validator.validate(original, refined)
    assert result.is_valid is False
    assert result.protected_tokens_preserved is False


def test_contextual_refiner_includes_visual_anchor(monkeypatch):
    config = RefinerConfig(llm_provider="ollama", llm_model="dummy")
    refiner = ContextualRefiner(config)
    captured = {}

    def fake_call(user_prompt: str):
        captured["prompt"] = user_prompt
        return "Gulfstream"

    monkeypatch.setattr(refiner, "_call_ollama", fake_call)
    result = refiner.refine_with_context(
        raw_text="Guiltream",
        visual_description="Gulfstream G650 on a runway",
        semantic_context=None,
    )

    assert result == "Gulfstream"
    assert "Gulfstream G650 on a runway" in captured["prompt"]


def test_semantic_refiner_uses_technical_density(monkeypatch):
    config = RefinerConfig(
        min_refine_threshold=0.2,
        technical_density_trigger=0.01,
        technical_density_threshold_delta=0.15,
        llm_provider="ollama",
        llm_model="dummy",
    )
    refiner = SemanticRefiner(config)
    called = {"value": False}

    def fake_scan(*_args, **_kwargs):
        return 0.12

    def fake_refine(*_args, **_kwargs):
        called["value"] = True
        return "ECU-001-A"

    monkeypatch.setattr(refiner.scanner, "calculate_corruption_score", fake_scan)
    monkeypatch.setattr(refiner.refiner, "refine_with_context", fake_refine)

    result = refiner.process(raw_text="ECU-001-A", visual_description=None, semantic_context=None)
    assert called["value"] is True
    assert result.refinement_applied is True

