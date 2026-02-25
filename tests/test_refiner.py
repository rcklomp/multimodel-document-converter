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


def test_semantic_refiner_blocks_layout_disorder_before_llm(monkeypatch):
    config = RefinerConfig(
        min_refine_threshold=0.1,
        layout_disorder_block_threshold=0.6,
        llm_provider="ollama",
        llm_model="dummy",
    )
    refiner = SemanticRefiner(config)

    called = {"value": False}

    def fake_scan(*_args, **_kwargs):
        return 0.95

    def fake_layout(*_args, **_kwargs):
        return 0.9

    def fake_refine(*_args, **_kwargs):
        called["value"] = True
        return "SHOULD_NOT_BE_USED"

    monkeypatch.setattr(refiner.scanner, "calculate_corruption_score", fake_scan)
    monkeypatch.setattr(refiner.scanner, "calculate_layout_disorder_score", fake_layout)
    monkeypatch.setattr(refiner.refiner, "refine_with_context", fake_refine)

    raw = "INT-KLCD\nINT-KLCDR\n5.3 Section"
    result = refiner.process(raw_text=raw, visual_description=None, semantic_context=None)

    assert called["value"] is False
    assert result.refinement_applied is False
    assert result.refined_text == raw
    assert result.rejection_reason == "Layout disorder detected"


def test_semantic_refiner_high_corruption_technical_dense_uses_conservative_path(monkeypatch):
    config = RefinerConfig(
        min_refine_threshold=0.1,
        high_corruption_threshold=0.7,
        technical_no_summarize_threshold=0.3,
        layout_disorder_block_threshold=0.95,  # ensure this branch does not pre-empt
        llm_provider="ollama",
        llm_model="dummy",
    )
    refiner = SemanticRefiner(config)

    called = {"value": False}

    def fake_scan(*_args, **_kwargs):
        return 0.9

    def fake_layout(*_args, **_kwargs):
        return 0.0

    def fake_refine(*_args, **_kwargs):
        called["value"] = True
        return "SHOULD_NOT_BE_USED"

    monkeypatch.setattr(refiner.scanner, "calculate_corruption_score", fake_scan)
    monkeypatch.setattr(refiner.scanner, "calculate_layout_disorder_score", fake_layout)
    monkeypatch.setattr(refiner.refiner, "refine_with_context", fake_refine)

    raw = "ECU-001-A  ,  SN: ABC123   P/N: ZX-900"
    result = refiner.process(raw_text=raw, visual_description=None, semantic_context=None)

    assert called["value"] is False
    assert result.refinement_applied is True
    assert result.provider == "conservative_guardrail"
    assert result.model == "whitespace_punctuation_only"
    assert "  " not in result.refined_text
    assert " ," not in result.refined_text
    assert "ECU-001-A" in result.refined_text

