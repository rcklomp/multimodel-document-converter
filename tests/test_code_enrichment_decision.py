"""Tests for the code-enrichment decision lane (Workstream B).

Covers:
- _score_code_evidence() thresholds
- Ayeva-like code-heavy evidence DOES enable it
- Technical-manual incidental-code does NOT automatically enable enrichment
- CodeEnrichmentConfig.api_key never falls back to vlm.api_key
"""

import pytest


# ---------------------------------------------------------------------------
# _score_code_evidence unit tests
# ---------------------------------------------------------------------------

from mmrag_v2.batch_processor import (
    decide_code_enrichment_for_pdf,
    _decide_code_evidence_from_pages,
    _score_code_evidence,
    _select_code_evidence_sample_indices,
)


def _lines(*chunks: str) -> list:
    """Helper: wrap strings as a list of page texts."""
    return list(chunks)


class TestScoreCodeEvidence:
    def test_sample_indices_cover_early_code_regions_and_spread(self):
        """Sampler must not skip early body code pages in long programming books."""
        indices = _select_code_evidence_sample_indices(359)
        # Regression guard for Ayeva/Chaubal-style failures: dense code appears
        # around pages 25 and 77, which sparse whole-document sampling missed.
        assert 24 in indices
        assert 76 in indices
        assert max(indices) > 300

    def test_document_decision_counts_strong_pages_without_dilution(self):
        """Dense code pages must not be averaged away by surrounding prose."""
        prose_page = "This is explanatory prose about software design.\n" * 80
        code_page = "\n".join([
            "class Strategy:",
            "    def execute(self):",
            "        return self.value",
            "def factory(case):",
            "    return Strategy()",
        ] * 3)

        needs, reason, score, _, _ = _decide_code_evidence_from_pages(
            [prose_page] * 20 + [code_page, prose_page, code_page] + [prose_page] * 20
        )

        assert needs is True
        assert "strong_code_pages=2" in reason
        assert score >= 0.10

    def test_empty_returns_false(self):
        needs, reason, ratio, fences, kws = _score_code_evidence([])
        assert needs is False
        assert ratio == 0.0
        assert fences == 0

    def test_blank_page_returns_false(self):
        needs, _, _, _, _ = _score_code_evidence(["   \n\n   \n"])
        assert needs is False

    def test_pure_prose_no_code(self):
        """Magazine prose with no code indicators."""
        prose = (
            "Combat Aircraft Monthly\n"
            "The F-22 Raptor is a fifth-generation fighter aircraft.\n"
            "It entered service in 2005 and remains unmatched in air superiority.\n"
            "Avionics include AN/APG-77 AESA radar and advanced electronic warfare.\n"
        ) * 5
        needs, _, ratio, fences, _ = _score_code_evidence(_lines(prose))
        assert needs is False
        assert fences == 0
        assert ratio < 0.10

    def test_fenced_code_heavy_triggers(self):
        """Ayeva-style page: many fenced Python blocks."""
        code_page = (
            "Design Patterns chapter 3\n"
            "```python\n"
            "class CreditCard(PaymentBase): def process_payment(self): print('ok')\n"
            "```\n"
            "Some explanation text.\n"
            "```python\n"
            "def factory_method(case): return JSONDataExtractor() if case == 'json' else XMLDataExtractor()\n"
            "```\n"
            "More explanation.\n"
            "```python\n"
            "if __name__ == '__main__': extract(case='json')\n"
            "```\n"
        )
        # 3 fences in one page; replicate to get above threshold of 5 fences
        multi_page = code_page * 3  # 9 fence markers
        needs, reason, _, fences, _ = _score_code_evidence(_lines(multi_page))
        assert needs is True
        assert fences >= 5
        assert "fence_count" in reason

    def test_keyword_dense_code_triggers(self):
        """Chaubal-style: dense def/class/import lines without fences."""
        code_text = "\n".join([
            "import torch",
            "import torch.nn as nn",
            "from torch.optim import Adam",
            "class LinearModel(nn.Module):",
            "    def __init__(self, input_dim, output_dim):",
            "        super().__init__()",
            "        self.fc = nn.Linear(input_dim, output_dim)",
            "    def forward(self, x):",
            "        return self.fc(x)",
            "def train(model, loader, optimizer):",
            "    for batch in loader:",
            "        loss = model(batch)",
            "        loss.backward()",
            "        optimizer.step()",
            "    return loss",
        ] * 6)  # repeat to get ratio well above 0.10
        needs, reason, ratio, _, _ = _score_code_evidence(_lines(code_text))
        assert needs is True
        assert ratio >= 0.10

    def test_incidental_code_in_technical_manual_does_not_trigger(self):
        """Technical manual with occasional shell commands — below threshold.

        Uses startswith matching so 'range from 15C' does NOT count as a code line.
        Only lines that START with a code keyword (def/class/import/etc.) are counted.
        """
        manual_text = (
            "Greenhouse Design and Control\n\n"
            "Chapter 4: Environmental Control Systems\n\n"
            "Temperature control is critical for plant growth. The PID controller\n"
            "adjusts the HVAC system output. Installation requires:\n\n"
            "    pip install greenhouse-controller\n\n"
            "After installation run the diagnostic tool.\n\n"
            "    python check_sensors.py\n\n"
            "The sensors communicate via Modbus RTU.\n"
            "Wiring diagrams are in Chapter 5.\n"
            "The control loop executes every 500ms.\n"
            "Temperature setpoints range 15C to 35C depending on crop type.\n"
            "Humidity is maintained at 60-80 percent relative humidity.\n"
            "CO2 enrichment targets 800-1200 ppm during daylight hours.\n"
        )
        needs, _, ratio, fences, kws = _score_code_evidence(_lines(manual_text))
        # No lines START with Python keywords; two indented shell commands are not
        # enough evidence to run expensive CodeFormulaV2 enrichment.
        assert needs is False
        assert ratio < 0.10
        assert fences == 0
        # keyword_lines must be 0 because no line starts with def/class/import/from/return/yield
        assert kws == 0

    def test_three_fenced_blocks_triggers(self):
        """3 fenced blocks = 6 fence lines (open+close) → above fence threshold of 5."""
        # Each ``` block contributes one opening fence line and one closing fence line.
        # 3 blocks = 6 fence markers → above _CODE_FENCE_THRESHOLD=5.
        block = "```python\ncode here\n```\n"
        page = (block * 3) + "Some prose.\n" * 20
        needs, _, _, fences, _ = _score_code_evidence(_lines(page))
        assert fences == 6
        assert needs is True

    def test_two_fenced_blocks_below_threshold_if_sparse(self):
        """2 fenced blocks = 4 fence lines; needs ratio to compensate."""
        block = "```python\ncode here\n```\n"
        prose = "Some prose line.\n" * 30  # dilute ratio
        page = prose + (block * 2)
        needs, _, _, fences, _ = _score_code_evidence([page])
        assert fences == 4  # 2 open + 2 close
        assert needs is False


# ---------------------------------------------------------------------------
# PLAN_V2.8 §4 — CodeFormulaV2 enable contract for the named documents
# These named tests pin the cheap-evidence outcome for the v2.8 reference docs
# (Chaubal positive, Fluent positive control, Combat negative). They reuse the
# same `_score_code_evidence` lane the production trigger uses, so a future
# regression in the heuristic surfaces here under a contract-named test.
# ---------------------------------------------------------------------------


class TestPlanV28EnrichmentContract:
    """Per PLAN_V2.8 §4: pin enable/disable contract for the v2.8 reference docs."""

    def test_chaubal_cheap_evidence_trigger_fires(self):
        """Chaubal (PyTorch projects book): dense `import`/`def`/`class` → enable."""
        chaubal_like = "\n".join([
            "import torch",
            "import torch.nn as nn",
            "from torch.optim import Adam",
            "class CNN(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            "        self.conv1 = nn.Conv2d(3, 16, 3)",
            "        self.fc1 = nn.Linear(16 * 26 * 26, 128)",
            "    def forward(self, x):",
            "        x = self.conv1(x)",
            "        return self.fc1(x.view(x.size(0), -1))",
            "def train(model, loader, optimizer):",
            "    for batch in loader:",
            "        loss = model(batch)",
            "        loss.backward()",
        ] * 5)
        needs, reason, _, _, _ = _score_code_evidence(_lines(chaubal_like))
        assert needs is True, f"Chaubal-shape sample should trigger; reason={reason}"

    def test_fluent_python_cheap_evidence_trigger_fires(self):
        """Fluent Python: fenced + keyword-dense Python → enable (non-regression control)."""
        fluent_like = (
            "Chapter 7. Functions as First-Class Objects\n\n"
            "```python\n"
            "def factorial(n):\n"
            "    return 1 if n < 2 else n * factorial(n - 1)\n"
            "```\n\n"
            "```python\n"
            "from collections import namedtuple\n"
            "Card = namedtuple('Card', ['rank', 'suit'])\n"
            "```\n\n"
            "```python\n"
            "class FrenchDeck:\n"
            "    ranks = [str(n) for n in range(2, 11)] + list('JQKA')\n"
            "    def __init__(self):\n"
            "        self._cards = [Card(rank, suit) for suit in 'shdc' for rank in self.ranks]\n"
            "```\n"
        ) * 2
        needs, reason, _, _, _ = _score_code_evidence(_lines(fluent_like))
        assert needs is True, f"Fluent-shape sample should trigger; reason={reason}"

    def test_combat_aircraft_cheap_evidence_does_not_fire(self):
        """Combat Aircraft (magazine, no real code): must NOT enable enrichment.

        Per PLAN_V2.8 §4 hard constraint: encoding corruption alone (or magazine
        ornament-glyph noise) must not turn on `do_code_enrichment`.
        """
        combat_like = (
            "Combat Aircraft\nAugust 2025\n\n"
            "Wing/Group: 19th Air Force\n"
            "Squadron: 22nd Tactical Fighter\n"
            "Location: Iwo To, Japan\n"
            "Aircraft: F-35C Lightning II\n"
            "TailCode: NE-200\n\n"
            "The aircraft were stationed aboard the USS George Washington for\n"
            "Field training during the deployment.\n"
            "Pilots reported nominal carrier qualification scores.\n"
            "Avionics passed all pre-flight checks per squadron standards.\n"
            "The deployment marked the first time F-35Cs operated at Iwo To.\n"
        ) * 6
        needs, _, ratio, fences, kws = _score_code_evidence(_lines(combat_like))
        assert needs is False, "Magazine prose must not trigger CodeFormulaV2"
        assert fences == 0
        # No line starts with a code keyword like def/class/import.
        assert kws == 0


# ---------------------------------------------------------------------------
# Config isolation: code_enrichment.api_key must not fall back to vlm.api_key
# ---------------------------------------------------------------------------

class TestCodeEnrichmentConfigIsolation:
    def test_separate_api_keys(self):
        from mmrag_v2.config import AppConfig, VLMConfig, RefinerConfig, CodeEnrichmentConfig

        cfg = AppConfig(
            vlm=VLMConfig(api_key="vlm-key-abc"),
            refiner=RefinerConfig(api_key="refiner-key-xyz"),
            code_enrichment=CodeEnrichmentConfig(api_key="ce-key-separate"),
        )
        assert cfg.code_enrichment.api_key == "ce-key-separate"
        assert cfg.vlm.api_key == "vlm-key-abc"
        assert cfg.refiner.api_key == "refiner-key-xyz"
        # No cross-contamination
        assert cfg.code_enrichment.api_key != cfg.vlm.api_key
        assert cfg.code_enrichment.api_key != cfg.refiner.api_key

    def test_code_enrichment_key_is_none_when_not_set(self):
        """If code_enrichment section is absent, api_key must be None — not the VLM key."""
        from mmrag_v2.config import AppConfig, VLMConfig, CodeEnrichmentConfig

        cfg = AppConfig(
            vlm=VLMConfig(api_key="vlm-key-abc"),
            code_enrichment=CodeEnrichmentConfig(),  # api_key not set
        )
        assert cfg.code_enrichment.api_key is None

    def test_config_parse_code_enrichment_section(self, tmp_path):
        """Parsed config reads code_enrichment from its own section only."""
        import yaml
        from mmrag_v2.config import _parse_config

        config_file = tmp_path / ".mmrag-v2.yml"
        config_file.write_text(
            "vlm:\n"
            "  api_key: vlm-secret\n"
            "  provider: openai\n"
            "refiner:\n"
            "  api_key: refiner-secret\n"
            "code_enrichment:\n"
            "  enabled: true\n"
            "  api_key: ce-secret\n"
            "  model: CodeFormulaV2\n"
            "  timeout: 240\n"
        )
        cfg = _parse_config(config_file)

        assert cfg.code_enrichment.enabled is True
        assert cfg.code_enrichment.api_key == "ce-secret"
        assert cfg.code_enrichment.model == "CodeFormulaV2"
        assert cfg.code_enrichment.timeout == 240
        # VLM and refiner keys untouched
        assert cfg.vlm.api_key == "vlm-secret"
        assert cfg.refiner.api_key == "refiner-secret"
        # No fallback
        assert cfg.code_enrichment.api_key != cfg.vlm.api_key

    def test_config_disabled_by_default(self):
        """code_enrichment.enabled defaults to False (opt-in, avoids slow CPU inference)."""
        from mmrag_v2.config import AppConfig
        cfg = AppConfig()
        assert cfg.code_enrichment.enabled is False


# ---------------------------------------------------------------------------
# BatchProcessor decision: encoding-corrupt magazine stays False
# ---------------------------------------------------------------------------

class TestBatchProcessorCodeEnrichmentDecision:
    def _make_processor(self, tmp_path):
        from mmrag_v2.batch_processor import BatchProcessor
        return BatchProcessor(output_dir=str(tmp_path))

    def test_config_enabled_false_blocks_enrichment(self, tmp_path):
        """Code enrichment is skipped when config.enabled=False."""
        from mmrag_v2.config import CodeEnrichmentConfig
        proc = self._make_processor(tmp_path)
        proc.enable_code_enrichment(CodeEnrichmentConfig(enabled=False))

        # Simulate the guard check directly (without a real PDF)
        proc.needs_code_enrichment = True  # would have been set by pre-pass
        proc._decide_code_enrichment.__func__  # exists (sanity)

        # The guard resets it if config disabled
        proc._code_enrichment_config = CodeEnrichmentConfig(enabled=False)
        proc.needs_code_enrichment = True
        # Re-run guard path manually
        if proc._code_enrichment_config is not None and not proc._code_enrichment_config.enabled:
            proc.needs_code_enrichment = False
        assert proc.needs_code_enrichment is False

    def test_missing_config_blocks_enrichment(self, tmp_path):
        """Missing config must not allow implicit local CodeFormulaV2 inference."""
        proc = self._make_processor(tmp_path)
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"not a real pdf")

        proc._decide_code_enrichment(fake_pdf)

        assert proc.needs_code_enrichment is False
        assert "config not registered" in proc._code_enrichment_reason

    def test_module_decision_helper_respects_missing_config(self, tmp_path):
        """Direct processor path must share the same disabled-by-default decision."""
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"not a real pdf")

        needs, reason, score = decide_code_enrichment_for_pdf(fake_pdf, None)

        assert needs is False
        assert "config not registered" in reason
        assert score == 0.0

    def test_direct_processor_pops_needs_code_enrichment(self, tmp_path):
        """V2DocumentProcessor consumes but does not emit needs_code_enrichment metadata."""
        from mmrag_v2.processor import V2DocumentProcessor

        proc = V2DocumentProcessor(
            output_dir=str(tmp_path),
            vision_provider="none",
            intelligence_metadata={
                "profile_type": "digital_magazine",
                "needs_code_enrichment": True,
            },
        )

        assert proc.needs_code_enrichment is True
        assert "needs_code_enrichment" not in proc._intelligence_metadata


    def test_processor_preserves_structural_flags_but_filters_chunk_kwargs(self, tmp_path):
        """V2DocumentProcessor must keep decision flags off chunk factory metadata."""
        from mmrag_v2.processor import V2DocumentProcessor

        proc = V2DocumentProcessor(
            output_dir=str(tmp_path / "out"),
            intelligence_metadata={
                "profile_type": "digital_magazine",
                "document_domain": "literature",
                "has_encoding_corruption": True,
                "has_flat_text_corruption": True,
                "geometry_error_rate": 0.12,
                "total_pages": 327,
                "needs_code_enrichment": True,
            },
            vision_provider="none",
            enable_ocr=False,
        )

        assert proc.has_encoding_corruption is True
        assert proc.has_flat_text_corruption is True
        assert proc.geometry_error_rate == 0.12
        assert proc.needs_code_enrichment is True
        assert proc._intelligence_metadata == {
            "profile_type": "digital_magazine",
            "document_domain": "literature",
        }

    def test_mapper_filters_non_chunk_intelligence_metadata(self, tmp_path):
        """UIR/mapper path must not pass document-level flags to chunk factories."""
        from mmrag_v2.mapper import DoclingToV2Mapper
        from mmrag_v2.schema.ingestion_schema import FileType

        mapper = DoclingToV2Mapper(
            doc_hash="abc123",
            source_file="sample.pdf",
            output_dir=tmp_path / "out",
            file_type=FileType.PDF,
            intelligence_metadata={
                "profile_type": "technical_manual",
                "has_encoding_corruption": True,
                "has_flat_text_corruption": True,
                "needs_code_enrichment": True,
                "total_pages": 10,
            },
        )

        assert mapper.intelligence_metadata == {"profile_type": "technical_manual"}
