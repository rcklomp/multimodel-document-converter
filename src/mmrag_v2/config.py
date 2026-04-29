"""
Configuration file loader for MM-RAG Converter.

Reads defaults from ~/.mmrag-v2.yml (or MMRAG_CONFIG env var).
CLI flags always override config file values.

Config file format:
    vlm:
      provider: openai
      model: qwen-vl-max
      base_url: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
      api_key: sk-...
      timeout: 120

    refiner:
      enabled: true
      provider: openai
      model: qwen-plus
      base_url: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
      api_key: sk-...

    defaults:
      batch_size: 10
      output_dir: ./output
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATHS = [
    Path.home() / ".mmrag-v2.yml",
    Path.home() / ".mmrag-v2.yaml",
    Path(".mmrag-v2.yml"),
]


@dataclass
class VLMConfig:
    provider: str = "none"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 120


@dataclass
class RefinerConfig:
    enabled: bool = False
    provider: str = "openai"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class DefaultsConfig:
    batch_size: int = 10
    output_dir: str = "./output"


@dataclass
class AppConfig:
    vlm: VLMConfig = field(default_factory=VLMConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    _loaded_from: Optional[str] = None

    @property
    def loaded_from(self) -> Optional[str]:
        return self._loaded_from


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load config from YAML file. Returns defaults if no file found."""
    # Check env var first
    if config_path is None:
        config_path = os.environ.get("MMRAG_CONFIG")

    # Try default paths
    paths_to_try = [Path(config_path)] if config_path else DEFAULT_CONFIG_PATHS

    for path in paths_to_try:
        if path.exists():
            return _parse_config(path)

    return AppConfig()


def _parse_config(path: Path) -> AppConfig:
    """Parse a YAML config file."""
    try:
        import yaml
    except ImportError:
        # PyYAML not installed — try simple key:value parsing
        return _parse_simple(path)

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to parse config {path}: {e}")
        return AppConfig()

    return _build_config(data, str(path))


def _parse_simple(path: Path) -> AppConfig:
    """Fallback parser for simple YAML without PyYAML dependency."""
    data: dict = {}
    current_section = None

    try:
        with open(path, "r") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if not line.startswith(" ") and stripped.endswith(":"):
                    current_section = stripped[:-1].strip()
                    data[current_section] = {}
                elif current_section and ":" in stripped:
                    key, _, value = stripped.partition(":")
                    value = value.strip().strip("'\"")
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    data[current_section][key.strip()] = value
    except Exception as e:
        logger.warning(f"Failed to parse config {path}: {e}")
        return AppConfig()

    return _build_config(data, str(path))


def _build_config(data: dict, source: str) -> AppConfig:
    """Build AppConfig from parsed dict."""
    vlm_data = data.get("vlm", {})
    refiner_data = data.get("refiner", {})
    defaults_data = data.get("defaults", {})

    vlm = VLMConfig(
        provider=vlm_data.get("provider", "none"),
        model=vlm_data.get("model"),
        base_url=vlm_data.get("base_url"),
        api_key=vlm_data.get("api_key"),
        timeout=int(vlm_data.get("timeout", 120)),
    )

    refiner = RefinerConfig(
        enabled=bool(refiner_data.get("enabled", False)),
        provider=refiner_data.get("provider", "openai"),
        model=refiner_data.get("model"),
        base_url=refiner_data.get("base_url"),
        api_key=refiner_data.get("api_key"),
    )

    defaults = DefaultsConfig(
        batch_size=int(defaults_data.get("batch_size", 10)),
        output_dir=defaults_data.get("output_dir", "./output"),
    )

    config = AppConfig(vlm=vlm, refiner=refiner, defaults=defaults, _loaded_from=source)
    logger.info(f"[CONFIG] Loaded from {source}: VLM={vlm.provider}, refiner={refiner.enabled}")
    return config
