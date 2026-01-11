# Changelog: MM-Converter-V2

All notable changes to this project will be documented in this file. This project adheres to the functional requirements defined in **SRS v2.4**.

## [v18.1] - 2026-01-11

### Added

* **JoyCaption VLM Integration:** Full implementation of `llama-joycaption-beta-one` via OpenAI-compatible API (LM Studio) for high-fidelity visual descriptions.
* **Asynchronous Batch Processing:** Decoupled the VLM inference from the text extraction pipeline. The processor now fills a queue, increasing throughput for text-heavy documents by 3x.
* **VLM Contextual Awareness:** Implementation of a 3-page sliding window for text-context injection into image prompts, significantly improving entity recognition (e.g., identifying the "USS Abraham Lincoln" via nearby captions).

### Fixed

* **REQ-COORD-02 (Spatial Anchor Fix):** Resolved the critical bug where `page_width` and `page_height` were returned as `null`. All assets now include correct physical page dimensions (612x792 for standard PDF points).
* **Metadata Sanitization:** Added post-processing filters to remove LLM internal monologues (e.g., `<think>` tags) from the final JSONL output.
* **Path Normalization:** Improved handling of relative asset paths, ensuring the `ingestion.jsonl` remains portable across different environments.

### Changed

* **Strategy Orchestrator Tuning:** Refined the `High-Fidelity` strategy with a balanced `Sensitivity: 0.5` setting, optimized for complex magazine layouts (validated against *Combat Aircraft*).