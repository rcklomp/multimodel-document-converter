# V3 EXECUTION MANDATE

## 1. THE ARCHITECTURAL CONTRACT
* **Engines:** (e.g., `engines/docling.py`) MUST parse source files and return a single `UniversalDocument`. Engines are strictly forbidden from chunking.
* **Processor:** `processor.py` MUST be 100% engine-agnostic (zero Docling imports). It consumes the `UniversalDocument` and passes it to the chunker.
* **Chunker:** The chunking logic MUST accept a `UniversalDocument` as input and emit `UIRChunk` objects.

## 2. THE ONLY DEFINITION OF DONE
An architectural phase is ONLY complete when:
1. `pytest tests/test_v3_security.py` returns Exit Code 0.
2. `pytest tests/` returns Exit Code 0 with zero skipped tests added.
3. The Identity Gate script runs and outputs a < 5% delta.

## 3. SCOPE CONSTRAINTS
* The port of `batch_processor.py` is strictly limited to engine-agnostic orchestration (batching, routing, JSONL writing). 
* All v2.16 heuristic reconciliation paths are permanently deferred per `V3_DEFERRED_TESTS.md`.
* Chunk count and content parity are explicitly excluded from the smoke test requirements, as the V3 chunker fundamentally alters chunking shape.

## 4. STATUS ENFORCEMENT
There is no "in-progress." 
There is no "rebooked." 
There is no "implemented but not validated." 
You either pass the strict programmatic gates, or you have failed the prompt.