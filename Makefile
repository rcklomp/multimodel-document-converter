.PHONY: test lint check smoke acceptance clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Quality ──────────────────────────────────────────────────────────────

test: ## Run full pytest suite
	python -m pytest tests/ -v

test-fast: ## Run tests without slow markers
	python -m pytest tests/ -v -x --timeout=30

lint: ## Run ruff + black checks
	ruff check src tests
	black --check src tests

fmt: ## Auto-format with black + ruff fix
	black src tests
	ruff check --fix src tests

typecheck: ## Run mypy on source
	mypy src/mmrag_v2

check: ## Run lint + typecheck + tests
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test

# ── Acceptance ───────────────────────────────────────────────────────────

smoke: ## Run multi-profile smoke test (10 pages, no VLM)
	PAGES=10 BATCH_SIZE=3 bash scripts/smoke_multiprofile.sh

acceptance: ## Run technical-manual deep acceptance (4 docs × 20 pages)
	bash scripts/acceptance_technical_manual.sh

# ── Utility ──────────────────────────────────────────────────────────────

version: ## Show current version
	python -m mmrag_v2.cli version

preflight: ## Run mmrag-v2 system check
	python -m mmrag_v2.cli check

clean: ## Remove output directories and caches
	rm -rf output/__pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
