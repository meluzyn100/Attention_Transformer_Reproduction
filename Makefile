.PHONY: help install lint format test ci act clean

help:
	@echo "Targets:"
	@echo "  install  Install project dependencies in editable mode"
	@echo "  lint     Run ruff"
	@echo "  format   Run black"
	@echo "  test     Run pytest"
	@echo "  ci       Run lint and tests"
	@echo "  act      Run the GitHub Actions workflow locally with act"
	@echo "  clean    Remove common cache/build artifacts"


download-data:
	@echo "Downloading WMT14 de-en raw data..."
	python scripts/download_data.py --output-dir data/raw/wmt14_de_en

install:
	python -m pip install -e '.[dev]'

lint:
	python -m ruff check src tests

format:
	python -m black .

test:
	python -m pytest

ci: lint test

act:
	act pull_request

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info