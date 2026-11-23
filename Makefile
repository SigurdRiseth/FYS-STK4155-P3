.PHONY: install lint format typecheck test check notebook data clean

install:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

notebook:
	uv run jupyter lab

# Merge, clean, and split the dataset. Refuses to overwrite existing splits;
# run `uv run python scripts/prepare_data.py --force` to regenerate them.
data:
	uv run python scripts/prepare_data.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
