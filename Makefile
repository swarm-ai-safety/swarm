.PHONY: install install-dev lint lint-fix typecheck test coverage ci clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,runtime]"
	$(PYTHON) -m pip install pre-commit
	pre-commit install

lint:
	ruff check src/ tests/

lint-fix:
	ruff check --fix src/ tests/

typecheck:
	mypy src/

test:
	pytest tests/ -v

coverage:
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

ci: lint typecheck coverage

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
