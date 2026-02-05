.PHONY: install install-dev lint typecheck test coverage clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,runtime]"
	pip install pre-commit
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

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
