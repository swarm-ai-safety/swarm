.PHONY: install install-dev lint lint-fix typecheck test coverage ci clean docs docs-serve test-changes test-parallel test-durations test-no-network claude-code-demo run-alignment-scenarios run-alignment-analyze run-alignment-all

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,runtime]"
	$(PYTHON) -m pip install pre-commit mkdocs-material pymdown-extensions
	pre-commit install

lint:
	ruff check swarm/ tests/

lint-fix:
	ruff check --fix swarm/ tests/

typecheck:
	mypy swarm/

test:
	pytest tests/ -v

test-changes:
	python scripts/test_changes.py -- pytest -m "not slow" -q

test-parallel:
	pytest -n auto -m "not slow" -q

test-durations:
	pytest --durations=20 -q

test-no-network:
	pytest --disable-socket -m "not slow" -q

coverage:
	pytest tests/ --cov=swarm --cov-report=term-missing --cov-report=html

ci: lint typecheck coverage

docs:
	mkdocs build --strict

docs-serve:
	mkdocs serve

claude-code-demo:
	bash scripts/run_claude_code_demo.sh $(SCENARIO)

run-alignment-scenarios:
	swarm run scenarios/alignment_waltz_targeted_feedback.yaml
	swarm run scenarios/macpo_weak_to_strong.yaml

run-alignment-analyze:
	swarm analyze logs/alignment_waltz_targeted_feedback_metrics.csv --metrics toxicity_rate,quality_gap,spread
	swarm analyze logs/macpo_weak_to_strong_metrics.csv --metrics toxicity_rate,quality_gap,spread

run-alignment-all: run-alignment-scenarios run-alignment-analyze

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage site/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
