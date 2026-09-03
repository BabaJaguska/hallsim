.PHONY: install install-dev format lint check test test-all run run-compose run-validate help all

# A uv-created .venv has no pip, and a CI runner has pip and no uv. Pick
# whichever is on PATH so one command installs in both.
PIP_INSTALL := $(shell command -v uv >/dev/null 2>&1 \
	&& echo 'uv pip install' || echo 'python -m pip install')

install:
	$(PIP_INSTALL) -e .

install-dev:
	$(PIP_INSTALL) -e ".[dev]"

format:
	black --line-length 79 src/
	black --line-length 79 tests/
	black --line-length 79 demos/

# black skips gitignored files when walking a directory; flake8 does not.
# Without the exclusion, lint gates on the scratch probes (demos/_*.py, in
# .gitignore) that format refuses to touch, and make check cannot be made to
# pass. Keep this pattern in step with .gitignore.
LINT_EXCLUDE = demos/_*.py

lint:
	flake8 --ignore E501,E402,W504,W503,E226,E203 src/
	flake8 --ignore E501,E402,W504,W503,E226,E203 tests/
	flake8 --ignore E501,E402,W504,W503,E226,E203 \
		--extend-exclude '$(LINT_EXCLUDE)' demos/

check:
	black --check --line-length 79 src/ tests/ demos/
	$(MAKE) lint
	python scripts/check_prose_ratio.py src/hallsim

hooks:
	pre-commit install

test:
	python -m pytest tests/ -m "not slow and not network and not demo"

test-all:
	python -m pytest tests/ -m "not network"

run:
	simulate multi-hallmark run

run-compose:
	simulate compose

run-validate:
	simulate validate-demo

all: install format lint test run

help:
	@echo "Available commands:"
	@echo "  make install      - Install runtime deps (editable) from pyproject.toml"
	@echo "  make install-dev  - Install runtime + dev deps (editable) from pyproject.toml"
	@echo "  make format       - Format the code using black"
	@echo "  make lint         - Lint the code using flake8"
	@echo "  make test         - Run tests using pytest"
	@echo "  make run          - Run the legacy simulation"
	@echo "  make run-compose  - Run the composable architecture demo"
	@echo "  make run-validate - Run the semantic validation demo"
	@echo "  make help         - Show this help message"
