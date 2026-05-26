.PHONY: install install-dev format lint test run run-compose run-validate help all

install:
	pip install --upgrade pip
	pip install -e .

install-dev:
	pip install --upgrade pip
	pip install -e ".[dev]"

format:
	black --line-length 79 src/
	black --line-length 79 tests/

lint:
	flake8 --ignore E501,E402,W504,W503,E226,E203 src/
	flake8 --ignore E501,E402,W504,W503,E226,E203 tests/

test:
	pytest tests/

run:
	simulate basic

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
