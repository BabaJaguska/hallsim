.PHONY: install format lint test run run-compose run-validate update help all

install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:
	pip install --upgrade pip
	pip install -r requirements_dev.txt
	pip install -e .

format:
	black --line-length 79 src/
	black --line-length 79 tests/

lint:
	flake8 --ignore E501,E402,W504,W503,E203 src/
	flake8 --ignore E501,E402,W504,W503,E203 tests/

test:
	pytest tests/

run:
	simulate basic

run-compose:
	simulate compose

run-validate:
	simulate validate-demo

update:
	pip freeze > requirements_freeze.txt
	@echo "Updated requirements_freeze.txt with current dependencies."
	python -m piptools compile -o requirements.txt pyproject.toml
	python -m piptools compile pyproject.toml \
		--output-file requirements_dev.txt\
		--constraint=requirements.txt \
		--extra dev
	@echo "Updated requirements.txt from pyproject.toml."


all: install format lint test run

help:
	@echo "Available commands:"
	@echo "  make format  - Format the code using black"
	@echo "  make lint    - Lint the code using flake8"
	@echo "  make test    - Run tests using pytest"
	@echo "  make run          - Run the legacy simulation"
	@echo "  make run-compose  - Run the composable architecture demo"
	@echo "  make run-validate - Run the semantic validation demo"
	@echo "  make help         - Show this help message"