.PHONY: install install-dev format lint check test test-all \
	test-single-process run run-compose run-validate hooks help all

# A uv-created .venv has no pip, and a CI runner has pip and no uv. Pick
# whichever is on PATH so one command installs in both.
PIP_INSTALL := $(shell command -v uv >/dev/null 2>&1 \
	&& echo 'uv pip install' || echo 'python -m pip install')

# A developer shell often has another project's venv ahead on PATH, so a bare
# `python3` is not necessarily this project's. Prefer the local one; fall back
# for CI, which has no .venv.
PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python \
	|| echo python3)

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
	$(PYTHON) scripts/check_prose_ratio.py src/hallsim

hooks:
	pre-commit install

# One interpreter cannot hold the whole suite: a full run aborts inside XLA
# compilation around 78% and reports steady-state failures that do not
# reproduce in isolation, with a failure count that varies run to run on
# identical code. Why is not known -- load, TMPDIR, the compile cache, any
# single test file and flatten's operand count have each been ruled out by
# measurement. The same files pass when split, so each chunk gets a fresh
# interpreter. Raise TEST_CHUNK to trade startup cost for headroom; lower it
# if the ceiling is hit again.
TEST_CHUNK ?= 12
TEST_FILES = $(sort $(wildcard tests/unit/test_*.py) \
	$(wildcard tests/integration/test_*.py))

# Each chunk is its own pytest process; the first failing chunk stops the run
# and its status is the target's status.
define run_chunked
	@set -e; \
	echo "$(TEST_FILES)" | tr ' ' '\n' | grep . | \
	xargs -n $(TEST_CHUNK) sh -c \
		'$(PYTHON) -m pytest "$$@" -m "$(1)" || exit 255' sh
endef

test:
	$(call run_chunked,not slow and not network and not demo)

test-all:
	$(call run_chunked,not network)

# The whole suite in one interpreter — what CI used to do. Kept so the
# accumulation ceiling stays reproducible rather than becoming folklore.
test-single-process:
	$(PYTHON) -m pytest tests/ -m "not slow and not network and not demo"

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
	@echo "  make test         - Run tests (chunked, one interpreter per chunk)"
	@echo "  make test-all     - Run every non-network test, chunked"
	@echo "  make run          - Run the legacy simulation"
	@echo "  make run-compose  - Run the composable architecture demo"
	@echo "  make run-validate - Run the semantic validation demo"
	@echo "  make help         - Show this help message"
