# ASL → English Translator -- developer shortcuts.
#
# `make install` creates .venv and installs the package with the dev extras.
# On a machine where .venv already exists, only the pip step runs.

PYTHON ?= python3
VENV   ?= .venv
PY     := $(VENV)/bin/python

.PHONY: install test lint smoke clean

install:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

smoke:
	$(PY) scripts/smoke_train.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
