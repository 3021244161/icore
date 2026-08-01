# icore — common development commands
#
# Usage:
#   make install      # install core deps in editable mode
#   make install-dev  # install core + dev/test deps
#   make install-all  # install everything (db, llm, mcp, ui, dev)
#   make run          # start the API server (production bootstrap)
#   make dev          # start with auto-reload
#   make test         # run the test suite
#   make lint         # ruff lint
#   make format       # ruff format (fix)
#   make typecheck    # mypy
#   make docker-build # build container image
#   make docker-run   # run container

PYTHON ?= python
PORT   ?= 8000
HOST   ?= 0.0.0.0

.PHONY: install install-dev install-all run dev test lint format typecheck clean docker-build docker-run

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

install-all:
	$(PYTHON) -m pip install -e ".[all,dev]"

run:
	$(PYTHON) -m icore --host $(HOST) --port $(PORT)

dev:
	$(PYTHON) -m icore --host $(HOST) --port $(PORT) --reload

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check icore tests

format:
	$(PYTHON) -m ruff check --fix icore tests
	$(PYTHON) -m ruff format icore tests

typecheck:
	$(PYTHON) -m mypy icore

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

docker-build:
	docker build -t icore:latest .

docker-run:
	docker run --rm -p $(PORT):8000 \
		--env-file .env \
		-v $(PWD)/config:/app/config \
		icore:latest
