SHELL := /bin/bash

.PHONY: fmt lint test etl_monthly etl_annual figures

# Config
PYTHONPATH := src
HS_MAP     := notes/hs_map.csv

fmt:
	@isort src tests scripts
	@black src tests scripts

lint:
	@black --check src tests scripts
	@isort --check-only src tests scripts
	@flake8 src tests scripts

test:
	@PYTHONPATH=$(PYTHONPATH) pytest -q

etl_monthly:
	@PYTHONPATH=$(PYTHONPATH) python -m china_ir.etl --mode monthly --hs-map $(HS_MAP)

etl_annual:
	@PYTHONPATH=$(PYTHONPATH) python -m china_ir.etl --mode annual --hs-map $(HS_MAP)

figures: etl_annual
	@PYTHONPATH=$(PYTHONPATH) ./scripts/make_figures.py
	@ls -lh figures/annual_stack.png figures/hhi_monthly.png
