PY=python

.PHONY: init lint test fmt clean

init:
	pre-commit install

lint:
	black --check src tests scripts
	isort --check-only src tests scripts
	flake8 src tests scripts

fmt:
	black src tests scripts
	isort src tests scripts

test:
	PYTHONPATH=src pytest -q

clean:
	rm -rf data_work/* outputs/* figures/* __pycache__ */__pycache__
pull:
	PYTHONPATH=src ./scripts/pull_comtrade.py --type C --freq M --class HS --period 202401,202402 --reporter 156 --partner 0 --flow X --cmd TOTAL

figures:
	PYTHONPATH=src ./scripts/make_figures.py

ci_figures: etl_annual etl_monthly figures
