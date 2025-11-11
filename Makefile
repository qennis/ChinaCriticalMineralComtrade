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
	black code
	isort code

test:
	PYTHONPATH=src pytest -q

clean:
	rm -rf data_work/* outputs/* figures/* __pycache__ */__pycache__
