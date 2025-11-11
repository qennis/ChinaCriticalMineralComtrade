PY=python

.PHONY: init lint test fmt clean

init:
	pre-commit install

lint:
	black --check src tests
	isort --check-only src tests
	flake8 src tests

fmt:
	black code
	isort code

test:
	PYTHONPATH=src pytest -q

clean:
	rm -rf data_work/* outputs/* figures/* __pycache__ */__pycache__
