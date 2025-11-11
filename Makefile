PY=python

.PHONY: init lint test fmt clean

init:
	pre-commit install

lint:
	black --check code
	isort --check-only code
	flake8 code

fmt:
	black code
	isort code

test:
	pytest -q

clean:
	rm -rf data_work/* outputs/* figures/* __pycache__ */__pycache__
