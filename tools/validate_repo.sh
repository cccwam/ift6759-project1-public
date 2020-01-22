#!/bin/bash
# Summary:
#   flake8 is used to enforce the pep8 standard
#   pytest is used to run unit tests
flake8 --exclude=venv* --statistics
pytest -v --cov=tests/
