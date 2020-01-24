:: Example usage:
::   tools\validate_repo.bat
:: Summary:
::   flake8 is used to enforce the pep8 standard
::   pytest is used to run unit tests
flake8 --statistics --config .flake8
pytest -v --cov=tests/
