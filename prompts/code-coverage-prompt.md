To verify the coverage of the test cases run following command 
```
uv run pytest tests/ -v --tb=short --cov=src --cov-report=xml --cov-report=html --cov-report=term
```

It will display the percentage as of the current state of MCP. 
Create the plan how to boost the coverage plan 


After you improve the coverage, you have to run following command to ensure that it passes MyPy, Ruff and security checks

```
uv run ruff format --check .
uv run mypy src/ --ignore-missing-imports --show-error-codes --no-error-summary
uv run ruff check --output-format=github . 
uv run pip-audit

```