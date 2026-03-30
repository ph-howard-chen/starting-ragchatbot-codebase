.PHONY: format check

# Format all Python source files in-place
format:
	uv run black backend/ main.py
	uv run isort backend/ main.py

# Check formatting without modifying files (for CI)
check:
	uv run black --check backend/ main.py
	uv run isort --check-only backend/ main.py
