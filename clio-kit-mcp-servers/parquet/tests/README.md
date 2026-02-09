# Parquet MCP Test Suite

## Overview

Complete test suite with **192 tests** across **17 test files**.

## Test Files (in priority order)

### 1. ⭐ `test_json_response_validation.py` (18 tests) - START HERE
**Purpose**: Validates that all MCP tools return proper JSON with correct structure and filtered data.

- **Suite 1**: JSON Structure Validation (6 tests)
  - Valid JSON from all tools
  - Success/error response structure
  - Filter metadata in responses

- **Suite 2**: JSON Data Validation with Filters (12 tests)
  - Validates every row matches filter criteria
  - Tests: equality, inequality, comparisons, AND, OR, IN, complex nested, boolean

### 2. `test_filtering.py` (15 tests)
**Purpose**: Tests filter parsing and PyArrow filter application.

- **Suite 1**: Simple Comparison Operators (6 tests)
  - `=`, `!=`, `<`, `>`, `<=`, `>=`

- **Suite 2**: Logical Operators (5 tests)
  - AND, OR, NOT, nested expressions, multiple conditions

- **Suite 3**: IN Clause and NULL Checks (4 tests)
  - IN with integers/floats
  - IS NULL, IS NOT NULL

### 3. `test_filtered_slice.py` (13 tests)
**Purpose**: Tests read_slice with various filter scenarios.

- **Suite 6**: Filtered Slice Operations (8 tests)
  - Empty results, all match, column projection
  - Multiple AND/OR conditions
  - Different data types, size limits, row order

- **Suite 7**: Error Handling (5 tests)
  - Invalid syntax, column not found, type mismatch
  - Empty filter, malformed parentheses

### 4. `test_filter_edge_cases.py` (13 tests)
**Purpose**: Tests edge cases and error handling.

- **Suite 9**: Edge Cases (13 tests)
  - NULL comparisons, empty/whitespace filters
  - Very long expressions, case sensitivity
  - Division by zero, boolean columns
  - Scientific notation, negative numbers
  - Float precision, multiple spaces

### 5. `test_performance.py` (12 tests)
**Purpose**: Performance benchmarks (run with `-m slow`).

- **Suite 10**: Aggregation Performance (6 tests)
  - MIN, MAX, filtered aggregation
  - COUNT_DISTINCT, sequential operations
  - Complex filter performance

- **Suite 11**: Filtering Performance (3 tests)
  - Large file filtering
  - Complex multi-condition filters
  - Memory usage

## Running Tests

### Quick Start
```bash
# From the parquet directory
cd iowarp-mcps/iowarp_mcp_servers/parquet

# Run JSON validation tests (most important)
uv run pytest tests/test_json_response_validation.py -v

# Run all fast tests (skip performance)
uv run pytest tests/ -v -m "not slow"

# Run all tests including performance
uv run pytest tests/ -v
```

### By Priority
```bash
# 1. JSON validation (START HERE)
uv run pytest tests/test_json_response_validation.py -v

# 2. Filter parsing
uv run pytest tests/test_filtering.py -v

# 3. Filtered slice operations
uv run pytest tests/test_filtered_slice.py -v

# 4. Edge cases
uv run pytest tests/test_filter_edge_cases.py -v

# 5. Performance (slow)
uv run pytest tests/test_performance.py -v -m slow
```

### Specific Tests
```bash
# Run a specific test
uv run pytest tests/test_json_response_validation.py::test_read_slice_returns_valid_json -v

# Run only slow tests
uv run pytest tests/ -v -m slow

# Run with coverage
uv run pytest tests/ --cov=src/parquet_mcp --cov-report=html
```

## Test Philosophy

This test suite follows a **JSON-first validation approach**:

1. **Primary Goal**: Ensure the MCP server returns correct JSON responses
2. **Data Validation**: Every row in filtered JSON responses must match the filter
3. **Error Handling**: Errors must be returned as properly formatted JSON
4. **Metadata Accuracy**: Filter metadata must correctly reflect what was applied

This ensures the MCP interface works correctly end-to-end, testing what users actually interact with.

## Success Criteria

- ✅ 192 tests
- ✅ All JSON responses are valid and parseable
- ✅ All filtered JSON data matches filter criteria
- ✅ Performance tests meet time constraints

