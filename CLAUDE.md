# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IoWarp MCPs is a collection of Model Context Protocol (MCP) servers designed for scientific computing research. These enable AI agents and LLMs to interact with data analysis tools, HPC resources, and research datasets through a standardized protocol.

## Architecture

### Repository Structure
- **`mcps/`**: Contains all individual MCP server packages (Adios, Arxiv, Chronolog, etc.)
- **`src/iowarp_mcps/`**: Unified launcher for all MCP servers
- **`bin/`**: WRP (Universal Scientific MCP Client) implementation
- **`scripts/`**: Utility scripts for documentation and maintenance

### MCP Server Pattern
Each MCP server follows a consistent structure:
- **`src/server.py`**: Main server implementation using fastmcp
- **`pyproject.toml`**: Package configuration with entry point definition
- **`tests/`**: Test files for the MCP server
- Entry points follow pattern: `<name>-mcp = "server:main"`

## Common Development Commands

### Building and Testing Individual MCPs
```bash
# Navigate to specific MCP directory
cd mcps/<MCP_NAME>

# Install dependencies with uv
uv sync --all-extras --dev

# Run tests
uv run pytest tests/ -v

# Run linting
uv run ruff check .
uv run ruff format .

# Run type checking
uv run mypy src/ --ignore-missing-imports

# Run security audit
uv run pip-audit
```

### Running MCP Servers
```bash
# Using the unified launcher (from repository root)
uvx iowarp-mcps <server-name>

# Running directly from MCP directory
cd mcps/<MCP_NAME>
uv run <name>-mcp

# Examples
uvx iowarp-mcps pandas
uvx iowarp-mcps plot
uvx iowarp-mcps slurm
```

### Quality Control
The repository uses GitHub Actions for comprehensive quality control:
- **Ruff**: Linting and formatting checks
- **MyPy**: Type checking
- **pytest**: Unit tests with coverage reporting
- **pip-audit**: Security vulnerability scanning
- **Python compatibility**: Tests across Python 3.10, 3.11, and 3.12

### WRP Framework Testing
```bash
# Run WRP tests (requires API keys)
python -m pytest test_wrp_framework.py -v --tb=short
```

## Key Development Patterns

### Adding a New MCP Server
1. Create directory under `mcps/` with consistent naming
2. Follow the standard structure with `src/server.py` and `pyproject.toml`
3. Define entry point as `<name>-mcp = "server:main"`
4. The launcher will auto-discover the new server

### Testing Strategy
- Each MCP has its own test suite in `tests/` directory
- Use pytest for all testing
- Tests are run in parallel across all MCPs in CI
- Coverage reports are uploaded to Codecov

### Dependency Management
- All projects use `uv` for dependency management
- Dependencies are specified in `pyproject.toml`
- Use `uv sync` to install dependencies
- Add development dependencies with `uv add --dev <package>`

## Important Notes

- The unified launcher (`iowarp-mcps`) auto-discovers MCP servers by scanning for `pyproject.toml` files
- MCP servers use the fastmcp framework for implementation
- The WRP client supports multiple LLM providers (Gemini, Ollama, OpenAI, Claude)
- All code should follow Python 3.10+ compatibility
- Security scanning is mandatory - no vulnerabilities should be introduced