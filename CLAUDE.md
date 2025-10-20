# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**IoWarp MCPs** is a production-grade monorepo containing 15+ MCP (Model Context Protocol) servers designed for scientific computing research. The project enables AI agents and LLMs to interact with HPC resources, scientific data formats, and research datasets through a standardized protocol.

The repository uses a **unified launcher with auto-discovery** pattern: each MCP server is independently developed and tested, but all are launched through a single `iowarp-mcps <server-name>` command.

**Key Technologies**: FastMCP, Python 3.10+, UV package manager, Pydantic, pytest, Ruff

## Project Structure

```
iowarp-mcps/                           # Monorepo root
├── src/iowarp_mcps/                   # Unified launcher CLI
├── iowarp_mcp_servers/                # 15 independent MCP servers
│   ├── hdf5/ ⭐                       # Flagship server (v2.0, 25+ tools)
│   ├── pandas/                        # Data analysis operations
│   ├── slurm/                         # HPC job management
│   ├── arxiv/                         # Research paper fetching
│   ├── chronolog/                     # Distributed logging
│   ├── compression/                   # File compression
│   ├── darshan/                       # I/O performance analysis
│   ├── jarvis/                        # Data pipeline management
│   ├── lmod/                          # Environment modules
│   ├── ndp/                           # Dataset discovery
│   ├── node-hardware/                 # System hardware info
│   ├── parallel-sort/                 # Large file sorting
│   ├── parquet/                       # Parquet file handling
│   ├── plot/                          # Data visualization
│   ├── adios/                         # ADIOS2 data I/O
│   └── [each has its own pyproject.toml, dependencies, tests]
├── iowarp_mcp_webpage/                # Docusaurus documentation site
├── scripts/                           # Utility scripts (generate_docs.py, etc)
├── .github/workflows/                 # CI/CD automation
└── [root config files]                # pyproject.toml, uv.lock

```

**Key Design Pattern:**
- Root `pyproject.toml` only includes launcher dependencies (click)
- Each MCP server in `iowarp_mcp_servers/` is a complete Python package with its own `pyproject.toml`, entry point, and isolated dependencies
- Launcher auto-discovers servers by scanning for `pyproject.toml` files
- Servers run via `uvx` with isolated environments for dependency isolation

## Common Development Commands

### Setup & Installation

```bash
# Install all dependencies (development mode)
uv sync --all-extras --dev

# For a specific MCP server
cd iowarp_mcp_servers/hdf5
uv sync --all-extras --dev
```

### Code Quality & Testing

#### Run all quality checks (mimics CI):
```bash
# For a specific MCP server
cd iowarp_mcp_servers/hdf5

# Ruff: Linting + formatting
uv run ruff check .
uv run ruff format . --check

# MyPy: Type checking
uv run mypy src/

# pytest: Tests with coverage
uv run pytest -v --cov=src/

# pip-audit: Security vulnerabilities
uv run pip-audit
```

#### Quick test runs:
```bash
# Run all tests in a server
cd iowarp_mcp_servers/hdf5
uv run pytest -v

# Run a single test file
uv run pytest tests/test_server.py -v

# Run a specific test
uv run pytest tests/test_server.py::TestClass::test_method -v

# Run with coverage report
uv run pytest --cov=src/ --cov-report=html
```

### Running Servers

```bash
# Via launcher (from root directory)
uvx iowarp-mcps hdf5

# Direct development mode (from server directory)
cd iowarp_mcp_servers/hdf5
uv run hdf5-mcp

# List all available MCPs
uvx iowarp-mcps
```

### Code Formatting & Fixing

```bash
# Format code automatically (Ruff)
cd iowarp_mcp_servers/hdf5
uv run ruff format .

# Fix linting issues automatically
uv run ruff check --fix .

# Verify types after changes
uv run mypy src/
```

## Architecture Patterns

### Standard MCP Server Structure

Each server follows this proven pattern:

```
ServerName/
├── pyproject.toml                     # Entry point: {name}-mcp = "module:server:main"
├── README.md                          # Server documentation
├── src/{name}_mcp/
│   ├── __init__.py
│   ├── server.py                      # @mcp.tool(), @mcp.resource() decorators
│   ├── config.py                      # Pydantic configuration models
│   └── [domain-specific modules]      # Implementation details
├── tests/
│   ├── conftest.py                    # Shared pytest fixtures
│   ├── test_server.py                 # Server lifecycle tests
│   ├── test_mcp_handlers.py           # MCP tool/resource tests
│   └── test_*.py                      # Feature-specific tests
└── uv.lock                            # Dependency lock
```

### FastMCP Decorators Pattern

```python
from fastmcp import FastMCP

mcp = FastMCP("server-name")

# Expose Python function as MCP tool
@mcp.tool(description="What this tool does")
def my_tool(param1: str, param2: int) -> str:
    return result

# Expose resources (URI scheme: scheme://path)
@mcp.resource(uri_template="scheme://{path}")
def get_resource(path: str) -> str:
    return content

# Multi-step workflows with prompts
@mcp.prompt()
def workflow() -> list[Message]:
    return [Message(...), Message(...)]

async def main():
    await mcp.run()
```

### Configuration with Pydantic

```python
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    """Configuration from environment variables"""
    param: str = Field(default="value", description="...")
    debug: bool = Field(default=False)

    class Config:
        env_prefix = "SERVER_"
```

## Important Implementation Details

### Dependency Isolation

- **UV Package Manager**: Modern, faster Python package manager. Use `uv` commands instead of `pip`
- **Per-Server Dependencies**: Each server has isolated dependencies via its own `pyproject.toml`
- **Lock Files**: `uv.lock` at server level ensures reproducible builds
- **Development Dependencies**: Specified in `[dependency-groups] dev` section

### Testing Strategy

- **Unit Tests**: Per-capability/feature testing
- **Integration Tests**: Server lifecycle and tool registration
- **Multi-Python Support**: CI tests against Python 3.10, 3.11, 3.12
- **Parallel Execution**: GitHub Actions runs tests in parallel (20 parallel jobs)
- **Coverage Tracking**: pytest-cov with Codecov integration

### Code Quality Standards

- **Linting**: Ruff (replaces black, flake8, isort)
- **Type Checking**: MyPy with `--ignore-missing-imports`
- **Security**: pip-audit scans for vulnerabilities
- **Format**: Single tool (Ruff) for consistent formatting - no manual formatting

### Performance Optimization (HDF5 v2.0 Reference)

The HDF5 server (v2.0) implements patterns useful for all MCPs:
- **LRU Cache**: 1000-item cache for repeated queries (100-1000x speedup)
- **Resource Pooling**: Lazy loading with proper cleanup
- **Performance Monitoring**: Adaptive units (B, KB, MB, etc.)
- **Async Handling**: asyncio for I/O-bound operations

## Adding a New MCP Server

1. Create directory: `iowarp_mcp_servers/my-server/` (use kebab-case)
2. Create `pyproject.toml` with:
   ```toml
   [project]
   name = "my-server-mcp"
   version = "1.0.0"

   [project.scripts]
   my-server-mcp = "my_server_mcp.server:main"
   ```
3. Implement `src/my_server_mcp/server.py` using FastMCP decorators
4. Add tests in `tests/` directory
5. Launcher auto-discovers it on next run

## CI/CD Pipeline

**Quality Control** (`.github/workflows/quality_control.yml`):
- Auto-discovers all MCPs with `pyproject.toml`
- Runs 4 checks in parallel per MCP:
  - Ruff linting + formatting
  - MyPy type checking
  - pytest with coverage
  - pip-audit security scan
- Tests Python 3.10, 3.11, 3.12
- Coverage uploaded to Codecov

**Key Note**: Chronolog MCP has dedicated workflow (requires system dependencies)

## Python Version & Dependencies

- **Minimum Python**: 3.10 (enforced in root `pyproject.toml`)
- **Package Manager**: UV (not pip/conda)
- **Build System**: Hatchling
- **Key Frameworks**: FastMCP 0.2.0+, Pydantic 2.4.2+

## Important Files Reference

| Purpose | Path |
|---------|------|
| Main Launcher | `src/iowarp_mcps/__init__.py` |
| HDF5 Server Example | `iowarp_mcp_servers/hdf5/src/hdf5_mcp/server.py` |
| Quality Control CI | `.github/workflows/quality_control.yml` |
| Main Configuration | `pyproject.toml` |
| Main Docs Site | `iowarp_mcp_webpage/` |

## Debugging Tips

### Server Won't Start

1. Check if `pyproject.toml` has correct entry point: `name-mcp = "module:server:main"`
2. Verify server file has `async def main()` and proper MCP initialization
3. Test directly: `cd iowarp_mcp_servers/hdf5 && uv run hdf5-mcp`

### Tests Failing

1. Check Python version: `python --version` (must be 3.10+)
2. Reinstall dependencies: `uv sync --all-extras --dev`
3. Run with verbose output: `uv run pytest -vv`
4. Check for missing conftest.py fixtures in test directory

### Type Errors

1. Run MyPy locally: `uv run mypy src/`
2. Don't add `# type: ignore` without understanding the issue
3. Use proper type hints for all function parameters and returns
4. Consider using `from typing import TypeVar, Generic` for complex types

### Performance Issues

1. Profile with: `python -m cProfile -s cumtime your_script.py`
2. Check for repeated file I/O or network calls
3. Consider caching with LRU (@lru_cache) for expensive operations
4. Use asyncio for I/O-bound operations

## Website Design System

The IoWarp MCPs website (`iowarp_mcp_webpage/`) follows a **neobrutalist design language** with the following principles:

### Brand Colors (Locked from Logo)

These colors are sacred and never change:

```css
--brand-teal: #217CA3;
--brand-teal-light: #6BC2E4;
--brand-orange: #EC7C26;
--brand-cream: #FAF8F5;
```

### Dark-First Theme Philosophy

**Dark Mode (Default):**
- Background: Pure black `#000000`
- Surface elements: Dark blue `#0F1F35`
- Text: Light gray `#E5E7EB`
- Shadows: Light teal `rgba(107, 194, 228, 0.3-0.6)` for visibility
- Accents: Teal and orange only for highlights

**Light Mode:**
- Background: Warm cream `#FAF8F5`
- Surface elements: White `#FFFFFF`
- Text: Black `#111827`
- Shadows: Black for visibility
- Accents: Teal and orange for highlights

### Neobrutalism Elements

Neobrutalism comes from **structure and typography**, not color contrast:

- **Thick borders**: 3px (`--border-width: 3px`)
- **Border radius**: Single value of 8px everywhere (`--border-radius: 8px`)
- **Hard shadows**: No blur, using `Xpx Ypx 0px 0px color` pattern
  - Small: `4px 4px 0px 0px`
  - Medium: `6px 6px 0px 0px`
  - Large: `8px 8px 0px 0px`
  - Extra large: `10px 10px 0px 0px`
- **Bold typography**: Font weights 700-900
- **Warp animations**: Slight rotate/skew transforms on hover
- **Millimeter grid**: Background pattern on hero using 20px grid

### Critical Design Rules

1. **NO white boxes on dark backgrounds** - All cards use `#0F1F35` in dark mode
2. **Shadows must be visible** - Light teal in dark mode, black in light mode
3. **Teal/Orange are accents only** - Not for large background areas
4. **Category tags are semantic** - Use meaning-based colors independent of theme
5. **Single border radius** - Always 8px, no variations
6. **Locked brand colors** - Never modify teal, orange, or cream values

### Component Patterns

**Hero Section:**
- Teal background with millimeter paper grid overlay
- Grid uses orange (35% opacity) in light mode, light teal (35% opacity) in dark mode
- Gnosis box: Dark/light background with theme-inverted outline
  - Light mode: White bg, dark blue `#0A1520` outline and shadow
  - Dark mode: Dark bg `#0F1F35`, orange outline and shadow
- Warp/rotation transforms for dynamic feel

**MCP Cards:**
- Featured cards: Large, detailed, with platform buttons
- Regular cards: Simple icon, category tag, description
- All cards: 8px radius, 3px border, hard shadow

**Search & Filters:**
- Search bar: Rotated -0.5deg, large shadow (8px)
- Orange shadow glow in dark mode for emphasis
- Category chips: Filled with semantic colors, active state with border

### File Structure

```
iowarp_mcp_webpage/
├── src/
│   ├── css/custom.css              # Global design tokens, hero styling
│   ├── pages/index.js              # Homepage hero and layout
│   ├── components/
│   │   └── MCPShowcase/
│   │       ├── index.js            # MCP cards, search, filters
│   │       └── styles.module.css   # Component-specific styling
│   └── data/mcpData.js             # MCP metadata and categories
├── static/
│   └── img/
│       ├── logos/                  # Platform and institution logos
│       └── iowarp_logo.png         # Main brand logo (360px)
└── docusaurus.config.js            # Site configuration
```

### Development Commands

```bash
# Start dev server
cd iowarp_mcp_webpage
npm start

# Build for production
npm run build

# Serve production build locally
npm run serve
```

### SEO & Metadata

The site includes comprehensive metadata for:
- Social sharing (Open Graph, Twitter cards)
- Search engines (keywords, descriptions)
- Official technology links (HDF5, ADIOS, Slurm, etc.)
- Institution branding (GRC, IIT, NSF)

## Key Resources

- **MCP Protocol**: https://modelcontextprotocol.io/
- **FastMCP Documentation**: https://github.com/jlowin/FastMCP
- **Project Website**: https://iowarp.github.io/iowarp-mcps/
- **Contribution Guide**: https://github.com/iowarp/iowarp-mcps/wiki/Contribution
- **Community**: Zulip chat at grc.zulipchat.com/#narrow/channel/518574-iowarp-mcps

## Branch Strategy

- **main**: Stable releases and merging PRs
- **dev**: Development branch for integration
- **feature/\***: Feature branches for new work

When creating PRs, target `main` branch for releases.
- "No ceremony. Only substance. Execute what's explicitly requested.
  Never create preparatory files, meta-work, explanatory documents,
  or planning artifacts. No files unless asked. Act, don't narrate.
  Substance over process. Code quality over documentation theater."

  Core principle: If not explicitly requested → don't create it.