# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CLIO Kit** is part of the IoWarp platform's tooling layer for AI agents. It is a production-grade monorepo containing 22 MCP (Model Context Protocol) servers designed for scientific computing research, alongside skills, workflow bundles and an indexed community marketplace. The project enables AI agents and LLMs to interact with HPC resources, scientific data formats, and research datasets through a standardized protocol.

The repository uses a **unified launcher with auto-discovery** pattern: each MCP server is independently developed and tested, but all are launched through a single `clio-kit mcp-server <server-name>` command.

**Platform Context**: CLIO Kit is the tooling layer of the IoWarp platform, providing comprehensive agent capabilities beyond just MCP servers.

**Key Technologies**: FastMCP 4, Python 3.10+, UV package manager, Pydantic, pytest, Ruff

## Project Structure

```
clio-kit/                           # Monorepo root
├── src/clio_kit/                   # Unified launcher CLI
├── clio-kit-mcp-servers/                # 22 independent MCP servers
│   ├── hdf5/ ⭐                       # HDF5 access (27 tools)
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
│   ├── paraview/                      # Scientific visualization
│   ├── parquet/                       # Parquet file handling
│   ├── plot/                          # Data visualization
│   ├── adios/                         # ADIOS2 data I/O
│   ├── geo/                           # Geospatial data and maps
│   ├── terrain/                       # Digital elevation models
│   ├── seismology/                    # Seismic waveforms and catalogs
│   ├── spack/                         # HPC package management
│   ├── scientific-catalog/            # Scientific dataset catalogs
│   ├── web/                           # Web fetching and search
│   └── [each has its own manifest, lock file, dependencies, tests]
├── clio-agentic-search/             # Standalone hybrid retrieval engine (not an MCP server)
├── clio-kit-website/                # Docusaurus documentation site
├── scripts/                           # Utility scripts (generate_docs.py, etc)
├── .github/workflows/                 # CI/CD automation
└── [root config files]                # pyproject.toml, uv.lock

```

**Key Design Pattern:**
- Root `pyproject.toml` includes only launcher dependencies; MCP SDK clients are a verification extra
- Each MCP server in `clio-kit-mcp-servers/` is a complete package with its own manifest, lock file, entry point, and isolated dependencies
- Launcher auto-discovers servers by reading each one's `clio-server.toml`, which names the server, its runtime (`python`, `node` or `go`) and its entry point
- Servers run in isolated, content-addressed environments built from their own lock file — `uv sync --frozen`, `npm ci` or `go build` — so a launch never resolves dependencies

## Common Development Commands

### Setup & Installation

```bash
# Install all dependencies (development mode)
uv sync --all-extras --dev

# For a specific MCP server
cd clio-kit-mcp-servers/hdf5
uv sync --all-extras --dev
```

### Code Quality & Testing

#### Run all quality checks (mimics CI):
```bash
# For a specific MCP server
cd clio-kit-mcp-servers/hdf5

# Ruff: Linting + formatting
uv run ruff check .
uv run ruff format . --check

# MyPy: Type checking
uv run mypy src/

# pytest: Tests with coverage
uv run pytest -v --cov=src/

# pip-audit: Security vulnerabilities
uv run pip-audit

# FastMCP 4 validation (instructions, annotations, tags, resources, prompts)
uv run python ../../scripts/validate_fastmcp.py
```

#### Quick test runs:
```bash
# Run all tests in a server
cd clio-kit-mcp-servers/hdf5
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
uv run clio-kit mcp-server hdf5

# Direct development mode (from server directory)
cd clio-kit-mcp-servers/hdf5
uv run hdf5-mcp

# List all available MCPs
uv run clio-kit mcp-servers
```

### Code Formatting & Fixing

```bash
# Format code automatically (Ruff)
cd clio-kit-mcp-servers/hdf5
uv run ruff format .

# Fix linting issues automatically
uv run ruff check --fix .

# Verify types after changes
uv run mypy src/
```

### clio-agentic-search (Standalone Service)

```bash
# Setup
cd clio-agentic-search
uv sync --all-extras --dev

# Quality checks
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
uv run pytest --ignore=tests/benchmarks -v

# Full quality gate (6 checks)
uv run python -m clio_agentic_search.evals.quality_gate

# Run API server
uv run uvicorn clio_agentic_search.api.app:app --reload

# CLI
uv run clio query --namespace local_fs --q "pressure between 190 and 360 kPa"
uv run clio index --namespace local_fs
```

**Note**: clio-agentic-search is NOT an MCP server. It's a standalone FastAPI service with its own CI jobs in `quality_control.yml`. The launcher does not discover it.

## Architecture Patterns

### Standard MCP Server Structure

Each server follows this proven pattern:

```
ServerName/
├── pyproject.toml                     # Entry point: {name}-mcp = "module.server:main"
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

### FastMCP 4 Server Pattern

All servers use FastMCP 4 and must include: instructions, tool annotations, tool tags, at least 1 resource, and at least 1 prompt.

```python
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.prompts import Message

mcp = FastMCP(
    "server-name",
    instructions="Brief description of what this server does and when to use each tool.",
)

# Tools: always include annotations and tags
@mcp.tool(
    description="1-2 sentence description of what this tool does.",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
    tags={"category", "subcategory"},
)
async def my_tool(param1: str, param2: int) -> str:
    try:
        return do_work(param1, param2)
    except Exception as e:
        raise ToolError(f"Operation failed: {e}") from e

# Resources: at least 1 per server
@mcp.resource("server-name://capabilities")
def capabilities() -> dict:
    """What this server can do."""
    return {"features": [...]}

# Resource templates (parameterized)
@mcp.resource("server-name://{file_path}/metadata")
def file_metadata(file_path: str) -> dict:
    return get_metadata(file_path)

# Prompts: at least 1 per server
@mcp.prompt()
def guided_workflow(input_path: str) -> list[Message]:
    """Guided workflow for common operations."""
    return [
        Message(f"Analyze the data at {input_path}. First summarize, then process."),
    ]

def main() -> None:
    import os, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "http"], default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    transport = args.transport or os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")
```

### Key FastMCP 4 Imports

```python
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError, ResourceError
from fastmcp.prompts import Message
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
- **Parallel Execution**: GitHub Actions discovers the applicable server matrix
- **Coverage Tracking**: pytest-cov with Codecov integration

### Code Quality Standards

- **Linting**: Ruff (replaces black, flake8, isort)
- **Type Checking**: MyPy with `--ignore-missing-imports`
- **Security**: pip-audit scans for vulnerabilities
- **Format**: Single tool (Ruff) for consistent formatting - no manual formatting

### HDF5 implementation patterns

The HDF5 server implements patterns useful for other MCPs:
- **LRU Cache**: bounded caching for repeated queries; measure performance on the target workload
- **Resource Pooling**: Lazy loading with proper cleanup
- **Performance Monitoring**: Adaptive units (B, KB, MB, etc.)
- **Async Handling**: asyncio for I/O-bound operations

## Adding a New MCP Server

1. Create directory: `clio-kit-mcp-servers/my-server/` (use kebab-case)
2. Create `pyproject.toml` with hatchling build backend:
   ```toml
   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [project]
   name = "my-server-mcp"
   version = "1.0.0"
   dependencies = ["fastmcp>=4.0.3,<5"]

   [project.scripts]
   my-server-mcp = "my_server_mcp.server:main"
   ```
3. Implement `src/my_server_mcp/server.py` following the FastMCP 4 pattern above. **Required**:
   - `instructions=` on `FastMCP()` constructor
   - `annotations=` and `tags=` on every `@mcp.tool()`
   - `ToolError` for all error paths (not error dicts)
   - At least 1 `@mcp.resource()`
   - At least 1 `@mcp.prompt()` returning `list[Message]`
   - Keep small tool inventories on one page; some clients do not follow pagination
4. Add tests in `tests/` directory
5. Validate: `uv run python ../../scripts/validate_fastmcp.py`
6. Add `clio-server.toml`, register the server in `mcp-server-versions.toml`, and generate its manifests. Follow the [contributor guide](CONTRIBUTING.md#adding-a-new-mcp-server).

## CI/CD Pipeline

**Quality Control** (`.github/workflows/quality_control.yml`):
- Auto-discovers all MCPs with `pyproject.toml`
- Uses reusable composite action (`.github/actions/setup-mcp/action.yml`) for setup
- Runs 5 checks in parallel per MCP:
  - Ruff linting + formatting
  - MyPy type checking (advisory, non-blocking)
  - pytest with coverage (matrix: Python 3.10, 3.11, 3.12)
  - pip-audit security scan
  - FastMCP 4 validation (`scripts/validate_fastmcp.py`)
- Coverage uploaded to Codecov (Python 3.12 only)

**Docs & Website** (`.github/workflows/docs-and-website.yml`):
- Generates Docusaurus docs via `scripts/generate_docs.py`
- Updates README files via `scripts/readme_filler.py`
- Both use `scripts/extract_mcp_metadata.py` (FastMCP async API, not AST parsing)

**Key Note**: Chronolog MCP has dedicated workflow (requires system dependencies)

## Python Version & Dependencies

- **Minimum Python**: 3.10 (enforced in root `pyproject.toml`)
- **Package Manager**: UV (not pip/conda)
- **Build System**: Hatchling
- **Key Frameworks**: FastMCP 4.0.3+ (below 5), MCP Python SDK 2.2+ (below 3), Pydantic 2.4.2+

## Important Files Reference

| Purpose | Path |
|---------|------|
| Main Launcher | `src/clio_kit/__init__.py` |
| HDF5 Server (reference) | `clio-kit-mcp-servers/hdf5/src/hdf5_mcp/server.py` |
| Quality Control CI | `.github/workflows/quality_control.yml` |
| Composite Action | `.github/actions/setup-mcp/action.yml` |
| FastMCP Validator | `scripts/validate_fastmcp.py` |
| Metadata Extractor | `scripts/extract_mcp_metadata.py` |
| Doc Generator | `scripts/generate_docs.py` |
| README Updater | `scripts/readme_filler.py` |
| Main Configuration | `pyproject.toml` |
| Main Docs Site | `clio-kit-website/` |

## Debugging Tips

### Server Won't Start

1. Check if `pyproject.toml` has correct entry point: `name-mcp = "module.server:main"`
2. Verify server file has `def main()` with argparse and `mcp.run()` call
3. Test directly: `cd clio-kit-mcp-servers/hdf5 && uv run hdf5-mcp`

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

The CLIO Kit website (`clio-kit-website/`) follows a **neobrutalist design language** with the following principles:

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
clio-kit-website/
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
cd clio-kit-website
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
- **Project Website**: https://toolkit.iowarp.ai/
- **Contribution Guide**: https://github.com/iowarp/clio-kit/blob/main/CONTRIBUTING.md
- **Community**: Zulip chat at https://iowarp.zulipchat.com/#narrow/channel/543872-Agent-Toolkit
- **Join Community**: Invitation link at https://iowarp.zulipchat.com/join/e4wh24du356e4y2iw6x6jeay/

## Branch Strategy

- **main**: Stable releases and merging PRs
- **feature/\***: Feature branches for new work

When creating PRs, target `main` branch for releases.
- "No ceremony. Only substance. Execute what's explicitly requested.
  Never create preparatory files, meta-work, explanatory documents,
  or planning artifacts. No files unless asked. Act, don't narrate.
  Substance over process. Code quality over documentation theater."

  Core principle: If not explicitly requested → don't create it.