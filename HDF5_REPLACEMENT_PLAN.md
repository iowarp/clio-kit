# HDF5 MCP: Full Replacement Plan

## Mission
Replace iowarp's basic HDF5 MCP with your advanced implementation while maintaining iowarp's simple installation (`uvx iowarp-mcps hdf5`). Make this the exemplary template for all iowarp MCPs.

## Current vs New

**Current (iowarp):**
- 4 tools, ~460 lines
- FastMCP framework
- Read-only operations
- No caching, no parallelism
- Simple but limited

**New (your implementation):**
- 25+ tools, ~3000 lines
- Full MCP with advanced features
- Caching, lazy loading, parallel ops
- Streaming, discovery, optimization
- Production-grade architecture

## Strategy: Direct Replacement

### Keep from iowarp:
- Installation method: `uvx iowarp-mcps hdf5`
- Entry point: `hdf5-mcp = "server:main"`
- PyPI publishing workflow
- Directory structure convention

### Bring from your implementation:
- Everything in `src/hdf5_mcp_server/` → `src/`
- All tools, resource management, caching
- Architecture patterns (ToolRegistry, decorators, etc.)
- Advanced features

## File Migration Map

```
YOUR IMPLEMENTATION                    IOWARP LOCATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
src/hdf5_mcp_server/
  ├── main.py                    →    src/main.py
  ├── server.py                  →    src/server.py (replace)
  ├── tools.py                   →    src/tools.py (new)
  ├── resources.py               →    src/resources.py (new)
  ├── cache.py                   →    src/cache.py (new)
  ├── config.py                  →    src/config.py (new)
  ├── utils.py                   →    src/utils.py (new)
  ├── prompts.py                 →    src/prompts.py (new)
  ├── protocol.py                →    src/protocol.py (new)
  ├── async_io.py                →    src/async_io.py (new)
  ├── batch_operations.py        →    src/batch_operations.py (new)
  ├── parallel_ops.py            →    src/parallel_ops.py (new)
  ├── streaming.py               →    src/streaming.py (new)
  ├── scanner.py                 →    src/scanner.py (new)
  ├── resource_pool.py           →    src/resource_pool.py (new)
  ├── task_queue.py              →    src/task_queue.py (new)
  └── transports/                →    src/transports/ (new)

DELETE from iowarp:
  ├── src/mcp_handlers.py              (replaced by tools.py)
  ├── src/capabilities/                (replaced by tools.py)
  └── src/server.py                    (replaced by your server.py)
```

## Entry Point Adjustment

**Current iowarp:**
```toml
[project.scripts]
hdf5-mcp = "server:main"
```

**Change to:**
```toml
[project.scripts]
hdf5-mcp = "main:main"
```

This matches your entry point structure.

## Dependencies Update

**Add to pyproject.toml:**
```toml
dependencies = [
  "mcp>=1.4.0",           # Core MCP (upgrade from fastmcp)
  "h5py>=3.9.0",          # Latest HDF5
  "numpy>=1.24.0,<2.0.0", # NumPy
  "pydantic>=2.4.2,<3.0.0", # Validation
  "aiofiles>=23.2.1",     # Async file ops
  "psutil>=5.9.0",        # System monitoring
  "jinja2>=3.1.0",        # Prompt templates
  "python-dotenv>=0.19.0" # Config
]
```

Optional (for advanced features):
```toml
[project.optional-dependencies]
performance = [
  "dask>=2023.0.0",       # Large-scale parallel
  "aiohttp>=3.9.0"        # Async HTTP
]
```

## Migration Steps

### Phase 1: Clean Slate (15 min)
```bash
cd /home/akougkas/projects/iowarp-mcps/mcps/HDF5/

# Backup current (just in case)
git checkout -b backup/hdf5-simple
git checkout main

# Remove old implementation
rm -rf src/capabilities/
rm src/mcp_handlers.py
rm src/server.py
```

### Phase 2: Copy Your Implementation (10 min)
```bash
# Copy all source files
cp -r /home/akougkas/projects/hdf5-mcp-server/src/hdf5_mcp_server/* src/

# Clean up package naming (remove hdf5_mcp_server references if any)
# Files should import relatively: from .tools import ToolRegistry
```

### Phase 3: Update Configuration (10 min)
**Update `pyproject.toml`:**
- Change entry point to `main:main`
- Update dependencies (add new ones)
- Bump version to 2.0.0 (major upgrade)

**Update `src/__init__.py`:**
```python
"""HDF5 MCP Server - Advanced HDF5 operations for AI agents."""

from .server import HDF5Server
from .tools import HDF5Tools, ToolRegistry

__version__ = "2.0.0"
__all__ = ["HDF5Server", "HDF5Tools", "ToolRegistry"]
```

### Phase 4: Documentation (30 min)
**Update `README.md`** with:
- New tool list (all 25+)
- Installation (same: `uvx iowarp-mcps hdf5`)
- Quick examples showing advanced features
- Architecture overview (simple diagram)

**Create `docs/ARCHITECTURE.md`:**
- Tool registry pattern
- Resource management
- Caching strategy
- Parallel operations

**Create `docs/EXAMPLES.md`:**
- Basic usage (backward compatible)
- Advanced features (streaming, batch, discovery)
- Common workflows

Keep docs concise, code-heavy, minimal prose.

### Phase 5: Testing (45 min)
**Copy your tests:**
```bash
cp -r /home/akougkas/projects/hdf5-mcp-server/tests/* tests/
```

**Update test structure to match iowarp conventions:**
- Use pytest (already in use)
- Add integration tests with sample HDF5 files
- Verify backward compatibility

**Key tests:**
- Installation: `uvx iowarp-mcps hdf5` works
- All 25+ tools functional
- Caching works (speed test)
- Parallel ops work (multi-file)
- Streaming works (large file)

### Phase 6: Verify Installation (15 min)
```bash
# Local install test
cd /home/akougkas/projects/iowarp-mcps
uv sync

# Test entry point
uvx --from . iowarp-mcps hdf5

# Should start server with your implementation
```

## Critical Adjustments

### 1. Server Entry Point
Your `main.py` likely has:
```python
def main():
    server = HDF5Server(...)
    server.run()
```

Ensure it matches iowarp's calling convention (stdio by default).

### 2. Import Paths
Change absolute imports:
```python
# From this:
from hdf5_mcp_server.tools import ToolRegistry

# To this:
from .tools import ToolRegistry
```

### 3. Configuration
iowarp uses environment variables. Ensure your `config.py` supports:
- `HDF5_DATA_DIR` - default data directory
- `HDF5_CACHE_SIZE` - cache capacity
- `HDF5_NUM_WORKERS` - parallel workers

## Documentation Structure

```
mcps/HDF5/
├── README.md                    # Main docs (simple, example-focused)
├── docs/
│   ├── ARCHITECTURE.md          # How it works (diagrams + code)
│   ├── EXAMPLES.md              # Common use cases
│   ├── TOOLS.md                 # All 25+ tools listed
│   └── MIGRATION.md             # For users of old version
├── src/                         # Your implementation
└── tests/                       # Your tests
```

**README.md structure:**
```markdown
# HDF5 MCP - Advanced Scientific Data Access

One-liner description.

## Installation
`uvx iowarp-mcps hdf5`

## Quick Start
3 code examples (basic, advanced, discovery)

## Features
- 25+ tools
- Caching (100-1000x speedup)
- Parallel ops (4-8x speedup)
- Streaming (unlimited size)
- Discovery tools

## Tools Overview
Table with categories

## Architecture
Link to docs/ARCHITECTURE.md

## Examples
Link to docs/EXAMPLES.md
```

Keep it visual, code-heavy, minimal marketing speak.

## Success Criteria

### Functional:
- ✅ `uvx iowarp-mcps hdf5` installs and runs
- ✅ All 25+ tools work
- ✅ Caching provides speedup (verify with test)
- ✅ Parallel ops work (verify with multi-file test)
- ✅ Streaming works (verify with 1GB+ file)

### Quality:
- ✅ Tests pass (pytest)
- ✅ Type checking passes (mypy)
- ✅ Linting passes (ruff)
- ✅ Docs are clear and code-focused

### Integration:
- ✅ Follows iowarp conventions
- ✅ CI/CD passes
- ✅ Published to PyPI as iowarp-mcps v2.0.0

## Timeline

**Total: ~2-3 hours**

| Phase | Time | Task |
|-------|------|------|
| 1 | 15 min | Clean old implementation |
| 2 | 10 min | Copy your implementation |
| 3 | 10 min | Update config files |
| 4 | 30 min | Write docs (README, ARCHITECTURE, EXAMPLES) |
| 5 | 45 min | Copy and adapt tests |
| 6 | 15 min | Verify installation |
| - | 30 min | Buffer for issues |

## Post-Migration

### Immediate:
1. Test with real HDF5 files (scientific datasets)
2. Benchmark vs old implementation
3. Create demo video/GIF for docs

### This Week:
1. Get team review
2. Merge to main
3. Publish to PyPI (v2.0.0)
4. Announce upgrade

### Template for Other MCPs:
Use this migration as pattern:
- Replace simple implementations with advanced ones
- Keep installation simple
- Docs: concise, code-heavy, visual
- Tests: comprehensive but fast

## Files to Create/Update

**Create:**
- `HDF5_REPLACEMENT_PLAN.md` (this file)
- `docs/ARCHITECTURE.md`
- `docs/EXAMPLES.md`
- `docs/TOOLS.md`
- `docs/MIGRATION.md`

**Update:**
- `README.md` (complete rewrite)
- `pyproject.toml` (dependencies, entry point, version)
- `src/__init__.py` (exports)
- `.gitignore` (if needed)

**Replace:**
- All of `src/` with your implementation



Ready to build the best HDF5 MCP in existence.
