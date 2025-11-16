# Hdf5 MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/hdf5-mcp.svg)](https://pypi.org/project/hdf5-mcp/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

**Part of [Agent Toolkit](https://iowarp.github.io/agent-toolkit/) - Gnosis Research Center**

**Part of the IoWarp MCP Server Collection** - Comprehensive HDF5 operations for AI agents working with scientific and engineering data.

## Quick Start

```bash
uvx iowarp-agent-toolkit mcp-server hdf5
```

## Documentation

- **Full Documentation**: [Agent Toolkit Website](https://iowarp.github.io/agent-toolkit/)
- **Installation Guide**: See [INSTALLATION.md](../../../CLAUDE.md#setup--installation)
- **Contributing**: See [Contribution Guide](https://github.com/iowarp/agent-toolkit/wiki/Contribution)

---

## ✨ What's New in v2.0

- **FastMCP Architecture** - Zero-boilerplate with `@mcp.tool()` decorators
- **AI-Powered Features** - LLM sampling, progress reporting, elicitation
- **Resource URIs** - Access HDF5 files via `hdf5://` scheme with wildcards
- **Client Roots** - Automatic file discovery from client directories
- **Context Integration** - Progress tracking and AI insights
- **Enhanced Error Handling** - ToolError/ResourceError exceptions
- **Workflow Prompts** - Pre-built analysis templates with Message()
- **26+ Tools** - All with tags, annotations, and full FastMCP support

## Installation

### Option 1: Direct Run (Recommended)
```bash
# Run from local directory
cd /path/to/agent-toolkit/agent-toolkit-mcp-servers/hdf5
uv run hdf5-mcp

# Or from GitHub
uvx --from "git+https://github.com/iowarp/agent-toolkit.git#subdirectory=agent-toolkit-mcp-servers/hdf5" hdf5-mcp
```

### Option 2: Claude Code Integration
```bash
# Add to Claude Code
claude add mcp hdf5 -- uv --directory /path/to/agent-toolkit/agent-toolkit-mcp-servers/hdf5 run hdf5-mcp
```

### Option 3: Global Install
```bash
# Install globally
uv pip install "git+https://github.com/iowarp/agent-toolkit.git#subdirectory=agent-toolkit-mcp-servers/hdf5"

# Run anywhere
hdf5-mcp
```

### Verify Installation
```bash
# Run quick test
./quick_test.sh

# Or manually
hdf5-mcp --help
```

See **[INSTALL.md](INSTALL.md)** for detailed installation instructions and troubleshooting.

## Features

- **26 Tools** - Comprehensive HDF5 operations with `@mcp.tool()` decorators
- **3 Resources** - HDF5 file URIs with `@mcp.resource()` and wildcard support
- **4 Prompts** - Analysis workflows using `Message()` objects
- **AI-Powered** - LLM sampling for insights, progress reporting, user elicitation
- **LRU Caching** - 100-1000x speedup on repeated queries
- **Parallel Ops** - 4-8x faster batch processing
- **Streaming** - Handle unlimited file sizes with chunked reading
- **Discovery** - AI-assisted exploration and similar dataset finding
- **Optimization** - Bottleneck detection with AI recommendations
- **Client Roots** - Automatic file discovery from client directories

## Quick Start

### Basic Usage (Tools)
```python
# Open HDF5 file
open_file(path="simulation.h5")

# Analyze structure
analyze_dataset_structure(path="/")

# Read dataset
read_full_dataset(path="/results/temperature")

# Close file
close_file()
```

### New: Resource URIs
```python
# Access file metadata
hdf5://simulation.h5/metadata

# Access dataset
hdf5://simulation.h5/datasets//results/temperature

# Access structure
hdf5://simulation.h5/structure
```

### New: Workflow Prompts
```python
# Explore file workflow
explore_hdf5_file(file_path="simulation.h5")

# Optimize access workflow
optimize_hdf5_access(file_path="simulation.h5", access_pattern="sequential")

# Compare datasets
compare_hdf5_datasets(file_path="data.h5", dataset1="/a", dataset2="/b")

# Batch processing
batch_process_hdf5(directory="data/", operation="statistics")
```

### Advanced Features
```python
# Stream large dataset
stream_dataset(path="/large_data", chunk_size=10000)

# Batch read multiple datasets in parallel
batch_read_datasets(paths=["/data1", "/data2", "/data3"])

# Find similar datasets
find_similar_datasets(reference_path="/template", threshold=0.8)
```

### Discovery & Optimization
```python
# Get exploration suggestions
suggest_next_exploration(current_path="/results/")

# Identify performance bottlenecks
identify_io_bottlenecks()

# Optimize access patterns
optimize_access_pattern(dataset_path="/data", access_pattern="sequential")
```

## Tool Categories

| Category | Tools | Description |
|----------|-------|-------------|
| **File** | open_file, close_file, get_filename, get_mode, get_by_path, list_keys, visit | File management and navigation |
| **Dataset** | read_full, read_partial, get_shape, get_dtype, get_size, get_chunks, export_dataset | Dataset operations, metadata, and export |
| **Attribute** | read_attribute, list_attributes | Metadata access |
| **Performance** | parallel_scan, batch_read, stream_data, aggregate_stats | High-performance parallel operations |
| **Discovery** | analyze_structure, find_similar, suggest_exploration, identify_bottlenecks, optimize_access | AI-powered intelligent exploration |
| **Admin** | refresh_hdf5_resources, list_available_hdf5_files | File discovery and management |

## Architecture (FastMCP v2.0)

### Core Components

**Server Layer** (server.py)
- FastMCP decorators: `@mcp.tool()`, `@mcp.resource()`, `@mcp.prompt()`
- Lifespan management for startup/shutdown
- Context integration (progress, LLM sampling, elicitation)
- 26+ tools with tags and annotations
- Consistent error handling (ToolError/ResourceError)

**Resource Layer** (resources.py)
- ResourceManager with LRU caching (1000 items)
- Lazy loading with LazyHDF5Proxy
- Client roots integration and file discovery
- Thread-safe operations with locks
- Automatic cleanup on shutdown

**Utilities** (utils.py)
- PerformanceMonitor with nanosecond precision
- FileHandleCache with time-based expiry
- HDF5Manager context manager
- Adaptive units (ns, μs, ms, s)

**Configuration** (config.py)
- Pydantic-based validation
- Environment variable support
- Runtime updates
- Hierarchical configuration

### Key Features
- **Zero Boilerplate** - FastMCP decorators eliminate registration code
- **LRU Caching** - 100-1000x speedup on repeated queries
- **Parallel Processing** - 4-8x faster batch operations
- **AI Integration** - LLM sampling for insights and recommendations
- **Client Roots** - Automatic discovery from client directories
- **Streaming** - Unlimited file size support

## Performance

```
Repeated Queries:    100-1000x faster (LRU cache)
Batch Operations:    4-8x faster (parallel processing)
Directory Scans:     3-5x faster (multi-threaded)
Large Files:         Unlimited (streaming)
```

## Configuration

Environment variables:
```bash
HDF5_DATA_DIR=/path/to/data          # Default data directory
HDF5_CACHE_SIZE=1000                 # LRU cache capacity
HDF5_NUM_WORKERS=4                   # Parallel worker count
HDF5_SHOW_PERFORMANCE=false          # Show timing in results (true for dev/debug)
```

**Performance Measurement**:
- Always captured with nanosecond precision
- Adaptive units (ns, μs, ms, s)
- Hidden by default (production)
- Enable with `HDF5_SHOW_PERFORMANCE=true` for debugging

## Transport Support

### stdio (Default)
```bash
uvx iowarp-agent-toolkit mcp-server hdf5
```
For local AI assistants (Claude Code, Cursor). Simple subprocess mode.

### SSE/HTTP (Advanced)
```bash
uvx iowarp-agent-toolkit mcp-server hdf5 --transport sse --port 8765
```
For streaming large datasets, multiple clients, remote servers.

**MCP Protocol 2025-06-18 Compliant**:
- ✅ Session management (`Mcp-Session-Id`)
- ✅ Resumable streams (`Last-Event-ID`)
- ✅ Origin validation (security)
- ✅ Protocol version negotiation

See [docs/TRANSPORTS.md](docs/TRANSPORTS.md) for details.

## Documentation

- **[TOOLS.md](docs/TOOLS.md)** - Complete tool reference (all 25 tools)
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design and components
- **[EXAMPLES.md](docs/EXAMPLES.md)** - Usage examples and workflows
- **[TRANSPORTS.md](docs/TRANSPORTS.md)** - Transport configuration and protocol details

## Requirements

- Python >= 3.10
- fastmcp >= 0.2.0
- h5py >= 3.9.0
- numpy >= 1.24.0, <2.0.0
- pydantic >= 2.4.2, <3.0.0
- psutil >= 5.9.0
- python-dotenv >= 0.19.0

## Advanced Features

**Resource Management**:
- Lazy loading (on-demand file opening)
- LRU caching (100-1000x speedup on repeated queries)
- File handle pooling

**Parallel Processing**:
- Multi-threaded batch operations
- Parallel directory scanning
- Configurable worker count

**Streaming**:
- Memory-bounded chunked reading
- Handle 100GB+ files
- Per-chunk statistics

**Discovery**:
- Find similar datasets
- Suggest exploration paths
- Identify performance bottlenecks

## License

MIT

---

**Part of [Agent Toolkit](https://github.com/iowarp/agent-toolkit)** - Scientific computing tools for AI agents

**Status**: v2.0.0 - Production Ready with FastMCP

## About IoWarp

IoWarp is a collection of Model Context Protocol (MCP) servers designed specifically for AI agents working with scientific and engineering data. Our servers provide:

- **Zero-boilerplate FastMCP architecture**
- **AI-first design** with LLM sampling and progress reporting
- **Production-grade performance** with caching and parallelization
- **Comprehensive tooling** for domain-specific workflows
- **Consistent interfaces** across all scientific data formats

The HDF5 server is the flagship implementation, showcasing the full power of FastMCP for scientific computing.
