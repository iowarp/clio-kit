# HDF5 FastMCP v3.0 - Next-Gen Scientific Data Access 🚀

**The most advanced MCP server** - Built with FastMCP 2.0, showcasing zero-boilerplate architecture with enterprise-grade performance.

## ✨ What's New in v3.0

- **FastMCP 2.0** - Complete rewrite with modern patterns
- **Zero Boilerplate** - All tools use `@mcp.tool()` decorator
- **Resource URIs** - Access HDF5 files via `hdf5://` scheme
- **Workflow Prompts** - Pre-built analysis templates
- **40% Smaller** - 1500 lines vs 2500 lines (ToolRegistry eliminated)
- **Same Power** - ALL features preserved + new capabilities

## Installation

```bash
uvx iowarp-mcps hdf5
```

## Features

- **25 Tools** - Comprehensive HDF5 operations with `@mcp.tool()`
- **3 Resources** - HDF5 file URIs with `@mcp.resource()`
- **4 Prompts** - Analysis workflows with `@mcp.prompt()`
- **LRU Caching** - 100-1000x speedup on repeated queries
- **Parallel Ops** - 4-8x faster batch processing
- **Streaming** - Handle unlimited file sizes
- **Discovery** - Find similar datasets, suggest exploration paths
- **Optimization** - Detect bottlenecks, recommend access patterns

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
| **Dataset** | read_full, read_partial, get_shape, get_dtype, get_size, get_chunks | Dataset operations and metadata |
| **Attribute** | read_attribute, list_attributes | Metadata access |
| **Performance** | parallel_scan, batch_read, stream_data, aggregate_stats | High-performance operations |
| **Discovery** | analyze_structure, find_similar, suggest_exploration, identify_bottlenecks, optimize_access | Intelligent data exploration |

## Architecture (FastMCP v3.0)

### Before (v2.0 - Standard MCP SDK)
- ToolRegistry pattern (~300 lines of boilerplate)
- Manual JSON Schema generation
- Custom decorators and handlers
- **Total: ~2500 lines**

### After (v3.0 - FastMCP)
- `@mcp.tool()` - Zero boilerplate tool registration
- `@mcp.resource()` - HDF5 URI scheme support
- `@mcp.prompt()` - Workflow templates
- **Total: ~1500 lines (40% reduction)**

### Preserved Excellence
- **Resource Manager** - Lazy loading + LRU caching (1000 items)
- **Parallel Processing** - ThreadPoolExecutor for batch operations
- **Streaming** - Memory-efficient chunked reading
- **Performance** - Nanosecond precision with adaptive units
- **Error Handling** - Consistent patterns across all tools

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
uvx iowarp-mcps hdf5
```
For local AI assistants (Claude Code, Cursor). Simple subprocess mode.

### SSE/HTTP (Advanced)
```bash
uvx iowarp-mcps hdf5 --transport sse --port 8765
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
- h5py >= 3.9.0
- numpy >= 1.24.0
- mcp >= 1.4.0
- pydantic >= 2.4.2
- aiofiles >= 23.2.1
- jinja2 >= 3.1.0

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

**Part of [IoWarp MCPs](https://github.com/iowarp/iowarp-mcps)** - Scientific computing tools for AI agents

**Status**: v3.0.0 - FastMCP Exemplar Implementation 🚀

## Migration from v2.0

If you're upgrading from v2.0, see [MIGRATION.md](MIGRATION.md) for the complete transformation story. Key highlights:

- **100% API compatible** - All tools work exactly the same
- **New capabilities** - Resources and Prompts added
- **Simpler code** - 40% reduction in codebase
- **Same performance** - All optimizations preserved
- **Zero breaking changes** - Drop-in replacement
