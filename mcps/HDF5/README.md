# HDF5 MCP - Advanced Scientific Data Access

Enterprise-grade HDF5 file operations for AI agents with caching, parallel processing, and intelligent data discovery.

## Installation

```bash
uvx iowarp-mcps hdf5
```

## Features

- **25+ Tools** - Comprehensive HDF5 operations
- **Caching** - 100-1000x speedup on repeated queries
- **Parallel Ops** - 4-8x faster batch processing
- **Streaming** - Handle unlimited file sizes
- **Discovery** - Find similar datasets, suggest exploration paths
- **Optimization** - Detect bottlenecks, recommend access patterns

## Quick Start

### Basic Usage
```python
# List HDF5 files
list_hdf5(directory="data/")

# Inspect file structure
inspect_hdf5(filename="simulation.h5")

# Read dataset
read_full_dataset(path="/results/temperature")
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

## Architecture

- **Tool Registry** - Centralized tool management with auto-documentation
- **Resource Manager** - Lazy loading and LRU caching
- **Parallel Processing** - ThreadPoolExecutor for batch operations
- **Streaming** - Memory-efficient chunked reading
- **Decorators** - Consistent error handling, logging, and performance tracking

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

**Status**: v2.0.0 - Enterprise-grade multi-transport HDF5 MCP
