# Migrating from HDF5 MCP v1.0 to v2.0

## Overview

HDF5 MCP v2.0 is a major upgrade featuring 25+ tools, multi-transport support, and enterprise-grade performance.

## Breaking Changes

### ⚠️ None - Fully Backward Compatible

All v1.0 tools continue to work with the same interface.

## What's New in v2.0

### More Tools
v1.0: 4 tools
v2.0: 25 tools (6.25x increase)

**New capabilities**:
- File management (open/close)
- Dataset slicing (partial reads)
- Streaming (large files)
- Batch operations (parallel)
- Statistics (mean, std, min, max)
- Discovery (find similar, suggest)
- Optimization (bottleneck detection)

### Multi-Transport Support
v1.0: stdio only
v2.0: stdio + SSE/HTTP

**stdio** (default):
```bash
uvx iowarp-mcps hdf5
```
Same as v1.0, no changes needed.

**SSE** (new):
```bash
uvx iowarp-mcps hdf5 --transport sse --port 8765
```
For streaming large datasets, multiple clients.

### Performance Improvements
- **Caching**: 100-1000x speedup on repeated queries
- **Parallel ops**: 4-8x speedup on batch operations
- **Streaming**: Handle 100GB+ files (was memory-limited)
- **Lazy loading**: On-demand file opening

### Protocol Upgrade
v1.0: Basic MCP
v2.0: MCP 2025-06-18 specification

**New features**:
- Session management (Mcp-Session-Id)
- Resumable streams (Last-Event-ID)
- Protocol version negotiation
- Enhanced security (Origin validation)

## Migration Guide

### For End Users

**No changes required!**

Installation and usage remain the same:
```bash
uvx iowarp-mcps hdf5
```

### For Developers

If you integrated with v1.0 tools programmatically:

**v1.0 code still works**:
```python
# These continue to work exactly as before
list_hdf5(directory="data/")
inspect_hdf5(filename="file.h5")
preview_hdf5(filename="file.h5", count=10)
read_all_hdf5(filename="file.h5")
```

**New v2.0 capabilities available**:
```python
# File management (new)
open_file(path="file.h5")
close_file()

# Slicing (new)
read_partial_dataset(path="/data", start=[0], count=[100])

# Streaming (new)
hdf5_stream_data(path="/large", chunk_size=10000)

# Batch operations (new)
hdf5_batch_read(paths=["/d1", "/d2", "/d3"])

# Discovery (new)
suggest_next_exploration(current_path="/")
find_similar_datasets(reference_path="/template")
```

### For MCP Client Configurations

**No changes needed** for stdio mode:
```json
{
  "mcpServers": {
    "hdf5-mcp": {
      "command": "uvx",
      "args": ["iowarp-mcps", "hdf5"]
    }
  }
}
```

**For SSE mode** (optional):
```json
{
  "mcpServers": {
    "hdf5-mcp-sse": {
      "command": "uvx",
      "args": ["iowarp-mcps", "hdf5", "--transport", "sse", "--port", "8765"]
    }
  }
}
```

## Deprecated Features

None. All v1.0 features remain available.

## New Dependencies

v2.0 adds these dependencies:
- `mcp>=1.4.0` (replaces fastmcp)
- `numpy>=1.24.0`
- `pydantic>=2.4.2`
- `aiofiles>=23.2.1`
- `aiohttp>=3.9.0`
- `jinja2>=3.1.0`

These are automatically installed via `uvx`.

## Performance Tips

### Enable Caching
Caching is automatic. For best performance:
```python
# First call: ~150ms (disk read)
read_full_dataset(path="/data")

# Second call: ~1ms (cache hit)
read_full_dataset(path="/data")

# 150x speedup!
```

### Use Batch Operations
For multiple datasets:
```python
# Slow (sequential):
for path in paths:
    read_full_dataset(path)

# Fast (parallel):
hdf5_batch_read(paths=paths)  # 4-8x faster
```

### Stream Large Files
For datasets >100MB:
```python
# Don't:
read_full_dataset(path="/100GB_dataset")  # Out of memory!

# Do:
hdf5_stream_data(path="/100GB_dataset", chunk_size=100000)  # Works!
```

## Configuration Changes

### v1.0 Configuration
Environment variables (all still work):
```bash
HDF5_DATA_DIR=/path/to/data
```

### v2.0 Additional Options
New environment variables:
```bash
HDF5_TRANSPORT=stdio           # or sse
HDF5_CACHE_SIZE=1000           # LRU cache capacity
HDF5_NUM_WORKERS=4             # Parallel worker count
HDF5_SSE_HOST=127.0.0.1        # SSE bind address
HDF5_SSE_PORT=8765             # SSE port
```

## Troubleshooting

### Issue: Tool not found
**v1.0**: Limited to 4 tools
**v2.0**: 25 tools available

Check `docs/TOOLS.md` for complete list.

### Issue: Out of memory on large files
**v1.0**: Full file loading
**v2.0**: Use streaming

```python
hdf5_stream_data(path="/large_file")
```

### Issue: Slow multi-dataset operations
**v1.0**: Sequential only
**v2.0**: Use batch operations

```python
hdf5_batch_read(paths=[...])
```

### Issue: Want to use SSE mode
**v2.0**: Multi-transport support

```bash
uvx iowarp-mcps hdf5 --transport sse --port 8765
```

## Rollback (If Needed)

v2.0 is fully backward compatible, but if you need to rollback:

```bash
# Pin to v1.0
uvx --from iowarp-mcps==1.0.0 iowarp-mcps hdf5
```

## Getting Help

- **Documentation**: `docs/` directory
- **Tool Reference**: `docs/TOOLS.md`
- **Examples**: `docs/EXAMPLES.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Transport Details**: `docs/TRANSPORTS.md`
- **Issues**: https://github.com/iowarp/iowarp-mcps/issues

## Summary

✅ **Backward compatible** - All v1.0 tools work
✅ **More features** - 25 tools vs 4
✅ **Better performance** - Caching, parallel, streaming
✅ **Multi-transport** - stdio + SSE
✅ **Protocol compliant** - MCP 2025-06-18
✅ **No breaking changes** - Drop-in replacement

Upgrade recommended for all users.
