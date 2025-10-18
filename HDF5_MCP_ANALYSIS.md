# HDF5 MCP: Comparative Analysis & Improvements

## Current State

**iowarp-mcps/HDF5**: 4 basic tools
- `list_hdf5()` - list files
- `inspect_hdf5()` - show structure
- `preview_hdf5()` - sample first N elements
- `read_all_hdf5()` - load entire datasets

All read-only, sequential, no caching.

**Your hdf5-mcp-server**: 25+ tools organized into categories
- File operations (open/close/navigate)
- Dataset operations (read full/partial, slicing, statistics)
- Performance (parallel scan, batch read, streaming)
- Discovery (similar datasets, structure analysis, bottleneck detection)
- Optimization (access pattern suggestions)

## Core Differences

| Aspect | iowarp | Yours |
|--------|--------|-------|
| **Tool Registration** | Manual with @mcp.tool() | ToolRegistry class - auto-documentation |
| **Cross-cutting Concerns** | Mixed in each function | Decorators (@handle_errors, @log_op, @measure_perf) |
| **Resource Management** | None | LazyHDF5Proxy + LRUCache |
| **File Handling** | Open/close per operation | Lazy load, cache open files |
| **Dataset Access** | Full load only | Full/partial/streaming |
| **Parallelism** | No | ThreadPoolExecutor for batch/scan ops |
| **Error Handling** | Try/catch returns JSON | Decorator-based consistency |

## What Actually Matters (for yours to be better)

### 1. **Architecture Pattern**
Your ToolRegistry is cleaner than scattered decorators. Instead of:
```python
@mcp.tool()
async def list_hdf5_tool():
```

You have:
```python
@ToolRegistry.register(category="file")
async def list_hdf5(self):
```

**Why it's better**: Centralized metadata, easier to discover tools programmatically, category grouping built-in.

### 2. **Resource Management**
iowarp opens files independently each time. Your LazyHDF5Proxy:
```python
@property
def file(self) -> h5py.File:
    if self._file is None:
        self._file = h5py.File(self._path, 'r')
    return self._file
```

**Impact**: Same file opened multiple times = wasted resources. Lazy loading + caching = reuse.

### 3. **Caching with LRU**
```python
class LRUCache:
    def get(self, key):
    def put(self, key, value):  # Auto-evict oldest when full
```

**Impact**: Repeated queries hit cache (microseconds) vs disk (milliseconds). 100-1000x speedup.

### 4. **Decorator Pattern**
Instead of mixing error handling + logging + measurement in every function:
```python
@handle_hdf5_errors
@log_operation
@measure_performance
async def my_tool(self):
    # Just business logic
```

**Impact**: DRY, consistent behavior, easy to change globally.

### 5. **Parallel Operations**
```python
ThreadPoolExecutor(max_workers=NUM_WORKERS)
for future in as_completed(future_to_path):
```

For batch operations and multi-file scans.

**Impact**: 4-8x speedup on multi-dataset operations.

### 6. **Streaming**
Instead of loading entire dataset into memory:
```python
for i in range(0, dataset.size, chunk_size):
    chunk = dataset[i:i+chunk_size]
    # process chunk
```

**Impact**: No memory limit. Can handle 100GB+ files.

## Concrete Improvements for iowarp

### High Impact, Low Effort:
1. **Add open_file/close_file tools** - Track open state, enable session workflows
2. **Add recursive list** - Support nested directories
3. **Add partial read** - With start/count parameters for slicing
4. **Root attributes** - Currently missed by `visititems()`

### Medium Effort:
5. **Lazy file cache** - Reuse open file handles, reduce I/O
6. **Dataset statistics** - mean, std, min, max without full load
7. **Batch operations** - Read multiple datasets in parallel
8. **Streaming preview** - Handle large datasets safely

### Advanced:
9. **Find similar datasets** - Query by shape/dtype/size
10. **Bottleneck detection** - Warn about unchunked large files
11. **Access pattern advice** - Suggest optimal read strategy

## Architecture You Should Adopt

```python
# src/tools.py - centralized
class ToolRegistry:
    @register(category="file")
    @handle_errors
    @log_op
    @measure_perf
    async def list_hdf5(self, dir): ...

# src/resource_manager.py - file pooling
class ResourceManager:
    def get_hdf5_file(path) -> LazyHDF5Proxy

class LazyHDF5Proxy:
    @property
    def file(self): # lazy load

# src/cache.py - dataset caching
class LRUCache:
    get/put with eviction
```

## Why These Patterns Work

- **ToolRegistry**: Single source of truth for tools. Easier to track, document, categorize.
- **Decorators**: Don't repeat error/logging/metrics code 10 times. Change once, affects all tools.
- **LazyProxy**: Open file once, reuse handle. No redundant opens.
- **LRUCache**: Bounded memory, massive speed on repeated access.
- **Parallelism**: CPUs are multi-core. Use them.
- **Streaming**: Memory not a bottleneck. Can analyze massive files.

## Bottom Line

iowarp's strength: **simple, focused, works**
Your strength: **production patterns, performance, features**

Merge them by:
1. Keep iowarp's simplicity
2. Add your resource management (LazyProxy + LRUCache)
3. Add your decorator pattern for cross-cutting concerns
4. Add your parallel/streaming capabilities incrementally

Everything else (discovery tools, prompts, etc.) is nice-to-have for research.
