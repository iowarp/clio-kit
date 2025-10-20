# FastMCP 2.11.3 - Latest Features Guide

**Source:** Official docs via Context7 (/jlowin/fastmcp)
**Date:** October 19, 2025
**For:** IoWarp MCP development reference

---

## Quick Reference: What's New in FastMCP 2.x

### Context API Enhancements

#### 1. LLM Sampling - Ask client's LLM for completions

```python
@mcp.tool
async def generate_summary(text: str, ctx: Context) -> str:
    """Use client's LLM for summaries."""
    response = await ctx.sample(
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
        max_tokens=100
    )
    return response.content
```

**Use cases for HDF5 MCP:**
- `analyze_dataset_structure()` - Generate AI insights about data patterns
- `suggest_next_exploration()` - AI-powered recommendations
- `identify_io_bottlenecks()` - AI analysis of performance issues

#### 2. Interactive Input - Request user choices

```python
@mcp.tool
async def export_dataset(path: str, ctx: Context) -> str:
    """Interactive export with format selection."""
    result = await ctx.elicit("Choose format:", response_type=str)
    if result.action == "accept":
        format = result.data
        # Export with chosen format
```

**Use cases for HDF5 MCP:**
- Already implemented in `export_dataset` ✅
- Could add to other interactive workflows

#### 3. Progress Reporting - Show operation status

```python
@mcp.tool
async def long_operation(items: list, ctx: Context) -> dict:
    """Report progress for visibility."""
    total = len(items)
    for i, item in enumerate(items):
        await process(item)
        await ctx.report_progress(
            progress=i + 1,
            total=total,
            message=f"Processed {i+1}/{total}"
        )
    return {"processed": total}
```

**Use cases for HDF5 MCP:**
- `hdf5_parallel_scan` - Show file scan progress
- `hdf5_stream_data` - Show streaming chunks progress
- `hdf5_batch_read` - Show parallel read progress
- `hdf5_aggregate_stats` - Show computation progress

#### 4. State Management - Persist data across requests

```python
@mcp.tool
def store_preference(key: str, value: str, ctx: Context) -> str:
    """Store user preferences."""
    ctx.set_state(key, value)
    return f"Stored {key}"

@mcp.tool
def get_preference(key: str, ctx: Context) -> str:
    """Retrieve preferences."""
    return ctx.get_state(key, default="not set")
```

**Use cases for HDF5 MCP:**
- Remember last opened file
- Store user's preferred export format
- Cache analysis parameters

#### 5. Request Metadata - Access client info

```python
@mcp.tool
def debug_info(ctx: Context) -> dict:
    """Get request information."""
    return {
        "request_id": ctx.request_id,
        "client_id": ctx.client_id,
        "has_sampling": ctx.has_sampling_capability
    }
```

---

### Advanced Annotations

```python
@mcp.tool(
    tags={"dataset", "read"},
    annotations={
        "title": "Read Full Dataset",          # Display name for LLM
        "readOnlyHint": True,                   # No side effects
        "idempotentHint": True,                 # Same result on repeat
        "destructiveHint": False,               # Won't delete data
        "openWorldHint": False                  # No external API calls
    }
)
def read_full_dataset(path: str) -> dict:
    """Read complete dataset."""
    return {}
```

**Current HDF5 usage:** Only `readOnlyHint` used
**Opportunity:** Add all 5 hints for better LLM understanding

---

### Resource Template Query Parameters

```python
# NEW: Query parameters for optional filters
@mcp.resource("hdf5://{filename}/datasets{?pattern,limit}")
def list_datasets(filename: str, pattern: str = "*", limit: int = 100) -> dict:
    """List datasets with optional filtering."""
    datasets = get_all_datasets(filename)
    if pattern != "*":
        datasets = [d for d in datasets if fnmatch(d, pattern)]
    return {"datasets": datasets[:limit]}
```

**Current HDF5:** Not using query parameters
**Opportunity:** Add filtering to resource URIs

---

### Middleware for Cross-Cutting Concerns

```python
from fastmcp.server.middleware import Middleware, MiddlewareContext

class PerformanceMiddleware(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        start = time.perf_counter()
        result = await call_next(context)
        elapsed = time.perf_counter() - start

        # Could log, store metrics, etc.
        logger.info(f"Tool {context.message.name} took {elapsed:.2f}s")
        return result

# Add to server
mcp.add_middleware(PerformanceMiddleware())
```

**Current HDF5:** Using custom decorators (`@with_performance_tracking`)
**Opportunity:** Migrate to middleware for cleaner code

---

## h5py 3.15.1 New Features (vs 3.9.0)

### 1. iter_chunks() - Efficient Chunk Iteration

```python
# OLD way (manual)
chunk_shape = dset.chunks
for i in range(0, dset.shape[0], chunk_shape[0]):
    for j in range(0, dset.shape[1], chunk_shape[1]):
        chunk = dset[i:i+chunk_shape[0], j:j+chunk_shape[1]]

# NEW way (efficient)
for chunk_slice in dset.iter_chunks():
    chunk = dset[chunk_slice]  # Much faster!
```

**Use in HDF5 MCP:** Improve `hdf5_stream_data` performance

### 2. nbytes - Accurate Size Reporting

```python
# More accurate than dset.size * dset.dtype.itemsize
actual_bytes = dset.nbytes  # Accounts for compression, fill values, etc.
```

**Use in HDF5 MCP:** Better reporting in `get_size()` and performance monitoring

### 3. Per-Dataset Chunk Cache

```python
# Tune cache for specific dataset access patterns
dset = f.create_dataset(
    "data",
    shape=(10000, 10000),
    chunks=(100, 100),
    rdcc_nbytes=10*1024*1024,  # 10MB cache
    rdcc_w0=0.75,               # Write policy
    rdcc_nslots=1009            # Hash slots
)
```

**Use in HDF5 MCP:** Optimize `hdf5_batch_read` for parallel access

### 4. astype() - Zero-Copy Type Conversion

```python
# Efficient type conversion on read
with dset.astype('float32'):
    data = dset[:]  # Read as float32, no intermediate copy
```

**Use in HDF5 MCP:** Improve `export_dataset` performance

---

## Implementation Quality Score

| Aspect | Score | Notes |
|--------|-------|-------|
| **FastMCP Usage** | 7/10 | Good basics, missing 2.x features |
| **h5py Usage** | 8/10 | Solid, missing iter_chunks |
| **Context API** | 5/10 | 19/27 tools use it, but only 6 method calls |
| **Error Handling** | 9/10 | Excellent, consistent patterns |
| **Performance** | 8/10 | Good caching/parallel, could optimize more |
| **Documentation** | 10/10 | Comprehensive, well-maintained |
| **Testing** | 4/10 | Basic tests only, no integration tests |
| **Code Organization** | 9/10 | Clean modules, good separation |

**Overall:** 7.5/10 - Solid foundation, needs modernization

---

## Comparison: HDF5 MCP vs FastMCP Best Practices

### Tool Definition

**Current (HDF5):**
```python
@mcp.tool(tags={"file", "core"}, annotations={"readOnlyHint": False})
@with_error_handling
@with_performance_tracking
async def open_file(path: str, mode: str = 'r') -> str:
    """Open an HDF5 file for operations."""
    # Implementation
```

**Best Practice (FastMCP 2.11.3):**
```python
@mcp.tool(
    tags={"file", "core"},
    annotations={
        "title": "Open HDF5 File",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def open_file(path: str, mode: str = 'r', ctx: Context) -> dict:
    """Open an HDF5 file for operations."""
    await ctx.info(f"Opening {path} in mode {mode}")

    start = time.perf_counter()
    # Implementation
    elapsed = time.perf_counter() - start

    await ctx.report_progress(100, 100, "File opened")

    return {
        "status": "opened",
        "path": path,
        "mode": mode,
        "_perf": {"elapsed_ms": elapsed * 1000}
    }
```

**Changes:**
- Return dict instead of string (more structured)
- Use ctx.info() instead of logger
- Use ctx.report_progress() for UX
- Add all 5 annotation hints
- Include performance in returned data

---

## Stored:**
- 2025-10-19
