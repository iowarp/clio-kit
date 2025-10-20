# HDF5 MCP Usage Examples

## Basic Operations

### List Available Files
```python
# List all HDF5 files in directory
result = list_hdf5(directory="data/")
# Returns: ["data/experiment1.h5", "data/experiment2.h5"]
```

### Open and Inspect File
```python
# Open file
open_file(path="data/simulation.h5")

# Inspect structure
analyze_dataset_structure(path="/")
# Shows groups, datasets, hierarchy

# Navigate to specific path
get_by_path(path="/results/temperature")
# Returns: "Dataset: /results/temperature, shape: (1000, 500), dtype: float64"
```

### Read Dataset
```python
# Read entire dataset
read_full_dataset(path="/results/temperature")

# Read partial dataset (slicing)
read_partial_dataset(
    path="/results/temperature",
    start=[0, 0],
    count=[100, 500]  # First 100 rows, all columns
)

# Get dataset metadata
get_shape(path="/results/temperature")  # (1000, 500)
get_dtype(path="/results/temperature")  # float64
get_size(path="/results/temperature")   # 500000
```

## Advanced Features

### Streaming Large Datasets
```python
# Stream 10GB dataset in chunks
hdf5_stream_data(
    path="/large_simulation/data",
    chunk_size=100000,     # 100K elements per chunk
    max_chunks=100         # Process 100 chunks
)

# Output includes per-chunk statistics:
# - Mean, std, min, max
# - Progress tracking
# - Memory usage: Only 1 chunk in memory at a time
```

### Batch Operations
```python
# Read multiple datasets in parallel
hdf5_batch_read(
    paths=[
        "/experiment1/temperature",
        "/experiment1/pressure",
        "/experiment1/velocity",
        "/experiment2/temperature",
    ],
    slice_spec="0:1000"  # First 1000 elements of each
)

# Output shows:
# - Which datasets succeeded/failed
# - Shape, dtype, size for each
# - Total data read
# - Parallel speedup (4-8x)
```

### Aggregate Statistics
```python
# Compute statistics across multiple datasets
hdf5_aggregate_stats(
    paths=[
        "/exp1/data",
        "/exp2/data",
        "/exp3/data"
    ],
    stats=["mean", "std", "min", "max", "sum"]
)

# Returns per-dataset stats + cross-dataset aggregation
# For large datasets (>500MB), automatically samples for efficiency
```

### Parallel Directory Scanning
```python
# Scan 100s of files quickly
hdf5_parallel_scan(
    directory="/simulation_results",
    pattern="**/*.h5"  # Recursive
)

# Output:
# - Files processed count
# - Total datasets found
# - Total size
# - Summary per file (first 10)
# - Uses multi-threading (3-5x faster than sequential)
```

## Discovery & Exploration

### Find Similar Datasets
```python
# Find datasets with similar characteristics
find_similar_datasets(
    reference_path="/template/data",
    similarity_threshold=0.8
)

# Similarity based on:
# - Shape matching
# - Data type matching
# - Size similarity

# Returns ranked list of similar datasets
```

### Smart Exploration
```python
# Get AI-suggested exploration paths
suggest_next_exploration(current_path="/results/")

# Returns scored suggestions based on:
# - Dataset size (optimal for exploration: 1-100MB)
# - Dimensionality (2D/3D more interesting)
# - Naming patterns (contains "data", "result")
# - Number of children (for groups)
```

### Performance Analysis
```python
# Identify I/O bottlenecks
identify_io_bottlenecks()

# Detects:
# - Large datasets without chunking
# - Very small chunk sizes
# - Missing compression
# - High-dimensional arrays

# Optimize access patterns
optimize_access_pattern(
    dataset_path="/large_data",
    access_pattern="random"  # or "sequential", "batch"
)

# Provides recommendations:
# - Chunk size suggestions
# - Read strategy
# - Tool selection
```

## Workflow Examples

### Explore Unknown HDF5 File
```python
# Step 1: Open file
open_file(path="unknown_file.h5")

# Step 2: Analyze structure
analyze_dataset_structure(path="/")
# Shows: 15 groups, 47 datasets, hierarchical organization

# Step 3: Get exploration suggestions
suggest_next_exploration(current_path="/")
# Suggests: /results/primary_data (score: 5, 25MB, 2D array)

# Step 4: Investigate suggested dataset
read_partial_dataset(
    path="/results/primary_data",
    start=[0, 0],
    count=[10, 10]  # Preview 10×10 subset
)

# Step 5: Find related data
find_similar_datasets(
    reference_path="/results/primary_data",
    threshold=0.7
)
# Finds: /results/secondary_data, /processed/output_data

# Step 6: Clean up
close_file()
```

### Process Multiple Large Files
```python
# Step 1: Scan directory
hdf5_parallel_scan(
    directory="/simulation_outputs",
    pattern="*.h5"
)
# Found: 50 files, 200 datasets, 150GB total

# Step 2: Batch read specific datasets
paths = [
    "/file1.h5/temperature",
    "/file2.h5/temperature",
    "/file3.h5/temperature"
]

hdf5_batch_read(paths=paths, parallel=True)
# 4-8x faster than sequential reads

# Step 3: Aggregate statistics
hdf5_aggregate_stats(
    paths=paths,
    stats=["mean", "std", "min", "max"]
)
# Per-dataset stats + global aggregation
```

### Stream Processing Large Dataset
```python
# Open 100GB file
open_file(path="/massive_simulation.h5")

# Check for bottlenecks first
identify_io_bottlenecks()
# ⚠️  /data/simulation (95GB) without chunking

# Stream with optimal chunk size
hdf5_stream_data(
    path="/data/simulation",
    chunk_size=100000,   # 100K elements
    max_chunks=1000      # Process 1000 chunks
)

# Memory usage: Only ~10MB per chunk
# Processing: Statistics computed per chunk
# Output: Aggregated chunk statistics
```

## Attribute Operations

### Read Metadata
```python
# List all attributes at a path
list_attributes(path="/experiment/metadata")

# Returns:
# {
#   "experiment_name": "Test Run 42",
#   "date": "2025-10-18",
#   "temperature": 298.15
# }

# Read specific attribute
read_attribute(
    path="/experiment/metadata",
    name="experiment_name"
)
# Returns: "Test Run 42"
```

## Transport-Specific Examples

### stdio Mode (Default)
```bash
# Run server
uvx iowarp-mcps hdf5 --data-dir ./my_data

# Server automatically handles stdio communication
# Client (Claude Code, Cursor) sends JSON-RPC via stdin
# Server responds via stdout
```

### SSE Mode (Advanced)
```bash
# Start SSE server
uvx iowarp-mcps hdf5 --transport sse --port 8765

# In another terminal, interact via HTTP:

# Health check
curl http://localhost:8765/health

# Open SSE stream
curl -N -H "Accept: text/event-stream" \
     -H "MCP-Protocol-Version: 2025-06-18" \
     http://localhost:8765/mcp

# Send request (in another terminal)
curl -X POST http://localhost:8765/mcp \
     -H "Content-Type: application/json" \
     -H "Accept: application/json" \
     -H "MCP-Protocol-Version: 2025-06-18" \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

### Resumable Streaming
```bash
# Initial connection
curl -N -H "Accept: text/event-stream" \
     -H "MCP-Protocol-Version: 2025-06-18" \
     -H "Mcp-Session-Id: <session-uuid>" \
     http://localhost:8765/mcp

# If connection drops, resume from last event:
curl -N -H "Accept: text/event-stream" \
     -H "MCP-Protocol-Version: 2025-06-18" \
     -H "Mcp-Session-Id: <session-uuid>" \
     -H "Last-Event-ID: client_1_42" \
     http://localhost:8765/mcp
```

## Performance Benchmarks

### Caching Impact
```
First query:  read_full_dataset("/data") → 150ms (disk read)
Second query: read_full_dataset("/data") → 1ms (cache hit)
Speedup: 150x
```

### Batch Operations
```
Sequential: 10 datasets × 100ms = 1000ms
Parallel:   10 datasets / 4 workers = 280ms
Speedup: 3.5x
```

### Directory Scanning
```
Sequential: 100 files × 50ms = 5000ms
Parallel:   100 files / 4 workers = 1300ms
Speedup: 3.8x
```

### Streaming
```
Without: 100GB file → Out of memory
With:    100GB file → 10MB memory usage
Efficiency: 10,000x reduction
```

## Error Handling

All tools have consistent error handling:

```python
# On error, tools return:
{
  "error": "Detailed error message",
  "error_type": "FileNotFoundError",
  "path": "/nonexistent/dataset"
}

# Logging includes:
# - Timestamp
# - Tool name
# - Error details
# - Stack trace (DEBUG level)
```

## Configuration Examples

### Environment Variables
```bash
export HDF5_DATA_DIR=/data/simulations
export HDF5_CACHE_SIZE=2000
export HDF5_NUM_WORKERS=8
export HDF5_TRANSPORT=sse
export HDF5_SSE_PORT=9000

uvx iowarp-mcps hdf5
```

### CLI Override
```bash
# Override environment variables
uvx iowarp-mcps hdf5 \
  --data-dir /custom/path \
  --transport sse \
  --port 8888 \
  --log-level DEBUG
```

## Common Patterns

### Pattern 1: Quick Data Preview
```python
open_file("experiment.h5")
analyze_dataset_structure("/")
suggest_next_exploration("/")
read_partial_dataset(path="/interesting/dataset", start=[0], count=[100])
close_file()
```

### Pattern 2: Batch Statistical Analysis
```python
paths = ["/exp1/data", "/exp2/data", "/exp3/data"]
hdf5_aggregate_stats(paths, stats=["mean", "std", "min", "max"])
```

### Pattern 3: Large File Processing
```python
open_file("massive.h5")
identify_io_bottlenecks()  # Check for issues
hdf5_stream_data(path="/data", chunk_size=50000)  # Stream safely
close_file()
```

## References

- Transport details: `TRANSPORTS.md`
- Architecture: `ARCHITECTURE.md`
- Tool reference: `README.md`
