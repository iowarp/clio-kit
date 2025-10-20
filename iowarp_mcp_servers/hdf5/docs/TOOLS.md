# HDF5 MCP Tool Reference

Complete reference for all 25 tools available in HDF5 MCP v2.0.

## File Operations (4 tools)

### `open_file`
Open an HDF5 file with lazy loading.

**Parameters**:
- `path` (str): Path to HDF5 file
- `mode` (str, optional): Access mode (default: 'r')

**Returns**: Success message with filename and mode

**Example**:
```python
open_file(path="/data/simulation.h5", mode="r")
```

---

### `close_file`
Close the currently open file.

**Parameters**: None

**Returns**: Confirmation message with filename

**Example**:
```python
close_file()
```

---

### `get_filename`
Get the current file's path.

**Parameters**: None

**Returns**: Current filename or "No file currently open"

---

### `get_mode`
Get the current file's access mode.

**Parameters**: None

**Returns**: Access mode ('r', 'r+', etc.) or "No file currently open"

---

## Navigation (4 tools)

### `get_by_path`
Get a dataset or group by path.

**Parameters**:
- `path` (str): Path to object within file

**Returns**: Object info (type, shape, dtype, keys)

**Example**:
```python
get_by_path(path="/results/temperature")
# → "Dataset: /results/temperature, shape: (1000, 500), dtype: float64"
```

---

### `list_keys`
List keys in the current group.

**Parameters**: None

**Returns**: JSON list of keys

---

### `visit`
Visit all nodes recursively.

**Parameters**:
- `callback_fn` (str): Callback function name

**Returns**: JSON list of all paths with types

---

### `visitnodes`
Visit items in the current group.

**Parameters**:
- `callback_fn` (str): Callback function name

**Returns**: JSON list of paths

---

## Dataset Operations (6 tools)

### `read_full_dataset`
Read an entire dataset with efficient chunked reading for large datasets.

**Parameters**:
- `path` (str): Path to dataset

**Returns**: Success message with data description

**Features**:
- Automatic chunked reading for datasets >100MB
- Memory-efficient

**Example**:
```python
read_full_dataset(path="/experiment/data")
```

---

### `read_partial_dataset`
Read a portion of a dataset with slicing.

**Parameters**:
- `path` (str): Path to dataset
- `start` (List[int], optional): Starting indices
- `count` (List[int], optional): Number of elements per dimension

**Returns**: Dataset slice with metadata

**Example**:
```python
read_partial_dataset(
    path="/data",
    start=[0, 0],
    count=[100, 50]  # Read 100×50 subset
)
```

---

### `get_shape`
Get the shape of a dataset.

**Parameters**:
- `path` (str): Path to dataset

**Returns**: Shape tuple as string

---

### `get_dtype`
Get the data type of a dataset.

**Parameters**:
- `path` (str): Path to dataset

**Returns**: Data type as string

---

### `get_size`
Get the size (total elements) of a dataset.

**Parameters**:
- `path` (str): Path to dataset

**Returns**: Total number of elements

---

### `get_chunks`
Get chunk information for a dataset.

**Parameters**:
- `path` (str): Path to dataset

**Returns**: Chunk shape and size in KB

**Example**:
```python
get_chunks(path="/large_dataset")
# → Chunk shape: (1000, 100), Chunk size: 781.25 KB
```

---

## Attribute Operations (2 tools)

### `read_attribute`
Read a specific attribute from an object.

**Parameters**:
- `path` (str): Path to object
- `name` (str): Attribute name

**Returns**: Attribute value

**Example**:
```python
read_attribute(path="/experiment", name="temperature")
# → "298.15"
```

---

### `list_attributes`
List all attributes of an object.

**Parameters**:
- `path` (str): Path to object

**Returns**: JSON dict of all attributes

**Example**:
```python
list_attributes(path="/experiment/metadata")
# → {"experiment_name": "Test 42", "date": "2025-10-18"}
```

---

## Performance Tools (4 tools)

### `hdf5_parallel_scan`
Fast multi-file scanning with parallel processing.

**Parameters**:
- `directory` (str): Directory to scan
- `pattern` (str, optional): File pattern (default: "*.h5")

**Returns**: Summary of scanned files with dataset counts and sizes

**Performance**: 3-5x faster than sequential scanning

**Example**:
```python
hdf5_parallel_scan(directory="/simulations", pattern="**/*.h5")
# → Scanned 100 files, 500 datasets, 50GB total (parallel workers: 4)
```

---

### `hdf5_batch_read`
Read multiple datasets in one call with parallel processing.

**Parameters**:
- `paths` (List[str]): List of dataset paths
- `slice_spec` (str, optional): Slice specification (e.g., "0:100")

**Returns**: Summary of batch read with per-dataset info

**Performance**: 4-8x faster than sequential reads

**Example**:
```python
hdf5_batch_read(
    paths=["/data1", "/data2", "/data3"],
    slice_spec="0:1000"
)
```

---

### `hdf5_stream_data`
Stream large datasets efficiently with memory management.

**Parameters**:
- `path` (str): Path to dataset
- `chunk_size` (int, optional): Elements per chunk (default: 1024)
- `max_chunks` (int, optional): Maximum chunks to process (default: 100)

**Returns**: Streaming report with per-chunk statistics

**Memory**: Only one chunk in memory at a time

**Example**:
```python
hdf5_stream_data(
    path="/massive_dataset",
    chunk_size=100000,
    max_chunks=500
)
# Processes 50M elements with <10MB memory
```

---

### `hdf5_aggregate_stats`
Parallel statistics computation across multiple datasets.

**Parameters**:
- `paths` (List[str]): List of dataset paths
- `stats` (List[str], optional): Statistics to compute (default: ["mean", "std", "min", "max", "sum", "count"])

**Returns**: Per-dataset stats + cross-dataset aggregation

**Features**:
- Automatic sampling for datasets >500MB
- Weighted aggregation across datasets
- Parallel computation

**Example**:
```python
hdf5_aggregate_stats(
    paths=["/exp1/data", "/exp2/data", "/exp3/data"],
    stats=["mean", "std", "min", "max"]
)
```

---

## Discovery Tools (5 tools)

### `analyze_dataset_structure`
Analyze and understand file organization and data patterns.

**Parameters**:
- `path` (str, optional): Path to analyze (default: "/")

**Returns**: Structural analysis with groups, datasets, sizes

**Example**:
```python
analyze_dataset_structure(path="/results")
# → Type: Group, Total items: 15, Groups: 3, Datasets: 12
```

---

### `find_similar_datasets`
Find datasets with similar characteristics to a reference.

**Parameters**:
- `reference_path` (str): Reference dataset path
- `similarity_threshold` (float, optional): Similarity threshold 0-1 (default: 0.8)

**Returns**: Ranked list of similar datasets

**Similarity criteria**:
- Shape matching
- Data type matching
- Size similarity

**Example**:
```python
find_similar_datasets(
    reference_path="/template/data",
    similarity_threshold=0.75
)
# → Found 5 similar datasets, ranked by similarity
```

---

### `suggest_next_exploration`
Suggest interesting data to explore based on current location.

**Parameters**:
- `current_path` (str, optional): Current location (default: "/")

**Returns**: Scored suggestions for next exploration

**Scoring based on**:
- Dataset size (1-100MB ideal)
- Dimensionality (2D/3D preferred)
- Naming patterns ("data", "result")
- Group size

**Example**:
```python
suggest_next_exploration(current_path="/")
# → 1. /results/primary_data (score: 5) - Shape: (1000, 1000), Size: 7.6 MB
```

---

### `identify_io_bottlenecks`
Identify potential I/O bottlenecks and performance issues.

**Parameters**:
- `analysis_paths` (List[str], optional): Paths to analyze (default: auto-discover)

**Returns**: List of datasets with performance issues

**Detects**:
- Large datasets without chunking
- Very small chunk sizes
- Missing compression
- High-dimensional arrays

**Example**:
```python
identify_io_bottlenecks()
# → Found 3 issues:
#   ⚠️  /large_data (500MB) without chunking
#   ⚠️  /compressed_data has very small chunks
```

---

### `optimize_access_pattern`
Suggest better approaches for data access based on usage patterns.

**Parameters**:
- `dataset_path` (str): Path to dataset
- `access_pattern` (str, optional): Pattern type (default: "sequential")
  - Options: "sequential", "random", "batch"

**Returns**: Optimization recommendations

**Example**:
```python
optimize_access_pattern(
    dataset_path="/large_dataset",
    access_pattern="random"
)
# → Random Access Recommendations:
#   • Enable chunking for better performance
#   • Chunk size (2.5 MB) is good for random access
#   • Use read_partial_dataset() with specific slices
```

---

## Tool Categories Summary

| Category | Count | Tools |
|----------|-------|-------|
| **File** | 4 | open_file, close_file, get_filename, get_mode |
| **Navigation** | 4 | get_by_path, list_keys, visit, visitnodes |
| **Dataset** | 6 | read_full_dataset, read_partial_dataset, get_shape, get_dtype, get_size, get_chunks |
| **Attribute** | 2 | read_attribute, list_attributes |
| **Performance** | 4 | hdf5_parallel_scan, hdf5_batch_read, hdf5_stream_data, hdf5_aggregate_stats |
| **Discovery** | 5 | analyze_dataset_structure, find_similar_datasets, suggest_next_exploration, identify_io_bottlenecks, optimize_access_pattern |
| **Total** | **25** | |

## Decorator Features

All tools include:

- **Error Handling** (`@handle_hdf5_errors`): Consistent error responses
- **Logging** (`@log_operation`): Standardized operation logging
- **Performance Tracking** (`@measure_performance`): Execution time measurement

## Response Format

All tools return `ToolResult` which is a list of `TextContent`:

```python
[
    TextContent(
        type="text",
        text="Result data or error message"
    )
]
```

Performance-tracked tools include timing:
```
Successfully read dataset /data: array of shape (1000, 500)

⏱️ Completed in 150ms
```

## Common Patterns

### Pattern 1: File Session
```python
open_file(path="file.h5")
# ... perform operations ...
close_file()
```

### Pattern 2: Quick Inspection
```python
analyze_dataset_structure(path="/")
suggest_next_exploration(path="/")
```

### Pattern 3: Performance-Critical
```python
identify_io_bottlenecks()
optimize_access_pattern(dataset_path="/data", access_pattern="batch")
hdf5_batch_read(paths=[...])  # Use recommended approach
```

### Pattern 4: Large Data
```python
hdf5_stream_data(path="/huge_dataset", chunk_size=50000)
```

## References

- Architecture: `ARCHITECTURE.md`
- Examples: `EXAMPLES.md`
- Transport details: `TRANSPORTS.md`
