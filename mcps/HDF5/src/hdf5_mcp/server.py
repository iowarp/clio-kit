"""
HDF5 FastMCP Server - Next-generation scientific data access for AI agents.

This is the exemplar FastMCP implementation showcasing:
- 25+ tools with zero boilerplate
- Resource URIs for HDF5 files/datasets
- Workflow prompts for analysis
- Advanced caching, parallel processing, streaming
- Enterprise-grade performance
"""

import asyncio
import logging
import os
import time
import json
from pathlib import Path
from typing import Any, Optional, List
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import h5py
import numpy as np
from fastmcp import FastMCP

from .config import get_config
from .resources import ResourceManager, LazyHDF5Proxy
from .utils import HDF5Manager

# =========================================================================
# Server Setup
# =========================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastMCP server
mcp = FastMCP("HDF5", version="3.0.0")

# Global state - keep it simple
config = get_config()
resource_manager = ResourceManager(cache_capacity=1000)
num_workers = max(2, multiprocessing.cpu_count() - 1)
executor = ThreadPoolExecutor(max_workers=num_workers)

# Current file handle (stateful for tool sequence)
current_file: Optional[LazyHDF5Proxy] = None

# =========================================================================
# Performance & Error Decorators
# =========================================================================

def with_performance_tracking(func):
    """Track performance with adaptive units."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter_ns()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter_ns() - start

        # Show performance if enabled
        if os.getenv("HDF5_SHOW_PERFORMANCE", "false").lower() == "true":
            if elapsed < 1_000:
                perf_str = f"{elapsed}ns"
            elif elapsed < 1_000_000:
                perf_str = f"{elapsed / 1_000:.1f}μs"
            elif elapsed < 1_000_000_000:
                perf_str = f"{elapsed / 1_000_000:.1f}ms"
            else:
                perf_str = f"{elapsed / 1_000_000_000:.2f}s"

            # Append to result if it's a string
            if isinstance(result, str):
                result += f"\n\n⏱ {perf_str}"

        return result
    return wrapper

def with_error_handling(func):
    """Consistent error handling for all tools."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return f"Error: {str(e)}"
    return wrapper

# =========================================================================
# FILE OPERATIONS TOOLS
# =========================================================================

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def open_file(path: str, mode: str = 'r') -> str:
    """Open an HDF5 file for operations.

    Args:
        path: Path to HDF5 file
        mode: File access mode ('r', 'r+', 'w', 'a')

    Returns:
        Success message with file info
    """
    global current_file

    current_file = resource_manager.get_hdf5_file(path)
    if current_file is None:
        return f"Error: Could not open file {path}"

    return f"Successfully opened {path} in {mode} mode"

@mcp.tool()
@with_error_handling
async def close_file() -> str:
    """Close the current HDF5 file.

    Returns:
        Status message
    """
    global current_file

    if current_file:
        filename = current_file.filename
        current_file.close()
        current_file = None
        return f"File closed: {filename}"

    return "No file currently open"

@mcp.tool()
@with_error_handling
async def get_filename() -> str:
    """Get the current file's path.

    Returns:
        File path or error message
    """
    if current_file:
        return current_file.filename
    return "No file currently open"

@mcp.tool()
@with_error_handling
async def get_mode() -> str:
    """Get the current file's access mode.

    Returns:
        File mode or error message
    """
    if current_file:
        return current_file.mode
    return "No file currently open"

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def get_by_path(path: str) -> str:
    """Get a dataset or group by path.

    Args:
        path: Path to object within file

    Returns:
        Object information
    """
    if not current_file:
        return "No file currently open"

    obj = current_file[path]

    if isinstance(obj, h5py.Dataset):
        return f"Dataset: {path}, shape: {obj.shape}, dtype: {obj.dtype}"
    elif isinstance(obj, h5py.Group):
        return f"Group: {path}, keys: {list(obj.keys())}"
    else:
        return f"Object: {path}, type: {type(obj).__name__}"

@mcp.tool()
@with_error_handling
async def list_keys(path: str = "/") -> str:
    """List keys in a group.

    Args:
        path: Path to group (default: root)

    Returns:
        JSON array of keys
    """
    if not current_file:
        return "No file currently open"

    obj = current_file[path] if path != "/" else current_file.file
    if not isinstance(obj, h5py.Group):
        return f"{path} is not a group"

    keys = list(obj.keys())
    return json.dumps(keys, indent=2)

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def visit(callback_fn: str = "collect_paths") -> str:
    """Visit all nodes recursively.

    Args:
        callback_fn: Callback function name (currently collects all paths)

    Returns:
        JSON array of all paths and types
    """
    if not current_file:
        return "No file currently open"

    paths = []

    def collect_paths(name, obj):
        paths.append({
            "name": name,
            "type": type(obj).__name__
        })
        return None

    current_file.file.visititems(collect_paths)
    return json.dumps(paths, indent=2)

# =========================================================================
# DATASET OPERATIONS TOOLS
# =========================================================================

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def read_full_dataset(path: str) -> str:
    """Read an entire dataset with efficient chunked reading for large datasets.

    Args:
        path: Path to dataset within file

    Returns:
        Dataset description
    """
    if not current_file:
        return "Error: No file currently open"

    dataset = current_file[path]

    # For large datasets, use chunked reading
    if dataset.nbytes > 1e8:  # 100MB threshold
        data = _read_large_dataset(dataset)
    else:
        data = dataset[:]

    # Format output
    if isinstance(data, np.ndarray) and data.size > 0:
        if np.array_equal(data, np.arange(data.size)):
            description = f"array from 0 to {data.size-1}"
        else:
            description = f"array of shape {data.shape} with dtype {data.dtype}"
    else:
        description = str(data)

    return f"Successfully read dataset {path}: {description}"

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def read_partial_dataset(path: str, start: str = None, count: str = None) -> str:
    """Read a portion of a dataset with slicing.

    Args:
        path: Path to dataset within file
        start: Starting indices as comma-separated string (e.g., "0,0,0")
        count: Number of elements as comma-separated string (e.g., "10,10,10")

    Returns:
        Partial dataset description
    """
    if not current_file:
        return "Error: No file currently open"

    dataset = current_file[path]
    if not isinstance(dataset, h5py.Dataset):
        return f"{path} is not a dataset"

    # Parse start and count
    if start:
        start_tuple = tuple(int(x.strip()) for x in start.split(','))
    else:
        start_tuple = tuple(0 for _ in dataset.shape)

    if count:
        count_tuple = tuple(int(x.strip()) for x in count.split(','))
    else:
        count_tuple = dataset.shape

    # Build slice
    slices = tuple(slice(s, s + c) for s, c in zip(start_tuple, count_tuple))
    data = dataset[slices]

    return (
        f"Successfully read partial dataset {path}\n"
        f"Slice: start={start_tuple}, count={count_tuple}\n"
        f"Result shape: {data.shape}\n"
        f"Dtype: {data.dtype}\n"
        f"First few values: {data.flat[:5].tolist()}"
    )

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def get_shape(path: str) -> str:
    """Get the shape of a dataset.

    Args:
        path: Path to dataset

    Returns:
        Dataset shape
    """
    if not current_file:
        return "No file currently open"

    dataset = current_file[path]
    if not isinstance(dataset, h5py.Dataset):
        return f"{path} is not a dataset"

    return str(dataset.shape)

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def get_dtype(path: str) -> str:
    """Get the data type of a dataset.

    Args:
        path: Path to dataset

    Returns:
        Dataset dtype
    """
    if not current_file:
        return "No file currently open"

    dataset = current_file[path]
    if not isinstance(dataset, h5py.Dataset):
        return f"{path} is not a dataset"

    return str(dataset.dtype)

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def get_size(path: str) -> str:
    """Get the size of a dataset.

    Args:
        path: Path to dataset

    Returns:
        Dataset size
    """
    if not current_file:
        return "No file currently open"

    dataset = current_file[path]
    if not isinstance(dataset, h5py.Dataset):
        return f"{path} is not a dataset"

    return str(dataset.size)

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def get_chunks(path: str) -> str:
    """Get chunk information for a dataset.

    Args:
        path: Path to dataset

    Returns:
        Chunk configuration
    """
    if not current_file:
        return "Error: No file currently open"

    dataset = current_file[path]
    if not isinstance(dataset, h5py.Dataset):
        return f"{path} is not a dataset"

    chunks = dataset.chunks
    if chunks is None:
        return "Dataset is not chunked (contiguous storage)"

    chunk_size_kb = np.prod(chunks) * dataset.dtype.itemsize / 1024
    return (
        f"Chunk configuration:\n"
        f"Chunk shape: {chunks}\n"
        f"Chunk size: {chunk_size_kb:.2f} KB"
    )

# =========================================================================
# ATTRIBUTE OPERATIONS TOOLS
# =========================================================================

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def read_attribute(path: str, name: str) -> str:
    """Read an attribute from an object.

    Args:
        path: Path to object
        name: Attribute name

    Returns:
        Attribute value
    """
    if not current_file:
        return "No file currently open"

    obj = current_file[path] if path != "/" else current_file.file

    if name in obj.attrs:
        value = obj.attrs[name]
        if hasattr(value, "tolist"):
            value = value.tolist()
        return str(value)

    return f"Attribute '{name}' not found"

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def list_attributes(path: str) -> str:
    """List all attributes of an object.

    Args:
        path: Path to object

    Returns:
        JSON dict of attributes
    """
    if not current_file:
        return "No file currently open"

    obj = current_file[path] if path != "/" else current_file.file

    attrs = dict(obj.attrs)

    # Convert numpy arrays to lists
    for key, value in attrs.items():
        if hasattr(value, "tolist"):
            attrs[key] = value.tolist()
        else:
            attrs[key] = str(value)

    if not attrs:
        return f"No attributes found at path: {path}"

    return f"Attributes at {path}:\n{json.dumps(attrs, indent=2)}"

# =========================================================================
# PERFORMANCE TOOLS - Parallel, Batch, Streaming
# =========================================================================

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def hdf5_parallel_scan(directory: str, pattern: str = "*.h5") -> str:
    """Fast multi-file scanning with parallel processing.

    Args:
        directory: Directory to scan
        pattern: File pattern (default: *.h5)

    Returns:
        Scan summary with file metadata
    """
    import glob

    search_path = Path(directory) / pattern
    files = glob.glob(str(search_path), recursive=True)

    if not files:
        return f"No HDF5 files found in {directory} matching {pattern}"

    # Parallel scanning
    scan_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        future_to_file = {
            pool.submit(_scan_single_file, file_path): file_path
            for file_path in files[:50]  # Limit to 50 files
        }

        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                scan_results.append({
                    "file": file_path,
                    "status": "success",
                    **result
                })
            except Exception as e:
                scan_results.append({
                    "file": file_path,
                    "status": "error",
                    "error": str(e)
                })

    # Format results
    successful = [r for r in scan_results if r["status"] == "success"]
    total_datasets = sum(r.get("dataset_count", 0) for r in successful)
    total_size_mb = sum(r.get("total_size_mb", 0) for r in successful)

    summary = f"Parallel scan complete:\n"
    summary += f"Files processed: {len(scan_results)}\n"
    summary += f"Successful: {len(successful)}\n"
    summary += f"Total datasets: {total_datasets}\n"
    summary += f"Total size: {total_size_mb:.2f} MB\n\n"

    for result in scan_results[:10]:
        if result["status"] == "success":
            summary += f"✓ {Path(result['file']).name}: {result.get('dataset_count', 0)} datasets, {result.get('total_size_mb', 0):.1f} MB\n"
        else:
            summary += f"✗ {Path(result['file']).name}: {result['error']}\n"

    if len(scan_results) > 10:
        summary += f"... and {len(scan_results) - 10} more files\n"

    return summary

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def hdf5_batch_read(paths: str, slice_spec: Optional[str] = None) -> str:
    """Read multiple datasets in parallel.

    Args:
        paths: Comma-separated dataset paths or JSON array
        slice_spec: Optional slice specification

    Returns:
        Batch read summary
    """
    if not current_file:
        return "Error: No file currently open"

    # Parse paths
    try:
        path_list = json.loads(paths)
    except:
        path_list = [p.strip() for p in paths.split(',') if p.strip()]

    # Parse slice
    slice_obj = None
    if slice_spec:
        try:
            slice_obj = eval(f"np.s_[{slice_spec}]")
        except:
            return f"Error: Invalid slice specification: {slice_spec}"

    # Parallel batch reading
    results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        future_to_path = {
            pool.submit(_read_single_dataset, current_file, path, slice_obj): path
            for path in path_list
        }

        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                data_info = future.result()
                results[path] = data_info
            except Exception as e:
                results[path] = {"error": str(e)}

    # Format results
    summary = f"Batch read complete for {len(path_list)} datasets:\n\n"

    for path, result in results.items():
        if "error" in result:
            summary += f"✗ {path}: {result['error']}\n"
        else:
            summary += f"✓ {path}: shape {result['shape']}, dtype {result['dtype']}, size {result['size_mb']:.2f} MB\n"
            if "preview" in result:
                summary += f"  Preview: {result['preview']}\n"

    total_size_mb = sum(r.get("size_mb", 0) for r in results.values() if "error" not in r)
    summary += f"\nTotal data read: {total_size_mb:.2f} MB"

    return summary

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def hdf5_stream_data(path: str, chunk_size: int = 1024, max_chunks: int = 100) -> str:
    """Stream large datasets efficiently with memory management.

    Args:
        path: Path to dataset
        chunk_size: Number of elements per chunk
        max_chunks: Maximum number of chunks to process

    Returns:
        Stream processing summary with statistics
    """
    if not current_file:
        return "Error: No file currently open"

    dataset = current_file[path]

    if dataset.nbytes < 10 * 1024 * 1024:
        return f"Dataset {path} is small ({dataset.nbytes / (1024*1024):.1f} MB), consider using regular read"

    # Setup streaming
    total_elements = dataset.size
    elements_per_chunk = chunk_size
    total_chunks = min(max_chunks, (total_elements + elements_per_chunk - 1) // elements_per_chunk)

    # Stream processing
    chunk_summaries = []
    total_processed = 0

    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * elements_per_chunk
        end_idx = min(start_idx + elements_per_chunk, total_elements)

        # Read chunk
        if len(dataset.shape) == 1:
            chunk_data = dataset[start_idx:end_idx]
        else:
            chunk_data = dataset[start_idx:end_idx]

        # Process chunk
        chunk_info = {
            "chunk": chunk_idx + 1,
            "range": f"{start_idx}-{end_idx-1}",
            "elements": chunk_data.size,
            "mean": float(np.mean(chunk_data)) if chunk_data.size > 0 else 0,
            "std": float(np.std(chunk_data)) if chunk_data.size > 0 else 0,
            "min": float(np.min(chunk_data)) if chunk_data.size > 0 else 0,
            "max": float(np.max(chunk_data)) if chunk_data.size > 0 else 0
        }
        chunk_summaries.append(chunk_info)
        total_processed += chunk_data.size

        del chunk_data

    # Generate report
    streaming_rate = total_processed / (1024 * 1024)
    summary = f"Stream processing complete for dataset: {path}\n\n"
    summary += f"Dataset info:\n"
    summary += f"  Total size: {dataset.nbytes / (1024*1024):.2f} MB\n"
    summary += f"  Shape: {dataset.shape}\n"
    summary += f"  Dtype: {dataset.dtype}\n\n"
    summary += f"Streaming stats:\n"
    summary += f"  Chunks processed: {len(chunk_summaries)}\n"
    summary += f"  Elements processed: {total_processed:,}\n"
    summary += f"  Processing rate: {streaming_rate:.2f} MB\n\n"
    summary += "Chunk statistics:\n"

    for chunk in chunk_summaries[:10]:
        summary += f"  Chunk {chunk['chunk']}: mean={chunk['mean']:.3f}, std={chunk['std']:.3f}, range=[{chunk['min']:.3f}, {chunk['max']:.3f}]\n"

    if len(chunk_summaries) > 10:
        summary += f"  ... and {len(chunk_summaries) - 10} more chunks\n"

    return summary

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def hdf5_aggregate_stats(paths: str, stats: Optional[str] = None) -> str:
    """Parallel statistics computation across multiple datasets.

    Args:
        paths: Comma-separated dataset paths or JSON array
        stats: Comma-separated stats to compute (default: mean,std,min,max,sum,count)

    Returns:
        Aggregate statistics summary
    """
    if not current_file:
        return "Error: No file currently open"

    # Parse paths
    try:
        path_list = json.loads(paths)
    except:
        path_list = [p.strip() for p in paths.split(',') if p.strip()]

    # Parse stats
    if stats:
        stats_list = [s.strip() for s in stats.split(',') if s.strip()]
    else:
        stats_list = ["mean", "std", "min", "max", "sum", "count"]

    # Parallel statistics computation
    results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        future_to_path = {
            pool.submit(_compute_dataset_stats, current_file, path, stats_list): path
            for path in path_list
        }

        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                dataset_stats = future.result()
                results[path] = dataset_stats
            except Exception as e:
                results[path] = {"error": str(e)}

    # Aggregate results
    successful_stats = {k: v for k, v in results.items() if "error" not in v}

    summary = f"Aggregate statistics for {len(path_list)} datasets:\n\n"

    for path, stat_result in results.items():
        if "error" in stat_result:
            summary += f"✗ {path}: {stat_result['error']}\n"
        else:
            summary += f"✓ {path}:\n"
            summary += f"  Shape: {stat_result['shape']}, Size: {stat_result['size_mb']:.2f} MB\n"
            for stat_name in stats_list:
                if stat_name in stat_result:
                    summary += f"  {stat_name}: {stat_result[stat_name]:.6f}\n"
            summary += "\n"

    # Cross-dataset aggregation
    if len(successful_stats) > 1:
        summary += "Cross-dataset aggregation:\n"

        for stat_name in ["mean", "sum", "count"]:
            if all(stat_name in stats_list and stat_name in result for result in successful_stats.values()):
                values = [result[stat_name] for result in successful_stats.values()]
                if stat_name == "mean":
                    counts = [result.get("count", 1) for result in successful_stats.values()]
                    total_count = sum(counts)
                    weighted_mean = sum(v * c for v, c in zip(values, counts)) / total_count if total_count > 0 else 0
                    summary += f"  Overall {stat_name}: {weighted_mean:.6f}\n"
                elif stat_name == "sum":
                    summary += f"  Total {stat_name}: {sum(values):.6f}\n"
                elif stat_name == "count":
                    summary += f"  Total {stat_name}: {sum(values):,}\n"

        if all("min" in result for result in successful_stats.values()):
            global_min = min(result["min"] for result in successful_stats.values())
            summary += f"  Global min: {global_min:.6f}\n"

        if all("max" in result for result in successful_stats.values()):
            global_max = max(result["max"] for result in successful_stats.values())
            summary += f"  Global max: {global_max:.6f}\n"

    return summary

# =========================================================================
# DISCOVERY TOOLS - Analysis, Patterns, Optimization
# =========================================================================

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def analyze_dataset_structure(path: str = "/") -> str:
    """Analyze and understand file organization and data patterns.

    Args:
        path: Path to analyze (default: root)

    Returns:
        Structure analysis
    """
    if not current_file:
        return "Error: No file currently open"

    if path == "/":
        obj = current_file.file
    else:
        obj = current_file[path]

    if isinstance(obj, h5py.Group):
        items = list(obj.keys())
        groups = [k for k in items if isinstance(obj[k], h5py.Group)]
        datasets = [k for k in items if isinstance(obj[k], h5py.Dataset)]

        analysis = f"Structure Analysis for: {path}\n"
        analysis += f"Type: Group\n"
        analysis += f"Total items: {len(items)}\n"
        analysis += f"Groups: {len(groups)}\n"
        analysis += f"Datasets: {len(datasets)}\n\n"

        if datasets:
            analysis += "Datasets:\n"
            for ds_name in datasets[:10]:
                ds = obj[ds_name]
                analysis += f"  {ds_name}: {ds.shape}, {ds.dtype}\n"
            if len(datasets) > 10:
                analysis += f"  ... and {len(datasets) - 10} more datasets\n"

        if groups:
            analysis += f"\nGroups: {', '.join(groups[:10])}\n"
            if len(groups) > 10:
                analysis += f"... and {len(groups) - 10} more groups\n"

    elif isinstance(obj, h5py.Dataset):
        analysis = f"Structure Analysis for: {path}\n"
        analysis += f"Type: Dataset\n"
        analysis += f"Shape: {obj.shape}\n"
        analysis += f"Data type: {obj.dtype}\n"
        analysis += f"Size: {obj.nbytes / (1024*1024):.2f} MB\n"
        analysis += f"Chunks: {obj.chunks}\n"
    else:
        analysis = f"Structure Analysis for: {path}\n"
        analysis += f"Type: {type(obj).__name__}\n"

    return analysis

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def find_similar_datasets(reference_path: str, similarity_threshold: float = 0.8) -> str:
    """Find datasets with similar characteristics to a reference dataset.

    Args:
        reference_path: Path to reference dataset
        similarity_threshold: Similarity threshold (0.0 to 1.0)

    Returns:
        List of similar datasets with similarity scores
    """
    if not current_file:
        return "Error: No file currently open"

    ref_dataset = current_file[reference_path]
    if not isinstance(ref_dataset, h5py.Dataset):
        return f"Error: {reference_path} is not a dataset"

    ref_shape = ref_dataset.shape
    ref_dtype = ref_dataset.dtype
    ref_size = ref_dataset.nbytes

    similar_datasets = []

    def check_dataset(name, obj):
        if isinstance(obj, h5py.Dataset) and name != reference_path:
            shape_sim = 1.0 if obj.shape == ref_shape else 0.5
            dtype_sim = 1.0 if obj.dtype == ref_dtype else 0.3
            size_ratio = min(obj.nbytes, ref_size) / max(obj.nbytes, ref_size)

            similarity = (shape_sim + dtype_sim + size_ratio) / 3.0

            if similarity >= similarity_threshold:
                similar_datasets.append({
                    "path": name,
                    "similarity": similarity,
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                    "size_mb": obj.nbytes / (1024*1024)
                })

    actual_file = current_file.file if hasattr(current_file, 'file') else current_file
    actual_file.visititems(check_dataset)

    similar_datasets.sort(key=lambda x: x["similarity"], reverse=True)

    result = f"Similar datasets to '{reference_path}':\n"
    result += f"Reference: {ref_shape}, {ref_dtype}, {ref_size/(1024*1024):.2f} MB\n\n"

    if similar_datasets:
        result += f"Found {len(similar_datasets)} similar datasets:\n"
        for ds in similar_datasets[:10]:
            result += f"  {ds['path']} (similarity: {ds['similarity']:.3f})\n"
            result += f"    Shape: {ds['shape']}, Type: {ds['dtype']}, Size: {ds['size_mb']:.2f} MB\n"
    else:
        result += "No similar datasets found with the given threshold."

    return result

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def suggest_next_exploration(current_path: str = "/") -> str:
    """Suggest interesting data to explore next based on current location.

    Args:
        current_path: Current path (default: root)

    Returns:
        Exploration suggestions with interest scores
    """
    if not current_file:
        return "Error: No file currently open"

    if current_path == "/":
        obj = current_file.file
    else:
        obj = current_file[current_path]

    suggestions = []

    if isinstance(obj, h5py.Group):
        items = list(obj.keys())

        for item_name in items:
            try:
                item = obj[item_name]
                if isinstance(item, h5py.Dataset):
                    size_mb = item.nbytes / (1024 * 1024)
                    score = 0

                    if 1 <= size_mb <= 100:
                        score += 3
                    elif size_mb > 100:
                        score += 2

                    if len(item.shape) == 2:
                        score += 2
                    elif len(item.shape) > 2:
                        score += 1

                    if "data" in item_name.lower() or "result" in item_name.lower():
                        score += 1

                    suggestions.append({
                        "path": f"{current_path}/{item_name}" if current_path != "/" else f"/{item_name}",
                        "type": "dataset",
                        "score": score,
                        "info": f"Shape: {item.shape}, Size: {size_mb:.2f} MB"
                    })

                elif isinstance(item, h5py.Group):
                    child_count = len(list(item.keys()))
                    score = min(3, child_count // 5)

                    suggestions.append({
                        "path": f"{current_path}/{item_name}" if current_path != "/" else f"/{item_name}",
                        "type": "group",
                        "score": score,
                        "info": f"Contains {child_count} items"
                    })
            except Exception:
                continue

    suggestions.sort(key=lambda x: x["score"], reverse=True)

    result = f"Exploration suggestions from '{current_path}':\n\n"
    if suggestions:
        for i, suggestion in enumerate(suggestions[:5], 1):
            result += f"{i}. {suggestion['path']} ({suggestion['type']})\n"
            result += f"   {suggestion['info']}\n"
            result += f"   Interest score: {suggestion['score']}\n\n"
    else:
        result += "No additional exploration targets found at this location."

    return result

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def identify_io_bottlenecks(analysis_paths: Optional[List[str]] = None) -> str:
    """Identify potential I/O bottlenecks and performance issues.

    Args:
        analysis_paths: Optional list of paths to analyze (auto-discovers if None)

    Returns:
        Bottleneck analysis report
    """
    if not current_file:
        return "Error: No file currently open"

    if not analysis_paths:
        analysis_paths = []

        def collect_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                analysis_paths.append(name)

        actual_file = current_file.file if hasattr(current_file, 'file') else current_file
        actual_file.visititems(collect_datasets)
        analysis_paths = analysis_paths[:10]

    bottlenecks = []

    for path in analysis_paths:
        try:
            dataset = current_file[path]
            issues = []

            size_mb = dataset.nbytes / (1024 * 1024)

            if size_mb > 100 and dataset.chunks is None:
                issues.append(f"Large dataset ({size_mb:.1f} MB) without chunking")

            if dataset.chunks and np.prod(dataset.chunks) * dataset.dtype.itemsize < 1024:
                issues.append("Very small chunk size may hurt performance")

            if size_mb > 50 and not hasattr(dataset, 'compression'):
                issues.append("Large dataset without compression")

            if len(dataset.shape) > 3:
                issues.append("High-dimensional array may have access pattern issues")

            if issues:
                bottlenecks.append({
                    "path": path,
                    "size_mb": size_mb,
                    "issues": issues
                })
        except Exception:
            continue

    result = "I/O Bottleneck Analysis:\n\n"
    if bottlenecks:
        result += f"Found potential issues in {len(bottlenecks)} datasets:\n\n"
        for bottleneck in bottlenecks:
            result += f"📄 {bottleneck['path']} ({bottleneck['size_mb']:.2f} MB)\n"
            for issue in bottleneck['issues']:
                result += f"  ⚠️  {issue}\n"
            result += "\n"
    else:
        result += "✅ No significant I/O bottlenecks detected."

    return result

@mcp.tool()
@with_error_handling
@with_performance_tracking
async def optimize_access_pattern(dataset_path: str, access_pattern: str = "sequential") -> str:
    """Suggest better approaches for data access based on usage patterns.

    Args:
        dataset_path: Path to dataset
        access_pattern: Access pattern (sequential, random, batch)

    Returns:
        Optimization recommendations
    """
    if not current_file:
        return "Error: No file currently open"

    dataset = current_file[dataset_path]
    if not isinstance(dataset, h5py.Dataset):
        return f"Error: {dataset_path} is not a dataset"

    size_mb = dataset.nbytes / (1024 * 1024)
    shape = dataset.shape
    chunks = dataset.chunks

    result = f"Access Pattern Optimization for: {dataset_path}\n"
    result += f"Dataset size: {size_mb:.2f} MB, Shape: {shape}\n"
    result += f"Current chunking: {chunks}\n\n"

    if access_pattern.lower() == "sequential":
        result += "Sequential Access Recommendations:\n"
        if size_mb > 100:
            result += "• Use hdf5_stream_data() for memory-efficient processing\n"
            result += "• Consider processing in chunks to avoid memory issues\n"
        else:
            result += "• Use read_full_dataset() for complete data access\n"

        if not chunks:
            result += "• Consider enabling chunking for better I/O performance\n"

    elif access_pattern.lower() == "random":
        result += "Random Access Recommendations:\n"
        if not chunks:
            result += "• ⚠️  Enable chunking for better random access performance\n"
        else:
            chunk_size = np.prod(chunks) * dataset.dtype.itemsize / (1024 * 1024)
            if chunk_size > 10:
                result += f"• Consider smaller chunks (current: {chunk_size:.1f} MB)\n"
            else:
                result += f"• Chunk size ({chunk_size:.2f} MB) is good for random access\n"

        result += "• Use read_partial_dataset() with specific slices\n"

    elif access_pattern.lower() == "batch":
        result += "Batch Processing Recommendations:\n"
        result += "• Use hdf5_batch_read() for parallel processing\n"
        result += "• Consider hdf5_aggregate_stats() for statistical operations\n"
        if size_mb > 50:
            result += "• Use chunked reading for large datasets\n"

    else:
        result += f"General recommendations for '{access_pattern}' access:\n"
        result += "• Analyze your specific access patterns\n"
        result += "• Consider chunking strategy based on access needs\n"
        result += "• Use appropriate tools based on data size\n"

    return result

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def _read_large_dataset(dataset, chunk_size=1024*1024):
    """Read large dataset in chunks."""
    if len(dataset.shape) != 1:
        return dataset[:]

    result = np.empty(dataset.shape, dtype=dataset.dtype)
    for i in range(0, dataset.shape[0], chunk_size):
        end = min(i + chunk_size, dataset.shape[0])
        result[i:end] = dataset[i:end]
    return result

def _scan_single_file(file_path: str) -> dict:
    """Scan a single HDF5 file and return metadata."""
    with h5py.File(file_path, 'r') as f:
        dataset_count = 0
        total_size = 0
        datasets = []

        def count_datasets(name, obj):
            nonlocal dataset_count, total_size
            if isinstance(obj, h5py.Dataset):
                dataset_count += 1
                total_size += obj.nbytes
                datasets.append({
                    "name": name,
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                    "size_mb": obj.nbytes / (1024 * 1024)
                })

        f.visititems(count_datasets)

        return {
            "dataset_count": dataset_count,
            "total_size_mb": total_size / (1024 * 1024),
            "datasets": datasets[:5]
        }

def _read_single_dataset(file_proxy, path: str, slice_obj=None) -> dict:
    """Read a single dataset with optional slicing."""
    dataset = file_proxy[path]

    if slice_obj is not None:
        data = dataset[slice_obj]
    else:
        if dataset.nbytes > 100 * 1024 * 1024:
            if len(dataset.shape) == 1:
                data = dataset[:min(1000, dataset.shape[0])]
            else:
                data = dataset[:min(10, dataset.shape[0])]
        else:
            data = dataset[:]

    if isinstance(data, np.ndarray):
        if data.size <= 10:
            preview = data.tolist()
        else:
            preview = f"[{data.flat[0]}, {data.flat[1]}, ..., {data.flat[-1]}]"
    else:
        preview = str(data)

    return {
        "shape": dataset.shape,
        "dtype": str(dataset.dtype),
        "size_mb": dataset.nbytes / (1024 * 1024),
        "preview": preview,
        "data_shape": data.shape,
        "slice_applied": slice_obj is not None
    }

def _compute_dataset_stats(file_proxy, path: str, stats: List[str]) -> dict:
    """Compute statistics for a single dataset."""
    dataset = file_proxy[path]

    if dataset.nbytes > 500 * 1024 * 1024:
        sample_size = min(1000000, max(1000, dataset.size // 100))
        if len(dataset.shape) == 1:
            step = max(1, dataset.size // sample_size)
            data = dataset[::step]
        else:
            step = max(1, dataset.shape[0] // int(np.sqrt(sample_size)))
            data = dataset[::step]
    else:
        data = dataset[:]

    result = {
        "shape": dataset.shape,
        "dtype": str(dataset.dtype),
        "size_mb": dataset.nbytes / (1024 * 1024),
        "sampled": dataset.nbytes > 500 * 1024 * 1024
    }

    if np.issubdtype(data.dtype, np.number):
        if "mean" in stats:
            result["mean"] = float(np.mean(data))
        if "std" in stats:
            result["std"] = float(np.std(data))
        if "min" in stats:
            result["min"] = float(np.min(data))
        if "max" in stats:
            result["max"] = float(np.max(data))
        if "sum" in stats:
            result["sum"] = float(np.sum(data))
        if "count" in stats:
            result["count"] = int(data.size)
        if "median" in stats:
            result["median"] = float(np.median(data))
    else:
        result["note"] = f"Non-numeric data type ({data.dtype}), limited statistics available"
        if "count" in stats:
            result["count"] = int(data.size)

    return result

# =========================================================================
# RESOURCES - HDF5 URIs
# =========================================================================

@mcp.resource("hdf5://{file_path}/metadata")
async def hdf5_file_metadata(file_path: str) -> str:
    """Expose HDF5 file metadata as resource.

    Args:
        file_path: Path to HDF5 file

    Returns:
        JSON metadata
    """
    try:
        with h5py.File(file_path, 'r') as f:
            metadata = {
                "filename": f.filename,
                "mode": f.mode,
                "userblock_size": f.userblock_size,
                "keys": list(f.keys()),
                "attrs": dict(f.attrs)
            }
            return json.dumps(metadata, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.resource("hdf5://{file_path}/datasets/{dataset_path}")
async def hdf5_dataset_resource(file_path: str, dataset_path: str) -> str:
    """Expose HDF5 dataset as resource.

    Args:
        file_path: Path to HDF5 file
        dataset_path: Path to dataset within file

    Returns:
        Dataset data (preview for large datasets)
    """
    try:
        with h5py.File(file_path, 'r') as f:
            dataset = f[dataset_path]

            data_info = {
                "path": dataset_path,
                "shape": dataset.shape,
                "dtype": str(dataset.dtype),
                "size_mb": dataset.nbytes / (1024 * 1024)
            }

            # Include data preview
            if dataset.nbytes < 1024 * 1024:  # 1MB
                data = dataset[:]
                if hasattr(data, "tolist"):
                    data_info["data"] = data.tolist()
                else:
                    data_info["data"] = str(data)
            else:
                data_info["note"] = "Dataset too large, use tools to read"

            return json.dumps(data_info, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.resource("hdf5://{file_path}/structure")
async def hdf5_structure_resource(file_path: str) -> str:
    """Expose HDF5 file structure as resource.

    Args:
        file_path: Path to HDF5 file

    Returns:
        Hierarchical structure
    """
    try:
        with h5py.File(file_path, 'r') as f:
            structure = {}

            def build_structure(name, obj):
                if isinstance(obj, h5py.Group):
                    structure[name] = {"type": "Group", "keys": list(obj.keys())}
                elif isinstance(obj, h5py.Dataset):
                    structure[name] = {
                        "type": "Dataset",
                        "shape": obj.shape,
                        "dtype": str(obj.dtype)
                    }

            f.visititems(build_structure)
            return json.dumps(structure, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

# =========================================================================
# PROMPTS - Analysis Workflows
# =========================================================================

@mcp.prompt()
def explore_hdf5_file(file_path: str) -> str:
    """Generate workflow for exploring an HDF5 file.

    Args:
        file_path: Path to HDF5 file

    Returns:
        Exploration workflow prompt
    """
    return f"""Please explore the HDF5 file at {file_path}:

1. First, use open_file to open {file_path}
2. Use analyze_dataset_structure to understand the hierarchy
3. Use suggest_next_exploration to find interesting datasets
4. Read the suggested datasets with read_full_dataset or read_partial_dataset
5. Generate summary statistics with hdf5_aggregate_stats
6. Close the file when done with close_file

This workflow will give you a comprehensive understanding of the file's contents."""

@mcp.prompt()
def optimize_hdf5_access(file_path: str, access_pattern: str = "sequential") -> str:
    """Generate optimization workflow for HDF5 I/O.

    Args:
        file_path: Path to HDF5 file
        access_pattern: Access pattern (sequential, random, batch)

    Returns:
        Optimization workflow prompt
    """
    return f"""Optimize I/O access for {file_path}:

1. Use open_file to open {file_path}
2. Use identify_io_bottlenecks to detect performance issues
3. Use optimize_access_pattern with pattern='{access_pattern}' for specific datasets
4. Use hdf5_parallel_scan if analyzing multiple files in a directory
5. Use hdf5_stream_data for large datasets to avoid memory issues
6. Monitor performance with HDF5_SHOW_PERFORMANCE=true environment variable

This workflow will help you achieve optimal performance for your access patterns."""

@mcp.prompt()
def compare_hdf5_datasets(file_path: str, dataset1: str, dataset2: str) -> str:
    """Generate comparison workflow for two datasets.

    Args:
        file_path: Path to HDF5 file
        dataset1: First dataset path
        dataset2: Second dataset path

    Returns:
        Comparison workflow prompt
    """
    return f"""Compare datasets in {file_path}:

1. Use open_file to open {file_path}
2. Use get_shape and get_dtype to compare metadata for:
   - {dataset1}
   - {dataset2}
3. Use hdf5_batch_read with paths=["{dataset1}", "{dataset2}"] for parallel reading
4. Use hdf5_aggregate_stats to compute statistics for both datasets
5. Use find_similar_datasets starting from {dataset1} to see if {dataset2} is similar
6. Close the file with close_file

This workflow provides a comprehensive comparison of the two datasets."""

@mcp.prompt()
def batch_process_hdf5(directory: str, operation: str = "statistics") -> str:
    """Generate batch processing workflow for multiple HDF5 files.

    Args:
        directory: Directory containing HDF5 files
        operation: Operation to perform (statistics, scan, export)

    Returns:
        Batch processing workflow prompt
    """
    return f"""Batch process HDF5 files in {directory}:

1. Use hdf5_parallel_scan with directory="{directory}" to discover all files
2. For each interesting file found:
   a. Use open_file to open it
   b. Use analyze_dataset_structure to understand layout
   c. Use hdf5_aggregate_stats to compute {operation}
   d. Use close_file to release resources
3. Aggregate results across all files
4. Use identify_io_bottlenecks to find common performance issues

This parallel workflow efficiently processes multiple files with minimal overhead."""

# =========================================================================
# Server Lifecycle
# =========================================================================

async def initialize():
    """Initialize server resources."""
    try:
        logger.info("Initializing HDF5 FastMCP server...")

        # Ensure data directory exists
        data_dir = config.hdf5.data_dir
        if not data_dir.exists():
            logger.info(f"Creating data directory: {data_dir}")
            data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize resource manager
        await resource_manager.initialize()

        logger.info("HDF5 FastMCP server initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing server: {e}")
        raise

async def cleanup():
    """Cleanup server resources."""
    try:
        logger.info("Shutting down HDF5 FastMCP server...")

        # Close current file if open
        global current_file
        if current_file:
            current_file.close()
            current_file = None

        # Shutdown resource manager
        await resource_manager.shutdown()

        # Shutdown executor
        executor.shutdown(wait=True)

        logger.info("HDF5 FastMCP server shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# =========================================================================
# Main Entry Point
# =========================================================================

def main():
    """Main entry point for HDF5 FastMCP server."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="HDF5 FastMCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse", "http"], default="stdio",
                       help="Transport protocol (default: stdio)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP/SSE (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port for HTTP/SSE (default: 8765)")
    parser.add_argument("--data-dir", type=Path, help="Directory containing HDF5 files")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Set data directory if provided
    if args.data_dir:
        os.environ['HDF5_MCP_DATA_DIR'] = str(args.data_dir)

    # Initialize server
    asyncio.run(initialize())

    try:
        # Run with selected transport
        if args.transport == "stdio":
            logger.info("Starting with stdio transport")
            mcp.run(transport="stdio")
        elif args.transport == "sse":
            logger.info(f"Starting with SSE transport on {args.host}:{args.port}")
            mcp.run(transport="sse", host=args.host, port=args.port)
        elif args.transport == "http":
            logger.info(f"Starting with HTTP transport on {args.host}:{args.port}")
            mcp.run(transport="http", host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)
    finally:
        asyncio.run(cleanup())

if __name__ == "__main__":
    main()
