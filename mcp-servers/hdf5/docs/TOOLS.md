# HDF5 MCP tool reference

The server advertises 27 tools. The table reflects the current server source;
inspect the live MCP input schema for types and constraints before calling.
Tool names here are MCP calls, not standalone Python imports.

| Tool and arguments | Purpose |
| --- | --- |
| `open_file(path, mode='r')` | Open an HDF5 file for operations. |
| `close_file()` | Close the current HDF5 file. Returns: Status message |
| `get_filename()` | Get the current file's path. Returns: File path |
| `get_mode()` | Get the current file's access mode. Returns: File mode |
| `get_by_path(path)` | Get a dataset or group by path. |
| `list_keys(path='/')` | List keys in a group. |
| `visit(callback_fn='collect_paths')` | Visit all nodes recursively. |
| `read_full_dataset(path)` | Read an entire dataset with efficient chunked reading for large datasets. |
| `read_partial_dataset(path, start=None, count=None)` | Read a portion of a dataset with slicing. |
| `get_shape(path)` | Get the shape of a dataset. |
| `get_dtype(path)` | Get the data type of a dataset. |
| `get_size(path)` | Get the size of a dataset. |
| `get_chunks(path)` | Get chunk information for a dataset. |
| `read_attribute(path, name)` | Read an attribute from an object. |
| `list_attributes(path)` | List all attributes of an object. |
| `hdf5_parallel_scan(directory, pattern='*.h5')` | Fast multi-file scanning with parallel processing. |
| `hdf5_batch_read(paths, slice_spec=None)` | Read multiple datasets in parallel. |
| `hdf5_stream_data(path, chunk_size=1024, max_chunks=100)` | Stream large datasets efficiently with memory management. |
| `hdf5_aggregate_stats(paths, stats=None)` | Parallel statistics, with explicit sampling coverage above 500 MiB. Sample statistics describe only the selected values, not full-dataset totals. |
| `analyze_dataset_structure(path='/')` | Analyze and understand file organization and data patterns with AI insights. |
| `find_similar_datasets(reference_path, similarity_threshold=0.8)` | Find datasets with similar characteristics to a reference dataset with AI analysis. |
| `suggest_next_exploration(current_path='/')` | Suggest interesting data to explore next based on current location with AI recommendations. |
| `identify_io_bottlenecks(analysis_paths=None)` | Identify potential I/O bottlenecks and performance issues with AI recommendations. |
| `optimize_access_pattern(dataset_path, access_pattern='sequential')` | Suggest better approaches for data access based on usage patterns. |
| `refresh_hdf5_resources()` | Re-scan client roots and update available HDF5 resources. FastMCP automatically sends notifications/resources/list_changed to clients. Returns: Summary of refreshed resources |
| `list_available_hdf5_files()` | List all registered HDF5 files with resource URIs for Claude Code @ mentions. Returns: List of available files with resource URIs |
| `export_dataset(path, output_path=None, export_format=None)` | Export dataset to various formats with user format selection. |

Open a file before tools that operate on its contents and close it after the
workflow. `read_partial_dataset` uses comma-separated strings for indices.
`hdf5_aggregate_stats` takes string arguments for `paths` and `stats`, such as
`"/values"` and `"sum,count,mean"`. Aggregate statistics label sampled coverage; exports and full reads can load
the complete dataset into memory. For bounded examples, see [EXAMPLES.md](EXAMPLES.md).
