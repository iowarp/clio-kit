# HDF5 MCP Server

HDF5 MCP provides 27 tools for inspecting scientific files, reading bounded
subsets, computing statistics and exporting data. It uses FastMCP 4 and MCP
Python SDK v2. Part of [CLIO Kit](https://toolkit.iowarp.ai/).

## Installation

Install the launcher with the [CLIO Kit setup guide](../../setup.md), then
configure your client with command `clio-kit` and arguments
`["mcp-server", "hdf5"]`. See [agent integrations](../../README.md#agent-integrations)
for Codex, Claude Code, Cursor, VS Code/Copilot, Antigravity and Claude Desktop.

```bash
clio-kit doctor --server hdf5 --connect
clio-kit mcp-server hdf5 -- --help
```

For development, from the repository root:

```bash
uv sync --frozen --dev --directory clio-kit-mcp-servers/hdf5
uv run --frozen --directory clio-kit-mcp-servers/hdf5 hdf5-mcp --help
```

## Workflow

1. Call `open_file` with an absolute path available to the MCP process.
2. Use `list_keys`, `get_shape` and `get_dtype` before reading values.
3. Use `read_partial_dataset` for a bounded preview; `start` and `count` are
   comma-separated strings, for example `"0,0"` and `"5,4"`.
4. Use `hdf5_aggregate_stats` or `hdf5_stream_data` for summaries. Check processed
   coverage before treating any sum, count or extrema as a whole-dataset result.
5. Export a suitably small dataset with `export_dataset`, specifying
   `export_format` as `csv`, `json` or `numpy`, then close the file.

Datasets above 500 MiB are sampled by aggregate statistics. Results label sample
coverage, and cross-dataset totals are omitted if any dataset was sampled.
Chunk summaries stop at `max_chunks`; full reads and exports can materialize the
dataset in memory. Chunked processing is not a promise of unlimited-size output.

## Configuration and transport

The default transport is stdio. Pass server options after `--`:

```bash
clio-kit mcp-server hdf5 -- --data-dir /path/to/data --log-level INFO
clio-kit mcp-server hdf5 -- --transport http --host 127.0.0.1 --port 8765
```

`HDF5_MCP_DATA_DIR` configures the discovery directory. Set
`HDF5_SHOW_PERFORMANCE=true` to include operation timings. The CLI accepts
`MCP_TRANSPORT` as its transport default. Use explicit host/port arguments for
HTTP; see [transport configuration](docs/TRANSPORTS.md) for deployment boundaries.

Resource templates expose metadata, bounded dataset previews and structure at
`hdf5://` URIs. Four prompts describe inspection and analysis workflows.
Optional legacy sampling/elicitation depends on the negotiated protocol and
client capabilities; modern exports use an explicit format or default to JSON.

## Documentation

- [Tool reference](docs/TOOLS.md)
- [Runnable workflow](docs/EXAMPLES.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Compatibility notes](docs/MIGRATION.md)
- [Testing](TESTING.md)
- [Contributing](../../CONTRIBUTING.md)

## Capabilities

### `open_file`
**Description**: Open an HDF5 file for operations.
**Tags**: core, file

### `close_file`
**Description**: Close the current HDF5 file.

Returns:
    Status message
**Tags**: core, file

### `get_filename`
**Description**: Get the current file's path.

Returns:
    File path
**Hints**: read-only, idempotent
**Tags**: file, info

### `get_mode`
**Description**: Get the current file's access mode.

Returns:
    File mode
**Hints**: read-only, idempotent
**Tags**: file, info

### `get_by_path`
**Description**: Get a dataset or group by path.
**Hints**: read-only, idempotent
**Tags**: dataset, navigation

### `list_keys`
**Description**: List keys in a group.
**Hints**: read-only, idempotent
**Tags**: dataset, navigation

### `visit`
**Description**: Visit all nodes recursively.
**Hints**: read-only, idempotent
**Tags**: dataset, navigation

### `read_full_dataset`
**Description**: Read an entire dataset with efficient chunked reading for large datasets.
**Hints**: read-only, idempotent
**Tags**: dataset, read

### `read_partial_dataset`
**Description**: Read a portion of a dataset with slicing.
**Hints**: read-only, idempotent
**Tags**: dataset, read

### `get_shape`
**Description**: Get the shape of a dataset.
**Hints**: read-only, idempotent
**Tags**: dataset, metadata

### `get_dtype`
**Description**: Get the data type of a dataset.
**Hints**: read-only, idempotent
**Tags**: dataset, metadata

### `get_size`
**Description**: Get the size of a dataset.
**Hints**: read-only, idempotent
**Tags**: dataset, metadata

### `get_chunks`
**Description**: Get chunk information for a dataset.
**Hints**: read-only, idempotent
**Tags**: dataset, metadata, performance

### `read_attribute`
**Description**: Read an attribute from an object.
**Hints**: read-only, idempotent
**Tags**: attribute, metadata

### `list_attributes`
**Description**: List all attributes of an object.
**Hints**: read-only, idempotent
**Tags**: attribute, metadata

### `hdf5_parallel_scan`
**Description**: Fast multi-file scanning with parallel processing.
**Hints**: read-only, idempotent
**Tags**: parallel, performance, scan

### `hdf5_batch_read`
**Description**: Read multiple datasets in parallel.
**Hints**: read-only, idempotent
**Tags**: parallel, performance, read

### `hdf5_stream_data`
**Description**: Stream large datasets efficiently with memory management.
**Hints**: read-only, idempotent
**Tags**: performance, streaming

### `hdf5_aggregate_stats`
**Description**: Parallel statistics, with explicit sampling coverage above 500 MiB.

Sample statistics describe only the selected values, not full-dataset totals.
**Hints**: read-only, idempotent
**Tags**: analysis, parallel, performance

### `analyze_dataset_structure`
**Description**: Analyze and understand file organization and data patterns with AI insights.
**Hints**: read-only, idempotent
**Tags**: ai-powered, analysis, discovery

### `find_similar_datasets`
**Description**: Find datasets with similar characteristics to a reference dataset with AI analysis.
**Hints**: read-only, idempotent
**Tags**: ai-powered, discovery, similarity

### `suggest_next_exploration`
**Description**: Suggest interesting data to explore next based on current location with AI recommendations.
**Hints**: read-only, idempotent
**Tags**: ai-powered, discovery, recommendation

### `identify_io_bottlenecks`
**Description**: Identify potential I/O bottlenecks and performance issues with AI recommendations.
**Hints**: read-only, idempotent
**Tags**: ai-powered, discovery, performance

### `optimize_access_pattern`
**Description**: Suggest better approaches for data access based on usage patterns.
**Hints**: read-only, idempotent
**Tags**: discovery, optimization, performance

### `refresh_hdf5_resources`
**Description**: Re-scan client roots and update available HDF5 resources.

FastMCP automatically sends notifications/resources/list_changed to clients.

Returns:
    Summary of refreshed resources
**Tags**: admin, discovery

### `list_available_hdf5_files`
**Description**: List all registered HDF5 files with resource URIs for Claude Code @ mentions.

Returns:
    List of available files with resource URIs
**Hints**: read-only, idempotent
**Tags**: discovery, helper

### `export_dataset`
**Description**: Export dataset to various formats with user format selection.
**Tags**: dataset, export, interactive

### Resources

- `hdf5://{file_path}/metadata` - Expose HDF5 file metadata as resource.

Args:
    file_path: Path to HDF5 file

Returns:
    JSON metadata
- `hdf5://{file_path}/datasets/{dataset_path*}` - Expose HDF5 dataset as resource.

Args:
    file_path: Path to HDF5 file
    dataset_path: Path to dataset within file (supports nested paths)

Returns:
    Dataset data (preview for large datasets)
- `hdf5://{file_path}/structure` - Expose HDF5 file structure as resource.

Args:
    file_path: Path to HDF5 file

Returns:
    Hierarchical structure

### Prompts

- **explore_hdf5_file**: Generate workflow for exploring an HDF5 file.
- **optimize_hdf5_access**: Generate optimization workflow for HDF5 I/O.
- **compare_hdf5_datasets**: Generate comparison workflow for two datasets.
- **batch_process_hdf5**: Generate batch processing workflow for multiple HDF5 files.
## Claude Code

```bash
claude mcp add clio-hdf5 -- clio-kit mcp-server hdf5
```

Or install via the CLIO Kit plugin marketplace:

```
/plugin marketplace add iowarp/clio-kit
/plugin install clio-hdf5@clio-kit
```
## Claude Desktop

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "clio-hdf5": {
      "command": "clio-kit",
      "args": [
        "mcp-server",
        "hdf5"
      ]
    }
  }
}
```