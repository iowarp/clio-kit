# HDF5 MCP architecture

The server registers 27 tools, three resource templates and four workflow
prompts through FastMCP 4. MCP Python SDK v2 handles the protocol. The public
entry point is `hdf5_mcp.server:main`; CLIO launches it from the embedded lock.

## Modules

| Module under `src/hdf5_mcp/` | Responsibility |
| --- | --- |
| `server.py` | FastMCP registration, file state, h5py operations, lifecycle and CLI |
| `resources.py` | Lazy HDF5 proxies, resource registration, bounded caching and file discovery |
| `statistics.py` | Dataset statistics, sample coverage and cross-dataset aggregation rules |
| `exports.py` | Explicit export format and optional legacy elicitation |
| `config.py` | Configuration models, environment variables and configuration-file loading |
| `utils.py` | Shared HDF5, formatting and performance helpers |

Tools that use the current file require `open_file` first and `close_file` after
use. The current file is process-level state; do not interleave unrelated file
workflows through the same server process. File paths are resolved where the
server runs, which may be a remote workspace or container.

## Resource and memory behavior

The server creates a `ResourceManager` with a 1,000-entry cache and uses worker
pools for some scans, reads and aggregations. Cache capacity is an entry count,
not a bound on every allocation. Speedups depend on file layout, storage and
workload; there is no general throughput multiplier.

`hdf5_stream_data` reads chunks up to `max_chunks` and returns a summary.
`hdf5_aggregate_stats` samples datasets above 500 MiB and labels sample coverage;
cross-dataset totals are omitted when sampling is involved. Full reads and
exports can allocate the dataset in memory. Choose bounded reads for exploration.

FastMCP owns transport dispatch; there is no separate CLIO `ToolRegistry`,
`tools.py` or custom SSE transport layer in this server. See
[transport configuration](TRANSPORTS.md), [tool reference](TOOLS.md) and
[testing](../TESTING.md).
