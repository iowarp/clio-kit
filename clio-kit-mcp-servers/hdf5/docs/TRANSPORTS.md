# HDF5 transport configuration

The server uses FastMCP 4 and MCP Python SDK v2. The CLI accepts `stdio` and
`http`; inspect the installed arguments with:

```bash
clio-kit mcp-server hdf5 -- --help
```

## stdio

For local agents, configure command `clio-kit` and arguments
`["mcp-server", "hdf5"]`. The agent owns the subprocess and exchanges MCP
messages on stdin/stdout. Operational logs go to stderr.

```bash
clio-kit mcp-server hdf5 -- --data-dir /path/to/data
clio-kit doctor --server hdf5 --connect
```

See [agent integrations](../../../README.md#agent-integrations) for each client's
JSON/TOML configuration. Skill installation is separate from MCP registration.

## HTTP

For a separately managed MCP service:

```bash
clio-kit mcp-server hdf5 -- --transport http --host 127.0.0.1 --port 8765
```

Connect an HTTP-capable MCP client to `http://127.0.0.1:8765/mcp`.
Pass `--host` explicitly: the CLI otherwise defaults to `0.0.0.0`.
`MCP_TRANSPORT=http` selects the same transport; CLI `--transport` takes
precedence. This server does not define custom `/health` or `/stats` routes.
It does not configure authentication; deployments beyond a local interface
need an appropriate access boundary.

## Data and protocol limits

Changing transport does not remove tool result or memory limits.
`hdf5_stream_data` processes bounded chunks and returns a summary, not an
unlimited stream of raw data. Check its processed coverage and `max_chunks`.
Some other operations, including export, materialize the dataset in memory.

The verification suite exercises modern `2026-07-28` and legacy `2025-11-25`
stdio connections. HTTP deployment, authentication and recovery need their own
site acceptance. Explicit `export_format` works in either protocol mode;
modern connections default to JSON. See [migration notes](MIGRATION.md).
