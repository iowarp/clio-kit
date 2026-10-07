# HDF5 MCP compatibility notes

The current server uses FastMCP 4 and MCP Python SDK v2. Package versions and
wire protocol versions are separate: installed-system verification covers
`2026-07-28` and legacy `2025-11-25` connections.

## Client configuration

Install CLIO Kit with the [setup guide](../../../setup.md), then register
command `clio-kit` with arguments `["mcp-server", "hdf5"]`. For HTTP, use
`--transport http`; the old `sse` CLI option is not supported. See
[transport configuration](TRANSPORTS.md).

## Tool calls

- `export_dataset` accepts `export_format="csv"`, `"json"` or `"numpy"`.
  Modern connections default to JSON. When omitted on a legacy connection,
  the server can ask through elicitation if the client supports it.
- `read_partial_dataset` takes comma-separated strings for `start` and `count`,
  such as `"0,0"` and `"10,5"`, rather than JSON arrays.
- `hdf5_aggregate_stats` labels sampled coverage. A sample sum or count is not
  a whole-dataset total; cross-dataset totals are omitted if any input is sampled.
- The small tool inventory is returned on one page for clients that do not
  follow pagination.

Some descriptive tools retain optional legacy sampling support. Modern
connections must not depend on server-initiated model sampling or mid-call
questions; inspect the returned data and perform interpretation in the agent.

Use the [tool reference](TOOLS.md) and the live client schema for current names
and parameters. After upgrading the launcher, restart MCP processes so they
use the updated locked environment.
