# ArXiv MCP installation

Install the launcher using the [CLIO Kit setup guide](../../../setup.md).
The server needs network access to the arXiv API; PDF tools also need a writable
output directory. Configure any supported stdio client with command `clio-kit`
and arguments `["mcp-server", "arxiv"]`. See the shared
[agent integrations](../../../README.md#agent-integrations).

Check the installed server:

```bash
clio-kit doctor --server arxiv --connect
```

Then ask the client to call `search_by_title` with a known paper title and check
the returned identifier and title. A successful connection alone does not verify
API access; public API availability and rate limits can affect queries.

## Develop from source

```bash
git clone https://github.com/iowarp/clio-kit.git
cd clio-kit/clio-kit-mcp-servers/arxiv
uv sync --frozen --dev
uv run --frozen arxiv-mcp --help
uv run --frozen pytest -q
```

Use `uv run --frozen arxiv-mcp` for stdio. Inspect `--help` for supported
transport options. The complete tool inventory is in the
[server reference](../README.md#capabilities).
