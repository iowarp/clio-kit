# Testing HDF5 MCP

From the repository root:

```bash
uv sync --frozen --dev --directory clio-kit-mcp-servers/hdf5
uv run --frozen --directory clio-kit-mcp-servers/hdf5 pytest -q
```

The tests cover configuration, resource management, formatting, scientific
statistics and MCP export behavior. In particular:

- `tests/test_statistics.py` checks actual HDF5 datasets, sample coverage and
  omission of misleading cross-dataset totals.
- `tests/test_export_protocol.py` exercises real FastMCP calls in modern and
  legacy protocol modes, including CSV contents and default JSON export.

Protocol tests import the server and use `fastmcp.Client`; decorators are not a
reason to avoid testing tool calls. Close files and let client contexts exit so
resources are released. Coverage exclusions in `pyproject.toml` affect reported
percentages, not whether protocol behavior needs testing.

For coverage, add `--cov=src/hdf5_mcp --cov-report=term-missing` to the server
pytest command. Read the actual report for the tested revision rather than
reusing an old percentage or test count.

## Installed launcher

```bash
clio-kit doctor --server hdf5 --connect
```

The root [setup guide](../../setup.md) includes a real stdio verification
example. The root `scripts/verify_marketplace_install.py` builds and installs a
wheel and exercises scientific results, including sampled HDF5 coverage. A
successful import or handshake alone does not verify numerical correctness.

The [educational examples](examples/README.md) use direct h5py calls and do not
replace protocol tests.
