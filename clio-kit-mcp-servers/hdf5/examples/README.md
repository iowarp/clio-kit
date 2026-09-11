# HDF5 educational examples

These scripts demonstrate HDF5 operations with direct h5py calls. They do not
launch an MCP server or verify client integration.

From the repository root:

```bash
cd clio-kit-mcp-servers/hdf5
uv run --frozen python examples/create_demo_data.py
uv run --frozen python examples/demo_script.py
```

The first script creates `examples/demo_data.h5` with climate-style temperature,
pressure and metadata arrays. The second inspects structure, attributes, slices
and storage layout. Its legacy display banner is not the MCP package version.

For real MCP calls, follow the [workflow examples](../docs/EXAMPLES.md) and
[testing guide](../TESTING.md).
