# HDF5 MCP workflow example

Install the launcher with the [setup guide](../../../setup.md). This example
creates a small HDF5 file, calls the installed server over stdio, exports a
bounded dataset and checks the exact exported values. All files are temporary.

```bash
uv run --no-project --with 'mcp>=2.2,<3' --with h5py python - <<'PY'
import asyncio
import json
import tempfile
from pathlib import Path

import h5py
from mcp import Client, StdioServerParameters

async def check():
    with tempfile.TemporaryDirectory(prefix="clio-hdf5-example-") as directory:
        source = Path(directory) / "data.h5"
        exported = Path(directory) / "data.json"
        expected = [[1, 2], [3, 4], [5, 6]]
        with h5py.File(source, "w") as file:
            file.create_dataset("values", data=expected)

        parameters = StdioServerParameters(
            command="clio-kit", args=["mcp-server", "hdf5"]
        )
        async with Client(parameters) as client:
            result = await client.call_tool("open_file", {"path": str(source)})
            assert not result.is_error, result
            try:
                calls = [
                    ("get_shape", {"path": "/values"}),
                    ("read_partial_dataset", {
                        "path": "/values", "start": "0,0", "count": "2,2"
                    }),
                    ("hdf5_aggregate_stats", {
                        "paths": "/values", "stats": "sum,count,mean"
                    }),
                    ("export_dataset", {
                        "path": "/values", "output_path": str(exported),
                        "export_format": "json"
                    }),
                ]
                for name, arguments in calls:
                    result = await client.call_tool(name, arguments)
                    assert not result.is_error, (name, result)
                    print(name, result.content)
                assert json.loads(exported.read_text())["data"] == expected
            finally:
                await client.call_tool("close_file", {})
    print("PASS: installed HDF5 MCP exported the exact dataset values")

asyncio.run(asyncio.wait_for(check(), timeout=300))
PY
```

For a large file, inspect metadata first and select bounded slices.
`hdf5_stream_data` returns summaries for up to `max_chunks`; it does not send all
raw values to the agent. `hdf5_aggregate_stats` can sample large datasets, so read
its coverage labels before interpreting a sum, count, minimum or maximum.
Full reads and exports may materialize the complete dataset in memory.

See the [tool reference](TOOLS.md) for parameters and
[transport configuration](TRANSPORTS.md) for HTTP deployment.
