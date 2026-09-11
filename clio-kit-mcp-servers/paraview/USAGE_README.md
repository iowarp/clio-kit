# ParaView native setup

Install CLIO Kit using the [setup guide](../../setup.md). ParaView also requires
a native `pvserver`, matching Python bindings and their shared libraries.
Installing the MCP's Python dependencies does not install ParaView itself.

## Match the native environment

Use a ParaView distribution or build with Python support. Set `UV_PYTHON` to a
Python interpreter matching that build's ABI, `PYTHONPATH` to its Python modules,
and `LD_LIBRARY_PATH` to required shared-library directories on Linux. Put these
settings in the MCP client's per-server environment as well as your test shell.
Do not mix a Python 3.11 extension with a Python 3.13 process.

Start the native service in a separate terminal:

```bash
/path/to/paraview/bin/pvserver --server-port=11111
```

Then configure the MCP client to launch:

```bash
clio-kit mcp-server paraview -- --server 127.0.0.1 --pv-port 11111
```

The default native server may exit after its client disconnects; restart it
before another session. A GUI and MCP connection may need separate server
instances or a native multi-client configuration.

## Verify rendering

Check `clio-kit doctor --server paraview --connect`, then use a real MCP client
to create a sphere, compute its surface area and save a PNG screenshot. Confirm
the returned file exists and opens. Headless rendering requires a compatible
display, such as Xvfb, or an EGL/OSMesa-enabled native build.

MPI rendering requires an MPI-enabled ParaView build. Starting multiple copies
of a serial `pvserver` with `mpirun` does not make it distributed. Verify the
native connection reports the expected data partitions before testing that path.

## Source build helpers

From the repository root:

```bash
cd clio-kit-mcp-servers/paraview
uv sync --frozen --dev
uv run --frozen paraview-mcp --help
```

The project's `scripts/` directory contains optional build, dependency and
configuration helpers. Review their platform assumptions before running them;
the native ABI and rendering checks above are still required. See the
[server README](README.md) for tools and the
[agent integration guide](../../README.md#agent-integrations) for client setup.
