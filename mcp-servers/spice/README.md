# SPICE MCP

Simulate circuits with ngspice or LTspice and plot their waveforms in an interactive
chart (an MCP Apps view).

## What runs

This server is [`ltspice-mcp`](https://pypi.org/project/ltspice-mcp/) **0.8.0** by
cognitohazard, run exactly as published. None of its code is copied here:
`pyproject.toml` pins the release, `uv.lock` locks its tree, and the launcher installs it
into this server's own environment. `spice_mcp.server` only runs the package's own
`ltspice-mcp` command from that environment as a separate program (it is never
imported), so its tools, its `ui://ltspice-mcp/plot` view and their results reach the
client unchanged, and it reports its own name and version.

```bash
clio-kit mcp-server spice
```

## The simulator is yours

ltspice-mcp runs **ngspice** or **LTspice** already installed on this machine and finds
it itself (see its `spice://config` resource); it downloads neither. Its own
configuration file (`ltspice-mcp.toml`, or `--config PATH` after the server name) is read
as ltspice-mcp documents it.

## Tools

The upstream server's own: `run_experiments`, `jobs`, `analyze_results`,
`plot_waveform` (opens the chart view), `inspect`, `edit_schematic`, `verify_circuit` and
`run_code`; read `spice://guide` first.

## Licence

The wrapper (these files) is BSD-3-Clause, like the rest of CLIO Kit. `ltspice-mcp` is
**GPL-3.0-or-later**. It is installed from PyPI by the lock, not redistributed here, and
it runs as its own program: the wrapper starts it and never imports or links it.
ngspice (BSD-3-Clause) and LTspice (Analog Devices' licence) are installed separately.

## Tests

```bash
uv sync --all-extras --dev && uv run pytest -v
```
