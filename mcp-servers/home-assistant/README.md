# Home Assistant MCP

Read sensors and their history from your own Home Assistant, control devices, and write
automations, scripts, scenes, helpers and dashboards.

## What runs

This server is [`ha-mcp`](https://pypi.org/project/ha-mcp/) **9.2.0** by
homeassistant-ai, run exactly as published. None of its code is copied here:
`pyproject.toml` pins the release, `uv.lock` locks its tree, and the launcher installs it
into this server's own environment (Python 3.13, which ha-mcp requires).
`home_assistant_mcp.server` only runs the package's own `ha-mcp` command from that
environment (as its own program), so its tools, resources and results reach the client
unchanged, and it reports its own name and version.

```bash
HOMEASSISTANT_URL=http://homeassistant.local:8123 \
HOMEASSISTANT_TOKEN=<long-lived access token> \
clio-kit mcp-server home-assistant
```

## Configuration

- `HOMEASSISTANT_URL`: the address you open Home Assistant at.
- `HOMEASSISTANT_TOKEN`: a long-lived access token, made in Home Assistant under your
  profile, Security.

Both are required. ha-mcp starts and lists its tools without reaching Home Assistant; a
wrong address or token shows at the first call. ha-mcp's own options (for example
`HA_MCP_DISABLE_SETTINGS_UI=1`, which stops it starting its settings page) are read as
ha-mcp documents them.

## Licence

The wrapper (these files) is BSD-3-Clause, like the rest of CLIO Kit. `ha-mcp` is MIT
and is installed from PyPI, not redistributed here. Home Assistant is a trademark of the
Open Home Foundation; this server is not affiliated with it.

## Tests

```bash
uv sync --all-extras --dev && uv run pytest -v
```
