# Earthdata MCP

Search NASA Earthdata (the Common Metadata Repository) for Earth-science datasets and
their files, and download them with a NASA Earthdata Login.

## What runs

This server is [`earthdata-mcp-server`](https://pypi.org/project/earthdata-mcp-server/)
**0.6.0** by Datalayer, run exactly as published. None of its code is copied here:
`pyproject.toml` pins the release, `uv.lock` locks its tree, and the launcher installs it
into this server's own environment. `earthdata_mcp.server` only runs the package's own
`earthdata-mcp-server start --transport stdio` from that environment (as its own
program), so its tools and their results reach the client unchanged.

```bash
clio-kit mcp-server earthdata
```

## Account

Searching datasets and their files is anonymous. Downloading files needs a free NASA
Earthdata Login (<https://urs.earthdata.nasa.gov>), read by `earthaccess` from
`EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` (or `~/.netrc`). Both are optional.

## Tools

The upstream server's own: `search_earth_datasets`, `search_earth_datagranules` and
`download_earth_data_granules`.

## Licence

The wrapper (these files) is BSD-3-Clause, like the rest of CLIO Kit.
`earthdata-mcp-server` is BSD-3-Clause (Datalayer, Inc.) and is installed from PyPI, not
redistributed here.

## Tests

```bash
uv sync --all-extras --dev && uv run pytest -v
```
