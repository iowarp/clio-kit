# Copernicus MCP

Search, subset and download Copernicus Marine Service (CMEMS) products and Copernicus
Climate Data Store (CDS) datasets such as ERA5. Large results are files with provenance,
and long jobs survive restarts.

## What runs

This server is [`copernicus-mcp`](https://pypi.org/project/copernicus-mcp/) **0.7.2** by
CliDyn, with its Marine and CDS backends, run exactly as published. None of its code is
copied here: `pyproject.toml` pins the release, `uv.lock` locks its tree, and the
launcher installs it into this server's own environment. `copernicus_kit_mcp.server`
only runs the package's own `copernicus-mcp serve` from that environment (as its own
program), so its tools and their results reach the client unchanged.

```bash
clio-kit mcp-server copernicus
```

## Accounts

Both services need a free account, read from the environment by the upstream server:

| Service | Variables | Where |
| --- | --- | --- |
| Copernicus Marine | `COPERNICUSMARINE_SERVICE_USERNAME`, `COPERNICUSMARINE_SERVICE_PASSWORD` | <https://data.marine.copernicus.eu/register> |
| Climate Data Store | `CDSAPI_KEY` (and optionally `CDSAPI_URL`) | <https://cds.climate.copernicus.eu>, your profile's API token; accept each dataset's licence once on its page |

The upstream server lists its CDS tools only when a CDS key is set; the Marine tools are
always listed and say when the Marine account is missing. In CLIO both come from the
Copernicus connector's fields, stored as secrets.

## Tools

The upstream server's own, among them `marine_search_datasets`,
`marine_describe_dataset`, `marine_estimate_subset`, `marine_subset_dataset`,
`marine_check_status`, `marine_get_files`, `cds_search_datasets`,
`cds_describe_dataset`, `cds_estimate_request`, `cds_submit_request`,
`cds_check_request_status` and `cds_download_request_result`.

## Licence

The wrapper (these files) is BSD-3-Clause, like the rest of CLIO Kit.
`copernicus-mcp` is BSD-3-Clause (Ivan Kuznetsov, CliDyn) and is installed from PyPI, not
redistributed here. Each Copernicus product carries its own data licence, recorded in
the upstream server's provenance.

## Tests

```bash
uv sync --all-extras --dev && uv run pytest -v
```
