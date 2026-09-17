# clio-scientific-io

Open a scientific data file, see its structure, and read part of it.

```bash
claude plugin install clio-scientific-io@clio-kit
```

Install the launcher and register the marketplace first; see [setup](../../setup.md).

- **MCPs:** `adios`, `compression`, `hdf5`, `parquet`.
- **Skills:** `choosing-a-storage-format`, `exploring-an-unfamiliar-dataset`, `reading-large-datasets-safely`.
- **Expected output:** File structure, dataset metadata and bounded reads with their limits stated.
- **Prerequisites:** Requires accessible data files and format dependencies. Bounded reads are samples, not whole-dataset statistics.

This primary bundle names dependencies; it contains no copy of server code or
skill instructions. Edit its membership in `mcp-server-versions.toml` and
regenerate the manifest. Task plugins can reuse these components across bundle
boundaries; see [plugin composition](../../docs/plugins.md).

Native installation targets Claude Code. Other clients need their own MCP
configuration and skill installation. Consult the [validation guide](../../docs/marketplace.md#scientific-acceptance-boundaries)
for tested coverage; installing a bundle does not prove every backend operation.
