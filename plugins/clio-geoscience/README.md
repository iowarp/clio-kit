# clio-geoscience

Inspect and map geospatial features, terrain models, and waveform archives.

```bash
claude plugin install clio-geoscience@clio-kit
```

Install the launcher and register the marketplace first; see [setup](../../setup.md).

- **MCPs:** `geo`, `seismology`, `terrain`.
- **Skills:** `seismic-analysis`, `geospatial-map`, `coordinate-systems`.
- **Expected output:** Maps or waveform summaries with their coordinate, unit and sampling assumptions recorded.
- **Prerequisites:** Requires supported vector, terrain or waveform files and the relevant backends. Confirm coordinate systems and units before combining results.

This primary bundle names dependencies; it contains no copy of server code or
skill instructions. Edit its membership in `mcp-server-versions.toml` and
regenerate the manifest. Task plugins can reuse these components across bundle
boundaries; see [plugin composition](../../docs/plugins.md).

Native installation targets Claude Code. Other clients need their own MCP
configuration and skill installation. Consult the [validation guide](../../docs/marketplace.md#scientific-acceptance-boundaries)
for tested coverage; installing a bundle does not prove every backend operation.
