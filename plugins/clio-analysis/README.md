# clio-analysis

Turn a dataset into summary statistics and a figure.

```bash
claude plugin install clio-analysis@clio-kit
```

Install the launcher and register the marketplace first; see [setup](../../setup.md).

- **MCPs:** `pandas`, `paraview`, `plot`.
- **Skills:** `chart-select`, `data-clean`, `results-summary`, `simulation-visualize`.
- **Expected output:** Checked summaries and figures, with input identity, transformations and assumptions recorded.
- **Prerequisites:** Requires accessible datasets; ParaView workflows also need a working ParaView environment. Check units and assumptions before interpreting figures.

This primary bundle names dependencies; it contains no copy of server code or
skill instructions. Edit its membership in `mcp-server-versions.toml` and
regenerate the manifest. Task plugins can reuse these components across bundle
boundaries; see [plugin composition](../../docs/plugins.md).

Native installation targets Claude Code. Other clients need their own MCP
configuration and skill installation. Consult the [validation guide](../../docs/marketplace.md#scientific-acceptance-boundaries)
for tested coverage; installing a bundle does not prove every backend operation.
