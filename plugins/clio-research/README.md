# clio-research

Find papers on a topic and the datasets behind them.

```bash
claude plugin install clio-research@clio-kit
```

Install the launcher and register the marketplace first; see [setup](../../setup.md).

- **MCPs:** `arxiv`, `ndp`, `scientific-catalog`, `web`.
- **Skills:** `bibliography`, `dataset-stage`, `research-survey`.
- **Expected output:** A sourced literature/dataset summary and staging status for any requested resources.
- **Prerequisites:** Requires network access; catalogue and staging workflows may require configured endpoints, credentials and storage.

This primary bundle names dependencies; it contains no copy of server code or
skill instructions. Edit its membership in `mcp-server-versions.toml` and
regenerate the manifest. Task plugins can reuse these components across bundle
boundaries; see [plugin composition](../../docs/plugins.md).

Native installation targets Claude Code. Other clients need their own MCP
configuration and skill installation. Consult the [validation guide](../../docs/marketplace.md#scientific-acceptance-boundaries)
for tested coverage; installing a bundle does not prove every backend operation.
