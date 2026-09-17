# clio-performance

Work out why a finished job was slower than it should have been.

```bash
claude plugin install clio-performance@clio-kit
```

Install the launcher and register the marketplace first; see [setup](../../setup.md).

- **MCPs:** `chronolog`, `darshan`, `parallel-sort`.
- **Skills:** `diagnosing-a-slow-job`, `interpreting-io-performance-numbers`, `recording-a-session-for-provenance`, `searching-large-log-files`.
- **Expected output:** An evidence-backed account of observed I/O or log behavior, with likely bottlenecks and uncertainties.
- **Prerequisites:** Requires supported logs and the relevant profiling/logging backends. Measurements depend on the workload and filesystem.

This primary bundle names dependencies; it contains no copy of server code or
skill instructions. Edit its membership in `mcp-server-versions.toml` and
regenerate the manifest. Task plugins can reuse these components across bundle
boundaries; see [plugin composition](../../docs/plugins.md).

Native installation targets Claude Code. Other clients need their own MCP
configuration and skill installation. Consult the [validation guide](../../docs/marketplace.md#scientific-acceptance-boundaries)
for tested coverage; installing a bundle does not prove every backend operation.
