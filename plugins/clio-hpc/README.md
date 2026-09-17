# clio-hpc

Install a code, wrap it in a pipeline, submit it, and follow it on a cluster.

```bash
claude plugin install clio-hpc@clio-kit
```

Install the launcher and register the marketplace first; see [setup](../../setup.md).

- **MCPs:** `jarvis`, `lmod`, `node-hardware`, `slurm`, `spack`.
- **Skills:** `managing-software-environments`, `running-a-simulation-on-a-cluster`, `writing-slurm-job-scripts`.
- **Expected output:** A prepared environment and pipeline, submission details and observed job status.
- **Prerequisites:** Requires the relevant cluster access, scheduler, module system and software tools; installation does not provision a cluster.

This primary bundle names dependencies; it contains no copy of server code or
skill instructions. Edit its membership in `mcp-server-versions.toml` and
regenerate the manifest. Task plugins can reuse these components across bundle
boundaries; see [plugin composition](../../docs/plugins.md).

Native installation targets Claude Code. Other clients need their own MCP
configuration and skill installation. Consult the [validation guide](../../docs/marketplace.md#scientific-acceptance-boundaries)
for tested coverage; installing a bundle does not prove every backend operation.
