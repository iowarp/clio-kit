# JARVIS MCP setup

Install the launcher using the [CLIO Kit setup guide](../../../setup.md).
JARVIS requires Python 3.11 or newer, a writable `JARVIS_ROOT`, and the site
software needed by the selected recipe. The launcher uses the server's locked
Python environment; it does not install a Slurm cluster or external applications.

```bash
clio-kit mcp-server jarvis -- --help
clio-kit doctor --server jarvis --connect
```

Configure your agent with command `clio-kit` and arguments
`["mcp-server", "jarvis"]`. For client-specific MCP and skill setup, use the
[agent integration guide](../../../README.md#agent-integrations).

## Initialize and verify a workflow

Use the administrative profile only for operator setup:

```bash
clio-kit mcp-server jarvis -- --profile all
```

Configure JARVIS's private/shared directories and available recipe repositories
for the site. Inspect the registered admin tool schemas before changing them.
Normal agent sessions should use the default user profile, which exposes six tools.

Search for a recipe with `jarvis_describe(target="package_search")`, then inspect
its canonical package name and configuration with `target="package"`. Create a
small `echo` pipeline, add the configured step, and call `jarvis_run` with
`submit=false`. Use the returned pipeline and execution IDs with
`jarvis_get_execution` and verify completion and expected output.

The shipped JARVIS 1.8.1 scheduler dependency fails when Slurm supplies the literal
`SLURM_CLUSTER_NAME=(null)`. Successful direct execution does not verify scheduler
execution. See the [native backend limits](../../../setup.md#native-backend-setup).

## Local development

From the repository root:

```bash
uv sync --frozen --dev --directory clio-kit-mcp-servers/jarvis
uv run --frozen --directory clio-kit-mcp-servers/jarvis jarvis-mcp --help
uv run --frozen --directory clio-kit-mcp-servers/jarvis pytest -q
```

The [server README](../README.md) documents profiles, Spack environment setup,
execution handles, progress and artifact contracts.
