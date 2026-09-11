# Set up CLIO Kit

Use this guide with any agent that supports Agent Skills and/or stdio MCP.
Follow it from the repository root. Install the launcher and marketplace
from the same checkout: the published PyPI package and GitHub default branch do
not yet represent `feat/360-meta-marketplace`. This checkout prepares `2.11.0`;
use the source installation below until that release is published.

## 1. Check prerequisites and the checkout

```bash
uv --version
git branch --show-current
test -f pyproject.toml && test -f .claude-plugin/marketplace.json
```

Skill discovery is tested with Codex 0.154.0; native plugin installation is
tested with Claude Code 2.1.266. Claude is not a prerequisite for portable
skills or MCP servers. If `uv` is missing, install it using [the official instructions](https://docs.astral.sh/uv/getting-started/installation/).

If you do not have this checkout yet:

```bash
git clone --branch feat/360-meta-marketplace https://github.com/iowarp/clio-kit.git
cd clio-kit
```

## 2. Install the launcher

```bash
uv tool install --force --reinstall ".[verification]"
clio-kit mcp-servers
```

Expect 22 servers. If `clio-kit` is not on PATH, run `uv tool update-shell`, open
a new shell, and repeat the inventory command. Do not proceed until it works.
The verification extra provides the MCP client used by `doctor --connect` and
`server inspect`. Server dependencies remain isolated and install on first use;
allow network access and time for that first start.

## 3. Choose a workflow and your agent's installation route

Use the user's stated work to select a bundle. Ask only if the work is unknown.

| Work | Bundle |
|---|---|
| Spack software, JARVIS pipelines, Slurm jobs | `clio-hpc` |
| I/O profiling, application logs, session provenance | `clio-performance` |
| HDF5, ADIOS BP5, Parquet, compressed files | `clio-scientific-io` |
| Tabular statistics, plots, ParaView | `clio-analysis` |
| Geospatial, terrain, seismic data | `clio-geoscience` |
| Papers and dataset discovery | `clio-research` |

### Client configuration and skill locations

Choose your client below. The [README integration guide](README.md#agent-integrations)
contains complete scientific I/O examples for every listed client, including all
four MCP servers and the three workflow skills.

| Client | MCP configuration | Project skill target |
|---|---|---|
| Codex CLI / IDE extension | `codex mcp add`; `~/.codex/config.toml` | `.agents/skills` |
| Claude Code | Native marketplace below, or `claude mcp add --scope project` | Bundle-managed, or `.claude/skills` |
| Cursor | `.cursor/mcp.json`, top-level `mcpServers` | `.cursor/skills` |
| VS Code / GitHub Copilot | `.vscode/mcp.json`, top-level `servers` | `.github/skills` |
| Antigravity | MCP settings → raw config; workspace `.agents/mcp_config.json` | `.agents/skills` |
| Claude Desktop | Developer settings → `claude_desktop_config.json` | Local MCP setup does not install skills |

For example, install the skills for Cursor with:

```bash
clio-kit skill install --bundle clio-scientific-io --target .cursor/skills
```

Substitute the table's target for your agent.
Current Cursor, VS Code, and Antigravity also discover `.agents/skills`,
so reuse an existing project installation there instead of duplicating it.
Antigravity's current global paths are `~/.gemini/config/mcp_config.json` and
`~/.gemini/config/skills`.

The Codex commands below apply to its IDE extension as well as its CLI. In VS
Code, Copilot and Codex use different MCP configurations. Install the launcher
where the MCP subprocess runs, including in a remote workspace or container.
Follow step 4 after configuring any client. The client-specific directories and
schemas are linked to official documentation in the README; that guidance does
not imply every client UI has been exercised in this release's acceptance tests.

### Codex and other agents that support Agent Skills

Install standard skill folders into your agent's documented discovery directory.
For Codex, use `.agents/skills` inside the project where you will work, or
`~/.agents/skills` for user-wide discovery. See [Codex skill discovery](https://developers.openai.com/codex/skills).
For example, from that project:

```bash
clio-kit skill list --bundle clio-scientific-io
clio-kit skill install --bundle clio-scientific-io --target .agents/skills
```

Omit `--bundle` to install all 20 skills, or give individual skill names before
`--target`. Other agents can use the same command with their own skill directory.
Skills are included in the installed CLIO Kit package; the checkout is not
needed for subsequent skill installation.

Skill folders contain instructions. Configure their required MCP servers
separately in your agent. For the scientific I/O workflow in Codex:

```bash
codex mcp add clio-hdf5 -- clio-kit mcp-server hdf5
codex mcp add clio-adios -- clio-kit mcp-server adios
codex mcp add clio-parquet -- clio-kit mcp-server parquet
codex mcp add clio-compression -- clio-kit mcp-server compression
codex mcp list
```

Start Codex in the target project and check `/skills` for the three installed
skills. Restart the client if it has not refreshed discovery. Check the client's
MCP tool inventory too: configuration registration alone does not prove a live
connection. Other MCP clients should configure command `clio-kit` with arguments
`["mcp-server", "NAME"]` using their own configuration schema.

The `.claude-plugin` catalogue, dependency bundles and `clio-agents` definitions
currently target Claude Code. They are not universal plugin/agent manifests;
Codex and other agents use the portable skill and MCP route above.

### Claude Code native marketplace

Use a client that supports plugin dependencies; older clients may not install
all bundle members. From the CLIO Kit checkout:

```bash
claude --version
claude plugin marketplace add "$PWD"
claude plugin marketplace list
```

Expect `clio-kit` registered with this checkout as its source. If that name is
already registered from another source, inspect it before replacing it. Keep
the checkout available for later marketplace updates.

For example, install the scientific file workflow:


```bash
claude plugin install clio-scientific-io@clio-kit
claude plugin details clio-scientific-io@clio-kit
```

This installs four servers and the associated skills as dependencies. A single
server is also installable, for example `clio-hdf5@clio-kit`. For procedures
without servers, use `clio-scientific-io-skills@clio-kit` or `clio-skills@clio-kit`
for all 20 skills. `clio-agents@clio-kit` adds planning and evidence-review agents.

`claude plugin marketplace list` lists marketplaces, not their entries. Refer
to the [README catalogue](README.md#workflow-bundles) for bundle names.

## 4. Verify connections and a real tool result

For every client, check the installed launcher independently:

```bash
clio-kit doctor --server hdf5 --connect
```

In Claude Code, also run `claude mcp list` and
`claude plugin details clio-scientific-io-skills@clio-kit`, then use
`/reload-plugins` or restart an existing session. Every installed CLIO server
should connect. `plugin list` showing `enabled` proves registration only.

For the scientific I/O bundle, run this bounded compression check. It creates
only temporary files, calls the real installed server over MCP, and verifies
that decompression restores the exact bytes:

```bash
uv run --no-project --with 'mcp>=1.20,<2' python - <<'PY'
import asyncio
import gzip
import tempfile
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def check():
    with tempfile.TemporaryDirectory(prefix="clio-setup-") as directory:
        source = Path(directory) / "check.txt.gz"
        expected = b"CLIO setup verification\n" * 10
        source.write_bytes(gzip.compress(expected))
        parameters = StdioServerParameters(
            command="clio-kit", args=["mcp-server", "compression"]
        )
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "decompress_file_tool", {"file_path": str(source)}
                )
                assert not result.isError, result
                assert source.with_suffix("").read_bytes() == expected
    print("PASS: installed MCP server restored the exact input bytes")

asyncio.run(asyncio.wait_for(check(), timeout=300))
PY
```

For another bundle, additionally exercise a small representative tool on known
input and check its output. HPC and native visualization workflows need their
site software/services; a connection is not evidence that those backends work.
Report which checks passed and which prerequisites remain unavailable.

Skill names and descriptions help the client select procedures; full bodies
load when used. `plugin details` is useful for inspecting installed components,
but its token estimate is not an exact per-conversation bill.

## Update and troubleshoot

After updating this checkout, reinstall the launcher with step 2. For portable
skills, rerun `clio-kit skill install` with the same selection and target.
Identical folders are left intact. If files differ, review your local edits
before adding `--replace`; it replaces only selected skill folders.

For Claude Code plugins:

```bash
claude plugin marketplace update clio-kit
claude plugin update clio-scientific-io@clio-kit
```

Upstream plugin content changes need version bumps. Maintainers can refresh
external catalogue snapshots with `clio-kit marketplace refresh --root .`.
See the [marketplace guide](clio-kit-website/docs/marketplace.md) for contribution
and multi-language runtime instructions.

- **Unknown plugin:** verify the marketplace source and the README name. A
  catalogue from `main` differs from this feature branch.
- **Executable not found:** check `command -v clio-kit` in the client's environment.
- **Connection failure:** run `clio-kit doctor --server NAME --connect` for the
  specific server; check network access and backend prerequisites. A directly
  launched stdio server may wait for protocol input rather than print a result.
- **Missing tools in an existing session:** reload plugins or restart the client.

## Remove this installation when requested

Do not run this section during setup. For portable skills, remove only the
skill folders you installed from the chosen target. Remove corresponding MCP
registrations through your client's settings or commands. Do not delete the
whole discovery directory if it contains other skills.

For Claude Code, substitute the bundle you installed:

```bash
claude plugin uninstall clio-scientific-io@clio-kit
claude plugin prune --dry-run
claude plugin prune --yes
claude plugin marketplace remove clio-kit
```

Inspect the prune preview first: it covers orphaned dependencies at the selected
scope. To reclaim old runtime environments, stop active servers, preview with
`clio-kit cache gc --keep 1 --dry-run`, then run `clio-kit cache gc --keep 1`.
This retains the newest environment per server. `--all` is not a supported flag.
Remove the launcher separately with `uv tool uninstall clio-kit` if it is no
longer needed by any client.
