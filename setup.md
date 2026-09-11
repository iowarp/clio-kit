# Set up CLIO Kit

Follow this guide from the repository root. Install the launcher and marketplace
from the same checkout: the published PyPI package and GitHub default branch do
not yet represent `feat/360-meta-marketplace`.

## 1. Check prerequisites and the checkout

```bash
claude --version
uv --version
git branch --show-current
test -f pyproject.toml && test -f .claude-plugin/marketplace.json
```

This guide is tested with Claude Code 2.1.266. Use a client that supports plugin
dependencies; older clients may not install a bundle's members. If `uv` is
missing, install it using [the official instructions](https://docs.astral.sh/uv/getting-started/installation/).

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

## 3. Register this marketplace

```bash
claude plugin marketplace add "$PWD"
claude plugin marketplace list
```

Expect `clio-kit` registered with this checkout as its source. If that name is
already registered from another source, inspect it before replacing it. Keep
the checkout available for later marketplace updates.

## 4. Install the relevant workflow

Use the user's stated work to select a bundle. Ask only if the work is unknown.

| Work | Bundle |
|---|---|
| Spack software, JARVIS pipelines, Slurm jobs | `clio-hpc` |
| I/O profiling, application logs, session provenance | `clio-performance` |
| HDF5, ADIOS BP5, Parquet, compressed files | `clio-scientific-io` |
| Tabular statistics, plots, ParaView | `clio-analysis` |
| Geospatial, terrain, seismic data | `clio-geoscience` |
| Papers and dataset discovery | `clio-research` |

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

## 5. Verify connections and a real tool result

```bash
claude mcp list
clio-kit doctor --server hdf5 --connect
claude plugin details clio-scientific-io-skills@clio-kit
```

Every installed CLIO server should connect. `plugin list` showing `enabled`
proves registration only. In an existing interactive Claude session, use
`/reload-plugins` or restart before using the new tools.

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

After updating this checkout, reinstall the launcher with step 2, then:

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

Do not run this section during setup. Substitute the bundle you installed.

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
