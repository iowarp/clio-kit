# Contributing to CLIO Kit

Contribute to the CLIO Kit meta-marketplace: add skills, MCP servers, plugins or agent definitions, or index an external marketplace. This guide covers development, validation and pull requests.

For a short authoring walkthrough, see [plugins, MCPs, skills and hooks](docs/authoring.md).
MCPs, skills, agents and hooks are components; a workflow plugin bundles the
components a task needs. Single-type collections and native installation wrappers
are listed separately from workflow plugins; see [the component model](docs/plugins.md).
For repository-owned packages, use the [folder contribution route](docs/authoring.md#add-a-component-folder-to-clio-kit): valid packages in `plugins/`, `skills/`, `agents/` and `hooks/` are discovered without central TOML registration.
Agents should start with [AGENTS.md](AGENTS.md).

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Contributing a Skill](#contributing-a-skill)
- [Contributing an MCP Server or Plugin You Maintain](#contributing-an-mcp-server-or-plugin-you-maintain)
- [Running Tests](#running-tests)
- [Code Quality Standards](#code-quality-standards)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Adding a New MCP Server](#adding-a-new-mcp-server)
- [Issue Reporting](#issue-reporting)
- [Community & Support](#community--support)

## Development Setup

### Prerequisites

- **Python 3.10+** (required)
- **[UV package manager](https://docs.astral.sh/uv/)** (recommended for dependency management)
- **Git** for version control

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/iowarp/clio-kit.git
cd clio-kit

# Install all dependencies (development mode)
uv sync --all-extras --dev

# Install the launcher from this checkout, so `clio-kit` runs the code you edit
uv tool install --force --reinstall --editable ".[verification]"
```

This guide calls `clio-kit` directly. A `clio-kit` installed any other way (for
example `uv tool install clio-kit`) reads the release it was installed from, not
your checkout, and nothing warns you. Check with `clio-kit --version`, or use
`uv run clio-kit ...` from the repository root, which always runs the checkout.
`clio-kit marketplace sync` and `marketplace refresh` act on the checkout named
by `--root`, which defaults to the current directory: run them from the
repository root or pass `--root` explicitly.

For client setup and contribution checks, use the
[Agent integration guide](README.md#agent-integrations). It covers MCP and skill
configuration for Codex, Claude Code, Cursor, VS Code / GitHub Copilot,
Antigravity, and Claude Desktop's local MCP route. Record which client you
actually exercised; a valid portable skill does not prove native plugin support.

### For a Specific MCP Server

```bash
# Navigate to the server directory
cd mcp-servers/hdf5

# Install dependencies
uv sync --all-extras --dev
```

## Project Structure

CLIO Kit uses a **monorepo architecture** with a unified launcher:

```
clio-kit/
├── src/clio_kit/              # Unified launcher CLI
├── mcp-servers/              # Independent MCP servers
│   ├── hdf5/                  # Each server has:
│   │   ├── src/               # - Source code
│   │   ├── tests/             # - Test suite
│   │   ├── pyproject.toml     # - Dependencies & entry points
│   │   ├── uv.lock            # - Pinned runtime
│   │   ├── clio-server.toml   # - Runtime descriptor (generated for Python,
│   │   │                      #   handwritten for Node and Go)
│   │   └── README.md          # - Documentation
│   └── ...
├── skills/                    # Workflow skills, grouped per bundle
│   └── clio-hpc-skills/
│       └── skills/<name>/     # SKILL.md + evals.md
├── plugins/                   # Server plugins, bundles and agent definitions
├── community/                 # External plugin and marketplace entries
├── .claude-plugin/            # Generated marketplace index
├── docs/                     # Shared human documentation
├── website/          # Website configuration and UI
├── AGENTS.md                 # Agent contribution entry point
├── .github/workflows/         # CI/CD automation
└── pyproject.toml             # Root configuration
```

**Key Principles:**
- Each MCP server is **independently developed and tested**
- Servers are **launched through a single unified command**: `clio-kit mcp-server <name>`
- **Dependency isolation** via per-server runtime descriptors and lock files
- **Auto-discovery** from each server's `clio-server.toml` descriptor
- **Bundles reference, never copy** the servers and skills they group
- **External contributions stay with their maintainers**; the marketplace indexes their sources

## Contributing a Skill

A skill is a written procedure for a tool sequence that is easy to get wrong. It
belongs here only when it spans more than one server, when call order matters
with a real penalty, or when two tools look interchangeable and are not. Anything
a single server can explain on its own belongs in that server's tool
descriptions, where it ships with the server and cannot fall out of sync.

Add a folder under the bundle it serves:

```
skills/clio-<bundle>-skills/skills/<your-skill>/
├── SKILL.md
└── evals.md
```

Skills must work with agents that support the standard
[Agent Skills format](https://agentskills.io/specification), including Codex.
`SKILL.md` frontmatter needs a `name` matching the folder and a `description`.
Put `bundle`, `servers`, `provenance` and `eval-status` under standard `metadata`
as string values. Use live MCP discovery to resolve client-specific tool names;
do not require Claude-only commands inside shared skill procedures.
Write the description as triggers only, in the words a user actually types, and
add a `Not for X; use Y` clause wherever another skill could plausibly claim the
same request. Descriptions support discovery; full bodies load when invoked. Keep
descriptions concise and put procedural detail in the body.

A minimal `SKILL.md` that passes validation as written:

```markdown
---
name: inspecting-simulation-output
description: 'Use when inspecting an unfamiliar simulation dataset. Triggers on "inspect simulation output". Not for plotting; use results-summary.'
metadata:
  bundle: clio-scientific-io
  servers: clio-hdf5
  provenance: designed
  eval-status: scenarios-recorded
---

Discover the available scientific I/O tools. Inspect the file metadata and
dataset shapes first, then read a bounded slice. Report the slice bounds and
units; do not treat a sample as a whole-dataset summary.
```

`metadata.bundle` is the bundle that lists the skill (`clio-<bundle>`, without
`-skills`); `servers` is a comma-separated list of plugin names, or `none`.
`eval-status` says how far the skill has been checked and must be one of,
weakest first: `untested`, `scenarios-recorded`, `trigger-checked`,
`smoke-checked`, `eval-run`.

`evals.md` is required and must not be empty. Record the scenarios that separate
the skill's behaviour from the baseline: the exact prompt, checkable
expectations, and the failure modes the agent shows without the skill. For the
example above:

```markdown
# Evals - inspecting-simulation-output

## S1 - unfamiliar HDF5 file

Setup: Prompt: "Inspect simulation output in run.h5 and tell me what it holds."

Expected:

- Lists the datasets with their shapes and units before reading any values.
- Reads a bounded slice and states its bounds.

Without the skill: reads a whole dataset, or reports a sample as the whole file.
```

Validate the skill, then the collection it joins:

```bash
clio-kit skill validate --maintained skills/clio-scientific-io-skills/skills/inspecting-simulation-output
clio-kit plugin validate --maintained skills/clio-scientific-io-skills
```

`skill validate` refuses a skill whose frontmatter does not parse, whose `name`
disagrees with its folder, that records no scenarios, whose description does not
open with `Use when`, or that carries no `Triggers on` clause. `--maintained`
adds the two things a skill in a `clio-` package owes: `metadata.bundle` and
`metadata.eval-status`. A missing `Not for X; use Y` boundary and an over-long
description are reported as advisories, because a first skill with nothing to
collide against is legitimately unbounded. `clio-kit plugin validate` applies
the same rules to every skill in a package and reports description character
counts.

A skill added to a maintained bundle also needs, in the same pull request:

1. **An evaluation case.** Add one `case("<skill-name>", "<prompt>", ...)` to
   `CASES` in `evals/codex_cases.py`. The root suite fails, naming the skill,
   while a maintained skill has none.
2. **A version bump in the inventory.** Raise `[bundles.clio-<bundle>].version`
   in `mcp-server-versions.toml`. The skills plugin's manifest
   (`skills/clio-<bundle>-skills/.claude-plugin/plugin.json`) is generated from
   that table, so a version edited in the manifest is reverted by the next sync.
3. **A catalogue sync before the root suite**, as CI does:

   ```bash
   clio-kit marketplace sync --root .
   uv run --frozen pytest tests -q
   ```

Skills in your own package (`plugins/<name>/skills/`, `skills/<package>/`) follow
the same blocking rules without `--maintained`; they need no evaluation case and
no inventory entry.

## Contributing a Server in Another Language

TypeScript and Go servers are supported two ways. Which to pick is a question
about ownership, not capability.

**Index it** — the server stays yours. Publish a plugin containing its manifest
and MCP configuration to npm, then add one entry under
[`community/`](community/README.md) with `type = "npm"`. Your code, your
dependencies and your release schedule stay in your repository, and it installs
through our marketplace exactly like ours do.

```toml
name        = "crystal-ts"
description = "Crystallography tools for materials workflows."
maintainer  = "some-lab"

[source]
type    = "npm"
package = "@some-lab/crystal-mcp"
version = "^1.0.0"
```

**Host it** — the server ships as part of the kit, and we maintain it with the
rest. Contribute it into `mcp-servers/` like any other server, with a
`clio-server.toml` declaring its runtime. Read the rest of this section first:
hosting is a real commitment on both sides.

All runtimes need their dependencies available on first build. Server source and
locks are separate release artifacts, fetched only when selected. The Python
runtime then uses `uv sync --frozen` to obtain missing dependencies.
Node uses `npm ci`; Go may download modules before compilation. Prepare and test
the runtime cache on the target platform before using a host without outbound
network access. Indexing an external server does not make its installation offline.

### Hosting one

The launcher builds and starts node and go from their own locks, the same way
it does Python, declared in `clio-server.toml`:

```toml
name    = "crystal"
runtime = "node"              # python | node | go
version = "1.0.0"
entry   = "bundle/server.js"  # go: the main package, e.g. ./cmd/server
description = "Crystallography tools."
```

Both paths are covered end to end in `tests/test_runtimes.py`, each building
a fixture server from its lock and reading back a JSON-RPC `initialize` reply.
The go fixture is deliberately a two-package module: while only node had such a
test, go shipped a build command that could not compile one. `go build -o
<file> ./...` fails with `cannot write multiple packages to non-directory`, so
it worked for a single-package module but failed for multi-package projects. The build
compiles the package `entry` names instead.

`entry` means the same thing in all three: the thing that runs. Python names
its console script, node the compiled JavaScript, go the main package to
compile. `lock` is not yours to choose -- which file pins a runtime follows
from the runtime, so a descriptor naming a different one is refused rather
than quietly ignored.

**A TypeScript server must commit its compiled JavaScript.** The build runs
`npm ci --omit=dev`, so `typescript` is a devDependency that is never installed
and no compile happens at launch. The compiled output is hashed into the
environment identity for node servers precisely because it is the artifact that
runs — unlike Python, where build output is throwaway and excluded.

**Name that output directory `bundle/`, not `dist/`, `lib/` or `build/`.** All
three of those are ignored repo-wide by `.gitignore`, and release component artifacts respect the repository's tracked and non-ignored files, so compiled output placed in any of them is silently
dropped and the server ships unable to start. `bundle/` is matched by nothing
and is included in the component artifact. (`node_modules` is already ignored, so it
never ships — which is what you want, since `npm ci` recreates it from the
lock.)

**Your server needs its own CI lane, and CI will not let you skip it.** The
shared matrix runs Python tools — ruff, mypy, pytest — and discovers what to
run them on by looking for `pyproject.toml`. A node or go server is not in it,
so `tests/test_ci_covers_every_server.py` fails on any server no job verifies,
naming it. Add a workflow that lints, type-checks and tests your server with
its own toolchain, then record it in that test's `DEDICATED_WORKFLOWS`. Without
this a hosted server would ship entirely unverified, which is why it is a gate
rather than a guideline.

The manifest generator reads your descriptor for the name, version, entry point
and description that a Python server states in `pyproject.toml`, so a hosted
server reaches the marketplace like any other. A server it cannot describe is
refused by name rather than skipped in silence.

**Every gate a hosted Node or Go server must pass.** Registration is step 7-9 of
[Adding a New MCP Server](#adding-a-new-mcp-server) with these differences; the
root suite (`uv run --frozen pytest tests -q`) checks each one:

| What you add | Checked by |
| --- | --- |
| `clio-server.toml`, written by hand, with `name` equal to the folder, `runtime`, `entry`, `version` and a `description`; the generator never overwrites a non-Python descriptor | `tests/test_discovery.py`, generation |
| The runtime's manifest and lock: `package.json` + `package-lock.json`, or `go.mod` + `go.sum` | `tests/test_discovery.py` |
| For TypeScript, the compiled JavaScript committed under `bundle/` | `clio-kit server inspect` warns about `dist/`, `lib/` and `build/` |
| The inventory edits of step 7: `[servers]`, one primary `[bundles.*]`, `[icons]`, `[mcp-registry-release].publish` | `tests/test_generate_server_json_contract.py` |
| `SERVER_TAGS` entry in `scripts/generate_server_json.py` | same file (marketplace keywords) |
| `README.md` rows: bundle table, server table, `mcp-name` comment, server count | same file |
| A workflow under `.github/workflows/` that lints, type-checks and tests the server, and its entry in `DEDICATED_WORKFLOWS` | `tests/test_ci_covers_every_server.py` |
| Generated files from step 8, including `mcp-servers/<name>/server.json` | `tests/test_generate_server_json_contract.py`, `tests/test_website_catalogue.py` |

Generation starts a hosted server over stdio to read its tools, so install the
verification extra (`uv sync --all-extras --dev`) and the runtime's toolchain
(`npm`, or `go`) before step 8. A server that cannot start is not published.

Selected Node and Go servers use a real stdio MCP session for registry metadata
extraction. Install `.[verification]` for this operation. Without a `[registry]`
table, a hosted server uses the shared `clio-kit` wheel coordinate. A descriptor
can instead declare `registryType`, `identifier`, `version`, and transport in a
`[registry]` table for npm, OCI, PyPI, NuGet, or MCPB distribution. Registry
coordinates must refer to artifacts you separately publish; generation does
not upload a package. Failed live extraction prevents publication.

The marketplace acceptance workflow exercises real SDK-based Node/TypeScript
and Go projects under `tests/fixtures/mcp-servers/`, using a freshly installed
wheel. These fixtures are not marketplace entries or wheel runtime components.
For your own project, use `clio-kit server run /path/to/project` and
`clio-kit server inspect /path/to/project`. Inspection requires the verification
extra and checks the actual stdio protocol.

## Contributing an MCP Server or Plugin You Maintain

Servers and plugins you maintain yourself are indexed rather than copied here.
Your code stays in your repository on your own release schedule.

```bash
clio-kit plugin init my-plugin
clio-kit plugin validate my-plugin
clio-kit plugin submit my-plugin --repo owner/name --open-pr
```

`validate` catches problems that a manifest check cannot, most importantly a
component path that leaves the plugin directory: it works in your checkout and
resolves to nothing once installed, because nothing outside the plugin root is
copied to the cache.

See [`community/README.md`](community/README.md) for accepted source types and
what we review.

For Claude Code event hooks, add `--hook` to `plugin init`. It creates a read-only
`SessionStart` example requiring `python3`. The validator checks default, custom-file
and inline hook definitions without executing them; still run the client's strict
validation. See the [hook guide](community/README.md#hooks) for runtime tests and host limits.

## Running Tests

### Test a Single Server

```bash
cd mcp-servers/hdf5

# Run all tests
uv run pytest -v

# Run specific test file
uv run pytest tests/test_server.py -v

# Run specific test
uv run pytest tests/test_server.py::test_function_name -v

# Run with coverage
uv run --with pytest-cov pytest --cov=src/ --cov-report=html --cov-report=term
```

Every server's own environment has `pytest` and `ruff`. `mypy`, `pip-audit` and
`pytest-cov` are missing from several servers' dev groups, and CI installs them
on top of the locked environment. `uv run --with <tool>` does the same locally
without changing the server's lock, so the commands below work in every server.

### Test All Servers

```bash
# From root directory
for server in mcp-servers/*/; do
    echo "Testing $server"
    cd "$server" && uv run pytest -v && cd - || exit 1
done
```

## Code Quality Standards

We enforce strict code quality standards through automated CI checks. **All checks must pass** before merging.

### Ruff (Linting + Formatting)

```bash
cd mcp-servers/hdf5

# Check linting
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Check formatting
uv run ruff format . --check

# Auto-format code
uv run ruff format .
```

### MyPy (Type Checking)

```bash
cd mcp-servers/hdf5

# Run type checking
uv run --with mypy mypy src/ --ignore-missing-imports
```

### pip-audit (Security)

```bash
cd mcp-servers/hdf5

# Scan for vulnerabilities
uv run --with pip-audit pip-audit
```

### Run All Quality Checks (Mimic CI)

```bash
cd mcp-servers/hdf5

uv run ruff check .
uv run ruff format . --check
uv run --with mypy mypy src/ --ignore-missing-imports
uv run --with pytest-cov pytest -v --cov=src/
uv run --with pip-audit pip-audit
```

## Submitting Pull Requests

### Before Submitting

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Ensure all tests pass**:
   ```bash
   cd mcp-servers/your-server
   uv run pytest -v
   ```

3. **Run quality checks**:
   ```bash
   uv run ruff check .
   uv run ruff format .
   uv run --with mypy mypy src/ --ignore-missing-imports
   ```

4. **Update documentation** if needed (README.md, docstrings)

### PR Guidelines

- **Target branch**: `main` (for releases)
- **Clear description**: Explain what changes you made and why
- **Reference issues**: Link related issues (e.g., "Fixes #123")
- **Small, focused changes**: One feature/fix per PR
- **Tests required**: Add tests for new features
- **Documentation**: Update READMEs and docstrings

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass locally
- [ ] Added tests for new features
- [ ] Updated documentation

## Checklist
- [ ] Code follows project style (Ruff)
- [ ] Type hints added (MyPy compliant)
- [ ] No security vulnerabilities (pip-audit)
- [ ] Documentation updated
```

## Adding a New MCP Server

These steps add a Python server named `my-server` and end with a green root
suite. Run them from the repository root unless a step says otherwise. For Node
and Go, read [Contributing a Server in Another Language](#contributing-a-server-in-another-language)
first; steps 7-9 apply to every runtime.

### 1. Create Directory Structure

```bash
# Use kebab-case for the directory, snake_case for the package
mkdir -p mcp-servers/my-server/src/my_server_mcp
mkdir -p mcp-servers/my-server/tests
touch mcp-servers/my-server/src/my_server_mcp/__init__.py
```

### 2. Create `pyproject.toml`

`mcp-servers/my-server/pyproject.toml`:

```toml
[project]
name = "my-server-mcp"
version = "1.0.0"
description = "Format a labeled count"
readme = "README.md"
requires-python = ">=3.10"
license = "BSD-3-Clause"
authors = [
    { name = "IoWarp Team - Gnosis Research Center", email = "grc@illinoistech.edu" }
]
dependencies = [
    "fastmcp>=4.0.3,<5",
    # Add your dependencies
]

[project.scripts]
my-server-mcp = "my_server_mcp.server:main"

[dependency-groups]
dev = [
    "mypy>=1.17.0",
    "pip-audit>=2.10.0",
    "pytest>=9.0.3",
    "pytest-asyncio>=1.1.0",
    "pytest-cov>=6.2.1",
    "ruff>=0.12.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/my_server_mcp"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
asyncio_mode = "auto"
```

The console script must end in `-mcp`. The `description` is what the
marketplace shows. Keep `requires-python = ">=3.10"`: the CI matrix runs every
server on each Python version it lists, and
`tests/test_ci_covers_every_server.py` fails when a server's range and the
matrix disagree.

### 3. Implement Server (`src/my_server_mcp/server.py`)

```python
from fastmcp import FastMCP
from fastmcp.prompts import Message

mcp = FastMCP(
    "my-server",
    version="1.0.0",  # keep equal to mcp-server-versions.toml
    instructions="Use my_tool to format a labeled count.",
)


@mcp.tool(
    description="Format a label and count.",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
    tags={"formatting"},
)
def my_tool(param1: str, param2: int) -> str:
    return f"Result: {param1} {param2}"


@mcp.resource("my-server://capabilities")
def capabilities() -> dict:
    return {"tools": ["my_tool"]}


@mcp.prompt()
def format_count(label: str) -> list[Message]:
    return [Message(f"Use my_tool to format the count for {label}.")]


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

Pass `version=` explicitly. Without it FastMCP reports its own library version
as the server's, and `tests/test_server_version_reporting.py` fails.

### 4. Create Tests (`tests/test_server.py`)

```python
from fastmcp import Client

from my_server_mcp.server import mcp


async def test_my_tool():
    async with Client(mcp) as client:
        result = await client.call_tool("my_tool", {"param1": "test", "param2": 42})
        assert result.data == "Result: test 42"
```

### 5. Create README.md

Use the standard template from existing servers (see `mcp-servers/hdf5/README.md` as reference).

### 6. Lock and Test Your Server

```bash
cd mcp-servers/my-server
uv lock                      # writes uv.lock; commit it
uv sync --all-extras --dev
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ --ignore-missing-imports
cd ../..
```

The launcher refuses a server without `uv.lock`, because it never resolves
dependencies at start-up. Run `uv lock` again whenever `pyproject.toml` changes.

### 7. Register the server

Three handwritten files name every server. Generation or the root suite fails
until all three agree.

**`mcp-server-versions.toml`** — four edits:

```toml
[mcp-registry-release]
publish = [..., "lmod", "my-server", "ndp", ...]    # sorted

[bundles.clio-analysis]                              # exactly one primary bundle
servers = ["my-server", "pandas", "paraview", "plot"]  # sorted

[servers]                                            # sorted; a bare name = "version" line
my-server = "1.0.0"

[icons]
my-server = "🔢"
```

The description comes from `pyproject.toml`. A server is published as
`scientific` unless you also add it to `[classification].general`. Add
`[prerequisites.my-server]` only if the server needs an external executable or
configuration file for `clio-kit doctor` to check.

**`scripts/generate_server_json.py`** — add the marketplace keywords to
`SERVER_TAGS`:

```python
    "my-server": ["formatting", "example"],
```

**`README.md`** — four edits, each checked against the inventory:

- the `<!-- mcp-name: io.github.iowarp/my-server-mcp -->` comment at the top
- the server's name in its bundle's row of the bundle table
- the count in the `# List all N available MCP servers` comment
- a row in the server table beginning `| **`my-server`** | 1.0.0 |`

### 8. Generate manifests and catalogues

```bash
uv run python scripts/generate_server_json.py
uv run python scripts/generate_docs.py mcp-servers website
uv run clio-kit marketplace sync --root .
```

The first command starts every Python server in its own environment to read
its tools, so its first run can take several minutes. Commit what these commands
write for your server:

- `mcp-servers/my-server/clio-server.toml` and `server.json` (both generated
  for a Python server; do not write them by hand)
- `plugins/clio-my-server/` and the bundle's `plugins/<bundle>/.claude-plugin/plugin.json`
- `.claude-plugin/marketplace.json`, `claude_desktop_config.json`, `gemini-extension.json`
- `docs/mcps/my_server.md` and `website/src/data/catalogue.json`

On an up-to-date checkout nothing else changes. If another server's
`server.json` or page is rewritten, leave it out of your pull request and
report it.

### 9. Verify

```bash
uv run clio-kit mcp-servers            # my-server is listed
uv run clio-kit mcp-server my-server   # starts on stdio; Ctrl+C to stop
uv run --frozen pytest tests -q        # root suite
uv run --frozen python scripts/check_file_size.py
```

CI needs no change for a Python server: the shared matrix discovers it through
`pyproject.toml`. Exercise at least one tool through a real MCP client before
opening the pull request; a passing unit test does not show that the launcher
can start the server.

## Issue Reporting

### Bug Reports

When reporting bugs, include:

- **Clear title**: Concise description of the issue
- **Steps to reproduce**: Exact commands/code to trigger the bug
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**:
  - Python version (`python --version`)
  - UV version (`uv --version`)
  - OS and version
  - MCP server name and version

### Feature Requests

When requesting features, include:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought about
- **Additional context**: Any relevant examples or documentation

### Template

```markdown
## Description
Clear description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version: 3.10.x
- UV version: 0.x.x
- OS: Ubuntu 22.04
- MCP server: hdf5-mcp v1.0.0
```

## Community & Support

### Get Help

- **Zulip Chat**: [CLIO Kit Community](https://iowarp.zulipchat.com/#narrow/channel/543872-Agent-Toolkit)
- **GitHub Issues**: [Report bugs](https://github.com/iowarp/clio-kit/issues)

### Contributing Guidelines

- **Be respectful**: Follow our code of conduct
- **Be clear**: Provide context and details
- **Be patient**: Maintainers are volunteers
- **Be collaborative**: Help review others' PRs

### Recognition

Contributors are recognized in:
- **GitHub Contributors**: Automatically listed
- **Release Notes**: Mentioned in CHANGELOG
- **Community**: Featured in project discussions

---

## Quick Reference

### Common Commands

```bash
# Setup
uv sync --all-extras --dev

# Test
uv run pytest -v

# Format
uv run ruff format .

# Lint
uv run ruff check --fix .

# Type check
uv run --with mypy mypy src/ --ignore-missing-imports

# Security scan
uv run --with pip-audit pip-audit

# Run server
uv run <server-name>-mcp
```

### Branch Strategy

- **main**: Stable releases (target for PRs)
- **feature/***: Feature branches

### Code Style

- **Formatting**: Ruff (automatic)
- **Imports**: Sorted by Ruff
- **Line length**: 88 characters (Ruff's default) for the launcher and most servers; `geo`, `ndp`, `scientific-catalog`, `spack` and `web` set 100 in their own `pyproject.toml`
- **Type hints**: Required for all public functions
- **Docstrings**: Required for all public functions

---

**Thank you for contributing to CLIO Kit!**

Your contributions help advance AI integration in scientific computing. 

For more information, visit:
- **Website**: [https://toolkit.iowarp.ai/](https://toolkit.iowarp.ai/)
- **Repository**: [https://github.com/iowarp/clio-kit](https://github.com/iowarp/clio-kit)
- **Gnosis Research Center**: [https://grc.iit.edu/](https://grc.iit.edu/)
