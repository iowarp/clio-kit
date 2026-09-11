# Contributing to CLIO Kit

Thank you for your interest in contributing to CLIO Kit! This guide will help you get started with development, testing, and submitting contributions.

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
```

For client setup and contribution checks, use the
[Agent integration guide](README.md#agent-integrations). It covers MCP and skill
configuration for Codex, Claude Code, Cursor, VS Code / GitHub Copilot,
Antigravity, and Claude Desktop's local MCP route. Record which client you
actually exercised; a valid portable skill does not prove native plugin support.

### For a Specific MCP Server

```bash
# Navigate to the server directory
cd clio-kit-mcp-servers/hdf5

# Install dependencies
uv sync --all-extras --dev
```

## Project Structure

CLIO Kit uses a **monorepo architecture** with a unified launcher:

```
clio-kit/
├── src/clio_kit/              # Unified launcher CLI
├── clio-kit-mcp-servers/      # Independent MCP servers
│   ├── hdf5/                  # Each server has:
│   │   ├── src/               # - Source code
│   │   ├── tests/             # - Test suite
│   │   ├── pyproject.toml     # - Dependencies & entry points
│   │   ├── uv.lock            # - Pinned runtime
│   │   ├── clio-server.toml   # - Runtime descriptor (generated)
│   │   └── README.md          # - Documentation
│   └── ...
├── skills/                    # Workflow skills, grouped per bundle
│   └── clio-hpc-skills/
│       └── skills/<name>/     # SKILL.md + evals.md
├── plugins/                   # Workflow bundles (manifests only)
├── community/                 # Entries pointing at outside repositories
├── .claude-plugin/            # Generated marketplace index
├── .github/workflows/         # CI/CD automation
└── pyproject.toml             # Root configuration
```

**Key Principles:**
- Each MCP server is **independently developed and tested**
- Servers are **launched through a single unified command**: `clio-kit mcp-server <name>`
- **Dependency isolation** via individual `pyproject.toml` files
- **Auto-discovery** from each server's `clio-server.toml` descriptor
- **Bundles reference, never copy** the servers and skills they group

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

`evals.md` is required. Generation fails without it. Record the scenarios that
separate the skill's behaviour from the baseline: the exact prompt, checkable
expectations, and the failure modes the agent shows without the skill.

`clio-kit skill validate /path/to/your-skill` enforces these rules for standalone
skills. `clio-kit plugin validate` also applies them to plugin contents, rather than trusting a
reviewer to notice. It refuses a skill whose frontmatter does not parse, whose
`name` disagrees with its folder, that records no scenarios, whose description
does not open with `Use when`, or that carries no `Triggers on` clause. A
missing `Not for X; use Y` boundary and an over-long description are reported
as advisories, because a first skill with nothing to collide against is
legitimately unbounded. Plugin validation also reports description character counts.

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
rest. Contribute it into `clio-kit-mcp-servers/` like any other server, with a
`clio-server.toml` declaring its runtime. Read the rest of this section first:
hosting is a real commitment on both sides.

All runtimes need their dependencies available on first build. Python source is
vendored in the wheel, but `uv sync --frozen` still downloads missing packages.
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
three of those are ignored repo-wide by `.gitignore`, and the wheel is built
from what git tracks, so compiled output placed in any of them is silently
dropped and the server ships unable to start. `bundle/` is matched by nothing
and rides into the wheel correctly. (`node_modules` is already ignored, so it
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

## Running Tests

### Test a Single Server

```bash
cd clio-kit-mcp-servers/hdf5

# Run all tests
uv run pytest -v

# Run specific test file
uv run pytest tests/test_server.py -v

# Run specific test
uv run pytest tests/test_server.py::test_function_name -v

# Run with coverage
uv run pytest --cov=src/ --cov-report=html --cov-report=term
```

### Test All Servers

```bash
# From root directory
for server in clio-kit-mcp-servers/*/; do
    echo "Testing $server"
    cd "$server" && uv run pytest -v && cd - || exit 1
done
```

## Code Quality Standards

We enforce strict code quality standards through automated CI checks. **All checks must pass** before merging.

### Ruff (Linting + Formatting)

```bash
cd clio-kit-mcp-servers/hdf5

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
cd clio-kit-mcp-servers/hdf5

# Run type checking
uv run mypy src/ --ignore-missing-imports
```

### pip-audit (Security)

```bash
cd clio-kit-mcp-servers/hdf5

# Scan for vulnerabilities
uv run pip-audit
```

### Run All Quality Checks (Mimic CI)

```bash
cd clio-kit-mcp-servers/hdf5

uv run ruff check .
uv run ruff format . --check
uv run mypy src/ --ignore-missing-imports
uv run pytest -v --cov=src/
uv run pip-audit
```

## Submitting Pull Requests

### Before Submitting

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Ensure all tests pass**:
   ```bash
   cd clio-kit-mcp-servers/your-server
   uv run pytest -v
   ```

3. **Run quality checks**:
   ```bash
   uv run ruff check .
   uv run ruff format .
   uv run mypy src/
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

Follow these steps to add a new MCP server to the monorepo:

### 1. Create Directory Structure

```bash
# Use kebab-case for directory name
mkdir -p clio-kit-mcp-servers/my-server/src/my_server_mcp
mkdir -p clio-kit-mcp-servers/my-server/tests
```

### 2. Create `pyproject.toml`

```toml
[project]
name = "my-server-mcp"
version = "1.0.0"
description = "Your server description"
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
authors = [
    {name = "IoWarp Team - Gnosis Research Center", email = "grc@illinoistech.edu"}
]

dependencies = [
    "fastmcp>=4.0.3,<5",
    # Add your dependencies
]

[project.scripts]
my-server-mcp = "my_server_mcp.server:main"

[dependency-groups]
dev = [
    "pytest>=9.0.3",
    "pytest-asyncio>=1.1.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "pip-audit>=2.0.0"
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 3. Implement Server (`src/my_server_mcp/server.py`)

```python
from fastmcp import FastMCP
from fastmcp.prompts import Message

mcp = FastMCP("my-server", instructions="Use my_tool to format a labeled count.")

@mcp.tool(
    description="Format a label and count.",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
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

### 4. Create Tests (`tests/test_server.py`)

```python
import pytest
from fastmcp import Client
from my_server_mcp.server import mcp

@pytest.mark.asyncio
async def test_my_tool():
    async with Client(mcp) as client:
        result = await client.call_tool("my_tool", {"param1": "test", "param2": 42})
        assert result.data == "Result: test 42"
```

### 5. Create README.md

Use the standard template from existing servers (see `clio-kit-mcp-servers/hdf5/README.md` as reference).

### 6. Test Your Server

```bash
cd clio-kit-mcp-servers/my-server
uv sync --all-extras --dev
uv run pytest -v
uv run ruff check .
uv run mypy src/
```

### 7. Register and verify discovery

Add `clio-server.toml` in the server directory:

```toml
name = "my-server"
runtime = "python"
version = "1.0.0"
lock = "uv.lock"
entry = "my-server-mcp"
```

Add the server's version, description and category to
`mcp-server-versions.toml`, following an existing entry. From the repository
root, generate manifests with `uv run python scripts/generate_server_json.py`
and website references with
`uv run python scripts/generate_docs.py clio-kit-mcp-servers clio-kit-website`.
Review the generated plugin and registry metadata, and add an installed-server
check to CI. For Node and Go, follow [Contributing a Server in Another Language](#contributing-a-server-in-another-language).

```bash
# From root directory
uv run clio-kit mcp-servers

# Your server should appear in the list
uv run clio-kit mcp-server my-server
```

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
uv run mypy src/

# Security scan
uv run pip-audit

# Run server
uv run <server-name>-mcp
```

### Branch Strategy

- **main**: Stable releases (target for PRs)
- **feature/***: Feature branches

### Code Style

- **Formatting**: Ruff (automatic)
- **Imports**: Sorted by Ruff
- **Line length**: 100 characters (Ruff default)
- **Type hints**: Required for all public functions
- **Docstrings**: Required for all public functions

---

**Thank you for contributing to CLIO Kit!**

Your contributions help advance AI integration in scientific computing. 

For more information, visit:
- **Website**: [https://toolkit.iowarp.ai/](https://toolkit.iowarp.ai/)
- **Repository**: [https://github.com/iowarp/clio-kit](https://github.com/iowarp/clio-kit)
- **Gnosis Research Center**: [https://grc.iit.edu/](https://grc.iit.edu/)
