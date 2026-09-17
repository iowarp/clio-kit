---
title: Selective installation
---

# Install only what you need

The distribution built from this branch separates the launcher from component
payloads. This behavior becomes available to public users when its release
artifacts and launcher are published; an older PyPI release keeps its old layout.
A source checkout remains the development route.

| Selection | Download behavior |
| --- | --- |
| Launcher | CLI, catalogue metadata, contracts and provenance; no server, skill or search payloads |
| MCP | Only that server's source and lock, followed by its isolated runtime dependencies on first launch |
| Skill | Only the selected skill folder, resources and license |
| Workflow project setup | Its skills and necessary local scripts; MCP configuration is written, and each server downloads when started |
| Native plugin | The selected package, dependency packages and their skills; server implementations download when started |
| Search or prompt | Its own payload when first used |

Listing servers, skills and prompts, or previewing a project installation with
`--dry-run`, does not download component payloads. Selecting all skills or a large
workflow intentionally selects all of that collection's members.

## Released launcher

After this distribution is released:

```bash
uv tool install clio-kit
clio-kit mcp-servers
clio-kit mcp-server hdf5
clio-kit skill install exploring-an-unfamiliar-dataset --target .agents/skills
clio-kit plugin install clio-scientific-io --client codex --project /path/to/project
```

No clone is required. The project installer also accepts `opencode`, `cursor`,
`antigravity`, `vscode` and `claude-code`. It preserves unrelated settings and
backs up changed configuration; conflicts require review and `--replace`.
Native agents, hooks and commands still require supported host adapters. Explicit
`--components-only` selects just the skills and MCP configuration.

Optional verification tools are installed with `uv tool install 'clio-kit[verification]'`.
The verification extra is not required to launch an MCP.

## Selected native Claude plugin

Download a local marketplace containing only your selected plugin and dependencies:

```bash
clio-kit plugin fetch clio-dataset-report --target /path/to/clio-selected
claude plugin marketplace add /path/to/clio-selected
claude plugin install clio-dataset-report@clio-kit
```

This route includes native agents and hooks without cloning the full repository.
`fetch` copies files but does not execute components or edit a client's profile.
The client runs hooks and MCPs after installation. Keep the selected marketplace
folder available for the client. The target must be new to protect existing files.

If `clio-kit` is already registered from another location, inspect that registration
and deliberately switch it using the client's marketplace controls. A selected
marketplace lists only its selected packages. To select multiple workflows, give multiple names to `plugin fetch`, for example
`clio-kit plugin fetch clio-hpc clio-scientific-io --target /path/to/clio-selected`.
Shared dependencies are downloaded once. This command does not merge existing
marketplace directories. Native removal uses `claude plugin uninstall`; shared
dependencies may remain until the client's orphan-pruning command is used.

The existing GitHub marketplace route remains available, but its repository
transfer is controlled by the client. It is not the selective-download route.
External indexed packages retain their publisher's source type and installation
behavior; CLIO Kit cannot promise a partial transfer for an upstream whole-repo plugin.

## Current checkout and local builds

To test unreleased code, install the launcher in editable mode:

```bash
uv tool install --force --reinstall --editable '.[verification]'
clio-kit plugin install clio-scientific-io --root "$PWD" --client codex --project /path/to/project
```

This deliberately uses a full developer checkout and does not fetch release assets.
Keep it available. For native Claude development, register the checkout as before.

`uv build` produces a small wheel and sdist plus `dist/components/`. Each payload
archive has a SHA-256 address. The wheel and sdist contain the same catalogue,
including archive hashes, sizes and per-file hashes. Building a wheel from that
sdist requires no component downloads. Rebuilding the sdist's wheel does not
recreate component archives; release publication uses the original build output.

A non-editable installation built from an unreleased checkout needs its matching
component mirror. For local testing, build first, install that wheel, and set:

```bash
export CLIO_KIT_COMPONENT_BASE_URL="file://$PWD/dist/components"
```

For production mirrors use HTTPS. Loopback HTTP is supported for acceptance
checks. Mirrors change only where payloads are fetched: expected hashes still
come from the installed launcher. Missing assets, changed bytes, unsafe archives
and incomplete downloads fail instead of falling back to an unpinned Git branch.

## Offline use and updates

Artifacts are cached under `components/` inside the normal CLIO Kit cache root
(`CLIO_KIT_CACHE_DIR` overrides it). Archives are checked before extraction;
extracted files are checked before reuse. Concurrent requests share a download
lock. Network interruptions never leave a completed cache entry.

Repeat installations reuse unchanged content. Updating the launcher selects its
new catalogue; only changed selected artifacts need another download. Rerun skill
or project installation to update copied files, reviewing conflicts before
`--replace`. Fetch native packages into a new directory and refresh the client's
registration/installations deliberately. Already installed project files do not
automatically change when the launcher is upgraded.

Before offline use, install selected skills and exercise each required server on
the target platform so its runtime dependencies are cached too.
`CLIO_KIT_OFFLINE=1` forbids component downloads; it does not configure uv, npm,
Go, or scientific tools' own network policies. Runtime cache pruning can remove
an environment that would need to be rebuilt. Removing a project skill or plugin
does not automatically remove its shared component cache.

## Verification and publication

```bash
uv run --frozen python scripts/verify_partial_install.py --output /tmp/clio-partial-install
python scripts/verify_component_release.py --dist dist
```

The first command builds and installs a wheel, records requests to a local HTTP
artifact mirror, runs a real HDF5 query, checks offline reuse, and installs a
selected native Claude plugin. Dependency downloads are separate from the recorded
CLIO component traffic. Root tests also check corrupt downloads, archive traversal,
links, duplicate entries, dry-run behavior and component identity across updates.

Release CI verifies that every payload referenced by the wheel exists with the
expected bytes, attests the artifacts, and publishes the immutable GitHub release
before uploading the launcher to PyPI. PyPI receives only the launcher wheel and
sdist. Existing release authorization still gates publication. This ordering
avoids publishing a launcher whose component URLs do not yet exist; the actual
public release workflow must still succeed before public availability is claimed.
