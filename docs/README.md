---
title: Documentation
slug: /
---

# CLIO Kit documentation

CLIO Kit brings scientific MCP servers, portable skills, workflow plugins,
agent definitions and community contributions into one meta-marketplace.

## Install and connect

- [Set up your agent](clients.md): Codex, Claude Code, OpenCode, Cursor, Antigravity, VS Code, Claude Desktop and Clio Coder.
- [Getting started](intro.md): launcher installation and the component model.
- [Selective installation](installation.md): release downloads, updates, removal considerations and offline preparation.

## Learn by doing

- [Install components](tutorials/install-components.md): choose individual MCPs, portable skills or a native workflow plugin.

- [Explore HDF5 with Codex](tutorials/codex-dataset.md): install a workflow, invoke its skill and inspect a scientific file through MCP.
- [Summarize and plot with Claude Code](tutorials/claude-analysis.md): use Pandas, Plot and a skill to produce checked means and a chart.
- [Archive and summarize](tutorials/archive-and-summarize.md): combine Compression, HDF5 and bounded statistics in Codex.
- [Clean and plot](tutorials/clean-and-plot.md): chain two skills with Pandas and Plot in Claude Code.
- [Choose a storage layout](tutorials/choose-storage.md): use a domain skill and review its assumptions.
- [Build, test and contribute a plugin](tutorials/contribute-plugin.md): run it in Claude Code and Codex, submit a community entry, and install it after acceptance.

## Contribute and extend

- [Contribution routes](contributing.md): where skills, MCPs, agents, hooks, plugins and external marketplaces belong.
- [Authoring contracts](authoring.md): manifests, runtime descriptors, skill metadata and checks.
- [Compose a workflow plugin](plugins.md): reuse components across scientific tasks.

## Reference

- [Marketplace guide](marketplace.md): external contributions, Clio Coder integration and tested coverage.
- [Agentic Search](agentic-search.md): the standalone scientific retrieval service.

Server references are in [`mcps/`](https://github.com/iowarp/clio-kit/tree/main/docs/mcps)
and the [website catalogue](https://toolkit.iowarp.ai/catalogue).
For agents, start with
[AGENTS.md](https://github.com/iowarp/clio-kit/blob/main/AGENTS.md).

These files are the documentation source for both GitHub and the website.
Website code lives separately in `website/`.
