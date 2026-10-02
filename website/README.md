# CLIO Kit Documentation Website

This Docusaurus site documents the scientific servers, workflow skills, plugins,
agents and community marketplace. The live site at
[toolkit.iowarp.ai](https://toolkit.iowarp.ai/) is deployed from `main`.

## Structure

```text
../docs/                    # Shared documentation source (GitHub and website)
├── README.md               # Documentation index
├── intro.md                # Installation and overview
├── clients.md              # Setup for each supported agent
├── contributing.md         # Component and external contribution routes
├── tutorials/              # Runnable tool, skill and plugin walkthroughs
├── authoring.md            # Plugins, MCPs, skills and hooks
├── marketplace.md          # Components, contribution and acceptance
└── mcps/                   # Server reference pages
website/
├── src/components/         # Documentation and catalogue UI
├── src/data/               # Generated server catalogue
├── src/pages/              # Overview, catalogue, component and publisher routes
├── static/
├── docusaurus.config.js
└── package.json
```

## Develop and build

From this directory, with Node 20 or newer, npm and Python 3.10 or newer installed:

```bash
npm ci
npm start
```

Open `http://localhost:5100/`. If the system has exhausted its file watchers, use
`CHOKIDAR_USEPOLLING=1 WATCHPACK_POLLING=2000 npm start`. For the production build
and local preview:

```bash
npm run build
npm run serve
```

The build checks links and reports missing anchors. Current verification uses
Node 24.15.0 and the dependencies pinned in `package-lock.json`.

## Marketplace preview

The homepage follows Clio Coder’s editorial structure: a concise introduction,
a three-step workflow, client routes, installation and contribution. It contains
no tutorial screenshots; the searchable catalogue stays directly accessible. `/tutorials` presents
walkthroughs stored in `docs/tutorials/`; titles, descriptions and thumbnails are
read from their Markdown metadata, rather than maintained in a second list. `/catalogue` browses workflows, skills, MCP servers, plugins,
agents, hooks and the standalone search service, separately from the overview.
Component details live at `/catalogue/<type>/<name>`; legacy `/component?id=…`
links redirect to their catalogue pages.
publisher profiles live at `/publishers`. Filters are encoded in the URL so a
filtered view can be shared or reopened.

`scripts/generate_website_catalogue.py` reads the marketplace index, bundle
inventory, local skill/agent definitions and community entries. `npm start` and
`npm run build` regenerate `src/data/catalogue.json` automatically using `uv` and
the root Python environment. No upstream network fetch or hook execution occurs.
Display-only workflow titles live in the generator; membership and versions
come from repository metadata. Metadata labels are not verification badges.

The overview adopts the editorial structure of [Clio Coder](https://coder.iowarp.ai/):
numbered sections, open layouts, practical examples and concise setup guidance.
Kit keeps its own teal/orange palette and catalogue data. Newsreader and IBM Plex
fonts are self-hosted in `static/fonts/`, adopted from Clio Coder with their SIL
licenses; no remote font service is used. `editorial.css` holds shared typography,
and `Overview.js` with `overview.module.css` holds the product introduction.
The tutorial screenshots in `static/img/tutorials/` capture real Claude Code and Codex CLI
processes in a PTY-backed terminal emulator. They show actual installations, skill
invocations, MCP calls and results. `runtime-chart.png` is the Plot MCP's output.
The corresponding CSV inputs, HDF5 creation scripts and downloadable contribution
package live in `docs/tutorials/`. Guides cover four MCPs, five maintained skills,
three main task workflows, component installation and contribution in both clients.
The storage-layout guide records a reviewed recommendation, not a benchmark.
The runs used isolated client profiles and disposable projects. Codex required
full local execution because the nested filesystem sandbox could not start on
this machine; the public tutorials use normal client approval settings. No client
credentials, private session traces or temporary profiles belong in the website.

When updating a walkthrough, run its actual agent and tools, independently check
the outputs, then capture the terminal again. Do not substitute a manually written
transcript or draw a fake agent screen. Keep the commands, screenshots, model/client
versions and stated results consistent. `sidebars.js` groups setup, tutorials,
contribution and references. Test both themes, narrow screens, keyboard navigation,
copy controls and catalogue filters when changing the UI.

## Maintain server reference pages

Documentation lives in root [`docs/`](../docs/README.md). Docusaurus reads
that directory directly; public `/docs/...` URLs stay the same. From the
repository root, regenerate server pages with:

```bash
uv run python scripts/generate_docs.py mcp-servers website
```

`scripts/generate_docs.py` regenerates contract metadata and the showcase from
`mcp-servers/` and `mcp-server-versions.toml`. Reviewed workflow text
between the `clio-kit:usage:start` and `clio-kit:usage:end` MDX comments is
preserved; edit it in `../docs/mcps/` when correcting usage. The generator does not substitute
invented Python calls when an example is absent. New servers need a real
workflow description plus their hosted-server CI coverage.

The `docs-and-website.yml` workflow generates and builds the site for matching
pull requests and publishes GitHub Pages after a push to `main`. No deployment
is performed by running the local build.

Developed by the [Gnosis Research Center](https://grc.iit.edu/) at
[Illinois Institute of Technology](https://www.iit.edu/), part of
[IoWarp](https://iowarp.ai).

## Image parser advisory

The locked Docusaurus dependency `image-size@2.0.2` has no published fix for
[ICNS](https://github.com/advisories/GHSA-w3rx-r6r6-pgpr) and
[JPEG XL/HEIF](https://github.com/advisories/GHSA-5p2g-fcmc-qvqq) parser loops.
`npm run build` and `npm start` run a byte-signature check before Docusaurus
starts, rejecting those containers even if their extensions are misleading.
Use PNG, JPEG, GIF, WebP or SVG website assets. Do not bypass the check with a
direct Docusaurus command when processing contributed files. The check limits
exposure; it does not patch the dependency, and `npm audit` still reports it.
The deployed site is static; this parser runs during builds, not in visitors'
browsers. Remove the workaround once an upstream patched release is available.
