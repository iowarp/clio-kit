# CLIO Kit Documentation Website

This Docusaurus site documents the scientific servers, workflow skills, plugins,
agents and community marketplace. The live site at
[toolkit.iowarp.ai](https://toolkit.iowarp.ai/) is deployed from `main`.

## Structure

```text
../docs/                    # Shared documentation source (GitHub and website)
├── README.md               # Documentation index
├── intro.md                # Installation and overview
├── authoring.md            # Plugins, MCPs, skills and hooks
├── marketplace.md          # Components, contribution and acceptance
├── agentic-search.md       # Standalone retrieval service
└── mcps/                   # Server reference pages
clio-kit-website/
├── src/components/         # Documentation and catalogue UI
├── src/data/               # Generated server catalogue
├── src/pages/              # Landing page
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

Open `http://localhost:5100/`. For the production build and local preview:

```bash
npm run build
npm run serve
```

The build checks links and reports missing anchors. Current verification uses
Node 24.15.0 and the dependencies pinned in `package-lock.json`.

## Marketplace preview

The homepage browses workflows, skills, MCP servers, plugins, agents, hooks and
the standalone search service. Component details live at `/component?id=…`;
publisher profiles live at `/publishers`. Filters are encoded in the URL so a
filtered view can be shared or reopened.

`scripts/generate_website_catalogue.py` reads the marketplace index, bundle
inventory, local skill/agent definitions and community entries. `npm start` and
`npm run build` regenerate `src/data/catalogue.json` automatically using `uv` and
the root Python environment. No upstream network fetch or hook execution occurs.
Display-only workflow titles live in the generator; membership and versions
come from repository metadata. Metadata labels are not verification badges.

This preview uses React and scoped CSS, inspired by portfolio-style catalogues.
It does not install ReUI or change documentation page components. Test both
themes and mobile filters when changing the catalogue UI.

## Maintain server reference pages

Documentation lives in root [`docs/`](../docs/README.md). Docusaurus reads
that directory directly; public `/docs/...` URLs stay the same. From the
repository root, regenerate server pages with:

```bash
uv run python scripts/generate_docs.py mcp-servers clio-kit-website
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
