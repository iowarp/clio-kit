# CLIO Kit Documentation Website

This Docusaurus site documents the scientific servers, workflow skills, plugins,
agents and community marketplace in this branch. The live site is deployed from
`main`; it may differ from this checkout until the feature is merged.

## Structure

```text
clio-kit-website/
├── docs/
│   ├── intro.md             # Installation and overview
│   ├── marketplace.md       # Components, contribution and acceptance
│   ├── agentic-search.md    # Standalone retrieval service
│   └── mcps/                # Server reference pages
├── src/components/          # Documentation and catalogue UI
├── src/data/                # Generated server catalogue
├── src/pages/               # Landing page
├── static/
├── docusaurus.config.js
└── package.json
```

## Develop and build

From this directory, with Node and npm installed:

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

## Maintain server reference pages

`scripts/generate_docs.py` regenerates contract metadata and the showcase from
`clio-kit-mcp-servers/` and `mcp-server-versions.toml`. Reviewed workflow text
between the `clio-kit:usage:start` and `clio-kit:usage:end` MDX comments is
preserved; edit it here when correcting usage. The generator does not substitute
invented Python calls when an example is absent. New servers need a real
workflow description plus their hosted-server CI coverage.

The `docs-and-website.yml` workflow generates and builds the site for matching
pull requests and publishes GitHub Pages after a push to `main`. No deployment
is performed by running the local build.

Developed by the [Gnosis Research Center](https://grc.iit.edu/) at
[Illinois Institute of Technology](https://www.iit.edu/), part of
[IoWarp](https://iowarp.ai).
