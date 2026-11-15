# PDF MCP

Read scientific papers from arXiv, bioRxiv, medRxiv, ChemRxiv, Zenodo, OSF, HAL and
other archives in an interactive PDF viewer (an MCP Apps view, PDF.js).

## What runs

This server is the PDF example server of the MCP Apps extension,
[`@modelcontextprotocol/server-pdf`](https://www.npmjs.com/package/@modelcontextprotocol/server-pdf)
**2.0.4**, run exactly as published. None of its code is copied here: `package.json`
pins the release, `package-lock.json` locks its tree, and the launcher installs it with
`npm ci`. `bundle/server.js` only starts the package's own command over stdio, so its
tools, its `ui://pdf-viewer/mcp-app.html` view and everything they return reach the
client unchanged; the server reports its own name (`PDF Server`) and version.

```bash
clio-kit mcp-server pdf                 # needs Node.js 20 or newer (node and npm)
clio-kit mcp-server pdf ~/papers        # also allow the PDFs in a local folder
```

Arguments after the server name pass through: a local file or folder to allow,
`--debug`. Over stdio the server also reads local files under the client's roots.

## Tools

The upstream server's own: `display_pdf` opens a PDF by URL or allowed path in the view;
`list_pdfs` lists the allowed local files and folders; `interact` drives an open view
(navigate, search, extract text, annotate, fill forms). The others (`read_pdf_bytes`, `submit_*`,
`poll_pdf_commands`, `save_pdf`) are called by the view itself (`visibility: ["app"]`).

## Licence

The wrapper (these files) is BSD-3-Clause, like the rest of CLIO Kit.
`@modelcontextprotocol/server-pdf` is MIT (modelcontextprotocol/ext-apps,
`examples/pdf-server`; the MCP project is moving new contributions to Apache-2.0) and is
installed from npm, not redistributed here.

## Tests

```bash
npm ci && npm run lint && npm test
```
