# Globe MCP

Show regions, places and data footprints on an interactive 3D globe (an MCP Apps view:
CesiumJS with OpenStreetMap tiles), and find places by name.

## What runs

This server is the map example server of the MCP Apps extension,
[`@modelcontextprotocol/server-map`](https://www.npmjs.com/package/@modelcontextprotocol/server-map)
**2.0.4**, run exactly as published. None of its code is copied here: `package.json`
pins the release, `package-lock.json` locks its tree, and the launcher installs it with
`npm ci`. `bundle/server.js` only starts the package's own command over stdio, so its
tools, its `ui://cesium-map/mcp-app.html` view (with the sources it declares: CesiumJS
from cesium.com, OpenStreetMap tiles) and everything they return reach the client
unchanged; the server reports its own name (`CesiumJS Map Server`) and version.

```bash
clio-kit mcp-server globe               # needs Node.js 20 or newer (node and npm)
```

It needs no account or key: no Cesium ion token is used.

## Tools

The upstream server's own: `show-map` opens the globe zoomed to a bounding box; `geocode` finds places by name through OpenStreetMap's Nominatim.

## Licence

The wrapper (these files) is BSD-3-Clause, like the rest of CLIO Kit.
`@modelcontextprotocol/server-map` is MIT (modelcontextprotocol/ext-apps,
`examples/map-server`; the MCP project is moving new contributions to Apache-2.0) and is
installed from npm, not redistributed here. CesiumJS (Apache-2.0) is loaded by the view
from cesium.com; map tiles are © OpenStreetMap contributors (ODbL).

## Tests

```bash
npm ci && npm run lint && npm test
```
