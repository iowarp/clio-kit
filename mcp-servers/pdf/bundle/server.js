#!/usr/bin/env node
// Runs the MCP Apps PDF server, @modelcontextprotocol/server-pdf, exactly as published.
//
// Nothing of it is copied here: package.json pins the release and package-lock.json
// locks its tree, which `npm ci` installs. This file only starts that package's own
// command-line entry (dist/index.js) over stdio, so its tools, its ui:// view and
// everything they return reach the client unchanged. Arguments after the server name
// (PDF URLs or local files to allow, --debug) pass through.
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const UPSTREAM = "@modelcontextprotocol/server-pdf";

// The package exports only its library (dist/server.js); its command-line entry sits
// beside it, so it is reached by file rather than through the exports map.
const library = fileURLToPath(import.meta.resolve(UPSTREAM));
if (!process.argv.slice(2).includes("--stdio")) {
  process.argv.splice(2, 0, "--stdio");
}
await import(pathToFileURL(join(dirname(library), "index.js")).href);
