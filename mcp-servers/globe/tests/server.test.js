// The wrapper starts the pinned @modelcontextprotocol/server-map over stdio and changes
// nothing it says: the same tools, the same ui:// view, served from the same package.
// Needs `npm ci` first (the locked tree in node_modules).
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const UPSTREAM = "@modelcontextprotocol/server-map";
const VIEW = "ui://cesium-map/mcp-app.html";

/** Start the server, send each request in turn, and return the replies by id. */
async function session(requests, args = []) {
  const child = spawn(process.execPath, [join(ROOT, "bundle", "server.js"), ...args], {
    stdio: ["pipe", "pipe", "pipe"],
  });
  const replies = new Map();
  let buffer = "";
  const done = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("no reply within 30 s")), 30_000);
    child.stdout.on("data", (chunk) => {
      buffer += chunk;
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const message = JSON.parse(buffer.slice(0, newline));
        buffer = buffer.slice(newline + 1);
        if (message.id !== undefined) replies.set(message.id, message);
        if (replies.size === requests.length) {
          clearTimeout(timer);
          resolve();
        }
      }
    });
    child.on("exit", (code) => reject(new Error(`server exited with ${code}`)));
  });
  const write = (message) => child.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", ...message })}\n`);
  write({
    id: 0,
    method: "initialize",
    params: { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "test", version: "1" } },
  });
  write({ method: "notifications/initialized" });
  requests.slice(1).forEach((request, index) => write({ id: index + 1, ...request }));
  try {
    await done;
  } finally {
    child.kill();
  }
  return replies;
}

test("it pins one exact upstream release", () => {
  const manifest = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
  const lock = JSON.parse(readFileSync(join(ROOT, "package-lock.json"), "utf8"));
  const pinned = manifest.dependencies[UPSTREAM];
  assert.match(pinned, /^\d+\.\d+\.\d+$/);
  assert.equal(lock.packages[`node_modules/${UPSTREAM}`].version, pinned);
});

test("the upstream tools and view pass through unchanged", async () => {
  const replies = await session([
    { method: "initialize" },
    { method: "tools/list" },
    { method: "resources/read", params: { uri: VIEW } },
  ]);
  assert.equal(replies.get(0).result.serverInfo.name, "CesiumJS Map Server");
  const tools = replies.get(1).result.tools;
  const show = tools.find((tool) => tool.name === "show-map");
  assert.equal(show._meta.ui.resourceUri, VIEW);
  assert.ok(tools.some((tool) => tool.name === "geocode"));
  const [content] = replies.get(2).result.contents;
  assert.equal(content.mimeType, "text/html;profile=mcp-app");
  assert.match(content.text, /<html/i);
  // The view's declared sources (CesiumJS, OpenStreetMap tiles) reach the host as well.
  assert.ok(content._meta.ui.csp.resourceDomains.includes("https://cesium.com"));
});
