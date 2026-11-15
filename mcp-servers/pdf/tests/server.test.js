// The wrapper starts the pinned @modelcontextprotocol/server-pdf over stdio and changes
// nothing it says: the same tools, the same ui:// view, served from the same package.
// Needs `npm ci` first (the locked tree in node_modules).
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const UPSTREAM = "@modelcontextprotocol/server-pdf";
const VIEW = "ui://pdf-viewer/mcp-app.html";

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
  assert.equal(replies.get(0).result.serverInfo.name, "PDF Server");
  const tools = replies.get(1).result.tools;
  const display = tools.find((tool) => tool.name === "display_pdf");
  assert.equal(display._meta.ui.resourceUri, VIEW);
  assert.ok(tools.some((tool) => tool.name === "list_pdfs"));
  const [content] = replies.get(2).result.contents;
  assert.equal(content.mimeType, "text/html;profile=mcp-app");
  assert.match(content.text, /<html/i);
});

test("an argument after the server name still reaches the upstream server", async () => {
  // --stdio is not added twice; a folder given on the command line is allowed.
  const folder = mkdtempSync(join(tmpdir(), "clio-pdf-"));
  try {
    const replies = await session(
      [{ method: "initialize" }, { method: "tools/call", params: { name: "list_pdfs", arguments: {} } }],
      ["--stdio", folder],
    );
    assert.deepEqual(replies.get(1).result.structuredContent.allowedDirectories, [folder]);
  } finally {
    rmSync(folder, { recursive: true, force: true });
  }
});
