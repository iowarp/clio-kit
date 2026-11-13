// A stand-in MCP Apps host for the ParaView view, and a headless browser check.
//
// It does what CLIO's host does for this view, and nothing more:
//   - serves a host page that mounts the view in a double-iframe sandbox from a
//     second origin (inner frame: sandbox="allow-scripts allow-forms");
//   - answers ui/initialize with host context carrying the view's service URL;
//   - proxies /service/* (HTTP and WebSocket) to the node half's forwarded
//     port, adding the token on the way through so the browser never sees it;
//   - records every ui/update-model-context and display-mode request.
//
// Then it drives the view in Chromium: waits for trame, drags to rotate while
// recording WebSocket frames, clicks to pick, and writes screenshots and a
// JSON report. Usage:
//
//   node standin.mjs --config run.json --out DIR [--width 1440 --height 900]
//
// run.json: {"view_html": "...", "port": 12345, "token": "...", "steps": [...]}

import { createRequire } from "node:module";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import http from "node:http";
import net from "node:net";
import path from "node:path";

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const config = JSON.parse(readFileSync(opt("config"), "utf-8"));
const out = opt("out", "standin-out");
const width = Number(opt("width", 1440));
const height = Number(opt("height", 900));
mkdirSync(out, { recursive: true });

const require = createRequire(path.join(process.env.PLAYWRIGHT_ROOT || "/opt/node-tools", "package.json"));
const { chromium } = require("playwright");

const SANDBOX = `<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%}body{display:flex}iframe{border:0;flex:1}</style></head><body><script>
const inner = document.createElement('iframe');
inner.setAttribute('sandbox', 'allow-scripts allow-forms');
document.body.appendChild(inner);
window.addEventListener('message', (e) => {
  if (e.source === window.parent) {
    if (e.data && e.data.method === 'stand-in/resource') { inner.srcdoc = e.data.params.html; return; }
    inner.contentWindow.postMessage(e.data, '*');
  } else if (e.source === inner.contentWindow) {
    window.parent.postMessage(e.data, '*');
  }
});
window.parent.postMessage({ method: 'stand-in/sandbox-ready' }, '*');
</script></body></html>`;

// JSON that is safe inside an inline <script>.
const safe = (value) => JSON.stringify(value).replace(/</g, "\\u003c");

function hostPage(sandboxOrigin, serviceUrl) {
  return `<!doctype html><html><head><meta charset="utf-8"><title>Stand-in host</title><style>
html,body{margin:0;height:100%;font:13px system-ui}#card{height:620px;border:1px solid #ccc;margin:8px}
body.full #card{position:fixed;inset:0;height:auto;margin:0}iframe{width:100%;height:100%;border:0}</style></head>
<body><div id="card"><iframe id="app" src="${sandboxOrigin}/sandbox.html"></iframe></div><script>
window.__log = []; window.__context = []; window.__modes = [];
const frame = document.getElementById('app');
const html = ${safe(config.view_html)};
function post(m) { frame.contentWindow.postMessage(m, '*'); }
const context = { displayMode: 'inline', availableDisplayModes: ['inline', 'fullscreen'],
  'ai.iowarp.clio/service': { url: ${JSON.stringify(serviceUrl)} } };
window.addEventListener('message', (e) => {
  if (e.source !== frame.contentWindow) return;
  const m = e.data; window.__log.push(m);
  if (m.method === 'stand-in/sandbox-ready') { post({ method: 'stand-in/resource', params: { html } }); return; }
  if (m.method === 'ui/initialize') post({ jsonrpc: '2.0', id: m.id, result: {
    protocolVersion: '2026-01-26', hostInfo: { name: 'stand-in', version: '1' }, hostCapabilities: {}, hostContext: context } });
  else if (m.method === 'ui/notifications/initialized') {
    post({ jsonrpc: '2.0', method: 'ui/notifications/tool-input', params: { arguments: ${safe(config.arguments || {})} } });
    post({ jsonrpc: '2.0', method: 'ui/notifications/tool-result', params: ${safe(config.result || {})} });
  } else if (m.method === 'ui/request-display-mode') {
    window.__modes.push(m.params.mode);
    const mode = m.params.mode === 'fullscreen' ? 'fullscreen' : 'inline';
    document.body.classList.toggle('full', mode === 'fullscreen');
    context.displayMode = mode;
    post({ jsonrpc: '2.0', id: m.id, result: { mode } });
    post({ jsonrpc: '2.0', method: 'ui/notifications/host-context-changed', params: { displayMode: mode } });
  } else if (m.method === 'ui/update-model-context') {
    window.__context.push({ at: Date.now(), params: m.params });
    post({ jsonrpc: '2.0', id: m.id, result: {} });
  } else if (m.id !== undefined && m.method) {
    post({ jsonrpc: '2.0', id: m.id, result: {} });
  }
});
</script></body></html>`;
}

const requests = [];
function proxyHttp(req, res) {
  const target = req.url.replace(/^\/service/, "") || "/";
  requests.push({ url: req.url, method: req.method });
  const upstream = http.request(
    { host: "127.0.0.1", port: config.port, path: target, method: req.method,
      headers: { ...req.headers, host: `127.0.0.1:${config.port}`, authorization: `Bearer ${config.token}` } },
    (up) => { res.writeHead(up.statusCode, up.headers); up.pipe(res); },
  );
  upstream.on("error", () => { res.writeHead(502); res.end("upstream gone"); });
  req.pipe(upstream);
}

const host = http.createServer((req, res) => {
  if (req.url.startsWith("/service")) return proxyHttp(req, res);
  res.writeHead(200, { "content-type": "text/html" });
  res.end(hostPage(`http://localhost:${sandboxPort}`, `http://127.0.0.1:${hostPort}/service/`));
});
host.on("upgrade", (req, socket, head) => {
  const target = req.url.replace(/^\/service/, "") || "/";
  requests.push({ url: req.url, method: "UPGRADE" });
  const upstream = net.connect(config.port, "127.0.0.1", () => {
    const headers = { ...req.headers, host: `127.0.0.1:${config.port}`, authorization: `Bearer ${config.token}` };
    let text = `GET ${target} HTTP/1.1\r\n`;
    for (const [k, v] of Object.entries(headers)) text += `${k}: ${v}\r\n`;
    upstream.write(text + "\r\n");
    if (head && head.length) upstream.write(head);
    upstream.pipe(socket); socket.pipe(upstream);
  });
  upstream.on("error", () => socket.destroy());
  socket.on("error", () => upstream.destroy());
});
const sandbox = http.createServer((_req, res) => {
  res.writeHead(200, { "content-type": "text/html" });
  res.end(SANDBOX);
});
await new Promise((r) => host.listen(0, "127.0.0.1", r));
await new Promise((r) => sandbox.listen(0, "127.0.0.1", r));
const hostPort = host.address().port;
const sandboxPort = sandbox.address().port;

const flags = (process.env.STANDIN_GL_FLAGS || "--use-angle=swiftshader,--enable-unsafe-swiftshader")
  .split(",").filter(Boolean);
const browser = await chromium.launch({ args: flags });
const page = await (await browser.newContext({ viewport: { width, height } })).newPage();
page.setDefaultTimeout(180000);
const report = { consoleErrors: [], failedRequests: [], frames: [], shots: [], gl_flags: flags };
page.on("console", (m) => { if (m.type() === "error") report.consoleErrors.push(m.text()); });
page.on("requestfailed", (r) => report.failedRequests.push(`${r.url()} ${r.failure()?.errorText}`));
page.on("websocket", (ws) => {
  ws.on("framereceived", (f) => {
    const p = f.payload;
    const text = typeof p === "string" ? p : "";
    report.frames.push({ at: Date.now(), size: p.length, binary: typeof p !== "string",
      image: /viewport\.image|"image"|image\/jpeg|image\/png/.test(text) });
  });
  ws.on("framesent", (f) => {
    const p = typeof f.payload === "string" ? f.payload : "";
    const m = p.match(/"method"\s*:\s*"([^"]+)"/);
    report.frames.push({ at: Date.now(), sent: true, method: m ? m[1] : null });
  });
});

async function shot(label) {
  const file = path.join(out, `${label}.png`);
  await page.screenshot({ path: file, timeout: 180000 });
  report.shots.push(file);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const viewFrame = () => page.frames().find((f) => f.url().includes("/service/"));
// The local and remote renderers each have a canvas; the hidden one sits off screen.
async function visibleCanvasBox(frame) {
  for (const canvas of await frame.locator(".clio-view canvas").elementHandles()) {
    const box = await canvas.boundingBox();
    if (box && box.x >= 0 && box.width > 0) return box;
  }
  throw new Error("no visible canvas in the view");
}

await page.goto(`http://127.0.0.1:${hostPort}/`);
for (let i = 0; i < 120 && !viewFrame(); i++) await sleep(250);
const trame = viewFrame();
if (!trame) {
  await shot("00-no-view");
  report.frameUrls = page.frames().map((f) => f.url());
  report.log = await page.evaluate(() => window.__log);
  writeFileSync(path.join(out, "report.json"), JSON.stringify(report, null, 2));
  console.error("the view never framed the service; see report.json");
  process.exit(2);
}
report.webgl = await trame.evaluate(() => {
  const c = document.createElement("canvas");
  return !!(c.getContext("webgl2") || c.getContext("webgl"));
});
await trame.waitForSelector("canvas", { timeout: 60000 });
await sleep(4000);
await shot("01-open");

for (const step of config.steps || []) {
  if (step.do === "drag") {
    const box = await visibleCanvasBox(trame);
    const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
    await sleep(1500);
    const start = Date.now();
    await page.mouse.move(cx, cy);
    await page.mouse.down();
    for (let i = 1; i <= 30; i++) { await page.mouse.move(cx + i * 6, cy + i * 2); await sleep(30); }
    await page.mouse.up();
    const end = Date.now();
    report.drag = { start, end };
    await sleep(1500);
  } else if (step.do === "click") {
    const box = await visibleCanvasBox(trame);
    await page.mouse.click(box.x + box.width * (step.x ?? 0.5), box.y + box.height * (step.y ?? 0.5));
    await sleep(2000);
  } else if (step.do === "wait") {
    await sleep(step.ms || 1000);
  } else if (step.do === "toggle-mode") {
    // The drawing control: press the choice that is not pressed.
    await trame.click('[aria-label="Drawing"] button[aria-pressed="false"]');
    await sleep(3000);
  }
  if (step.shot) await shot(step.shot);
}

report.context = await page.evaluate(() => window.__context);
report.modes = await page.evaluate(() => window.__modes);
report.viewText = await trame.evaluate(() => document.body.innerText);
report.requests = requests;
report.tokenInBrowser = JSON.stringify(report).includes(config.token);
await shot("99-final");
writeFileSync(path.join(out, "report.json"), JSON.stringify(report, null, 2));
await browser.close();
host.close(); sandbox.close();
console.log(path.join(out, "report.json"));
process.exit(0);
