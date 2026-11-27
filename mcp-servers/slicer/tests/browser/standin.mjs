// A stand-in MCP Apps host for the slicer layer view, and a headless browser check.
//
// It does what CLIO's host does for this view, and nothing more:
//   - serves a host page that mounts the view in a double-iframe sandbox on a second
//     origin; the sandbox document carries CLIO's view policy (the resource's declared
//     CSP domains, scripts may evaluate code) and writes the view into a same-origin
//     inner frame with document.write, as CLIO and the reference host do;
//   - answers ui/initialize with a host context (theme, style variables, display modes);
//   - sends the opening call's tool-input and tool-result, then tools/list;
//   - forwards the view's own tools/call requests to the real slicer server, through the
//     driver's bridge (run_standin.py);
//   - records every ui/update-model-context.
//
// Then it drives the view in Chromium with software WebGL: waits for every layer's
// toolpaths, moves the layer slider, toggles feature types and travel, calls the view's
// tools (go_to_layer, set_feature_visibility, pick_segment) as a host would, clicks a
// toolpath, switches the theme, and writes screenshots and a JSON report. Usage:
//
//   node standin.mjs --config run.json --out DIR [--width 1440 --height 900]
//
// run.json: {"view_html", "csp", "arguments", "result", "bridge", "go_to_z", "pick"}

import { createRequire } from "node:module";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import http from "node:http";
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
const label = opt("label", `${width}x${height}`);
mkdirSync(out, { recursive: true });

const require = createRequire(path.join(process.env.PLAYWRIGHT_ROOT || "/opt/node-tools", "package.json"));
const { chromium } = require("playwright");

// CLIO's view policy (gact/mcp_app_sandbox.py::_csp_header) for the declared domains.
function cspHeader(csp, hostOrigin) {
  const resources = (csp.resourceDomains || []).join(" ");
  const connects = (csp.connectDomains || []).join(" ");
  return [
    "default-src 'self' 'unsafe-inline'",
    `script-src 'self' 'unsafe-inline' 'unsafe-eval' blob: data: ${resources}`,
    `style-src 'self' 'unsafe-inline' blob: data: ${resources}`,
    `img-src 'self' data: blob: ${resources}`,
    `font-src 'self' data: blob: ${resources}`,
    `media-src 'self' data: blob: ${resources}`,
    `connect-src 'self' ${connects}`,
    `worker-src 'self' blob: ${resources}`,
    "frame-src 'self' data: blob:",
    "base-uri 'none'",
    "object-src 'none'",
    `frame-ancestors ${hostOrigin}`,
  ].join("; ");
}

const SANDBOX = `<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%}body{display:flex}iframe{border:0;flex:1}</style></head><body><script>
const inner = document.createElement('iframe');
inner.setAttribute('sandbox', 'allow-scripts allow-same-origin allow-forms');
document.body.appendChild(inner);
window.addEventListener('message', (e) => {
  if (e.source === window.parent) {
    if (e.data && e.data.method === 'stand-in/resource') {
      const doc = inner.contentDocument; doc.open(); doc.write(e.data.params.html); doc.close(); return;
    }
    inner.contentWindow.postMessage(e.data, '*');
  } else if (e.source === inner.contentWindow) {
    window.parent.postMessage(e.data, '*');
  }
});
window.parent.postMessage({ method: 'stand-in/sandbox-ready' }, '*');
</script></body></html>`;

const safe = (value) => JSON.stringify(value).replace(/</g, "\\u003c");
const LIGHT = { "--color-background-primary": "#fbfbfa", "--color-background-secondary": "#f1f1ef",
  "--color-text-primary": "#1d1f23", "--color-text-secondary": "#5d636c", "--color-border-primary": "#d9dadd",
  "--font-sans": "Inter, system-ui, sans-serif" };
const DARK = { "--color-background-primary": "#17181b", "--color-background-secondary": "#202226",
  "--color-text-primary": "#e8e9ec", "--color-text-secondary": "#a2a8b1", "--color-border-primary": "#34373d",
  "--font-sans": "Inter, system-ui, sans-serif" };

function hostPage(sandboxOrigin) {
  return `<!doctype html><html><head><meta charset="utf-8"><title>Stand-in host</title><style>
html,body{margin:0;height:100%;font:13px system-ui}#card{position:fixed;inset:8px;border:1px solid #ccc}
iframe{width:100%;height:100%;border:0}</style></head>
<body><div id="card"><iframe id="app" src="${sandboxOrigin}/sandbox.html"></iframe></div><script>
window.__log = []; window.__context = []; window.__viewCalls = []; window.__initialized = false;
const frame = document.getElementById('app');
const html = ${safe(config.view_html)};
const pending = new Map(); let nextId = 1;
function post(m) { frame.contentWindow.postMessage(m, '*'); }
const context = { theme: 'light', styles: { variables: ${safe(LIGHT)} }, displayMode: 'inline',
  availableDisplayModes: ['inline', 'fullscreen'] };
window.__hostRequest = (method, params) => new Promise((resolve, reject) => {
  const id = 'host-' + nextId++; pending.set(id, { resolve, reject });
  post({ jsonrpc: '2.0', id, method, params });
});
window.__setTheme = (theme) => {
  context.theme = theme; context.styles = { variables: theme === 'dark' ? ${safe(DARK)} : ${safe(LIGHT)} };
  document.body.style.background = context.styles.variables['--color-background-primary'];
  post({ jsonrpc: '2.0', method: 'ui/notifications/host-context-changed', params: { theme, styles: context.styles } });
};
window.addEventListener('message', async (e) => {
  if (e.source !== frame.contentWindow) return;
  const m = e.data; window.__log.push(m);
  if (m.method === 'stand-in/sandbox-ready') { post({ method: 'stand-in/resource', params: { html } }); return; }
  if (m.id !== undefined && !m.method && pending.has(m.id)) {
    const p = pending.get(m.id); pending.delete(m.id);
    if (m.error) p.reject(new Error(m.error.message)); else p.resolve(m.result); return;
  }
  if (m.method === 'ui/initialize') {
    window.__appCapabilities = m.params.appCapabilities;
    post({ jsonrpc: '2.0', id: m.id, result: { protocolVersion: '2026-01-26', hostInfo: { name: 'stand-in', version: '1' },
      hostCapabilities: { serverTools: {}, updateModelContext: { text: {}, structuredContent: {} } }, hostContext: context } });
  } else if (m.method === 'ui/notifications/initialized') {
    window.__initialized = true;
    post({ jsonrpc: '2.0', method: 'ui/notifications/tool-input', params: { arguments: ${safe(config.arguments || {})} } });
    post({ jsonrpc: '2.0', method: 'ui/notifications/tool-result', params: ${safe(config.result || {})} });
    window.__tools = await window.__hostRequest('tools/list', {});
  } else if (m.method === 'ui/update-model-context') {
    window.__context.push({ at: Date.now(), params: m.params });
    post({ jsonrpc: '2.0', id: m.id, result: {} });
  } else if (m.method === 'tools/call') {
    window.__viewCalls.push(m.params);
    const res = await fetch(${JSON.stringify(config.bridge)}, { method: 'POST', body: JSON.stringify(m.params) });
    post({ jsonrpc: '2.0', id: m.id, result: await res.json() });
  } else if (m.id !== undefined && m.method) {
    post({ jsonrpc: '2.0', id: m.id, result: {} });
  }
});
</script></body></html>`;
}

const host = http.createServer((_req, res) => {
  res.writeHead(200, { "content-type": "text/html" });
  res.end(hostPage(`http://localhost:${sandboxPort}`));
});
const sandbox = http.createServer((_req, res) => {
  res.writeHead(200, {
    "content-type": "text/html",
    "content-security-policy": cspHeader(config.csp || {}, `http://127.0.0.1:${hostPort}`),
  });
  res.end(SANDBOX);
});
await new Promise((r) => host.listen(0, "127.0.0.1", r));
await new Promise((r) => sandbox.listen(0, "127.0.0.1", r));
const hostPort = host.address().port;
const sandboxPort = sandbox.address().port;

const flags = (process.env.STANDIN_GL_FLAGS || "--use-angle=swiftshader,--enable-unsafe-swiftshader,--ignore-gpu-blocklist")
  .split(",").filter(Boolean);
const browser = await chromium.launch({ args: flags });
const contextOptions = { viewport: { width, height } };
if (process.env.STANDIN_VIDEO) contextOptions.recordVideo = { dir: out, size: { width, height } };
const bctx = await browser.newContext(contextOptions);
const page = await bctx.newPage();
page.setDefaultTimeout(120000);
const report = { label, consoleErrors: [], failedRequests: [], shots: [], checks: {}, gl_flags: flags };
page.on("console", (m) => { if (m.type() === "error") report.consoleErrors.push(m.text()); });
page.on("requestfailed", (r) => report.failedRequests.push(`${r.url()} ${r.failure()?.errorText}`));
page.on("response", (r) => { if (r.status() >= 400) report.failedRequests.push(`${r.url()} HTTP ${r.status()}`); });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function shot(name) {
  const file = path.join(out, `${label}-${name}.png`);
  await page.screenshot({ path: file });
  report.shots.push(file);
}
const viewFrame = () => page.frames().find((f) => f.parentFrame() && f.parentFrame().url().includes("sandbox.html"));
const lastContext = () => page.evaluate(() => {
  const all = window.__context; return all.length ? all[all.length - 1].params : null;
});
const call = (name, a) => page.evaluate(([n, x]) => window.__hostRequest("tools/call", { name: n, arguments: x }), [name, a]);
const drawn = (frame) => frame.evaluate(() => {
  const out = { layers: 0, objects: 0, visibleObjects: 0 };
  for (const layer of window.slicerView.layers.values()) {
    out.layers += 1;
    for (const f of Object.values(layer.feats)) { out.objects += 1; if (f.object.visible) out.visibleObjects += 1; }
  }
  return out;
});

await page.goto(`http://127.0.0.1:${hostPort}/`);
for (let i = 0; i < 120 && !viewFrame(); i++) await sleep(250);
const view = viewFrame();
await view.waitForFunction(() => window.slicerView && window.slicerView.state, null, { timeout: 90000 });
await view.waitForFunction(() => document.getElementById("status").hidden, null, { timeout: 120000 });
await sleep(800);
report.appCapabilities = await page.evaluate(() => window.__appCapabilities);
report.tools = await page.evaluate(() => (window.__tools?.tools || []).map((t) => t.name));
report.checks.open = { state: await view.evaluate(() => window.slicerView.state), drawn: await drawn(view) };
await shot("01-open");

// The person drags the layer slider down to the middle of the print.
const middle = Math.floor(report.checks.open.state.layer_count / 2);
await view.evaluate((v) => { const el = document.getElementById("to"); el.value = String(v); el.dispatchEvent(new Event("input")); }, middle);
await sleep(900);
report.checks.slider = { state: (await lastContext())?.structuredContent, text: await view.evaluate(() => document.getElementById("layer-text").textContent) };
await shot("02-slider-middle");

// The host moves the view by height (an A2UI channel or the agent): go_to_layer z_mm.
const moved = await call("go_to_layer", { z_mm: config.go_to_z });
await sleep(900);
report.checks.goToLayer = { result: moved?.content?.[0]?.text, isError: !!moved?.isError, state: (await lastContext())?.structuredContent };
await shot("03-go-to-layer");

// Feature types: hide sparse infill and show travel (through the view's tool).
const vis = await call("set_feature_visibility", { features: { internal_infill: false, travel: true } });
await sleep(1200);
report.checks.visibility = { result: vis?.content?.[0]?.text, isError: !!vis?.isError, drawn: await drawn(view),
  reported: (await lastContext())?.structuredContent?.visible_features };
await shot("04-travel-no-infill");
const unknown = await call("set_feature_visibility", { features: { lasers: true } });
report.checks.unknownFeature = { result: unknown?.content?.[0]?.text, isError: !!unknown?.isError };

// pick_segment from the host, then a person's click on the canvas.
const picked = await call("pick_segment", { x_mm: config.pick[0], y_mm: config.pick[1] });
await sleep(700);
report.checks.pickTool = { result: picked?.content?.[0]?.text, isError: !!picked?.isError,
  selection: (await lastContext())?.structuredContent?.selection };
await shot("05-picked");
const frameBox = await (await view.frameElement()).boundingBox();
const stageBox = await view.evaluate(() => { const r = document.getElementById("stage").getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; });
await page.mouse.move(frameBox.x + stageBox.x + stageBox.w * 0.5, frameBox.y + stageBox.y + stageBox.h * 0.45);
await page.mouse.down(); await page.mouse.move(frameBox.x + stageBox.x + stageBox.w * 0.62, frameBox.y + stageBox.y + stageBox.h * 0.4, { steps: 6 }); await page.mouse.up();
await page.mouse.wheel(0, -300);
await sleep(700);
report.checks.orbit = await view.evaluate(() => window.slicerView.state.layer);
await shot("06-orbit-zoom");

// Theme follows the host.
await page.evaluate(() => window.__setTheme("dark"));
await call("set_feature_visibility", { features: { internal_infill: true, travel: false } });
await call("go_to_layer", { layer: report.checks.open.state.layer_count - 1 });
await sleep(1200);
report.checks.darkBackground = await view.evaluate(() => getComputedStyle(document.body).backgroundColor);
await shot("07-dark");

report.viewCalls = await page.evaluate(() => window.__viewCalls.map((c) => c.name));
report.contextUpdates = await page.evaluate(() => window.__context.length);
writeFileSync(path.join(out, `${label}-report.json`), JSON.stringify(report, null, 2));
await bctx.close();
await browser.close();
host.close(); sandbox.close();
console.log(path.join(out, `${label}-report.json`));
process.exit(0);
