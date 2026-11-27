// A stand-in MCP Apps host for the geo globe view, and a headless browser check.
//
// It does what CLIO's host does for this view, and nothing more:
//   - serves a host page that mounts the view in a double-iframe sandbox on a second
//     origin; the sandbox document carries CLIO's view policy (the resource's declared
//     CSP domains, scripts may evaluate code) and writes the view into a same-origin
//     inner frame with document.write, as CLIO and the reference host do;
//   - answers ui/initialize with a host context (theme, style variables, display modes);
//   - sends the opening call's tool-input and tool-result, then tools/list;
//   - forwards the view's own tools/call requests to the real geo server, through the
//     driver's bridge (run_standin.py);
//   - records every ui/update-model-context.
//
// Then it drives the globe in Chromium with software WebGL: counts the drawn entities,
// clicks a station's dot, sets selected_feature from the host (select_feature), flies,
// adds/restyles/removes a layer, switches the theme, and writes screenshots and a
// JSON report. Usage:
//
//   node standin.mjs --config run.json --out DIR [--width 1440 --height 900]
//
// run.json: {"view_html", "csp", "arguments", "result", "bridge", "click", "select",
//            "add_layer"}

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
// Wait until the globe has drawn everything it asked for.
async function settle(frame, extra = 1500) {
  await frame.waitForFunction(() => {
    const v = window.geoGlobe && window.geoGlobe.viewer;
    if (!v) return false;
    v.scene.requestRender();
    return v.scene.globe.tilesLoaded;
  }, null, { timeout: 90000, polling: 500 }).catch(() => { report.tilesTimeout = (report.tilesTimeout || 0) + 1; });
  await sleep(extra);
}
const lastContext = () => page.evaluate(() => {
  const all = window.__context; return all.length ? all[all.length - 1].params : null;
});
const call = (name, a) => page.evaluate(([n, x]) => window.__hostRequest("tools/call", { name: n, arguments: x }), [name, a]);
const cameraAt = (frame) => frame.evaluate(() => {
  const v = window.geoGlobe.viewer; const c = v.camera.positionCartographic;
  return { lon: c.longitude * 180 / Math.PI, lat: c.latitude * 180 / Math.PI, height: c.height };
});

await page.goto(`http://127.0.0.1:${hostPort}/`);
for (let i = 0; i < 120 && !viewFrame(); i++) await sleep(250);
const globe = viewFrame();
await globe.waitForFunction(() => window.geoGlobe && window.geoGlobe.viewer, null, { timeout: 90000 });
await globe.evaluate(() => window.geoGlobe.ready);
await settle(globe, 3000);
report.webgl = await globe.evaluate(() => !!window.geoGlobe.viewer.scene.context);
report.appCapabilities = await page.evaluate(() => window.__appCapabilities);
report.tools = await page.evaluate(() => (window.__tools?.tools || []).map((t) => t.name));
const counts = () => globe.evaluate(() => {
  const out = {};
  for (const [name, layer] of window.geoGlobe.layers) {
    out[name] = { features: layer.features.size, entities: layer.source ? layer.source.entities.values.length : 0,
      points: layer.source ? layer.source.entities.values.filter((e) => e.point).length : 0,
      labels: layer.source ? layer.source.entities.values.filter((e) => e.label).length : 0 };
  }
  return out;
});
report.checks.drawn = await counts();
report.checks.openCamera = await cameraAt(globe);
await shot("01-open");

// A person clicks a station's dot.
const target = config.click;
const where = await globe.evaluate((id) => {
  const rec = window.geoGlobe.find(id);
  const v = window.geoGlobe.viewer;
  const pos = rec.entities[0].position.getValue(Cesium.JulianDate.now());
  const toWindow = Cesium.SceneTransforms.worldToWindowCoordinates || Cesium.SceneTransforms.wgs84ToWindowCoordinates;
  const p = toWindow(v.scene, pos);
  return p ? { x: p.x, y: p.y } : null;
}, target);
const box = await page.locator("#app").boundingBox();
const innerBox = await (await globe.frameElement()).boundingBox();
report.checks.clickAt = where;
await page.mouse.move(innerBox.x + where.x, innerBox.y + where.y);
await sleep(600);
await shot("02-hover");
await page.mouse.click(innerBox.x + where.x, innerBox.y + where.y);
await sleep(1200);
const afterClick = await lastContext();
report.checks.clickReported = afterClick?.structuredContent?.selected_feature ?? null;
report.checks.clickText = afterClick?.content?.[0]?.text ?? null;
report.checks.infoPanel = await globe.evaluate(() => !document.getElementById("info").hidden
  && document.getElementById("info-title").textContent);
await settle(globe, 500);
await shot("03-clicked");

// The host sets selected_feature (a linked table row): the globe turns to that dot.
const before = await cameraAt(globe);
const selected = await call("select_feature", { selected_feature: config.select });
await settle(globe, 1000);
const after = await cameraAt(globe);
report.checks.hostSelect = { result: selected?.content?.[0]?.text, before, after,
  selected: await globe.evaluate(() => window.geoGlobe.selected) };
const afterSelect = await lastContext();
report.checks.hostSelectReported = afterSelect?.structuredContent?.selected_feature ?? null;
await shot("04-host-selected");

// fly_to an area.
const flown = await call("fly_to", { bbox: config.fly_bbox });
await settle(globe, 800);
report.checks.flyTo = { result: flown?.content?.[0]?.text, camera: await cameraAt(globe),
  view: await globe.evaluate(() => {
    const r = window.geoGlobe.viewer.camera.computeViewRectangle();
    return r ? [r.west, r.south, r.east, r.north].map((v) => v * 180 / Math.PI) : null;
  }) };
await shot("05-fly-to");

// Layers through the view's tools (each edit goes through the geo server).
const added = await call("add_layer", { layer: config.add_layer, fly: false });
report.checks.addLayer = { result: added?.content?.[0]?.text, isError: !!added?.isError, drawn: await counts() };
const styled = await call("set_style", { layer: config.add_layer.name, style: { color: "magenta", linewidth: 3 } });
report.checks.setStyle = { result: styled?.content?.[0]?.text, isError: !!styled?.isError };
await call("fly_to", { bbox: config.overview_bbox, duration_s: 0 });
await settle(globe, 800);
await shot("06-added-layer");
const removed = await call("remove_layer", { layer: config.add_layer.name });
report.checks.removeLayer = { result: removed?.content?.[0]?.text, isError: !!removed?.isError, drawn: await counts() };
const view = await call("get_view", {});
report.checks.getView = { text: view?.content?.[0]?.text, keys: Object.keys(view?.structuredContent || {}) };
report.viewCalls = await page.evaluate(() => window.__viewCalls.map((c) => ({ name: c.name, action: c.arguments.action })));

// A person clicks inside the polygon: the area is selected and outlined.
const inside = await globe.evaluate(([lon, lat]) => {
  const v = window.geoGlobe.viewer;
  const toWindow = Cesium.SceneTransforms.worldToWindowCoordinates || Cesium.SceneTransforms.wgs84ToWindowCoordinates;
  const p = toWindow(v.scene, Cesium.Cartesian3.fromDegrees(lon, lat));
  return p ? { x: p.x, y: p.y } : null;
}, config.area_click);
await page.mouse.click(innerBox.x + inside.x, innerBox.y + inside.y);
await sleep(1200);
report.checks.areaClickReported = (await lastContext())?.structuredContent?.selected_feature ?? null;
await settle(globe, 500);
await shot("06b-area-selected");

// Theme follows the host.
await page.evaluate(() => window.__setTheme("dark"));
await call("select_feature", { selected_feature: config.click, fly: false });
await settle(globe, 1200);
report.checks.darkBackground = await globe.evaluate(() => getComputedStyle(document.body).backgroundColor);
await shot("07-dark");
await call("select_feature", { selected_feature: [] });
report.checks.cleared = await globe.evaluate(() => window.geoGlobe.selected.length);

report.contextUpdates = await page.evaluate(() => window.__context.length);
writeFileSync(path.join(out, `${label}-report.json`), JSON.stringify(report, null, 2));
await bctx.close();
await browser.close();
host.close(); sandbox.close();
console.log(path.join(out, `${label}-report.json`));
process.exit(0);
