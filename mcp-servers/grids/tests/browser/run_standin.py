"""Drive the grid editor through the real grids server and a stand-in MCP Apps host.

Opens a synthetic bathymetry with ``grid_open`` in memory, serves the
``ui://grids/editor`` resource inside a stand-in host page, forwards the
editor's own ``tools/call`` requests to the same server, and runs
``standin.mjs`` in headless Chromium: drag a box, set it to land, pick a cell,
undo, and fill enclosed lakes. Then checks the edit log on disk. Usage (from
mcp-servers/grids)::

    uv run python tests/browser/run_standin.py --out /tmp/grids-standin [--width 390 --height 844]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from fastmcp import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_grids import bathymetry  # noqa: E402

from grids_mcp.server import EDITOR_URI, mcp  # noqa: E402

HERE = Path(__file__).resolve().parent

HOST = """<!doctype html><html><head><meta charset="utf-8"><title>stand-in host</title>
<style>body{margin:0;background:#1e1f22}iframe{border:0;width:100%;height:100vh}</style></head>
<body><iframe id="app" src="/editor.html"></iframe><script>
const frame = document.getElementById('app');
const opened = __OPENED__;
window.hostLog = [];
window.addEventListener('message', async (event) => {
  const m = event.data; if (!m || m.jsonrpc !== '2.0') return;
  const reply = (result) => frame.contentWindow.postMessage({jsonrpc:'2.0', id:m.id, result}, '*');
  window.hostLog.push(m.method);
  if (m.method === 'ui/initialize') reply({protocolVersion: m.params.protocolVersion, hostContext: {theme: 'dark'}});
  else if (m.method === 'ui/notifications/initialized')
    frame.contentWindow.postMessage({jsonrpc:'2.0', method:'ui/notifications/tool-result', params: opened}, '*');
  else if (m.method === 'tools/call') {
    const response = await fetch('/call', {method:'POST', body: JSON.stringify(m.params)});
    reply(await response.json());
  }
});
</script></body></html>"""


async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async with Client(mcp) as client:
        result = await client.call_tool(name, arguments, raise_on_error=False)
    return {
        "isError": result.is_error,
        "structuredContent": result.structured_content,
        "content": [{"type": "text", "text": c.text} for c in result.content if hasattr(c, "text")],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = bathymetry(out / "topo.nc")
    opened = asyncio.run(call("grid_open", {"path": str(base)}))

    async def editor_html() -> str:
        async with Client(mcp) as client:
            return str((await client.read_resource(EDITOR_URI))[0].text)

    editor = asyncio.run(editor_html())
    host = HOST.replace("__OPENED__", json.dumps(opened))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass

        def _send(self, body: str, kind: str) -> None:
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            self._send(editor if self.path == "/editor.html" else host, "text/html")

        def do_POST(self) -> None:  # noqa: N802
            params = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send(json.dumps(asyncio.run(call(params["name"], params["arguments"]))),
                       "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        run = subprocess.run(
            ["node", str(HERE / "standin.mjs"), f"http://127.0.0.1:{server.server_port}/",
             str(out), str(args.width), str(args.height)],
            check=False, capture_output=True, text=True, timeout=300,
        )
    finally:
        server.shutdown()
    print(run.stdout)
    if run.returncode:
        print(run.stderr, file=sys.stderr)
        sys.exit(run.returncode)
    log = json.loads((out / "topo.edited.nc.edits.json").read_text())
    summary = {"edits": [(e["op"], e["cells_changed"]) for e in log["edits"]],
               "base_hash": log["base"]["hash"], "result_hash": log["result"]["hash"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
