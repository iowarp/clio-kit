"""Drive the slice view through the real slicer server and the stand-in host.

Slices a model with the PrusaSlicer installed here (``slice``), opens it with
``show_slice``, reads the ``ui://slicer/layers`` resource and its declared CSP, serves
a bridge that forwards the view's own ``tools/call`` requests (``slice_layers``,
``remember_slice_view``) to the same server, and runs ``standin.mjs`` in headless
Chromium with software WebGL. Afterwards it reopens the slice and checks the view's
last layer was saved. Usage (from mcp-servers/slicer)::

    uv run --frozen python tests/browser/run_standin.py --model MODEL.step \\
        --out /tmp/slicer-standin [--width 390 --height 844]

Needs PrusaSlicer, node with Playwright (``PLAYWRIGHT_ROOT``, default /opt/node-tools)
and network access to cdn.jsdelivr.net (three.js). Writes screenshots and a summary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


class Bridge:
    """Runs the server's tools for the view, on the driver's event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop, client: Any) -> None:
        self.loop, self.client = loop, client

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        future = asyncio.run_coroutine_threadsafe(
            self.client.call_tool(name, arguments, raise_on_error=False), self.loop
        )
        result = future.result(timeout=120)
        return {
            "content": [c.model_dump(mode="json") for c in result.content],
            "structuredContent": result.structured_content,
            "isError": result.is_error,
        }


def serve_bridge(bridge: Bridge) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            answer = json.dumps(bridge.call(body["name"], body.get("arguments") or {}))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("access-control-allow-origin", "*")
            self.end_headers()
            self.wfile.write(answer.encode())

        def log_message(self, *_args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


async def run(opts: argparse.Namespace) -> dict[str, Any]:
    from fastmcp import Client

    from slicer_mcp.server import mcp

    out: Path = opts.out
    out.mkdir(parents=True, exist_ok=True)
    async with Client(mcp) as client:
        sliced = await client.call_tool(
            "slice",
            {
                "model": str(opts.model),
                "printer": opts.printer,
                "material": "PLA",
                "quality": "0.2 mm",
            },
        )
        slice_id = sliced.structured_content["slice_id"]
        opened = await client.call_tool("show_slice", {"slice_id": slice_id})
        resource = (await client.read_resource("ui://slicer/layers"))[0]
        csp = ((resource.meta or {}).get("ui") or {}).get("csp") or {}
        server = serve_bridge(Bridge(asyncio.get_running_loop(), client))
        bounds = opened.structured_content["bounds"]
        config = {
            "view_html": resource.text,
            "csp": csp,
            "arguments": {"slice_id": slice_id},
            "result": {
                "content": [c.model_dump(mode="json") for c in opened.content],
                "structuredContent": opened.structured_content,
            },
            "bridge": f"http://127.0.0.1:{server.server_address[1]}/call",
            "go_to_z": opts.go_to_z,
            "pick": [bounds[0], (bounds[1] + bounds[3]) / 2],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(config, handle)
        done = await asyncio.to_thread(
            subprocess.run,
            [
                "node",
                str(HERE / "standin.mjs"),
                "--config",
                handle.name,
                "--out",
                str(out),
                "--width",
                opts.width,
                "--height",
                opts.height,
            ],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
            env=dict(os.environ),
        )
        os.unlink(handle.name)
        server.shutdown()
        if done.returncode != 0:
            raise RuntimeError(f"stand-in failed:\n{done.stdout}\n{done.stderr}")
        report = json.loads(Path(done.stdout.strip().splitlines()[-1]).read_text())
        reopened = await client.call_tool("show_slice", {"slice_id": slice_id})
    report["slice_text"] = sliced.content[0].text
    report["opening_text"] = opened.content[0].text
    report["opening_meta"] = opened.meta
    report["reopened_layer"] = reopened.structured_content["layer"]
    report["reopened_visible"] = reopened.structured_content["visible_features"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--printer", default="Prusa MK4")
    parser.add_argument("--go-to-z", dest="go_to_z", type=float, default=14.6)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--width", default="1440")
    parser.add_argument("--height", default="900")
    opts = parser.parse_args()
    report = asyncio.run(run(opts))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    raise SystemExit(main())
