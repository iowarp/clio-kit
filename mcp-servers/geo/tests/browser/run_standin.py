"""Drive the globe through the real geo server and the stand-in host (``standin.mjs``).

Calls ``show_on_globe`` in memory with a small synthetic station layer and a polygon
layer, reads the ``ui://geo/globe`` resource and its declared CSP, serves a bridge that
forwards the view's own ``tools/call`` requests to the same server, and runs the
stand-in host in headless Chromium with software WebGL. Afterwards it reopens the view
by id and checks the edits the view made were saved. Usage (from mcp-servers/geo)::

    uv run --frozen python tests/browser/run_standin.py --out /tmp/geo-globe-standin \\
        [--width 390 --height 844]

Needs node with Playwright (``PLAYWRIGHT_ROOT``, default /opt/node-tools) and network
access to cesium.com and tile.openstreetmap.org. Writes screenshots and a summary.
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

# Synthetic stations around a volcano, two kinds, for the check only.
STATIONS = [
    ("R01", -121.76, 46.85, "continuous"),
    ("R02", -121.98, 46.78, "continuous"),
    ("R03", -121.55, 46.93, "campaign"),
    ("R04", -121.70, 46.62, "continuous"),
    ("R05", -122.20, 47.00, "campaign"),
    ("R06", -121.35, 46.75, "continuous"),
    ("R07", -122.45, 46.70, "campaign"),
    ("R08", -121.90, 47.15, "continuous"),
]


def fixture_layers() -> list[dict[str, Any]]:
    stations = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"station": code, "kind": kind, "elevation_m": 400 + 90 * i},
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }
            for i, (code, lon, lat, kind) in enumerate(STATIONS)
        ],
    }
    zone = {
        "type": "Feature",
        "id": "study-area",
        "properties": {"name": "Study area", "kind": "box"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[-122.6, 46.5], [-121.2, 46.5], [-121.2, 47.25], [-122.6, 47.25], [-122.6, 46.5]]
            ],
        },
    }
    return [
        {
            "name": "study area",
            "geojson": zone,
            "style": {"facecolor": "tab:blue", "alpha": 0.18, "edgecolor": "navy"},
        },
        {
            "name": "stations",
            "geojson": stations,
            "id_property": "station",
            "style": {
                "color_by": "kind",
                "category_colors": {"continuous": "#d62728", "campaign": "#ff7f0e"},
                "markersize": 70,
            },
        },
    ]


TRACK = {
    "name": "survey track",
    "geojson": {
        "type": "LineString",
        "coordinates": [[-122.5, 46.55], [-122.0, 46.9], [-121.6, 46.7], [-121.25, 47.2]],
    },
}


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
        def do_POST(self) -> None:  # noqa: N802 - http.server API
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


async def run(out: Path, width: str, height: str, video: bool) -> dict[str, Any]:
    from fastmcp import Client

    from geo_mcp.server import mcp

    out.mkdir(parents=True, exist_ok=True)
    async with Client(mcp) as client:
        opened = await client.call_tool(
            "show_on_globe", {"layers": fixture_layers(), "title": "Stations around the volcano"}
        )
        resource = (await client.read_resource("ui://geo/globe"))[0]
        csp = ((resource.meta or {}).get("ui") or {}).get("csp") or {}
        server = serve_bridge(Bridge(asyncio.get_running_loop(), client))
        config = {
            "view_html": resource.text,
            "csp": csp,
            "arguments": {"title": "Stations around the volcano"},
            "result": {
                "content": [c.model_dump(mode="json") for c in opened.content],
                "structuredContent": opened.structured_content,
            },
            "bridge": f"http://127.0.0.1:{server.server_address[1]}/call",
            "click": "R03",
            "select": "R07",
            "fly_bbox": [-122.1, 46.75, -121.6, 47.0],
            "overview_bbox": [-122.7, 46.45, -121.1, 47.3],
            "add_layer": TRACK,
            "area_click": [-121.35, 47.18],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(config, handle)
        env = dict(os.environ)
        if video:
            env["STANDIN_VIDEO"] = "1"
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
                width,
                "--height",
                height,
            ],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
            env=env,
        )
        os.unlink(handle.name)
        server.shutdown()
        if done.returncode != 0:
            raise RuntimeError(f"stand-in failed:\n{done.stdout}\n{done.stderr}")
        report = json.loads(Path(done.stdout.strip().splitlines()[-1]).read_text())
        reopened = await client.call_tool(
            "show_on_globe", {"view": opened.structured_content["view"]}
        )
    report["opening_text"] = opened.content[0].text
    report["reopened_layers"] = [layer["name"] for layer in reopened.structured_content["layers"]]
    report["opening_meta"] = opened.meta
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--width", default="1440")
    parser.add_argument("--height", default="900")
    parser.add_argument("--video", action="store_true", help="record a webm of the run")
    opts = parser.parse_args()
    report = asyncio.run(run(opts.out, opts.width, opts.height, opts.video))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    raise SystemExit(main())
