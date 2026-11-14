"""Drive the ParaView front through a real MCP client and the stand-in host.

Starts the front over stdio (as an MCP host would), opens a dataset, adds an
isosurface, then mounts the view in ``standin.mjs`` and drives Chromium:
rotate, pick (and, with ``--slider``, arrow keys and a drag on the edit form's
first isovalue slider). Afterwards it asks the front what was picked, measures the
field and captures a PNG, and checks the capture's camera against the camera
the view last reported. Usage (from mcp-servers/paraview)::

    uv run python tests/browser/run_standin.py --dataset /path/wave.vti \\
        --field RTData --iso 150 --out /tmp/pv-standin
    uv run python tests/browser/run_standin.py --dataset /path/clouds.vti \\
        --field q --iso 1e-5 --iso2 1e-4 --slider --out /tmp/pv-slider

Environment: everything the front reads (PARAVIEW_ADDRESS, PARAVIEW_VIA,
PARAVIEW_PREPARE, PARAVIEW_REMOTE_RENDER_THRESHOLD_MB, ...) passes through.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

HERE = Path(__file__).resolve().parent
SERVER = HERE.parents[1]


def _isosurfaces(found: Any) -> list[float] | None:
    """The Contour's applied isovalues, anywhere in ``scene_state``'s structured result."""
    if isinstance(found, dict):
        if found.get("type") == "Contour" and "properties" in found:
            return (found["properties"] or {}).get("Isosurfaces")
        found = list(found.values())
    for item in found if isinstance(found, list) else []:
        value = _isosurfaces(item)
        if value is not None:
            return value
    return None


def _slider_check(sliders: dict[str, Any] | None, applied: list[float] | None) -> Any:
    """Whether each slider input moved it, and the drag's value is the one applied."""
    if not sliders:
        return None
    by = {s["at"]: s for s in sliders["samples"]}
    shown = [float(v) for v in by["after drag"]["values"].replace(",", " ").split()]
    first = [float(v) for v in by["before"]["values"].replace(",", " ").split()]
    return {
        "keys_moved": by["after 3 ArrowRight"]["valuenow"] != by["before"]["valuenow"],
        "drag_moved_live": by["mid-drag"]["valuenow"]
        != by["after 3 ArrowRight"]["valuenow"],
        "drag_moved": by["after drag"]["valuenow"] != by["mid-drag"]["valuenow"],
        "applied": applied,
        "applied_is_shown": applied is not None
        and len(applied) == len(shown)
        and all(abs(a - b) <= 1e-9 + 1e-6 * abs(b) for a, b in zip(applied, shown)),
        "others_kept": shown[1:] == first[1:],
    }


def _texts(result: Any) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--iso", type=float, required=True)
    parser.add_argument(
        "--iso2", type=float, help="a second isovalue, which the slider must keep"
    )
    parser.add_argument(
        "--slider", action="store_true", help="drive the edit form's value slider"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--width", default="1440")
    parser.add_argument("--height", default="900")
    parser.add_argument("--remote", action="store_true", help="expect remote mode")
    opts = parser.parse_args()
    opts.out.mkdir(parents=True, exist_ok=True)
    transport = StdioTransport(
        command="uv",
        args=["run", "--directory", str(SERVER), "paraview-mcp"],
        env=dict(os.environ),
    )
    summary: dict[str, Any] = {}
    async with Client(transport, timeout=900) as client:
        opened = await client.call_tool(
            "open_view", {"path": opts.dataset, "new": True}
        )
        service = (opened.meta or {}).get("ai.iowarp.clio/service") or {}
        summary["open_view_text"] = _texts(opened)
        summary["token_in_open_view_text"] = service.get("token", "") in json.dumps(
            [c.model_dump() for c in opened.content]
        ) + json.dumps(opened.structured_content)
        await client.call_tool(
            "create_filter",
            {
                "filter": "Contour",
                "properties": {
                    "ContourBy": ["POINTS", opts.field],
                    "Isosurfaces": [opts.iso]
                    + ([opts.iso2] if opts.iso2 is not None else []),
                    "ComputeScalars": 1,
                },
            },
        )
        view = await client.read_resource("ui://paraview/view")
        run = {
            "view_html": view[0].text,
            "port": service["port"],
            "token": service["token"],
            "arguments": {"path": opts.dataset},
            "result": {
                "content": [c.model_dump() for c in opened.content],
                "structuredContent": opened.structured_content,
            },
            "steps": [
                {"do": "drag", "shot": "02-rotated"},
                {"do": "click", "x": 0.5, "y": 0.5, "shot": "03-picked"},
                *(
                    [{"do": "edit-slider", "shot": "05-slider-dragged"}]
                    if opts.slider
                    else []
                ),
            ],
        }
        config = opts.out / "run.json"
        config.write_text(json.dumps(run))
        done = await asyncio.to_thread(
            subprocess.run,
            [
                "node",
                str(HERE / "standin.mjs"),
                "--config",
                str(config),
                "--out",
                str(opts.out),
                "--width",
                opts.width,
                "--height",
                opts.height,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        config.unlink()
        if done.returncode != 0:
            print(done.stdout, done.stderr, file=sys.stderr)
            return 1
        report = json.loads((opts.out / "report.json").read_text())
        picked = await client.call_tool("inspect_selection", {})
        measured = await client.call_tool("measure_field", {"field": opts.field})
        captured = await client.call_tool("capture", {"width": 800, "height": 600})
        state = await client.call_tool("scene_state", {})
    images = [c for c in captured.content if getattr(c, "type", "") == "image"]
    if images:
        import base64

        (opts.out / "capture.png").write_bytes(base64.b64decode(images[0].data))
    drag = report.get("drag") or {}
    during = [
        f
        for f in report["frames"]
        if drag and drag["start"] <= f["at"] <= drag["end"] + 200
    ]
    contexts = report.get("context") or []
    last_camera = None
    for entry in reversed(contexts):
        camera = (entry["params"].get("structuredContent") or {}).get("camera")
        if camera:
            last_camera = camera
            break
    summary.update(
        webgl=report.get("webgl"),
        gl_flags=report.get("gl_flags"),
        modes_requested=report.get("modes"),
        context_updates=len(contexts),
        last_context_text=(
            contexts[-1]["params"]["content"][0]["text"] if contexts else None
        ),
        frames_during_drag=len(during),
        image_frames_during_drag=sum(
            1
            for f in during
            if f.get("image") or (f.get("binary") and f.get("size", 0) > 2000)
        ),
        render_requests_during_drag=sorted(
            {f["method"] for f in during if f.get("sent") and f.get("method")}
        ),
        console_errors=report["consoleErrors"],
        failed_requests=report["failedRequests"],
        requests_outside_service=[
            r["url"] for r in report["requests"] if not r["url"].startswith("/service")
        ],
        inspect_selection=_texts(picked),
        measure_field=_texts(measured),
        capture_text=_texts(captured),
        capture_camera_matches_view=bool(
            last_camera
            and _texts(captured).find(f"{last_camera['position'][0]:.6g}") >= 0
        ),
        view_text=report.get("viewText"),
        frames_after_drag_start=sum(
            1
            for f in report["frames"]
            if drag and f["at"] >= drag["start"] and not f.get("sent")
        ),
        scene_state=_texts(state),
        sliders=report.get("sliders"),
        slider_check=_slider_check(
            report.get("sliders"), _isosurfaces(state.structured_content)
        ),
        screenshots=report["shots"] + [str(opts.out / "capture.png")],
    )
    (opts.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
