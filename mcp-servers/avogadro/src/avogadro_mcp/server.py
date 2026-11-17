"""Avogadro MCP server: Avogadro 2 for building and remodelling structures in 3D, streamed
into a view, with what the person builds and selects handed back as files.

Like the KiCad connector, the server finds and starts the application; installing it is
the agent's job with the person's consent (the avogadro plugin's install-avogadro skill).
The server runs Avogadro on a virtual display streamed over VNC to the view through the
host's loopback service. Burette stays the light in-conversation
viewer; Avogadro is for fragments, element changes, optimizing while dragging, crystals,
surfaces and supercells.
"""

from __future__ import annotations

import atexit
import base64
import json
import time
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, TypeVar

from fastmcp import FastMCP
from fastmcp.apps import AppConfig
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import ImageContent, TextContent
from pydantic import Field

from . import cjson, install, present
from .errors import AvogadroError
from .session import Session

VIEW_URI = "ui://avogadro/view"
SERVICE_META = "ai.iowarp.clio/service"
PATHS_META = "ai.iowarp.clio/paths"
STATE_META = "ai.iowarp.clio/state"
ARTIFACTS_ENV = "CLIO_KIT_ARTIFACTS"
VIEW_STATE = [
    {"key": "opened", "description": "The file Avogadro was opened with."},
    {"key": "sent", "description": "What the person last sent: formula, selected atoms."},
]

mcp = FastMCP(
    "avogadro",
    version="1.0.0",  # keep equal to mcp-server-versions.toml
    instructions=(
        "Avogadro 2 in a view, for building and remodelling structures in 3D: fragments, "
        "element changes, optimizing while dragging, crystals, surfaces, supercells. "
        "avogadro_status says whether Avogadro and the streaming programs are here and what "
        "to install when not. open_in_avogadro opens a "
        "structure file in the view; the person edits and sends it back with Extensions > "
        "CLIO > Send to CLIO; read_avogadro writes what they sent as a structure file with "
        "their selected atoms. avogadro_snapshot saves a picture; close_avogadro stops it."
    ),
)

T = TypeVar("T")
_session: Session | None = None


def session() -> Session:
    global _session
    if _session is None:
        _session = Session()
        atexit.register(_session.stop)
    return _session


def _run(fn: Callable[[], T]) -> T:
    try:
        return fn()
    except AvogadroError as exc:
        raise ToolError(exc.text()) from exc


def _text(text: str) -> TextContent:
    return TextContent(type="text", text=text)


def _out_dir(output_dir: str | None) -> Path:
    import os

    base = Path(output_dir or os.environ.get(ARTIFACTS_ENV) or Path.cwd()).expanduser()
    folder = base.resolve() / "chemistry" / "avogadro"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _existing(path: str) -> Path:
    import os

    p = Path(path).expanduser()
    if not p.is_absolute() and not p.exists() and os.environ.get(ARTIFACTS_ENV):
        p = Path(os.environ[ARTIFACTS_ENV]).expanduser() / p
    if not p.is_file():
        raise AvogadroError("not_found", f"The structure file {path!r} does not exist.")
    return p.resolve()


def _status() -> dict[str, Any]:
    ok, why = install.platform_supported()
    return {
        "platform": {"supported": ok, "why": why},
        "avogadro": install.find(),
        "version": install.VERSION,
        "display": install.display_tools(),
        "running": _session is not None and _session.running(),
    }


@mcp.tool(
    name="avogadro_status",
    title="Avogadro Status",
    description=(
        "Whether Avogadro 2 is installed here (and where), whether the virtual display and VNC "
        "server the streamed view needs (Xvfb, x11vnc) are present, what to install when "
        "something is missing, and whether Avogadro is running."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
)
async def avogadro_status() -> ToolResult:
    data = _status()
    return ToolResult(content=[_text(present.status_text(data))], structured_content=data)


@mcp.tool(
    name="open_in_avogadro",
    title="Open in Avogadro",
    description=(
        "Open Avogadro in a view, with a structure file (XYZ, SDF/MOL, PDB, CIF, CJSON, …) or "
        "empty to build from scratch. The person works in it; when they choose Extensions > "
        "CLIO > Send to CLIO, the view tells you, and read_avogadro writes what they sent. "
        "Starts Avogadro the first time (seconds)."
    ),
    app=AppConfig(resource_uri=VIEW_URI),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    meta={PATHS_META: {"path": {"machine": "local", "access": "read"}}, STATE_META: VIEW_STATE},
)
async def open_in_avogadro(
    path: Annotated[
        str | None, Field(description="Structure file to open; empty to build.")
    ] = None,
) -> ToolResult:
    def go() -> dict[str, Any]:
        s = session()
        target = _existing(path) if path else None
        s.start()
        if target:
            s.rpc("openFile", {"fileName": str(target)})
            s.opened.append(str(target))
        return {"opened": str(target) if target else None, "avogadro": s.command}

    data = _run(go)
    s = session()
    return ToolResult(
        content=[_text(present.open_text(data))],
        structured_content=data,
        meta={SERVICE_META: {"port": s.service_port, "token": s.token}},
    )


@mcp.tool(
    name="read_avogadro",
    title="Read from Avogadro",
    description=(
        "The molecule open in Avogadro now, with the person's edits and the atoms they have "
        "selected (0-based indices, file order), written as an XYZ structure file plus "
        "Avogadro's CJSON under output_dir/chemistry/avogadro/. An Avogadro too old to report "
        "its molecule gives what the person last sent with Extensions > CLIO > Send to CLIO."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    meta={PATHS_META: {"output_dir": {"machine": "local", "access": "write"}}},
)
async def read_avogadro(
    output_dir: Annotated[
        str | None, Field(description="Folder for the files; default the workspace.")
    ] = None,
) -> ToolResult:
    def go() -> dict[str, Any]:
        s = session()
        doc, source, sent = None, "live", None
        try:
            reply = s.rpc("getMolecule", {"format": "cjson"})
            doc = json.loads(reply["content"])
        except AvogadroError as exc:
            # Not open, or an Avogadro without getMolecule (before 2.1, JSON-RPC -32601):
            # what the person last sent with Send to CLIO, if anything.
            unknown = "-32601" in str(exc) or "not found" in str(exc).lower()
            if exc.reason != "not_running" and not unknown:
                raise
            sent = s.latest_sent()
            if sent is None and exc.reason == "not_running":
                raise
            if sent is None:
                raise AvogadroError(
                    "nothing_sent",
                    "This Avogadro cannot report its molecule, and the person has not sent "
                    "anything yet. Ask them to choose Extensions > CLIO > Send to CLIO in the "
                    "Avogadro view (selecting the atoms they mean first).",
                ) from exc
            doc, source = json.loads(sent.read_text()), "sent"
        mol = cjson.read(doc)
        folder = _out_dir(output_dir)
        stem = time.strftime("avogadro-%Y%m%d-%H%M%S")
        xyz, cj = folder / f"{stem}.xyz", folder / f"{stem}.cjson"
        n = 2
        while xyz.exists() or cj.exists():
            xyz, cj = folder / f"{stem}-{n}.xyz", folder / f"{stem}-{n}.cjson"
            n += 1
        cjson.write_xyz(xyz, mol, f"from Avogadro; selected {mol['selected']}")
        cj.write_text(json.dumps(doc))
        return {
            "formula": mol["formula"],
            "n_atoms": mol["n_atoms"],
            "selected": mol["selected"],
            "selected_elements": [mol["symbols"][i] for i in mol["selected"]],
            "unit_cell": mol["unit_cell"],
            "structure": {"path": str(xyz), "format": "xyz", "n_atoms": mol["n_atoms"]},
            "cjson": str(cj),
            "source": source,
            **({"sent": sent.stem} if sent is not None else {}),
        }

    data = _run(go)
    return ToolResult(content=[_text(present.read_text(data))], structured_content=data)


@mcp.tool(
    name="avogadro_snapshot",
    title="Avogadro Snapshot",
    description="A PNG of Avogadro's 3D view as it is now, saved and returned as an image.",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
    meta={PATHS_META: {"output_dir": {"machine": "local", "access": "write"}}},
)
async def avogadro_snapshot(
    output_dir: Annotated[
        str | None, Field(description="Folder for the PNG; default the workspace.")
    ] = None,
) -> ToolResult:
    def go() -> Path:
        folder = _out_dir(output_dir)
        png = folder / time.strftime("avogadro-%Y%m%d-%H%M%S.png")
        session().rpc("saveGraphic", {"fileName": str(png)})
        deadline = time.monotonic() + 10
        while not png.is_file() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not png.is_file():
            raise AvogadroError("rpc_failed", "Avogadro did not write the snapshot.")
        return png

    png = _run(go)
    data = base64.b64encode(png.read_bytes()).decode()
    return ToolResult(
        content=[
            _text(f"Snapshot of Avogadro's view: {png}."),
            ImageContent(type="image", data=data, mimeType="image/png"),
        ],
        structured_content={"path": str(png)},
    )


@mcp.tool(
    name="close_avogadro",
    title="Close Avogadro",
    description="Stop Avogadro and its virtual display; what was not sent is lost.",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True},
)
async def close_avogadro() -> ToolResult:
    global _session
    if _session is not None:
        _session.stop()
        _session = None
    return ToolResult(content=[_text("Avogadro is closed.")], structured_content={"closed": True})


@mcp.resource(
    VIEW_URI,
    name="avogadro_view",
    description="Avogadro 2, streamed through the host's service route.",
    mime_type="text/html;profile=mcp-app",
    app=AppConfig(prefers_border=False),
)
def view_resource() -> str:
    return resources.files("avogadro_mcp").joinpath("view.html").read_text("utf-8")


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
