"""One Avogadro per server process: a virtual display, Avogadro, a VNC server on it, and
a loopback service the view reaches through the host.

Processes are started in their own process groups and tracked; ``stop`` (and exit)
terminates each group and removes the session's temporary folder. The service binds
127.0.0.1 on a free port and refuses any request without the session's bearer token,
which CLIO's proxy adds (``ai.iowarp.clio/service``); it serves the stream page, the
vendored noVNC, ``state`` (the last thing the person sent) and a WebSocket bridged to
the VNC server's TCP port.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from importlib import resources
from pathlib import Path
from typing import Any

from . import install, rpc
from .errors import AvogadroError

PKG = resources.files("avogadro_mcp")
#: The virtual screen; Avogadro's window fills it (there is no window manager to do it).
WIDTH, HEIGHT = 1440, 900  # the window's size until the view reports its own
#: The virtual screen; the window and the streamed region follow the view inside it.
MAX_WIDTH, MAX_HEIGHT = 3840, 2160
#: Below this Avogadro's desktop interface does not fit; the view scales it down instead.
MIN_WIDTH, MIN_HEIGHT = 560, 420
RPC_NAME = "avogadro"  # Avogadro's default RPC socket name, under its TMPDIR


def workspace_dir() -> Path:
    """``chemistry/avogadro`` under $CLIO_KIT_ARTIFACTS (the workspace) or the cwd."""
    base = Path(os.environ.get("CLIO_KIT_ARTIFACTS") or Path.cwd()).expanduser()
    folder = base.resolve() / "chemistry" / "avogadro"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


#: A Unix socket path holds at most 107 bytes on Linux (108 with the terminator).
SOCKET_PATH_MAX = 107


def socket_dir(preferred: Path) -> Path:
    """Avogadro's TMPDIR, which holds its RPC socket: ``preferred`` when the socket path
    fits, else a fresh folder under /tmp. A sandbox's TMPDIR can be deep enough that
    ``<TMPDIR>/avogadro`` is over the limit, and Avogadro then fails to listen (EINVAL)."""
    if len(str(preferred / RPC_NAME).encode()) <= SOCKET_PATH_MAX:
        preferred.mkdir(parents=True)
        return preferred
    return Path(tempfile.mkdtemp(prefix="clio-avo-", dir="/tmp"))


def fitted_size(width: int, height: int) -> tuple[int, int]:
    """The window size for a view of ``width`` x ``height`` CSS pixels, within the limits."""
    return (
        max(MIN_WIDTH, min(MAX_WIDTH, int(width))),
        max(MIN_HEIGHT, min(MAX_HEIGHT, int(height))),
    )


class Session:
    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="clio-avogadro-"))
        self.tmp = socket_dir(self.root / "tmp")
        self.data = self.root / "data"
        self.handoff = self.root / "handoff"
        for p in (self.data, self.root / "config", self.handoff):
            p.mkdir(parents=True)
        self.token = secrets.token_urlsafe(32)
        self.procs: dict[str, subprocess.Popen[bytes]] = {}
        self.display: str = ""
        self.vnc_port = 0
        self.service_port = 0
        self.size = (WIDTH, HEIGHT)
        self.command = ""
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self.opened: list[str] = []
        self.workdir: Path | None = None

    # --- processes -------------------------------------------------------------
    def _spawn(
        self,
        name: str,
        argv: list[str],
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        log = (self.root / f"{name}.log").open("wb")
        self.procs[name] = subprocess.Popen(
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            cwd=cwd or self.root,
        )

    def _alive(self, name: str) -> bool:
        p = self.procs.get(name)
        return p is not None and p.poll() is None

    def running(self) -> bool:
        return all(self._alive(n) for n in ("xvfb", "avogadro", "x11vnc")) and bool(self._server)

    def _log_tail(self, name: str) -> str:
        try:
            return (self.root / f"{name}.log").read_text(errors="replace")[-600:]
        except OSError:
            return ""

    def start(self) -> None:
        if self.running():
            return
        ok, why = install.platform_supported()
        if not ok:
            raise AvogadroError("unsupported_platform", why)
        found = install.find()
        if not found["found"]:
            raise AvogadroError("not_installed", install.missing_text(found))
        tools = install.display_tools()
        if tools["missing"]:
            raise AvogadroError(
                "display_missing",
                f"The streamed view needs {', '.join(tools['missing'])} on this machine. "
                + tools["install"],
            )
        self.command = found["command"]
        self.stop(keep_root=True)
        # The display: Xvfb picks a free number and writes it to the pipe.
        rfd, wfd = os.pipe()
        log = (self.root / "xvfb.log").open("wb")
        self.procs["xvfb"] = subprocess.Popen(
            [
                tools["found"]["Xvfb"],
                "-displayfd",
                str(wfd),
                "-screen",
                "0",
                f"{MAX_WIDTH}x{MAX_HEIGHT}x24",
                "-nolisten",
                "tcp",
            ],
            pass_fds=(wfd,),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        os.close(wfd)
        number = b""
        deadline = time.monotonic() + 15
        while not number.endswith(b"\n") and time.monotonic() < deadline:
            chunk = os.read(rfd, 16)
            if not chunk:
                break
            number += chunk
        os.close(rfd)
        if not number.strip():
            raise AvogadroError("start_failed", "Xvfb did not start: " + self._log_tail("xvfb"))
        self.display = ":" + number.decode().strip()
        # Avogadro, with its own TMPDIR (its RPC socket), data and config folders.
        cmd_dir = self.data / "OpenChemistry" / "Avogadro" / "commands"
        cmd_dir.mkdir(parents=True, exist_ok=True)
        (cmd_dir / "send_to_clio.py").write_bytes(PKG.joinpath("handoff_command.py").read_bytes())
        env = dict(os.environ)
        env.update(
            DISPLAY=self.display,
            TMPDIR=str(self.tmp),
            XDG_DATA_HOME=str(self.data),
            XDG_CONFIG_HOME=str(self.root / "config"),
            QT_QPA_PLATFORM="xcb",
            LIBGL_ALWAYS_SOFTWARE="1",
            CLIO_AVOGADRO_HANDOFF=str(self.handoff),
            APPIMAGE_EXTRACT_AND_RUN="1",
        )
        # Started in the workspace, so Avogadro's Save and Open dialogs begin there and a
        # file the person saves is a workspace file.
        self.workdir = workspace_dir()
        self._spawn(
            "avogadro",
            # --skip-dialogs (Avogadro 2.1): a first-run dialog otherwise waits for a
            # click before the RPC listener starts, so in an unattended run the socket
            # never appears. The socket keeps its default name: 2.1 refuses to listen
            # on a name given with --rpc-name here (QLocalServer error 22).
            [self.command, "--skip-dialogs", "-qwindowgeometry", f"{WIDTH}x{HEIGHT}+0+0"],
            env,
            cwd=self.workdir,
        )
        sock = self.tmp / RPC_NAME
        deadline = time.monotonic() + 60
        while not sock.exists():
            if not self._alive("avogadro") or time.monotonic() > deadline:
                raise AvogadroError(
                    "start_failed", "Avogadro did not start: " + self._log_tail("avogadro")
                )
            time.sleep(0.3)
        # The VNC server on that display, loopback only, behind the service's token.
        self.vnc_port = free_port()
        self._spawn(
            "x11vnc",
            [
                tools["found"]["x11vnc"],
                "-display",
                self.display,
                "-localhost",
                "-rfbport",
                str(self.vnc_port),
                "-nopw",
                "-forever",
                "-shared",
                # Only the region the window fills is streamed; fit() moves it.
                "-clip",
                f"{WIDTH}x{HEIGHT}+0+0",
                "-quiet",
            ],
        )
        deadline = time.monotonic() + 15
        while True:
            try:
                socket.create_connection(("127.0.0.1", self.vnc_port), 1).close()
                break
            except OSError:
                if not self._alive("x11vnc") or time.monotonic() > deadline:
                    raise AvogadroError(
                        "start_failed", "x11vnc did not start: " + self._log_tail("x11vnc")
                    ) from None
                time.sleep(0.2)
        self._start_service()

    def stop(self, keep_root: bool = False) -> None:
        if self._server is not None:
            self._server.should_exit = True
            if self._thread:
                self._thread.join(timeout=5)
            self._server = None
        for name in ("x11vnc", "avogadro", "xvfb"):
            p = self.procs.pop(name, None)
            if p and p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                    p.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        if not keep_root:
            shutil.rmtree(self.root, ignore_errors=True)
            shutil.rmtree(self.tmp, ignore_errors=True)

    # --- what the agent does ---------------------------------------------------
    def fit(self, width: int, height: int) -> tuple[int, int]:
        """Make the Avogadro window and the streamed region the view's size, in CSS pixels.

        The view then draws the window 1:1 instead of shrinking a fixed desktop into its
        panel. Narrower than MIN_WIDTH x MIN_HEIGHT stays at that size and the view
        scales it down (Avogadro's desktop interface does not fit a phone).
        """
        w, h = fitted_size(width, height)
        if (w, h) == self.size or not self.running():
            return self.size
        from Xlib import X, display  # only with a running display

        d = display.Display(self.display)
        try:
            root = d.screen().root
            for win in root.query_tree().children:
                attrs = win.get_attributes()
                if attrs.map_state != X.IsViewable or attrs.override_redirect:
                    continue
                geo = win.get_geometry()
                if geo.width >= 300 and geo.height >= 200:  # the main window, not a tooltip
                    win.configure(x=0, y=0, width=w, height=h)
            d.sync()
        finally:
            d.close()
        subprocess.run(
            [
                install.display_tools()["found"]["x11vnc"],
                "-display",
                self.display,
                "-sync",
                "-remote",
                f"clip:{w}x{h}+0+0",
            ],
            capture_output=True,
            timeout=15,
            check=False,
        )
        self.size = (w, h)
        return self.size

    def rpc(self, method: str, params: dict[str, Any]) -> Any:
        if not self.running():
            raise AvogadroError("not_running", "Avogadro is not open; call open_in_avogadro.")
        return rpc.call(self.tmp / RPC_NAME, method, params)

    def latest_sent(self) -> Path | None:
        files = sorted(self.handoff.glob("sent-*.cjson"))
        return files[-1] if files else None

    def state(self) -> dict[str, Any]:
        from . import cjson

        sent = self.latest_sent()
        out: dict[str, Any] = {"opened": self.opened[-1] if self.opened else None, "sent": None}
        if sent:
            try:
                mol = cjson.read(json.loads(sent.read_text()))
                out["sent"] = {
                    "id": sent.stem,
                    "formula": mol["formula"],
                    "n_atoms": mol["n_atoms"],
                    "selected": mol["selected"],
                    "selected_elements": [mol["symbols"][i] for i in mol["selected"]],
                }
            except (ValueError, KeyError, json.JSONDecodeError):
                out["sent"] = {"id": sent.stem, "unreadable": True}
        return out

    # --- the loopback service -----------------------------------------------------
    def _start_service(self) -> None:
        import uvicorn
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Route, WebSocketRoute
        from starlette.websockets import WebSocket, WebSocketDisconnect

        token = self.token
        novnc = Path(str(PKG.joinpath("novnc")))

        def authorized(headers: Any) -> bool:
            return secrets.compare_digest(headers.get("authorization", ""), f"Bearer {token}")

        async def page(request: Request) -> Response:
            if not authorized(request.headers):
                return Response(status_code=401)
            path = request.path_params.get("path", "")
            if path in ("", "index.html"):
                body = PKG.joinpath("stream_page.html").read_bytes()
                return Response(body, media_type="text/html")
            if path == "state":
                return JSONResponse(self.state())
            if path == "size" and request.method == "POST":
                body = await request.json()
                w, h = await asyncio.to_thread(
                    self.fit, int(body.get("width", 0)), int(body.get("height", 0))
                )
                return JSONResponse({"width": w, "height": h})
            if path.startswith("novnc/"):
                target = (novnc / path[len("novnc/") :]).resolve()
                if novnc.resolve() in target.parents and target.is_file():
                    kind = "text/javascript" if target.suffix == ".js" else "text/plain"
                    return Response(target.read_bytes(), media_type=kind)
            return Response(status_code=404)

        async def bridge(ws: WebSocket) -> None:
            if not authorized(ws.headers):
                await ws.close(code=1008)
                return
            protocols = ws.scope.get("subprotocols") or []
            await ws.accept(subprotocol="binary" if "binary" in protocols else None)
            reader, writer = await asyncio.open_connection("127.0.0.1", self.vnc_port)

            async def up() -> None:
                try:
                    while True:
                        data = await ws.receive_bytes()
                        writer.write(data)
                        await writer.drain()
                except (WebSocketDisconnect, RuntimeError):
                    pass
                finally:
                    writer.close()

            async def down() -> None:
                try:
                    while chunk := await reader.read(65536):
                        await ws.send_bytes(chunk)
                except (WebSocketDisconnect, RuntimeError, ConnectionError):
                    pass
                finally:
                    try:
                        await ws.close()
                    except RuntimeError:
                        pass

            await asyncio.gather(up(), down())

        app = Starlette(
            routes=[
                WebSocketRoute("/websockify", bridge),
                Route("/{path:path}", page, methods=["GET", "POST"]),
            ]
        )
        self.service_port = free_port()
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.service_port,
            log_level="warning",
            ws="websockets-sansio",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline:
                raise AvogadroError("start_failed", "The view's service did not start.")
            time.sleep(0.05)
