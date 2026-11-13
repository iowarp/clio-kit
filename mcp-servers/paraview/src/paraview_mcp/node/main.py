"""Start the node half: one loopback port, one token, detached from the link.

Run by the front as::

    pvpython --dr --force-offscreen-rendering <folder>/code/run_node.py --root <folder>

It writes ``<folder>/node.json`` (mode 0600) with its pid, port and token once
the port is listening, keeps its scene when the link drops, and shuts down
after ``--idle-seconds`` with no link, writing the scene manifest first.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import socket
import sys
import time
from pathlib import Path
from typing import Any


def _pvserver_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def write_private(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as JSON readable only by this user."""
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _lock(root: Path) -> Any:
    import fcntl

    handle = open(root / "node.lock", "w")  # noqa: SIM115 - held for the process life
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another node half is running in this folder", flush=True)
        sys.exit(3)
    return handle


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="clio-paraview-node")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--idle-seconds", type=float, default=1800.0)
    parser.add_argument("--threshold-mb", type=float, default=None)
    parser.add_argument("--pvserver-port", type=int, default=11111)
    parser.add_argument("--code-hash", default="")
    parser.add_argument("--growth-interval", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse(argv)
    root: Path = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    lock = _lock(root)
    site = root / "site"
    if site.is_dir():
        # ParaView's own packages (wslink, aiohttp) stay first on the path.
        sys.path.append(str(site))

    from paraview import simple

    session = None
    if _pvserver_listening(args.pvserver_port):
        simple.Connect("localhost", args.pvserver_port)
        session = f"localhost:{args.pvserver_port}"
    threshold = args.threshold_mb
    if threshold is None:
        settings = simple.GetSettingsProxy("RenderViewSettings")
        threshold = float(settings.RemoteRenderThreshold)

    from .app import NodeApp
    from .engine import Engine

    engine = Engine(root, threshold_mb=threshold, session=session)
    token = secrets.token_urlsafe(32)
    node = NodeApp(engine, token)
    server = node.server
    record_path = root / "node.json"

    def ready(**_kwargs: Any) -> None:
        write_private(
            record_path,
            {
                "pid": os.getpid(),
                "port": int(server.port),
                "token": token,
                "code_hash": args.code_hash,
                "session": session,
                "threshold_mb": threshold,
                "idle_seconds": args.idle_seconds,
                "started": time.time(),
            },
        )
        print(f"node half listening on 127.0.0.1:{server.port}", flush=True)
        loop = asyncio.get_event_loop()
        loop.create_task(node.watch_growth(args.growth_interval))
        loop.create_task(idle(node, args.idle_seconds, record_path, lock))

    server.controller.on_server_ready.add(ready)
    server.start(
        host="127.0.0.1",
        port=0,
        open_browser=False,
        show_connection_info=False,
        disable_logging=True,
        timeout=0,
        exec_mode="main",
    )


async def idle(node: Any, limit: float, record_path: Path, lock: Any) -> None:
    """Shut down after ``limit`` seconds with no link, saving the scene first."""
    from . import manifest

    step = max(0.5, min(30.0, limit / 4))
    while True:
        await asyncio.sleep(step)
        if node.idle_for() < limit:
            continue
        engine = node.engine
        if engine.scene is not None:
            engine.scene["camera"] = engine.read_camera()
            manifest.write(engine.root, engine.scene)
        print(f"no link for {limit:g} s: shutting down", flush=True)
        try:
            record_path.unlink()
        except FileNotFoundError:
            pass
        lock.close()
        os._exit(0)


if __name__ == "__main__":
    main()
