"""Start the node half: one loopback port, one token, detached from the link.

Run by the front as::

    pvpython --dr --force-offscreen-rendering <folder>/code/run_node.py --root <folder>

It writes ``<folder>/node.json`` (mode 0600) with its pid, port and token once
the port is listening, keeps its scene when the link drops, and shuts down
after ``--idle-seconds`` with no link, writing the scene manifest first.
``--name`` picks another record and lock name, for a second node half in a
network namespace that cannot reach the first one. Each start removes the
``node-<ns>`` slots of node halves that have gone.

A front that cannot signal it, such as one confined in another sandbox, asks it
to stop by writing ``<name>.stop`` in the folder; it then saves its scene and
lets go of its lock.
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


def network_namespace() -> str:
    """This process's network namespace id on Linux, else empty."""
    try:
        link = os.readlink("/proc/self/ns/net")
    except OSError:
        return ""
    return "".join(ch for ch in link if ch.isdigit())


def _try_lock(handle: Any) -> bool:
    """Take ``handle``'s lock without waiting; False when another process holds it."""
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _lock(root: Path, name: str = "node") -> Any:
    handle = open(root / f"{name}.lock", "a")  # noqa: SIM115 - held for the process life
    if not _try_lock(handle):
        print("another node half is running in this folder", flush=True)
        sys.exit(3)
    return handle


def prune_slots(root: Path, own: str, min_age: float = 60.0) -> list[str]:
    """Remove ``<base>-<ns>`` slots whose node half has gone, so they do not pile up.

    A slot is gone when nobody holds its lock. Slots touched in the last
    ``min_age`` seconds are left alone, since their node may still be starting.
    A pvserver's own slot (``node-pv<hash>``) is kept: its log and lost note say
    why its node half went.
    """
    import re

    removed = []
    for lock_path in sorted(root.glob("node-*.lock")):
        name = lock_path.stem
        if name == own or not re.search(r"-(\d+|w\d+)$", name):
            continue
        try:
            if time.time() - lock_path.stat().st_mtime < min_age:
                continue
            handle = open(lock_path, "a")  # noqa: SIM115 - closed below
        except OSError:
            continue
        if not _try_lock(handle):
            handle.close()
            continue
        # A held lock blocks removing its file on Windows; let it go first.
        handle.close()
        for suffix in (".json", ".log", ".prev.log", ".lost.json", ".stop", ".lock"):
            try:
                (root / f"{name}{suffix}").unlink()
            except OSError:
                pass
        removed.append(name)
    return removed


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="clio-paraview-node")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--name", default="node")
    parser.add_argument("--idle-seconds", type=float, default=1800.0)
    parser.add_argument("--threshold-mb", type=float, default=None)
    parser.add_argument("--pvserver-port", type=int, default=11111)
    parser.add_argument(
        "--pvserver",
        default="",
        help="host:port of a pvserver to join; the data is read on its machine",
    )
    parser.add_argument("--code-hash", default="")
    parser.add_argument("--growth-interval", type=float, default=2.0)
    return parser.parse_args(argv)


def write_through() -> None:
    """Send ``print`` straight to the log, a line at a time.

    Some pvpython builds (ParaView 5.13) hold Python's output until a normal exit,
    ignoring flush, so a node half that ends with ``os._exit`` would leave an empty
    log and a hung start would show nothing of what it waits for.
    """
    for fd, name in ((1, "stdout"), (2, "stderr")):
        try:
            stream = open(  # noqa: SIM115 - lives as long as the process
                fd, "w", buffering=1, encoding="utf-8", errors="replace", closefd=False
            )
        except OSError:
            continue
        setattr(sys, name, stream)


def main(argv: list[str] | None = None) -> None:
    write_through()
    args = parse(argv)
    root: Path = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(root, 0o700)
    lock = _lock(root, args.name)
    stop_path = root / f"{args.name}.stop"
    # A request left for a node half that had already gone is not for this one.
    stop_path.unlink(missing_ok=True)
    pruned = prune_slots(root, args.name)
    if pruned:
        print(f"removed slots of node halves that have gone: {pruned}", flush=True)
    site = root / "site"
    if site.is_dir():
        # ParaView's own packages (wslink, aiohttp) stay first on the path.
        sys.path.append(str(site))
        from . import vtkjs_patch

        # Logged every start, so a bundle the guard does not fit is visible, not silent.
        print(f"vtk.js display-actor guard: {vtkjs_patch.apply(site)}", flush=True)

    from paraview import simple

    from . import liveness

    session = None
    if args.pvserver:
        host, _, port = args.pvserver.rpartition(":")
        # Said first, so a join that hangs shows what it is waiting for.
        print(f"joining pvserver at {args.pvserver}", flush=True)
        # A pvserver serves one client, so it is joined without a probe first.
        try:
            joined = simple.Connect(host or "localhost", int(port))
        except Exception as exc:  # noqa: BLE001 - reported to the front
            joined, reason = None, f"{type(exc).__name__}: {exc}"
        else:
            reason = "it did not answer, or its ParaView version is not this one"
        if not joined:
            print(f"pvserver join failed: {args.pvserver}: {reason}", flush=True)
            sys.exit(4)
        session = args.pvserver
        print(f"joined pvserver at {args.pvserver}", flush=True)
    elif _pvserver_listening(args.pvserver_port):
        simple.Connect("localhost", args.pvserver_port)
        session = f"localhost:{args.pvserver_port}"
    watched = liveness.pvserver_socket() if session else None
    if watched is not None and not liveness.watchable(watched):
        watched = None
    threshold = args.threshold_mb
    if threshold is None:
        settings = simple.GetSettingsProxy("RenderViewSettings")
        threshold = float(settings.RemoteRenderThreshold)

    from .app import NodeApp
    from .engine import Engine

    engine = Engine(
        root,
        threshold_mb=threshold,
        session=session,
        data_on_server=bool(args.pvserver),
    )
    token = secrets.token_urlsafe(32)
    node = NodeApp(engine, token)
    server = node.server
    record_path = root / f"{args.name}.json"

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
                "net": network_namespace(),
                "started": time.time(),
            },
        )
        print(f"node half listening on 127.0.0.1:{server.port}", flush=True)
        loop = asyncio.get_event_loop()
        loop.create_task(node.watch_growth(args.growth_interval))
        loop.create_task(idle(node, args.idle_seconds, record_path, lock))
        loop.create_task(watch_stop(node, stop_path, record_path, lock))
        if watched is not None and session:
            print(f"watching the pvserver connection to {session}", flush=True)
            loop.create_task(watch_pvserver(node, watched, session, record_path, lock))

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


def shut_down(
    node: Any, record_path: Path, lock: Any, why: str, render: bool = True
) -> None:
    """Save the scene, drop the record and the lock, and exit.

    Without ``render`` (the pvserver has gone) the camera is not read back: the
    manifest keeps the camera of the last commit.
    """
    from . import manifest

    engine = node.engine
    if engine.scene is not None:
        if render:
            engine.scene["camera"] = engine.read_camera()
        manifest.write(engine.root, engine.scene)
    print(f"{why}: shutting down", flush=True)
    try:
        record_path.unlink()
    except FileNotFoundError:
        pass
    lock.close()
    os._exit(0)


#: Seconds between looks at the pvserver connection, and how long the view is
#: given to show that it went before the node half exits.
PVSERVER_LOOK = 2.0
LOST_NOTICE = 3.0


async def watch_pvserver(
    node: Any, fd: int, pvserver: str, record_path: Path, lock: Any
) -> None:
    """Stop drawing once the pvserver closes the connection, and say so.

    Rendering after it has gone yields blank frames or aborts the process, so
    the node half tells its view, leaves ``<name>.lost.json`` for the front, and
    exits.
    """
    from . import liveness

    while True:
        await asyncio.sleep(PVSERVER_LOOK)
        if not liveness.closed(fd):
            continue
        reason = f"the pvserver at {pvserver} closed the connection"
        print(reason, flush=True)
        scene = node.engine.scene
        write_private(
            record_path.with_name(record_path.stem + ".lost.json"),
            {
                "reason": reason,
                "pvserver": pvserver,
                "scene": scene["scene"] if scene else None,
                "time": time.time(),
            },
        )
        node.pvserver_lost(reason)
        await asyncio.sleep(LOST_NOTICE)
        shut_down(node, record_path, lock, reason, render=False)


async def idle(node: Any, limit: float, record_path: Path, lock: Any) -> None:
    """Shut down after ``limit`` seconds with no link, saving the scene first."""
    step = max(0.5, min(30.0, limit / 4))
    while True:
        await asyncio.sleep(step)
        if node.idle_for() >= limit:
            shut_down(node, record_path, lock, f"no link for {limit:g} s")


async def watch_stop(node: Any, stop_path: Path, record_path: Path, lock: Any) -> None:
    """Shut down when a front asks by writing ``stop_path``."""
    while True:
        await asyncio.sleep(0.5)
        if stop_path.exists():
            stop_path.unlink(missing_ok=True)
            shut_down(node, record_path, lock, "a front asked this node half to stop")


if __name__ == "__main__":
    main()
