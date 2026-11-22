"""A process beside the node half that sees its pvserver go while ParaView blocks.

A ParaView call that waits on a pvserver which stopped answering holds Python's
lock, so no thread or task of the node half runs until it returns, and after
the pvserver has gone it may never return. The node's own watcher then cannot
say what happened, and the front waits on a call that will not finish.

This child holds a share of the pvserver socket and looks at it every few
seconds, without reading or changing it. Once the far end has closed, it gives
the node half a few seconds to stop by itself; if the node half is still held
up then, the child leaves the lost note the node would have left, drops the
node's record and ends it, so the front fails the call at once with the reason.
It ends with the node half, and it uses only the standard library, so any
Python can run it.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: Seconds between looks, and how long the node half has to stop by itself.
LOOK = 2.0
GRACE = 10.0


def interpreter() -> str:
    """A Python to run the child: a plain one when there is one, else this one.

    pvpython runs a script that never imports ParaView too, at some more memory.
    """
    here = Path(sys.executable)
    for name in ("python.exe", "python3.exe") if os.name == "nt" else ("python3",):
        beside = here.with_name(name)
        if beside.is_file():
            return str(beside)
    if os.name != "nt":
        found = shutil.which("python3")
        if found:
            return found
    return sys.executable


def start(fd: int, pvserver: str, record: Path) -> subprocess.Popen[bytes] | None:
    """Start the child that watches socket ``fd`` for the node half with ``record``."""
    command = [
        interpreter(),
        str(Path(__file__).resolve()),
        "--parent",
        str(os.getpid()),
        "--pvserver",
        pvserver,
        "--record",
        str(record),
    ]
    try:
        if os.name == "nt":
            child = subprocess.Popen(
                [*command, "--fd", "share"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=None,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            share = socket.socket(fileno=fd)
            try:
                data = share.share(child.pid)
            finally:
                share.detach()
            assert child.stdin is not None
            child.stdin.write(data)
            child.stdin.close()
            return child
        return subprocess.Popen(
            [*command, "--fd", str(fd)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=None,
            pass_fds=(fd,),
        )
    except (OSError, ValueError) as exc:
        print(f"the pvserver watchdog did not start: {exc}", flush=True)
        return None


def _alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return False
        try:
            return bool(kernel.WaitForSingleObject(handle, 0) == 0x102)  # timed out
        finally:
            kernel.CloseHandle(handle)
    # An orphan is adopted by another process, so this also covers a reused pid.
    return os.getppid() == pid


def _end(pid: int) -> None:
    try:
        if os.name == "nt":
            os.kill(pid, 9)  # TerminateProcess
        else:
            import signal

            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _write_private(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle)
    os.replace(temporary, path)


def watch(
    probe: socket.socket,
    parent: int,
    pvserver: str,
    record: Path,
    look: float = LOOK,
    grace: float = GRACE,
    alive: Callable[[int], bool] = _alive,
    end: Callable[[int], None] = _end,
) -> str:
    """Watch until the node half ends; return what ended the watch."""
    try:
        from .liveness import closed
    except ImportError:  # run as a script, beside liveness.py
        from liveness import closed  # type: ignore[import-not-found,no-redef]

    lost = record.with_name(record.stem + ".lost.json")

    def leave_note(reason: str) -> None:
        """What the node half would have left, unless it did; and no stale record."""
        if not lost.exists():
            _write_private(
                lost,
                {
                    "reason": f"the pvserver at {pvserver} {reason}",
                    "pvserver": pvserver,
                    "scene": None,
                    "time": time.time(),
                },
            )
        try:
            record.unlink()
        except OSError:
            pass

    while True:
        time.sleep(look)
        gone = closed(probe.fileno())
        if not alive(parent):
            if gone:
                leave_note("closed the connection, and the node half ended with it")
            return "the node half ended"
        if not gone:
            continue
        # The node half's own watcher says so, if its loop is free.
        until = time.monotonic() + grace
        while time.monotonic() < until:
            if not alive(parent):
                leave_note("closed the connection, and the node half ended with it")
                return "the node half ended after its pvserver"
            time.sleep(0.25)
        leave_note("closed the connection while ParaView was waiting on it")
        print(
            f"the pvserver at {pvserver} closed the connection while ParaView was "
            "waiting on it; the node half was held up, so it is ended",
            flush=True,
        )
        end(parent)
        return "ended the node half"

def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="clio-paraview-watchdog")
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--pvserver", required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--fd", required=True)
    args = parser.parse_args(argv)
    # pvpython holds print output back; the node half's log is on descriptor 2.
    try:
        sys.stdout = open(2, "w", buffering=1, closefd=False)  # noqa: SIM115
    except OSError:
        pass
    if args.fd == "share":
        # pvpython replaces sys.stdin, so the share is read from the descriptor.
        with open(0, "rb", closefd=False) as given:
            probe = socket.fromshare(given.read())
    else:
        probe = socket.socket(fileno=int(args.fd))
    try:
        watch(probe, args.parent, args.pvserver, args.record)
    finally:
        probe.close()


if __name__ == "__main__":
    main()
