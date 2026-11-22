"""The pvserver watchdog without ParaView: a socket pair stands in for the link."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

from paraview_mcp.node import watchdog


def _watch(
    record: Path, alive: list[bool], ended: list[int]
) -> tuple[socket.socket, threading.Thread, list[str]]:
    near, far = socket.socketpair()
    out: list[str] = []

    def run() -> None:
        out.append(
            watchdog.watch(
                near,
                4242,
                "127.0.0.1:11111",
                record,
                look=0.05,
                grace=0.3,
                alive=lambda pid: alive[0],
                end=ended.append,
            )
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return far, thread, out


def test_a_held_up_node_half_is_ended_with_the_reason(tmp_path: Path) -> None:
    """The live failure: the pvserver goes while a ParaView call blocks the node half,
    so the child leaves the note the node could not, drops its record and ends it."""
    record = tmp_path / "node-pv1.json"
    record.write_text("{}")
    ended: list[int] = []
    far, thread, out = _watch(record, [True], ended)
    far.close()
    thread.join(10)
    assert out == ["ended the node half"] and ended == [4242]
    note = json.loads((tmp_path / "node-pv1.lost.json").read_text())
    assert "127.0.0.1:11111" in note["reason"] and "waiting on it" in note["reason"]
    assert not record.exists()


def test_a_node_half_that_stops_by_itself_keeps_its_own_note(tmp_path: Path) -> None:
    """A free loop says so itself: the child then ends nothing and keeps its note."""
    record = tmp_path / "node-pv1.json"
    own = tmp_path / "node-pv1.lost.json"
    own.write_text(json.dumps({"reason": "its own", "scene": "s1"}))
    alive = [True]
    ended: list[int] = []
    far, thread, out = _watch(record, alive, ended)
    far.close()
    alive[0] = False
    thread.join(10)
    assert out[0].startswith("the node half ended") and ended == []
    assert json.loads(own.read_text()) == {"reason": "its own", "scene": "s1"}


def test_a_node_half_that_dies_with_its_pvserver_leaves_a_note(tmp_path: Path) -> None:
    """ParaView may abort the node half once the socket closes: the note says why."""
    record = tmp_path / "node-pv1.json"
    record.write_text("{}")
    alive = [True]
    ended: list[int] = []
    far, thread, out = _watch(record, alive, ended)
    far.close()
    alive[0] = False
    thread.join(10)
    assert out and ended == []
    note = json.loads((tmp_path / "node-pv1.lost.json").read_text())
    assert "ended with it" in note["reason"] and not record.exists()


def test_data_on_the_link_is_not_a_close(tmp_path: Path) -> None:
    """Unread data from the pvserver is only peeked at, never taken from ParaView."""
    record = tmp_path / "node-pv1.json"
    record.write_text("{}")
    alive = [True]
    ended: list[int] = []
    near, far = socket.socketpair()
    far.sendall(b"frame")
    out: list[str] = []
    thread = threading.Thread(
        target=lambda: out.append(
            watchdog.watch(
                near,
                1,
                "x:1",
                record,
                look=0.05,
                grace=0.1,
                alive=lambda pid: alive[0],
                end=ended.append,
            )
        ),
        daemon=True,
    )
    thread.start()
    thread.join(0.5)
    assert thread.is_alive() and ended == []
    alive[0] = False
    thread.join(10)
    assert near.recv(5) == b"frame"
    far.close()
    near.close()
