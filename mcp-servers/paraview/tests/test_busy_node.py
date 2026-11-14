"""A node half that is busy drawing is waited for, not taken for dead and restarted.

A software volume render holds the node's loop for tens of seconds. Restarting the node
then (the old behaviour) dropped the open scene, so the next call found "No scene is open".
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator

import pytest

from paraview_mcp import present
from paraview_mcp.attach import Attacher
from paraview_mcp.config import Settings, Target
from paraview_mcp.link import LinkError, NodeClient


@pytest.fixture
def silent_port() -> Iterator[int]:
    """A port that accepts connections and never answers, like a node mid-render."""

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    held: list[socket.socket] = []
    stop = threading.Event()

    def accept() -> None:
        server.settimeout(0.1)
        while not stop.is_set():
            try:
                held.append(server.accept()[0])
            except OSError:
                continue

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield server.getsockname()[1]
    stop.set()
    thread.join()
    for conn in held:
        conn.close()
    server.close()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_a_silent_node_is_busy(silent_port: int) -> None:
    with pytest.raises(LinkError) as caught:
        NodeClient(silent_port, "t")._request("GET", "/clio/health", timeout=0.3)
    assert caught.value.busy


def test_a_refused_link_is_not_busy() -> None:
    with pytest.raises(LinkError) as caught:
        NodeClient(_free_port(), "t")._request("GET", "/clio/health", timeout=0.3)
    assert not caught.value.busy


class _Client:
    def __init__(self, failures: list[LinkError]) -> None:
        self.failures = failures
        self.calls = 0

    def health(self) -> dict[str, str]:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return {"ok": "yes"}


def _attacher(busy_timeout: float) -> Attacher:
    return Attacher(Settings(target=Target(), busy_timeout=busy_timeout))


def test_a_busy_node_is_waited_for() -> None:
    client = _Client(
        [LinkError("timed out", busy=True), LinkError("timed out", busy=True)]
    )
    assert _attacher(60)._answers(client)  # type: ignore[arg-type]
    assert client.calls == 3


def test_a_refused_node_fails_at_once() -> None:
    client = _Client([LinkError("refused")])
    assert not _attacher(60)._answers(client)  # type: ignore[arg-type]
    assert client.calls == 1


def test_a_node_busy_past_the_limit_is_taken_for_hung() -> None:
    client = _Client([LinkError("timed out", busy=True)] * 3)
    assert not _attacher(0)._answers(client)  # type: ignore[arg-type]


def test_a_slow_frame_is_reported_with_its_cause() -> None:
    view = {
        "mode": "remote",
        "reason": "volume",
        "volume_grids": ["vtkRectilinearGrid"],
    }
    assert present.frame_note({**view, "frame_seconds": 0.4}) == ""
    slow = present.view_line({**view, "frame_seconds": 17.25})
    assert "its frames take up to 17.25 s each to draw" in slow
    assert "vtkRectilinearGrid" in slow and "ResampleToImage" in slow
    image = present.view_line(
        {**view, "volume_grids": ["vtkImageData"], "frame_seconds": 3.0}
    )
    assert "up to 3 s" in image and "ResampleToImage" not in image
    note = present.capture_note(17.25, view)
    assert note.startswith(" It took 17.25 s to draw, because ParaView draws a volume")
    assert note.endswith("much faster.")
    assert present.capture_note(17.25, {"mode": "local"}) == " It took 17.25 s to draw."
    assert present.capture_note(1.0, view) == ""


def test_a_solid_colour_is_shown_in_the_object_line() -> None:
    line = present.object_line(
        {
            "id": "o3",
            "label": "snow shell",
            "type": "Contour",
            "display": {"representation": "Surface", "solid_color": [1.0, 0.5, 0.0]},
        }
    )
    assert "solid colour #ff8000" in line
