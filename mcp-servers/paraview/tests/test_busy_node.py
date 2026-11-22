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


def _attacher() -> Attacher:
    return Attacher(Settings(target=Target()))


def test_a_busy_node_is_waited_for_and_the_wait_is_said() -> None:
    client = _Client([LinkError("timed out", busy=True)] * 40)
    attacher = _attacher()
    said: list[str] = []
    attacher._call = (None, said.append)
    assert attacher._answers(client)  # type: ignore[arg-type]
    # There is no limit: forty busy checks in a row are still waited for.
    assert client.calls == 41
    assert len(said) == 40
    assert "it is busy (a long render holds it), not gone" in said[0]


def test_a_refused_node_fails_at_once() -> None:
    client = _Client([LinkError("refused")])
    assert not _attacher()._answers(client)  # type: ignore[arg-type]
    assert client.calls == 1


def test_the_person_ends_a_busy_wait_by_cancelling() -> None:
    from paraview_mcp.errors import ParaViewError

    client = _Client([LinkError("timed out", busy=True)] * 3)
    attacher = _attacher()
    stop = threading.Event()
    stop.set()
    attacher._call = (stop, None)
    with pytest.raises(ParaViewError) as caught:
        attacher._answers(client)  # type: ignore[arg-type]
    assert caught.value.code == "cancelled"
    assert client.calls == 1


class _JobNode:
    """A stand-in node half that speaks the job protocol: a call is pending for
    ``pending`` asks, then answers; ``hold`` asks are not answered in time at all."""

    def __init__(self, pending: int, hold: int = 0) -> None:
        import http.server
        import json

        self.pending, self.hold = pending, hold
        self.asks: list[dict] = []
        self.cancelled: list[str] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/clio/cancel":
                    outer.cancelled.append(body["job"])
                    answer: dict = {"ok": True}
                else:
                    outer.asks.append(body)
                    if outer.hold:
                        outer.hold -= 1
                        outer.release.wait(5)
                        return
                    if outer.pending:
                        outer.pending -= 1
                        done = len(outer.asks)
                        answer = {
                            "pending": True,
                            "seconds": done,
                            "progress": {"done": done, "total": 9},
                        }
                    else:
                        answer = {"result": {"job": body["job"]}}
                data = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.release = threading.Event()
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


def test_a_long_call_is_asked_after_until_it_answers() -> None:
    """A call runs on the node under one id, however long it takes: each ask that
    finds it pending is progress, and every ask carries the same id."""
    node = _JobNode(pending=5)
    seen: list[dict] = []
    try:
        result = NodeClient(node.port, "t").command(
            "measure_over_time", {"field": "u"}, progress=seen.append, wait=0.01
        )
    finally:
        node.close()
    assert len(node.asks) == 6
    assert len({ask["job"] for ask in node.asks}) == 1
    assert result == {"job": node.asks[0]["job"]}
    assert [note["done"] for note in seen] == [1, 2, 3, 4, 5]


def test_a_node_that_holds_its_loop_is_waited_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ask the node does not answer in time means it is busy, not gone: the
    client says so and asks again, and the call still ends with its result."""
    from paraview_mcp import link

    monkeypatch.setattr(link, "ANSWER_GRACE", 0.2)
    node = _JobNode(pending=0, hold=2)
    seen: list[dict] = []
    try:
        result = NodeClient(node.port, "t").command(
            "capture", {}, progress=seen.append, wait=0.01
        )
    finally:
        node.close()
    assert result["job"] == node.asks[0]["job"]
    assert len(seen) == 2
    assert "it is busy (a long render holds it), not gone" in seen[0]["note"]


def test_a_cancelled_call_is_cancelled_on_the_node_too() -> None:
    from paraview_mcp.errors import ParaViewError

    node = _JobNode(pending=1000)
    stop = threading.Event()

    def progress(note: dict) -> None:
        if note["done"] >= 3:
            stop.set()

    try:
        with pytest.raises(ParaViewError) as caught:
            NodeClient(node.port, "t").command(
                "export_animation", {}, stop=stop, progress=progress, wait=0.01
            )
    finally:
        node.close()
    assert caught.value.code == "cancelled"
    assert node.cancelled == [node.asks[0]["job"]]


def test_a_node_that_is_gone_ends_the_call_at_once() -> None:
    with pytest.raises(LinkError) as caught:
        NodeClient(_free_port(), "t").command("capture", {}, wait=0.01)
    assert not caught.value.busy


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


def test_an_image_slice_view_says_why_it_is_drawn_by_paraview() -> None:
    growing = present.view_line({"mode": "remote", "reason": "growing"})
    assert "each new step arrives as one image" in growing
    line = present.view_line({"mode": "remote", "reason": "slice"})
    assert "an image slice is shown, which only ParaView draws" in line
