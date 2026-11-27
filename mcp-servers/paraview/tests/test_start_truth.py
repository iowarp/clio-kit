"""A start that stalls, a node half that went, and frames that show nothing (W17 gaps)."""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest
from fastmcp import Client
from test_front import SUMMARY, _FakeAttacher, _FakeClient

from paraview_mcp import attach, present
from paraview_mcp.attach import StartLook, wait_for_start
from paraview_mcp.errors import ParaViewError
from paraview_mcp.node import drawn
from paraview_mcp.node.main import prune_slots
from paraview_mcp.server import mcp


class _Clock:
    """A monotonic clock the wait moves forward, instead of sleeping."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.now = 1000.0
        monkeypatch.setattr(attach.time, "monotonic", lambda: self.now)

    def cancel(self) -> threading.Event:
        clock = self

        class _Cancel(threading.Event):
            def wait(self, timeout: float | None = None) -> bool:
                clock.now += timeout or 0
                return self.is_set()

        return _Cancel()


def _looks(*seen: StartLook) -> Any:
    items = list(seen)

    def look() -> StartLook:
        return items.pop(0) if len(items) > 1 else items[0]

    return look


def test_a_start_that_keeps_logging_is_waited_for_past_any_old_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock(monkeypatch)
    cancel = clock.cancel()
    told: list[str] = []
    count = {"n": 0}

    def look() -> StartLook:
        count["n"] += 1
        if count["n"] > 400:  # 400 s, well past the old 180 s limit
            return StartLook({"code_hash": "h"}, None, True, count["n"], "ready")
        return StartLook(None, None, True, count["n"], f"line {count['n']}")

    record = wait_for_start(look, "h", "here", "/l", 30.0, cancel, told.append)
    assert record == {"code_hash": "h"}
    assert told[0].startswith("Starting ParaView here, 0 s: line 1")
    assert len(told) >= 300


def test_a_quiet_start_is_stopped_after_the_stall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock(monkeypatch)
    stopped: list[bool] = []
    look = _looks(StartLook(None, None, True, 10, "loading paraview"))
    with pytest.raises(ParaViewError) as caught:
        wait_for_start(
            look,
            "h",
            "here",
            "/l",
            30.0,
            clock.cancel(),
            stop=lambda: stopped.append(1),
        )
    assert caught.value.code == "node_start_stalled"
    assert stopped == [1]
    assert "loading paraview" in caught.value.message


def test_a_join_that_hangs_says_so_in_plain_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock(monkeypatch)
    look = _looks(StartLook(None, None, True, 10, "joining pvserver at 127.0.0.1:9"))
    with pytest.raises(ParaViewError) as caught:
        wait_for_start(look, "h", "here", "/l", 30.0, clock.cancel())
    assert caught.value.code == "pvserver_join_stalled"
    assert caught.value.message.startswith("ParaView could not reach its server")
    assert "127.0.0.1:9" not in caught.value.message


def test_a_process_that_ended_fails_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(monkeypatch)
    look = _looks(
        StartLook(None, None, False, 10, "x", tail="Aborted\nstd::length_error")
    )
    with pytest.raises(ParaViewError) as caught:
        wait_for_start(look, "h", "here", "/l", 600.0, clock.cancel())
    assert caught.value.code == "node_failed"
    assert "std::length_error" in caught.value.message
    assert clock.now - 1000.0 < 10


def test_a_cancelled_start_stops_the_node(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(monkeypatch)
    cancel = clock.cancel()
    stopped: list[bool] = []

    def look() -> StartLook:
        if clock.now > 1005:
            cancel.set()
        return StartLook(None, None, True, int(clock.now), "busy")

    with pytest.raises(ParaViewError) as caught:
        wait_for_start(
            look, "h", "here", "/l", 60.0, cancel, stop=lambda: stopped.append(1)
        )
    assert caught.value.code == "cancelled"
    assert stopped == [1]


def test_each_pvserver_has_its_own_node_record() -> None:
    assert attach.node_base(None) == "node"
    one, two = attach.node_base("127.0.0.1:1"), attach.node_base("127.0.0.1:2")
    assert one.startswith("node-pv") and one != two
    script = attach.probe_script("3.3.0", "/f", one)
    assert f'"$D/{one}.json"' in script and f'"$D/{one}.lock"' in script
    assert attach.node_slot({"NODE": '{"pid": 1}'}, one) == (one, {"pid": 1})
    assert attach.node_slot({"NS": "5", "LOCKED": "1"}, one) == (f"{one}-5", None)


def test_a_lost_note_says_the_pvserver_went() -> None:
    note = json.dumps({"reason": "the pvserver at h:1 closed the connection"})
    error = attach.gone_error(note, "", "on ares", "tgv-1")
    assert error.code == "pvserver_lost"
    assert "closed the connection" in error.message
    assert "open_view(scene='tgv-1')" in error.message
    other = attach.gone_error("", "Segmentation fault", "on ares")
    assert other.code == "node_lost" and "Segmentation fault" in other.message


def test_prune_keeps_a_pvservers_own_slot(tmp_path: Any) -> None:
    import os

    for name in ("node-123", "node-w9", "node-pv0a1b2c3d"):
        (tmp_path / f"{name}.lock").write_text("")
        (tmp_path / f"{name}.log").write_text("x")
        os.utime(tmp_path / f"{name}.lock", (0, 0))
    removed = prune_slots(tmp_path, "node")
    assert sorted(removed) == ["node-123", "node-w9"]
    assert (tmp_path / "node-pv0a1b2c3d.log").exists()


def test_compare_counts_differing_pixels() -> None:
    background = bytes([10, 20, 30] * 100)
    same = drawn.compare(background, background, 3)
    assert same == 0.0
    painted = bytearray(background)
    for i in range(0, 30 * 3, 3):
        painted[i] = 250
    assert drawn.compare(bytes(painted), background, 3) == pytest.approx(0.3)
    # A faint difference (antialiasing noise) is not drawing.
    faint = bytes(v + 5 for v in background)
    assert drawn.compare(faint, background, 3) == 0.0


def test_count_sees_a_one_pixel_line_in_a_large_frame() -> None:
    width, height = 1280, 800
    background = bytes([82, 87, 110]) * (width * height)
    outline = bytearray(background)
    column = 377
    for row in range(137, 662):
        outline[(row * width + column) * 3] = 255
    assert drawn.count(bytes(outline), background, 3) == 662 - 137
    assert drawn.count(background, background, 3) == 0
    sparse = {"fraction": 0.0004, "pixels": 525, "blank": False, "sparse": True}
    assert present.blank_warning(sparse) == ""
    assert "thin lines" in present.drawn_text(sparse, None)


def test_drawn_text_never_claims_a_blank_picture() -> None:
    blank = {"fraction": 0.0, "blank": True}
    assert "blank" in present.drawn_text(blank, None)
    assert "Do not tell the person" in present.drawn_text(blank, None)
    assert present.blank_warning(blank).startswith("This frame is blank")
    assert present.blank_warning({"fraction": 0.4, "blank": False}) == ""
    seen = present.drawn_text(
        {"fraction": 0.42, "blank": False},
        {"viewers": 1, "frames_shown": 3, "last_frame_age": 2.0, "lost": None},
    )
    assert "42%" in seen and "3 picture(s) shown" in seen
    lost = present.drawn_text(None, {"viewers": 0, "lost": "the pvserver went"})
    assert "The view stopped: the pvserver went." in lost


async def test_open_view_on_another_pvserver_opens_beside_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from paraview_mcp import server

    first = _FakeAttacher(_FakeClient({"open_view": SUMMARY}))
    other_summary = {
        **SUMMARY,
        "scene": "tgv-2",
        "drawn": {"fraction": 0.0, "blank": True},
    }
    second = _FakeAttacher(_FakeClient({"open_view": other_summary, "capture": {}}))
    monkeypatch.setattr(server, "_attachers", {"": first, "127.0.0.1:9": second})
    monkeypatch.setattr(server, "_current", "")
    monkeypatch.setattr(server, "_scene_keys", {})
    async with Client(mcp) as client:
        one = await client.call_tool("open_view", {"path": "/d/cube.vti"})
        two = await client.call_tool(
            "open_view", {"path": "/run/data.bp5", "pvserver": "127.0.0.1:9"}
        )
        bad = await client.call_tool(
            "open_view", {"pvserver": "h; rm -rf ~:1"}, raise_on_error=False
        )
        await client.call_tool("open_view", {"scene": "cube-1"})
    assert one.meta["ai.iowarp.clio/reopen"] == {"arguments": {"scene": "cube-1"}}
    assert two.meta["ai.iowarp.clio/reopen"] == {
        "arguments": {"scene": "tgv-2", "pvserver": "127.0.0.1:9"}
    }
    assert "nothing shows yet" in two.content[0].text
    assert bad.is_error
    # The first view's scene goes back to its own attacher.
    assert len(first.link.client.calls) == 2
    assert first.opened == [True, True] and second.opened == [True]


async def test_capture_leads_with_a_blank_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from paraview_mcp import server

    client = _FakeClient(
        {
            "capture": {
                "png_base64": "iVBORw0KGgo=",
                "path": "/f/c.png",
                "camera": SUMMARY["camera"],
                "drawn": {"fraction": 0.0, "blank": True},
                "shown": {"viewers": 1, "frames_shown": 0, "lost": None},
            },
            "set_timestep": {"time": {"steps": 1}},
        }
    )
    fake = _FakeAttacher(client)
    monkeypatch.setattr(server, "_attachers", {"": fake})
    monkeypatch.setattr(server, "_current", "")
    async with Client(mcp) as mcp_client:
        captured = await mcp_client.call_tool("capture", {})
        await mcp_client.call_tool("set_timestep", {"index": 1})
    assert captured.content[0].text.startswith("This frame is blank")
    # Only open_view may start a node half; other tools report one that went.
    assert fake.opened == [False, False]
