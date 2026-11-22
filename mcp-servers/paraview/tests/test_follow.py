"""Following a growing source from the front: the view descriptor, watch_steps as an
MCP task and as a direct call, and a field measured over time. No ParaView needed."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp_tasks.client import call_tool_task
from test_front import SUMMARY, _FakeAttacher, _FakeClient

from paraview_mcp import follow, present
from paraview_mcp.node.disk import disk_key
from paraview_mcp.server import mcp


def steps_result(
    count: int,
    index: int | None = None,
    growing: bool = True,
) -> dict[str, Any]:
    """What the node's ``steps`` returns: ``count`` steps half a time unit apart."""
    times = [0.5 * i for i in range(count)]
    shown = count - 1 if index is None else index
    return {
        "scene": "run-1",
        "growing": growing,
        "steps": [
            {"index": i, "time": t, "seen": 100.0 + i} for i, t in enumerate(times)
        ],
        "time": {
            "steps": count,
            "index": shown,
            "time": times[shown],
            "latest_index": count - 1,
            "latest_time": times[-1],
            "follow_latest": index is None,
            "newer_steps": count - 1 - shown,
        },
        "last_growth": None,
        "checked": 0.0,
    }


class _Run:
    """A fake node and clock: each fetch returns the next result, the last one again."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.now = 0.0
        self.fetched: list[int] = []
        self.reports: list[tuple[int, str]] = []
        self.totals: list[int | None] = []

    async def fetch(self, after: int) -> dict[str, Any]:
        self.fetched.append(after)
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]

    async def report(self, written: int, total: int | None, message: str) -> None:
        self.reports.append((written, message))
        self.totals.append(total)

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds

    async def watch(self, **options: Any) -> dict[str, Any]:
        return await follow.watch(
            self.fetch, self.report, clock=self.clock, sleep=self.sleep, **options
        )


async def test_each_new_step_is_reported_once_following_then_parked() -> None:
    run = _Run(
        [
            steps_result(2),
            steps_result(2),
            steps_result(4),
            steps_result(5, index=3),
        ]
    )
    outcome = await run.watch(idle_seconds=3)
    started = (
        "Watching scene run-1: 2 step(s) so far, the source is still being written; "
        "the view follows it."
    )
    parked = (
        "Step 5 at time 2 is written; the view stays parked at step 4, 1 newer step(s) "
        "exist."
    )
    # Progress is the number of steps written so far, so it only rises.
    assert run.reports == [
        (2, started),
        (3, "Step 3 at time 1 is written; the view follows it."),
        (4, "Step 4 at time 1.5 is written; the view follows it."),
        (5, parked),
    ]
    assert run.totals == [None] * 4
    assert outcome["ended"] == "idle"
    assert outcome["new_steps"] == 3
    assert [s["index"] for s in outcome["steps"]] == [2, 3, 4]
    assert outcome["time"]["newer_steps"] == 1
    # It asks only for steps it has not reported.
    assert run.fetched[:3] == [0, 2, 2]
    assert run.fetched[-1] == 5
    text = follow.watch_text(outcome)
    assert "3 new step(s), step 3 (time 1) to step 5 (time 2)" in text
    assert "no new step came for the idle time" in text
    assert "\u00b7" not in text


async def test_expected_steps_give_the_total_and_end_the_watch_when_written() -> None:
    """With the descriptor's follow.expected_steps the progress has a total, each
    message says "of N", the watch polls at the descriptor's pace, and it ends as
    complete once the last step is written."""
    follow_part = {"expected_steps": 4, "poll_seconds": 7}
    run = _Run(
        [
            {**steps_result(1), "follow": follow_part},
            {**steps_result(3), "follow": follow_part},
            {**steps_result(4), "follow": follow_part},
        ]
    )
    outcome = await run.watch(idle_seconds=60)
    assert [w for w, _ in run.reports] == [1, 2, 3, 4]
    assert run.totals == [4, 4, 4, 4]
    assert run.reports[0][1].startswith("Watching scene run-1: 1 of 4 step(s) so far")
    assert run.reports[1][1].startswith("Step 2 of 4 at time 0.5 is written")
    assert run.reports[-1][1].startswith("Step 4 of 4 at time 1.5 is written")
    assert outcome["ended"] == "complete"
    assert outcome["expected_steps"] == 4
    assert outcome["seconds"] == 14
    assert "all 4 expected steps are written" in follow.watch_text(outcome)


async def test_a_source_not_marked_growing_ends_the_watch_at_once() -> None:
    run = _Run([steps_result(3, growing=False)])
    outcome = await run.watch()
    assert outcome["ended"] == "not_growing"
    assert outcome["new_steps"] == 0
    assert "not marked as growing" in run.reports[0][1]
    assert "no new step" in follow.watch_text(outcome)


async def test_after_reports_steps_already_written_and_max_seconds_stops() -> None:
    run = _Run([steps_result(3)])
    outcome = await run.watch(after=1, max_seconds=4, idle_seconds=60)
    assert [count for count, _ in run.reports] == [1, 2, 3]
    assert outcome["ended"] == "max_seconds"
    assert outcome["seconds"] >= 4


async def test_open_view_takes_a_view_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from paraview_mcp import server

    client = _FakeClient({"open_view": SUMMARY})
    monkeypatch.setattr(server, "_attachers", {"": _FakeAttacher(client)})
    descriptor = {
        "path": "/scratch/run/out_*.vti",
        "format": "vti",
        "reader": "XMLImageDataReader",
        "reader_properties": {"TimeArray": "None"},
        "growing": True,
        "name": "a field a package also reports",
    }
    async with Client(mcp) as mcp_client:
        await mcp_client.call_tool("open_view", {"view": descriptor})
        with pytest.raises(ToolError, match="bad_arguments"):
            await mcp_client.call_tool(
                "open_view", {"path": "/other.vti", "view": descriptor}
            )
    assert client.calls == [
        (
            "open_view",
            {
                "path": "/scratch/run/out_*.vti",
                "growing": True,
                "new": False,
                "format": "vti",
                "reader": "XMLImageDataReader",
                "reader_properties": {"TimeArray": "None"},
                "descriptor": {"name": "a field a package also reports"},
            },
        )
    ]


def test_the_descriptor_model_takes_every_field_of_the_vendored_schema() -> None:
    """The model follows CLIO's view descriptor v1: each property of the vendored
    schema is a field, and the schema's own examples load and keep their parts."""
    from paraview_mcp.descriptor import SCHEMA_ID, ViewDescriptor, schema

    vendored = schema()
    assert vendored["properties"]["schema"]["const"] == SCHEMA_ID
    fields = {
        field.alias or name for name, field in ViewDescriptor.model_fields.items()
    }
    assert set(vendored["properties"]) <= fields
    for part in ("follow", "time", "display", "producer"):
        nested = vendored["properties"][part].get("properties") or {}
        model = ViewDescriptor.model_fields[part].annotation
        part_fields = next(
            a for a in getattr(model, "__args__", (model,)) if a is not type(None)
        ).model_fields
        assert set(nested) <= set(part_fields), part
    for example in vendored.get("examples") or []:
        view = ViewDescriptor.model_validate(example)
        assert view.path == example["path"]
        assert view.schema_ == SCHEMA_ID
        extras = view.extras() or {}
        if "follow" in example:
            assert extras["follow"] == example["follow"]
        if "display" in example:
            assert extras["display"] == example["display"]
    # Only the path is required, so a host that knows nothing else can open data.
    assert ViewDescriptor.model_validate({"path": "/x.vti"}).extras() is None


class _GatedClient(_FakeClient):
    """Each ``steps`` command waits for the test to let it through."""

    def __init__(self, sequence: list[dict[str, Any]]) -> None:
        super().__init__({})
        self.sequence = sequence
        self.gate = threading.Semaphore(0)

    def command(
        self, name: str, args: dict, timeout: float = 900.0, by: str = "agent"
    ) -> dict:
        assert name == "steps"
        self.calls.append((name, args))
        assert self.gate.acquire(timeout=20)
        return self.sequence.pop(0) if len(self.sequence) > 1 else self.sequence[0]


async def test_watch_steps_as_a_task_reports_each_step_as_its_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RL44: each new step goes out as task progress, the task's status message."""
    from paraview_mcp import server

    client = _GatedClient(
        [
            steps_result(2),
            steps_result(3),
            steps_result(4, index=2),
            steps_result(4, index=2, growing=False),
        ]
    )
    monkeypatch.setattr(server, "_attachers", {"": _FakeAttacher(client)})
    monkeypatch.setattr(follow, "POLL_SECONDS", 0.01)
    async with Client(mcp) as mcp_client:
        task = await call_tool_task(mcp_client, "watch_steps", {"idle_seconds": 60})

        async def status_says(text: str) -> None:
            seen = None
            for _ in range(1000):
                status = await task.status()
                seen = status.status_message
                if seen and text in seen:
                    assert status.status == "working"
                    return
                await asyncio.sleep(0.01)
            raise AssertionError(f"status never said {text!r}; last: {seen!r}")

        client.gate.release()
        await status_says("Watching scene run-1: 2 step(s) so far")
        client.gate.release()
        await status_says("Step 3 at time 1 is written; the view follows it.")
        client.gate.release()
        await status_says(
            "Step 4 at time 1.5 is written; the view stays parked at step 3"
        )
        client.gate.release()
        result = await task.result()
    outcome = result.structured_content
    assert outcome is not None
    assert outcome["ended"] == "not_growing"
    assert outcome["new_steps"] == 2
    assert "2 new step(s)" in getattr(result.content[0], "text", "")


async def test_watch_steps_called_directly_sends_progress_notifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host without tasks still gets each step, as progress on the call."""
    from paraview_mcp import server

    client = _FakeClient({})
    sequence = [steps_result(1), steps_result(2), steps_result(2, growing=False)]
    client.command = lambda name, args, timeout=900.0, by="agent": (  # type: ignore[method-assign]
        sequence.pop(0) if len(sequence) > 1 else sequence[0]
    )
    monkeypatch.setattr(server, "_attachers", {"": _FakeAttacher(client)})
    monkeypatch.setattr(follow, "POLL_SECONDS", 0.01)
    messages: list[str] = []

    async def progress(value: float, total: float | None, message: str | None) -> None:
        messages.append(message or "")

    # A legacy (pre-2026) connection cannot negotiate tasks: the call runs directly.
    async with Client(mcp, progress_handler=progress, mode="legacy") as mcp_client:
        result = await mcp_client.call_tool("watch_steps", {"max_seconds": 30})
    assert result.structured_content is not None
    assert result.structured_content["new_steps"] == 1
    assert any("Step 2 at time 0.5" in m for m in messages), messages


def test_series_text_lists_each_step() -> None:
    result: dict[str, Any] = {
        "object": "o1",
        "label": "out_*.vti",
        "field": "u",
        "association": "POINTS",
        "component": "magnitude",
        "steps": 3,
        "series": [
            {
                "index": i,
                "time": 0.5 * i,
                "min": 0.0,
                "max": 1.0 + i,
                "mean": 0.5 * i,
                "std": 0.1,
                "median": 0.5,
                "count": 8,
                "finite": 8,
            }
            for i in range(3)
        ],
    }
    text = present.series_text(result)
    assert text.splitlines()[0] == (
        "u (points, magnitude) on o1 out_*.vti, over 3 of 3 step(s):"
    )
    assert "- step 3 (time 1): min 0, max 3, mean 1, std 0.1" in text
    long = {**result, "series": result["series"] * 30, "steps": 90}
    lines = present.series_text(long).splitlines()
    assert len(lines) <= present.SERIES_LINES + 2
    assert lines[-1].startswith("Every 3th step is written here")
    assert "\u00b7" not in text


def test_a_growing_folder_or_pattern_is_seen_to_grow(tmp_path: Path) -> None:
    """An ADIOS2 BP output is a folder: an entry added or grown is a change. A
    pattern changes when a file is added."""
    folder = tmp_path / "out.bp"
    folder.mkdir()
    (folder / "data.0").write_bytes(b"x")
    before = disk_key(str(folder))
    (folder / "md.idx").write_bytes(b"y")
    grown = disk_key(str(folder))
    assert grown != before
    (folder / "data.0").write_bytes(b"xx")
    assert disk_key(str(folder)) != grown
    (tmp_path / "s_0.vti").write_text("a")
    pattern = str(tmp_path / "s_*.vti")
    one = disk_key(pattern)
    (tmp_path / "s_1.vti").write_text("b")
    assert len(disk_key(pattern)) == 2 and disk_key(pattern) != one
