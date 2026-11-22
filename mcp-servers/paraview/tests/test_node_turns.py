"""A command comes before the follow: the node half's loop gives it its turns."""

from __future__ import annotations

import asyncio
import types
from typing import Any

from paraview_mcp.node import app as node_app


class _State:
    def __enter__(self) -> Any:
        return self

    def __exit__(self, *_: object) -> bool:
        return False


def _node(engine: Any) -> Any:
    node = object.__new__(node_app.NodeApp)
    node.engine = engine
    node.lost = None
    node.waiting = 0
    node.server = types.SimpleNamespace(state=_State())
    return node


def test_the_follow_skips_its_turns_while_a_command_waits() -> None:
    checks: list[int] = []
    node = _node(types.SimpleNamespace(check_growth=lambda: checks.append(1)))

    async def go() -> tuple[int, int]:
        node.waiting = 1
        task = asyncio.create_task(node.watch_growth(interval=0.01))
        await asyncio.sleep(0.15)
        held = len(checks)
        node.waiting = 0
        await asyncio.sleep(0.15)
        task.cancel()
        return held, len(checks)

    held, after = asyncio.run(go())
    assert held == 0 and after > 0


def test_a_command_counts_as_waiting_until_its_work_starts() -> None:
    node = _node(types.SimpleNamespace())
    seen: list[int] = []

    async def run(body: Any, started: Any) -> str:
        seen.append(node.waiting)
        started()
        seen.append(node.waiting)
        started()
        seen.append(node.waiting)
        return "done"

    node._run_command = run
    assert asyncio.run(node._execute({})) == "done"
    assert seen == [1, 0, 0] and node.waiting == 0

    async def fails(body: Any, started: Any) -> str:
        raise RuntimeError("bad")

    node._run_command = fails
    try:
        asyncio.run(node._execute({}))
    except RuntimeError:
        pass
    assert node.waiting == 0
