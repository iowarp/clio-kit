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
    node.shared = node_app.Shared()
    node.engine = engine
    node.lost = None
    node.waiting = 0
    node.name = "view"
    node.sockets = 0
    node.growth_busy = False
    node.server = types.SimpleNamespace(state=_State())
    return node


def _growing(checks: list[int], stages: int = 1) -> Any:
    def growth_stages() -> Any:
        for _ in range(stages - 1):
            yield "stage"
        checks.append(1)
        return None

    return types.SimpleNamespace(
        growth_stages=growth_stages,
        opening=False,
        scene=None,
        behind=lambda: False,
    )


def test_the_follow_skips_its_turns_while_a_command_waits() -> None:
    checks: list[int] = []
    node = _node(_growing(checks))

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


def test_a_growth_check_gives_the_loop_back_between_its_stages() -> None:
    checks: list[int] = []
    node = _node(_growing(checks, stages=4))
    ticks: list[bool] = []

    async def other() -> None:
        while True:
            ticks.append(node.growth_busy)
            await asyncio.sleep(0)

    async def go() -> None:
        task = asyncio.create_task(other())
        await node._grow_staged()
        task.cancel()

    asyncio.run(go())
    assert checks == [1] and node.growth_busy is False
    assert sum(ticks) >= 3


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
